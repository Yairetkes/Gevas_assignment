import cv2
# import dlib
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
import mediapipe as mp


# plt settings for ipython
plt.ion()
plt.figure()

# drawing constants
font = cv2.FONT_HERSHEY_SIMPLEX

# Landmark ROI
# 160 & 158 correspond to top l->r, 144 & 153 bottom l->r

leftEyeLandmarks = [160, 158, 153, 144]
# same pattern as above
rightEyeLandmarks = [385, 387, 380, 373]

class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# display image
def showImage(image, win_name="image"):
    cv2.imshow(win_name, image)
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()

# captures the face ROI, then finds facial landmarks on that face
def getFaceLandmarks(image_file):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    image = cv2.imread(image_file)
    
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # get the landmarks
    landmarks = results.multi_face_landmarks
    # print(landmarks)

    if len(landmarks) > 0:
        # return the landmarks
        return landmarks
    else:
        return None

# draws the facial landmarks on the face
def drawEyeLandmarks(
    im, landmarks: list[Landmark], landmarksToDraw, color=(0, 255, 0), radius=1, test=False
):
    points = landmarks
    for landmark in landmarksToDraw:
        p = points[landmark]
        cv2.circle(im, (p.x, p.y), radius, color, -1)
        if test:
            cv2.putText(
                im,
                str(landmark),
                (p.x - 10, p.y - 10),
                font,
                0.5,
                color,
                radius,
                cv2.LINE_AA,
            )
    showImage(im)

# based on the eye radius, and distance between the eyes try and find circles that best represent
# the eye region
def houghTransformCircleDetector(image, eye_radius, eye_distance):
    # parameters
    edge_sensitivity_threshold = 450
    edge_accumulator_threshold = 10
    inverse_accumulator_ratio = 1
    min_circle_distance = eye_distance / 2
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(gray, 5)
    # Apply hough transform on the images
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        inverse_accumulator_ratio,
        min_circle_distance,
        param1=edge_sensitivity_threshold,
        param2=edge_accumulator_threshold,
        minRadius=int(eye_radius - eye_radius / 4),
        maxRadius=int(eye_radius + eye_radius / 2),
    )
    return circles

# given circles draw them on the image
def drawCircles(img, circles):
    color = (0, 255, 0)
    thickness = 1
    # Draw detected circles
    if circles is not None:
        # round the values
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, color, thickness)
        showImage(img)

# eyes are tough to find with the current version of the landmark detector and houghCircles
# so we end up opting to find more circles, and then filtering out the circles to find the one that
# best represents the eye
def filterOutCircles(circles, landmarks: list[Landmark]):
    circles = np.uint16(np.around(circles))
    # average x & y of eye region landmarks
    left_distances = []
    right_distances = []
    left_x = left_y = right_x = right_y = 0
    for i in leftEyeLandmarks:
        left_x += landmarks[i].x
        left_y += landmarks[i].y
    for i in rightEyeLandmarks:
        right_x += landmarks[i].x
        right_y += landmarks[i].y
    left_x /= 4
    right_x /= 4
    left_y /= 4
    right_y /= 4
    # find the circle with the smallest distance from the left & right eye region
    for circle in circles[0, :]:
        left_distances.append(
            math.sqrt((circle[0] - left_x) ** 2 + (circle[1] - left_y) ** 2)
        )
        right_distances.append(
            math.sqrt((circle[0] - right_x) ** 2 + (circle[1] - right_y) ** 2)
        )
    left_eye = circles[0, :][left_distances.index(min(left_distances))]
    right_eye = circles[0, :][right_distances.index(min(right_distances))]
    return [left_eye, right_eye]

# creates a histogram of the hues in the eye region
def createHistogramOfColorsInEyeRegion(img, circles, eye_radius, test=False, radius_constant = 0):
    # Convert image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to float so to create a mask
    gray = gray / 255.0
    for circle in circles:
        center = (circle[0], circle[1])
        # get all pixels inside eye radius
        cv2.circle(gray, center, int(eye_radius + eye_radius / 2) - radius_constant, 2, -1)
    if test:
        showImage(gray)
    eye_pixels = np.where(gray == 2)
    # get histrogram of colors for eye_pixels
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    eye_pixel_hue = h[eye_pixels]
    # h[0]/h[180] = red, h[60] = green, h[120] = blue
    histogram = np.histogram(eye_pixel_hue, np.arange(180))
    return histogram, eye_pixels, eye_pixel_hue, h, s, v

# finds the largets n consecutive numbers in an array when the array is regarded as circular
def findLargestNConsecutiveBins(histogram, n):
    bins = histogram[0]
    maxSum = 0
    maxBins = []
    lengthOfBins = len(bins)
    for i in range(0, lengthOfBins):
        # max numbers can start at the end of the array and wrap to the beggining so we need to account for those
        # ex: [44, 2, 3, 40, 42] -> we need two arrays [0] & [3:4]
        overflow = i + n > lengthOfBins
        maxNum = i + n if not overflow else lengthOfBins
        maxNum2 = 0 if not overflow else (i + n) % lengthOfBins
        binRange = np.concatenate((np.arange(i, maxNum), np.arange(0, maxNum2)))
        newSum = sum(bins[binRange])

        if newSum > maxSum:
            maxSum = newSum
            maxBins = binRange

    return maxBins

# changes the most dominate hues of the eye region to another color space
def changeHueOfHistogram(
    histogram, eye_pixels, eye_pixel_hue, h, s, v, color, test=False
):
    largestBins = findLargestNConsecutiveBins(histogram, 30)
    if test:
        print(histogram[0])
        print(largestBins)
    if color == "brown":
        destinationHue = np.concatenate((np.arange(170, 180), np.arange(0, 20)))
    elif color == "blue":
        destinationHue = np.arange(100, 131)
    elif color == "green":
        destinationHue = np.arange(40, 71)
    # loop through the larget bins of hue colors, and map them to the destination color
    for i, bin in enumerate(largestBins):
        eye_pixel_hue[eye_pixel_hue == bin] = destinationHue[i]
    # update hue of eye pixel region
    h[eye_pixels] = eye_pixel_hue
    # recreate hsv image with updated hues, covert back to BGR and display
    newImage = cv2.merge([h, s, v])
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    # showImage(newImage)
    return newImage

def landmarks_2_coordinates(landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList, image_file) -> list[Landmark]:
    if not landmarks:
        return None
    
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    
    # edit each landmark - from normalised [0,1] value to exact pixel coordinates location.
    new_landmarks = []
    for facial_landmarks in landmarks:
        for i in range(468):
            point = facial_landmarks.landmark[i]
            x = int(point.x * width)
            y = int(point.y * height)
            
            new_landmarks.append(Landmark(x,y))
    
    return new_landmarks

def changeEyeColor(img_file, eye_color, test=False, radius_constant = 0):
    # grab the landmark points
    landmarks = getFaceLandmarks(img_file)
    points: list[Landmark] = landmarks_2_coordinates(landmarks, img_file)
    
    # from landmark points get the eye measurements
    # TODO - choose the right radius and distance, maybe with different landmarks so it will be generalised to other photos.
    eye_radius = np.uint16((points[158].x - points[160].x) / 2) 
    eye_distance = np.uint16(points[385].x - points[158].x)
    # eye_radius = np.uint16((points[158].x - points[160].x) / 2 - 2) 
    # eye_distance = np.uint16(points[385].x - points[158].x + 11)

    image = cv2.imread(img_file)

    # if we're in test mode display the landmarks on the image
    if test:
        drawEyeLandmarks(image.copy(), points, leftEyeLandmarks + rightEyeLandmarks)
        print("eye_radius", eye_radius, "eye_distance", eye_distance)

    circles = houghTransformCircleDetector(image, eye_radius, eye_distance)
    # drawCircles(img.copy(), circles, eye_radius)

    try:
        circles = filterOutCircles(circles, points)
    except Exception as e:
        print(e)
        raise e

    histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistogramOfColorsInEyeRegion(
        image, circles, eye_radius, test, radius_constant
    )
    return changeHueOfHistogram(
        histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, test
    )

def main():
    pic_file = "photos/blue1.jpg"
    
    image = cv2.imread(pic_file)

    radius_const = 4 # this constant should improve iris segmentation results. 4 seem to work good
    
    # TODO: implementing Raz's idea - remove space between circle and eye arc, using the landmark in between 2 upper eye used landmarks.

    destination_color = "brown"

    new_image = changeEyeColor(pic_file, destination_color, True, radius_const)
    plt.figure()
    # plt.subplot(121)
    # plt.imshow(image[:, :, ::-1])
    # plt.subplot(122)
    # plt.imshow(new_image[:, :, ::-1])

    output_file_name = "output_changed_2_" + destination_color + ".jpg"
    # output_file_name = "output.jpg"
    cv2.imwrite(output_file_name, new_image)
    showImage(new_image, "new image")


if __name__ == "__main__":
    main()