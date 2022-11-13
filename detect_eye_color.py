# import glob
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.cluster import KMeans

def dominantColors(image_file: str, num_of_clusters: int):
    
        #read image
        img = cv.imread(image_file)
        
        #convert to rgb from bgr
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        edited_image = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = num_of_clusters)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        dominant_colors = kmeans.cluster_centers_
        
        #save labels
        lables = kmeans.labels_
        
        #returning after converting to integer from float
        return dominant_colors.astype(int)

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    #return colors[count.argmax() - 40]
    return colors[300:350]


def detect_circles(img_file):
    
    #TODO: detect iris and crop it.

    img1 = cv.imread(img_file)
    img = cv.imread(img_file ,0)
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 135, 255, cv.THRESH_BINARY)

    cv.imshow('thresh',thresh)
    cv.waitKey(0)

    # Create mask
    height,width = img.shape
    mask = np.zeros((height,width), np.uint8)

    cv.imshow('mask',mask)
    cv.waitKey(0)

    edges = cv.Canny(thresh, 100, 200)

    # edges = cv.GaussianBlur()

    #cv.imshow('detected ',gray)
    cimg=cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1.3, 150, param1 = 70, param2 = 30, minRadius = 10, maxRadius = 30)

    print("circles = ", circles)
    for i in circles[0,:]:
        print("*** i = ", i)
        i[2]=i[2]+4
        # Draw on mask
        cv.circle(mask,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),thickness=-1)

    # Copy that image using that mask
    masked_data = cv.bitwise_and(img1, img1, mask=mask)

    cv.imshow('masked_data', masked_data)
    cv.waitKey(0)

    # Apply Threshold
    _,thresh = cv.threshold(mask,1,255,cv.THRESH_BINARY)

    cv.imshow('thresh', thresh)
    cv.waitKey(0)

    # Find Contour
    contours, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #contours = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    print("*** contours = ", contours)
    x,y,w,h = cv.boundingRect(contours[0]) # last voted comment in https://stackoverflow.com/questions/43852023/detecting-small-circles-using-houghcircles-opencv
    # might help

    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]

    #Code to close Window
    cv.imshow('detected Edge',img1)
    cv.imshow('Cropped Eye',crop)
    cv.waitKey(0)


def analyze(img_file_name: str):
    img = cv.imread(img_file_name)

    cv.imshow("cropped image", img)
    
    ##(2) convert to hsv-space, then split the channels
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
   
    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv.inRange(hsv, (36, 25, 25), (86, 255,255))
    # mask = cv.inRange(hsv, (20, 20, 20), (80, 255,255)) # original (with blur)
    mask = cv.inRange(hsv, (30, 30, 30), (80, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    # mask of blue (36,25,25) ~ (86, 255,255)
    mask = cv.inRange(hsv, (70, 70, 70), (130, 255,255))

    ## slice the blue
    imask = mask>0
    blue = np.zeros_like(img, np.uint8)
    blue[imask] = img[imask]

    black_pix_green_mask = np.sum(green == 0)
    black_pix_blue_mask = np.sum(blue == 0)

    print("The eye's colors of the choosed photo is - ")
    if black_pix_green_mask < black_pix_blue_mask:
        print("Green !!")
        mask = cv.inRange(hsv, (20, 20, 20), (70, 255,255))
    else:
        print("Blue !!")


    # Get the BGR value
    # blur=cv.GaussianBlur(img,(5,5),0)
    mask_upstate=cv.bitwise_and(img, img, mask=mask)
    mean = cv.mean(mask_upstate)
    multiplier = float(mask.size)/cv.countNonZero(mask)
    mean = tuple([multiplier * x for x in mean])
    print("mean[0:3] = ", mean[0:3])

    #rgb_val = matplotlib.colors.hsv_to_rgb(mean[0:3])
    #rgb_val = matplotlib.colors.hsv_to_rgb(mean[0:3])
    #rgb_val = (rgb_val[2], rgb_val[1], rgb_val[0])
    # Create a blank 300x300 black image
    image = np.zeros((300, 300, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    #mean = (mean[2], mean[1], mean[0], mean[3])
    image[:] = mean[0:3]

    cv.imshow('mean eyes color',image)
    cv.waitKey(0)

    # TODO: make sure that the mean of the eye really is mathematical mean of samplings.
    # TODO: fill eyes with different color.



def get_image(img_file_name: str):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')


    my_path = r"C:\Users\y0e00l5\Documents\Gevas_assignment\photos" + img_file_name
    img = cv.imread(my_path)

    cv.imshow('choosed photo',img)
    cv.waitKey(0)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        choosed_eyes = eyes[0]

        for (ex,ey,ew,eh) in eyes:
            if choosed_eyes[0] < ex:
                choosed_eyes = [ex,ey,ew,eh]

        (ex,ey,ew,eh) = choosed_eyes
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)
        cropped_eye = img[ey+y:ey + eh +y, ex+x:ex + ew+x]
        cv.imwrite("output_cropped_eye.jpg", cropped_eye)

    #print("**** unique_count_app(cropped_eye) = " ,unique_count_app(cropped_eye))

    #detect_circles("output_cropped_eye.jpg")
    analyze("output_cropped_eye.jpg")

    #print("&&&&&&& dominantColors(output_cropped_eye.jpg, 5) = ", dominantColors("output_cropped_eye.jpg", 3))

    # for i, col in enumerate(['b', 'g', 'r']):
    #     hist = cv.calcHist([cropped_eye], [i], None, [256], [0, 256])
    #     plt.plot(hist, color = col)
    #     plt.xlim([0, 256])
    
    #plt.show()

    #cv.imshow("cropped", cropped)

    #cv.imshow('img', img)
    #cv.waitKey(0)
    
    
    #return img


def main():
    get_image("/green1.jpg")


if __name__ == "__main__":
    main()


