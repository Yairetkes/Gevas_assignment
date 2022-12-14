import cv2 as cv
import numpy as np
import random
import math

def turn_to_green(img, height, width):    
    for i in range(height):
        for j in range(width):
            img[i][j][0] = 0
            img[i][j][1] = 255
            img[i][j][2] = 0

    return img


def draw_line(img, height, width):    
    for i in range(width):
        img[height//2 - 70][i][0] = 0
        img[height//2 - 70][i][1] = 0
        img[height//2 - 70][i][2] = 255

    return img

def canny_detection(img):
    edges = cv.Canny(img,25,200)

    return edges

def showImage(image, win_name="image"):
    cv.imshow(win_name, image)
    # (this is necessary to avoid Python kernel form crashing)
    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows()


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        prob = 0.3
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


def find_and_draw_lines(img):
    # this function Finds lines in a binary image using the standard Hough transform
    edges1 = canny_detection(img)

    lines = cv.HoughLines(edges1, 1, np.pi / 180, 150, None, 0, 0)

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(img, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return img

def main():
    img = cv.imread("photos\lines_img.jpg")
    
    img = find_and_draw_lines(img)
    showImage(img)

if __name__ == "__main__":
    main()
