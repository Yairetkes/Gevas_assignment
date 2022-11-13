import cv2 as cv
import numpy as np 

def main():
    my_path = r"C:\Users\y0e00l5\Documents\Gevas_assignment\photos\pic.jpg"
    img = cv.imread(my_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    cv.waitKey(0)
    np.
    


if __name__ == "__main__":
    main()


