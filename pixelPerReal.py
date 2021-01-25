import os
import datetime
import math
import warnings
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="filmage001.avi")
args = vars(ap.parse_args())

def main():

    video_capture = cv2.VideoCapture(args["input"])
    constFrame=0
    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if ret != True:
            break
        constFrame=frame
        cv2.namedWindow("Pix", 0);
        cv2.resizeWindow('Pix', 2574,1440);
        cv2.imshow('Pix', constFrame)
        cv2.circle(constFrame, (695, 1045), 1, (0,255,255), 30)




    cv2.imshow('YOLO3_Deep_SORT', constFrame)
    # video_capture.release()
    #
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
