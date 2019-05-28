
import sys
import time

import cv2
import os

import datetime



def main():

    cap = cv2.VideoCapture("video/Uni+ DVR_ch5_main_20190511120000_20190511130000.dav")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ret,img = cap.read()

    out = cv2.VideoWriter('avi_video/Uni+ DVR_ch5_main_20190511120000_20190511130000.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))
    while  ret:
        ret,img = cap.read()
        frame_width,frame_height,c = img.shape

        cv2.imwrite("out.jpg",img)

        cv2.imshow("window",img)

    cap.release()
    out.release()

main()
