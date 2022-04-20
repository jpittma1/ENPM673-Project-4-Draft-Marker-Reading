#!/usr/bin/env python3

# ENPM673 Spring 2022
# Section 0101
# Jerry Pittman, Jr. UID: 117707120; jpittma1@umd.edu
# Sri Sai Charan V (UID: 117509755); svellise@umd.edu
# Vaishanth Ramaraj (UID: 118154237); vrmrj@umd.edu
# Pranav Limbekar (UID: 118393711); pranav05@umd.edu
# Yash Kulkarni (UID: 117386967); ykulkarn@umd.edu

#Project #4: Water level detections from ship hull

import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import histogram_bin_edges, linalg as LA
import matplotlib.pyplot as plt
import sys
import math
import os
from os.path import isfile, join
import timeit

#********************************************
# Requires the following in same folder to run:
#1) "Vessel Draft Mark-(480p).mp4"
#********************************************

'''Read in the video'''
#---Input Video Parameters---###
threshold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('Vessel Draft Mark-(480p).mp4')


vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')
    
# ####----Setup For Optical Flow-------#########
# cap = cv2.VideoCapture(0)
# frame_previous = cap.read()[1]
# gray_previous = cv2.cvtColor(frame_previous, cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame_previous)
# hsv[:, :, 1] = 255
# param = {
#     'pyr_scale': 0.5,
#     'levels': 3,
#     'winsize': 15,
#     'iterations': 3,
#     'poly_n': 5,
#     'poly_sigma': 1.1,
#     'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
# }
# ############################################

###############----Output Video Parameters-----##############
make_video=True    #Toggle this to make an output video

if make_video == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps_out = 29
    fps = vid.get(cv2.CAP_PROP_FPS)
    # print("frames_per_second is: ", fps)
    fps_out=fps
    output_hough = cv2.VideoWriter("proj4_houghTransform_output.avi", fourcc, fps_out, (640, 480))
    output_contour=cv2.VideoWriter("proj4_Findcontours_output.avi", fourcc, fps_out, (640, 480))
    print("Making a video...this will take some time...")
###########################################################

while(vid.isOpened()):
    count+=1
    success,image = vid.read()
    
    # height, width = image.shape[:2]
    # print("Height: ", height)
    # print("width: ", width)
    
    if success:
        '''Pre-processing'''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        '''Harris corner detection'''
        # gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        # dst = np.uint8(dst)
        
        if count == 8:
            cv2.imwrite('proj4_harrisCorners.jpg', dst)
        
        '''Canny operator??'''
        edges = cv2.Canny(gray,100,200, apertureSize = 3)

        if count == 8:
            cv2.imwrite('proj4_cannyCorners.jpg', edges)
            
        '''Draw over the edges using blur and findContours'''
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgray= cv2.medianBlur(imgray,5)
        ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print("(frame) contours found are ", contours)
        
        
        img_plus_contours=image.copy()
        cv2.drawContours(img_plus_contours, contours,-1,(0,255,0), 4) #Green
            
        '''Draw over the edges using blur and Hough Transform'''
        img=image.copy()
        # minLineLength = 1
        # maxLineGap = 5
        # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        
        lines = cv2.HoughLines(edges,1,np.pi/180,10)
        # print("lines ", lines)
        if lines is not None:
            # print("lines is not None")
            # for x1,y1,x2,y2 in lines[0]: #for HoughLinesP
            #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                # print("x1, y1, x2, y2", x1, " ", y1, " ", x2, " ", y2)
                if y2>300:
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        if count == 8:
            cv2.imwrite('proj4_houghlines.jpg',img)
            cv2.imwrite('proj4_contours.jpg',img_plus_contours)
        
        
        '''Hough Transform Detection'''
        
        '''Optical Flow?'''
        # flow = cv2.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

        '''Detect/determine Lowest Numbers/Drafts; Place box around the number'''


        '''Translate numbers using CNN?'''


        '''Print text on screen of what draft number is'''
        
        ###----Save frame to create Output Video----####
        if make_video == True:
            output_hough.write(img)
            output_contour.write(img_plus_contours)
    
    else: #read video is not success; exit loop
        vid.release()
            
    # print("Count is: ", count)  #657

vid.release()
output_hough.release()
output_contour.release()
cv2.destroyAllWindows()
plt.close('all')