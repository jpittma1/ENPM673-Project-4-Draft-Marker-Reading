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
thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('Vessel Draft Mark-(480p).mp4')

vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')
    
####----Setup For Optical Flow-------#########
cap = cv2.VideoCapture(0)
frame_previous = cap.read()[1]
gray_previous = cv2.cvtColor(frame_previous, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame_previous)
hsv[:, :, 1] = 255
param = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.1,
    'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
}
############################################

###############----Output Video Parameters-----##############
make_video=True    #Toggle this to make an output video

if make_video == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps_out = 29
    fps = vid.get(cv2.CAP_PROP_FPS)
    # print("frames_per_second is: ", fps)
    fps_out=fps
    videoname=('proj4_output')
    output = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (720, 480))
    print("Making a video...this will take some time...")
###########################################################

while(vid.isOpened()):
    count+=1
    success,image = vid.read()
    
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
            
        '''Hough Transform to Find lowest corner point'''
        img=image.copy()
        minLineLength = 5
        maxLineGap = 100
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        # print("lines ", lines)
        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        if count == 8:
            cv2.imwrite('proj4_houghlines.jpg',img)
        
        
        '''Hough Transform Detection'''
        
        '''Optical Flow?'''
        flow = cv2.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

        '''Detect/determine Lowest Numbers/Drafts; Place box around the number'''


        '''Translate numbers using CNN?'''


        '''Print text on screen of what draft number is'''
        
        ###----Save frame to create Output Video----####
        if make_video == True:
            output.write(img)
    print("Count is: ", count)  #

vid.release()
output.release()
cv2.destroyAllWindows()
plt.close('all')