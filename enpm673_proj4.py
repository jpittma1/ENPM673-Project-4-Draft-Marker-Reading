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
# import scipy
# from scipy import fft, ifft
# from numpy import histogram_bin_edges, linalg as LA
# import matplotlib.pyplot as plt
# import sys
# import math
# import os
# from os.path import isfile, join
# import timeit
# import argparse
import torch
# import models.crnn
# from models.crnn import CRNN
from crnn import *
from DNN_functions import *


#********************************************
# Requires the following in same folder to run:
#1) "Vessel Draft Mark-(480p).mp4"
#********************************************

###TODO: Cut???
# ############ Add argument parser for command line arguments ############
# parser = argparse.ArgumentParser(
#     description="Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
#                 "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)"
#                 "The OCR model can be obtained from converting the pretrained CRNN model to .onnx format from the github repository https://github.com/meijieru/crnn.pytorch"
#                 "Or you can download trained OCR model directly from https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing")
# parser.add_argument('--input',
#                     help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# parser.add_argument('--model', '-m',default="CRNN", required=False,
# # parser.add_argument('--model', '-m', default="CRNN(32, 1, 37, 256)",required=True,
#                     help='Path to a binary .pb file contains trained detector network.')
# parser.add_argument('--ocr', default="crnn.onnx",
#                     help="Path to a binary .pb or .onnx file contains trained recognition network", )
# parser.add_argument('--width', type=int, default=320,
#                     help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
# parser.add_argument('--height', type=int, default=320,
#                     help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
# parser.add_argument('--thr', type=float, default=0.5,
#                     help='Confidence threshold.')
# parser.add_argument('--nms', type=float, default=0.4,
#                     help='Non-maximum suppression threshold.')
# args = parser.parse_args()

# model = crnn.CRNN(32, 1, 37, 256)
# model_path = './data/crnn.pth'
# model = CRNN(32, 1, 37, 256)
# # args.model = CRNN(32, 1, 37, 256)
# model.load_state_dict(torch.load('crnn.pth'))
# dummy_input = torch.randn(1, 1, 32, 100)
# torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)

###---Arguments for DNN/CNN---###
# Read and store arguments
# confThreshold = args.thr
# nmsThreshold = args.nms
# modelDetector = args.model
# modelDetector = CRNN(32, 1, 37, 256)
# modelRecognition = args.ocr

confThreshold=0.5
nmsThreshold=0.4
inpWidth = 320
inpHeight = 320
modelDetector = "east.pb"
modelRecognition = "crnn.onnx"


# Load network
print("[INFO] loading CRNN text detector model...")
# net = cv2.dnn.readNet(args["east"])
detector = cv2.dnn.readNet(modelDetector)
recognizer = cv2.dnn.readNet(modelRecognition)

# Create a new named window
kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)
outNames = []
outNames.append("feature_fusion/Conv_7/Sigmoid")
outNames.append("feature_fusion/concat_3")

'''Read in the video'''
#---Input Video Parameters---###
threshold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('Vessel Draft Mark-(480p).mp4')


vid.set(1,start)
count = start

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')

##TODO: Optical Flow??###
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
    output_dnn=cv2.VideoWriter("proj4_DNN_output.avi", fourcc, fps_out, (640, 480))
    print("Making video(s)...this will take some time...")
###########################################################

while(vid.isOpened()):
    count+=1
    success,image = vid.read()
    
    # height, width = image.shape[:2]
    # print("Height: ", height)
    # print("width: ", width)
    
     # Get frame height and width
    height_ = image.shape[0]
    width_ = image.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)
    
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
            cv2.imwrite('proj4_testImage.jpg', image)
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
        
        ##TODO: Optical Flow??###
        '''Optical Flow???'''
        # flow = cv2.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

        '''Translate numbers using CNN/DNN CRNN Text recognition model'''
        # Create a 4D blob from frame.
        frame=image.copy()
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Run the detection model
        detector.setInput(blob)
        outs = detector.forward(outNames)
        
        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            # get cropped image using perspective transform
            if modelRecognition:
                cropped = fourPointsTransform(frame, vertices)
                cropped = cv2.cvtColor(cropped, cv.COLOR_BGR2GRAY)

                # Create a 4D blob from cropped image
                blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
                recognizer.setInput(blob)

                # Run the recognition model
                result = recognizer.forward()

                '''Print text on screen of what draft number is'''
                # decode the result into text
                wordRecognized = decodeText(result)
                cv2.putText(frame, wordRecognized, (int(vertices[1][0]), int(vertices[1][1])), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 0))

            '''Place box around the number'''
            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 1)
                
        cv2.imshow(kWinName, frame)

        if count == 8:
            cv2.imwrite('proj4_DNN_result.jpg',frame)
        
        ###----Save frame to create Output Videos----####
        if make_video == True:
            output_hough.write(img)
            output_contour.write(img_plus_contours)
            output_dnn.write(frame)
    
    else: #read video is not success; exit loop
        vid.release()
            
    # print("Count is: ", count)  #657
    


vid.release()
output_hough.release()
output_contour.release()
output_dnn.release()
cv2.destroyAllWindows()
plt.close('all')