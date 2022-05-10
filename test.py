from pickletools import float8
import sys
sys.path.append('core')

import argparse
import os
import cv2 as cv
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

####-------so works on Jerry's computer-------###
# from core.utils import flow_viz    
# from core.utils.utils import InputPadder     
# DEVICE = 'cpu'  
#################################################


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().detach().numpy()
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # print(flo.shape)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return flo[:, :, [2,1,0]]/255.0

def load_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    vid_cap = cv.VideoCapture("input/water_level.mp4")

    _, prev_frame = vid_cap.read()    
    prev_frame = load_image(prev_frame)

    while True:
        success, current_frame = vid_cap.read()
        box_img = current_frame.copy()  #for slidedeck
        crop = np.zeros_like(current_frame)
        crop[:,180:350] = 255
        current_frame = cv.bitwise_and(crop,current_frame)
        
        if not success:
            break
        temp_img = current_frame.copy()
        cv.imshow("Source",current_frame)
        
        current_frame = load_image(current_frame)        

        padder = InputPadder(current_frame.shape)
        image1, image2 = padder.pad(current_frame, prev_frame)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        result = viz(image1, flow_up)

        prev_frame = current_frame
        result = result*255
        heat_map_img = result.astype(np.uint8)
        
        orginal_gray = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
        original_grey = cv.cvtColor(box_img, cv.COLOR_BGR2GRAY) #for slidedeck
        gray_img = cv.cvtColor(heat_map_img, cv.COLOR_BGR2GRAY)
        ret,thresh_img = cv.threshold(gray_img , 230 ,255,cv.THRESH_BINARY)

        masked_img = cv.bitwise_and(orginal_gray,thresh_img)
        crop = np.zeros_like(masked_img)
        crop[:,180:350] = 255
        masked_img = cv.bitwise_and(crop,masked_img)


        _ , masked_thresh = cv.threshold(masked_img , 230 ,255,cv.THRESH_BINARY)
        kernel = np.ones((7,7),np.uint8)


        masked_thresh_gradient = cv.morphologyEx(masked_thresh, cv.MORPH_GRADIENT, kernel)
        # masked_thresh_gradient_closing = cv.morphologyEx(masked_thresh_gradient, cv.MORPH_CLOSE, kernel)
        # masked_thresh_gradient_closing_canny = cv.Canny(masked_thresh_gradient_closing,100,200,2)
        masked_thresh_gradient_canny = cv.Canny(masked_thresh_gradient,100,200,2)


        contours, hierarchy  = cv.findContours(masked_thresh_gradient_canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        blank_testing = np.zeros_like(temp_img)
        # avg_arc_length = np.array([])
        avg_arc_length = []
        for i in contours:
            
            if 300 > cv.arcLength(i,False)>250:
                cv.drawContours(blank_testing, [i], -1, (0,255,0), 3)
                avg_arc_length.append(cv.arcLength(i,False))
                cv.imshow("mid_drawing",blank_testing)
                print(cv.arcLength(i,False))  

                rect = cv.minAreaRect(i)
                box = cv.boxPoints(rect)
                print(box)
                box = np.int0(box)
                cv.drawContours(temp_img,[box],0,(0,0,255),2)
                cv.drawContours(box_img,[box],0,(0,0,255),2)    #for slidedeck
                warp_image(box,masked_thresh)
                cv.waitKey(0)


            if 250> cv.arcLength(i,False) > 110:
                cv.drawContours(blank_testing, [i], -1, (255,255,255), 3)
                avg_arc_length.append(cv.arcLength(i,False))
                cv.imshow("mid_drawing",blank_testing)
                print(cv.arcLength(i,False))
                
                rect = cv.minAreaRect(i)
                box = cv.boxPoints(rect)
                print(box)
                box = np.int0(box)
                cv.drawContours(temp_img,[box],0,(0,0,255),2)
                cv.drawContours(box_img,[box],0,(0,0,255),2)    #for slidedeck
                warp_image(box,masked_thresh)
                cv.waitKey(0)


        # cv.imshow("grey_img", original_grey)   #for slidedeck
        # cv.imshow("Canny",canny_img)
        cv.imshow("Masked", masked_img)
        # cv.namedWindow("Masked_thresh",cv.WINDOW_NORMAL)
        # cv.imshow("Masked_thresh",masked_thresh)
        # cv.imshow("masked_thresh_gradient_canny",masked_thresh_gradient_canny)  #for slidedeck
        # cv.imshow("Warped",warped_img)
        # cv.imshow("Thresh", thresh_img)
        cv.imshow("Heat Map", heat_map_img)
        cv.imshow("Original w/Box", box_img)    #for slidedeck
        # cv.imshow("masked_thresh_closing",masked_thresh_closing)
        # cv.imshow("masked_thresh_closing_canny",masked_thresh_closing_canny)
        # cv.imshow("blank_testing",blank_testing)
        # cv.imshow("temp_img",temp_img)


        if cv.waitKey(0) == ord('q'):
            break

    vid_cap.release()
    cv.destroyAllWindows()


    # with torch.no_grad():
    #     images = glob.glob(os.path.join(args.path, '*.png')) + \
    #              glob.glob(os.path.join(args.path, '*.jpg'))

    #     print(images)
        
    #     images = sorted(images)
    #     for imfile1, imfile2 in zip(images[:-1], images[1:]):
    #         image1 = load_image(imfile1)
    #         image2 = load_image(imfile2)

    #         padder = InputPadder(image1.shape)
    #         image1, image2 = padder.pad(image1, image2)

    #         flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    #         viz(image1, flow_up)

def warp_image(input_pts,frame):
    inputs = np.float32(input_pts)

    # corners = np.int0(corners)
    x_arr = []
    y_arr  = []

    for i in inputs:
        x,y = i.ravel()
        x_arr.append(x)
        y_arr.append(y)
    
    x_max = np.max(x_arr[:])
    y_max = np.max(y_arr[:])
    x_min = np.min(x_arr[:])
    y_min = np.min(y_arr[:])

    my_corners =[]
    my_corners.append([x_min,y_min]) # top left    
    my_corners.append([x_max,y_min]) # top right
    my_corners.append([x_max,y_max]) # bottom right
    my_corners.append([x_min,y_max]) # Bottom left

    my_corners = np.float32(my_corners)
    output_pts = np.array([[0,0],
                    [200,0],
                    [200,100],
                    [0,100]],dtype=np.float32)
    M = cv.getPerspectiveTransform(my_corners,output_pts)
    out = cv.warpPerspective(frame,M,(200, 100),flags=cv.INTER_LINEAR)
    cv.imshow("warped_image",out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)