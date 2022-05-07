#!/usr/bin/env python3

import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'




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

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey(2000)

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

    vid_cap = cv2.VideoCapture("input/water_level.mp4")

    _, prev_frame = vid_cap.read()    
    prev_frame = load_image(prev_frame)

    while True:
        success, current_frame = vid_cap.read()
        cv2.imshow("Output", current_frame)
        original = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if not success:
            break

        current_frame = load_image(current_frame)        

        padder = InputPadder(current_frame.shape)
        image1, image2 = padder.pad(current_frame, prev_frame)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        result = viz(image1, flow_up)
        
        result = result*255
        # print(result*255)

        gray = cv2.cvtColor(result.astype('uint8'), cv2.COLOR_BGR2GRAY)       


        temp = np.argwhere(gray < 230)
        print(temp.shape)

        blank = np.ones_like(gray) * 255
        blank[temp[:,0], temp[:,1]] = 0

        output = cv2.bitwise_and(blank, original)
        # for x,y in temp:
        #     blank[x,y] = 255

        prev_frame = current_frame
        # cv2.imshow("out 2", output)
        
        cv2.imshow("gray", gray)
        if cv2.waitKey(24) == ord('q'):
            break

    vid_cap.release()
    cv2.destroyAllWindows()



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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)
