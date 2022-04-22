#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
import torch
from collections import OrderedDict

from utils import *
from utils.utils import *
from utils.flow_viz import *
from utils import flow_viz
# from flow_viz import *
# from utils.flow_viz import flow_to_image, flow_uv_to_colors
from utils.frame_utils import *
from utils.augmentor import *

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

# try:
#     autocast = torch.cuda.amp.autocast
# except:

# dummy autocast for PyTorch < 1.6
class autocast:
    def __init__(self, enabled):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        # if args.small:
        #     self.hidden_dim = hdim = 96
        #     self.context_dim = cdim = 64
        #     args.corr_levels = 4
        #     args.corr_radius = 3
        
        # else:
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        # args.corr_levels = 4
        # args.corr_radius = 4

        # if 'dropout' not in self.args:
        #   self.args.dropout = 0
        self.dropout = 0

        # if 'alternate_corr' not in self.args:
            # self.args.alternate_corr = False
        self.alternate_corr = False

        # feature network, context network, and update block
        # if args.small:
        #     self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
        #     self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
        #     self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        # else:
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        mixed_precision=True
        alternate_corr = False
        corr_levels = 4
        corr_radius = 4
        
        # run the feature network
        # with autocast(enabled=self.args.mixed_precision):
        with autocast(enabled=mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=corr_radius)

        # run the context network
        with autocast(enabled=mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def vizualize_flow(img, flo, save, counter):
    # permute the channels and change device is necessary
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 30    #match input video
    
    
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)

    # concatenate, save and show images
    img_flo = np.concatenate([img, flo], axis=0)
    
    height_ = img_flo.shape[0]
    width_ = img_flo.shape[1]
    
    print("size of img_flow is ", height_, width_)
    output_OF = cv2.VideoWriter("proj4_OF_output.avi", fourcc, fps_out, (640, 480))
    # output_OF = cv2.VideoWriter("proj4_OF_output.avi", fourcc, fps_out, (640, 960))
    
    if save:
        if counter = 8: #Don't want to save all images everytime
            cv2.imwrite(f"OF_frame_{str(counter)}.jpg", img_flo)
        # cv2.imwrite(f"OF_frames/frame_{str(counter)}.jpg", img_flo)
        output_OF.write(np.uint8(img_flo))
    # cv2.imshow("Optical Flow", img_flo / 255.0) ##REALLY SLOW!!!
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        return False
    return True

def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def opticalFlow(mode, iter, video):
    # get the RAFT model
    model = RAFT(mode)
    # load pretrained weights
    pretrained_weights = torch.load(mode, map_location=torch.device('cpu'))
    # torch.load with map_location=torch.device('cpu')
    save = True
    if save:
        if not os.path.exists("OF_frames"):
            os.mkdir("OF_frames")

    # if torch.cuda.is_available():
    #     device = "cuda"
    #     # parallel between available GPUs
    #     model = torch.nn.DataParallel(model)
    #     # load the pretrained weights into model
    #     model.load_state_dict(pretrained_weights)
    #     model.to(device)
    # else:
    device = "cpu"
    # change key names for CPU runtime
    pretrained_weights = get_cpu_model(pretrained_weights)
    # load the pretrained weights into model
    model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    # video_path = video
    # capture the video and get the first frame
    cap = video
    ret, frame_1 = cap.read()

    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)

    counter = 0
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break
            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=iter, test_mode=True)
            # transpose the flow output and convert it into numpy array
            ret = vizualize_flow(frame_1, flow_up, save, counter)
            if not ret:
                break
            frame_1 = frame_2
            counter += 1

            # output.write(frame_1)
            # output_video=append.
    output_OF.release()
    # return output

