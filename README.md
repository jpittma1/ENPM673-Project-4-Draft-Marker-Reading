Project 4 (Water Level Detection for Ship Hull)
Conducted for ENPM673 Spring 2022

Sri Sai Charan V 
Vaishanth Ramaraj
Pranav Limbekar
Yash Kulkarni
Jerry Pittman, Jr.

-------------
# RAFT
This repository contains the source code for RAFT:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1 as well as with CPU computer.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Run our Code on the Vessel Movie
if have GPU, run as is:
python test.py --model=models/raft-things.pth --path=demo-frames

if don't have a GPU:
1) Comment out line 20
2) Uncomment line 23

## Hardware Implementation using Raspberry Pi



