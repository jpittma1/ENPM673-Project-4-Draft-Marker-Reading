# Project 4 (Water Level Detection for Ship Hull)
Conducted for ENPM673 Spring 2022

Sri Sai Charan V 
Vaishanth Ramaraj
Pranav Limbekar
Yash Kulkarni
Jerry Pittman, Jr.

-------------
## RAFT
This repository contains the source code and paper for RAFT:

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

## To Run our Code on the Vessel Movie to detect and mark water levels
If have GPU, run code as is:
```Shell
python project4.py --model=models/raft-things.pth --path=demo-frames
```
if don't have a GPU:
1) Comment out line 20
2) Uncomment line 23
3) Run Code:
```Shell
python project4.py --model=models/raft-things.pth --path=demo-frames
```
## Hardware Implementation using Raspberry Pi

```Shell
python project4_train.py 
```

Sadly, for the the hardware implemention that due to slow processing onboard the robot with raspberry pi the number detected in the video doesn’t change. Level detection with water especially with flow takes a great deal amount of processing.

