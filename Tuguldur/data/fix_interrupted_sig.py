# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:01:39 2024

@author: animu
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

fs = 1000

dir = Path('W:/home/nero/datasets/RGS14/OverlappingOR/OS_Ephys_RGS14_Rat6_373726_SD2_OR_11-12_02_2020/')
path1 = dir/'2020-02-06_14-56-02_Post_Trial5'
path2 = dir/'2020-02-06_16-32-47_Post_Trial5_2'

lfp1 = loadmat(next(path1.glob("*HPC*continuous*")))['HPC']
lfp2 = loadmat(next(path2.glob("*HPC*continuous*")))['HPC']

gap = 3*60*60*fs - len(lfp1) - len(lfp2)
fill = np.zeros((gap, 1)).astype(int)

lfp = np.concatenate((lfp1, fill, lfp2))

hypno1 = loadmat(next(path1.glob('*states*')))['states']
hypno2 = loadmat(next(path2.glob('*states*')))['states']

fill2 = np.zeros(gap//fs).astype(np.uint8)
hypno = np.concatenate((np.squeeze(hypno1), fill2, np.squeeze(hypno2)))

data = {'HPC':lfp}
savemat(path1/"combined_HPC.mat", data)

data2 = {'states':hypno}
savemat(path1/"states.mat", data2)
