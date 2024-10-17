# -*- coding: utf-8 -*-
import sys
module_path = 'W:/home/nero/phasic_tonic/notebooks/buzsaki_method'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
    
from src.DatasetLoader import DatasetLoader
from src.runtime_logger import logger_setup
from src.utils import get_segments, get_sequences, phasic_detect, get_tonic, create_hypnogram
from src.helper import get_metadata

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yasa
import random

from pathlib import Path
from scipy.io import loadmat
from scipy.signal import spectrogram
from mne.filter import resample
from scipy.signal import hilbert
from neurodsp.filt import filter_signal

plt.style.use('seaborn-v0_8-white')
random.seed(0)

CBD_DIR = "W:/home/nero/datasets/CBD"
RGS_DIR = "W:/home/nero/datasets/RGS14"
OS_DIR = "W:/home/nero/datasets/OSbasic/"
CONF = "W:/home/nero/phasic_tonic/configs/dataset_loading.yaml"

fs_cbd = 2500
fs_os = 2500
fs_rgs = 1000

targetFs = 500
n_down_cbd = fs_cbd/targetFs
n_down_rgs = fs_rgs/targetFs
n_down_os = fs_os/targetFs

datasets = {
# 'dataset_name' : {'dir' : '/path/to/dataset', 'pattern_set': 'pattern_set_in_config'}
    "CBD": {"dir": CBD_DIR, "pattern_set": "CBD"},
    "RGS": {"dir": RGS_DIR, "pattern_set": "RGS"},
    "OS": {"dir": OS_DIR, "pattern_set": "OS"}
}

def get_name(m):
    return f"Rat{m['rat_id']}_SD{m['study_day']}_{m['condition']}_{m['treatment']}_posttrial{m['trial_num']}"

Datasets = DatasetLoader(datasets, CONF)
mapped_datasets = Datasets.load_datasets()

#%% Axillary functions
def _despine_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def create_title(metadata):
    treatment = {0: "Negative CBD", 1: "Positive CBD", 
                 2: "Negative RGS14", 3:"Positive RGS14", 4:"OS basic"}
    
    title = "Rat " + str(metadata["rat_id"])
    title += " Study Day: " + str(metadata["study_day"])
    title += " Treatment: " + treatment[metadata["treatment"]]
    title += " Post-trial: " + str(metadata["trial_num"])
    return title

def ensure_duration(rem_idx, min_dur):
    for rem_start, rem_end in rem_idx:
      if(rem_end-rem_start) < min_dur:
        rem_idx.remove(rem_idx)
    return rem_idx

def phasic_tonic_v1(eeg, hypno, fs, min_dur=3):
    
    rem_seq = get_sequences(np.where(hypno == 5)[0])
    rem_idx = [(start * fs, (end+1) * fs) for start, end in rem_seq]
    
    rem_idx = ensure_duration(rem_idx, min_dur=min_dur)
    if len(rem_idx) == 0:
        raise ValueError("No REM epochs greater than min_dur.")

    # get REM segments
    rem_epochs = get_segments(rem_idx, eeg)
    
    # Combine the REM indices with the corresponding downsampled segments
    rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}

    # Detect phasic
    phasicREM = phasic_detect(rem=rem, fs=fs)
        
    phasic = []
    tonic = []
    
    for rem_idx in phasicREM:
        rem_start, rem_end = rem_idx
        phasic += phasicREM[rem_idx]
        
        # Tonic epochs are determined as everywhere that is not phasic in the REM epoch
        tonic += get_tonic(rem_start*fs, rem_end*fs, phasicREM[rem_idx])
        
    return phasic, tonic

def phasic_tonic_v2(rem, fs):
    # Detect phasic
    phasicREM = phasic_detect(rem=rem, fs=fs)
        
    phasic = []
    tonic = []
    
    for rem_idx in phasicREM:
        rem_start, rem_end = rem_idx
        phasic += phasicREM[rem_idx]
        
        # Tonic epochs are determined as everywhere that is not phasic in the REM epoch
        tonic += get_tonic(rem_start*fs, rem_end*fs, phasicREM[rem_idx])
        
    return phasic, tonic

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx
#%% Load the dataset

# Pick specific sample
name = "Rat6_SD1_HC_2_posttrial5"

state, hpc, pfc = mapped_datasets[name]

metadata = get_metadata(name)

if metadata["treatment"] == 0 or metadata["treatment"] == 1:
    n_down = n_down_cbd
elif metadata["treatment"] == 2 or metadata["treatment"] == 3:
    n_down = n_down_rgs
elif metadata["treatment"] == 4:
    n_down = n_down_os

#%% Test sampling
#state = random.choice(list(mapped.keys()))
# Load the LFP data
lfpHPC = loadmat(hpc)['HPC']
lfpHPC = lfpHPC.flatten()

# Load the states
hypno = loadmat(state)['states']
hypno = hypno.flatten()

# Skip if no REM epoch is detected
if(np.any(hypno == 5)):
    print("REM detected.")
else:
    print("No REM.")
#%% Run the pipeline
data = resample(lfpHPC, down=n_down, method='fft', npad='auto')
#del lfpHPC

# Remove artifacts
art_std, _ = yasa.art_detect(data, targetFs , window=1, method='std', threshold=4)
art_up = yasa.hypno_upsample_to_data(art_std, 1, data, targetFs)
data[art_up] = 0
del art_up

data -= data.mean()
#%% Detect phasic
rem_seq = get_sequences(np.where(hypno == 5)[0])
rem_idx = [(start * targetFs, (end+1) * targetFs) for start, end in rem_seq]

rem_idx = ensure_duration(rem_idx, min_dur=3)
if len(rem_idx) == 0:
    raise ValueError("No REM epochs greater than min_dur.")

# get REM segments
rem_epochs = get_segments(rem_idx, data)

# Combine the REM indices with the corresponding downsampled segments
rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}

w1 = 5.0
w2 = 12.0
nfilt = 11
thr_dur = 900

trdiff_list = []
rem_eeg = np.array([])
eeg_seq = {}
sdiff_seq = {}
tridx_seq = {}
filt = np.ones((nfilt,))
filt = filt / filt.sum()

for idx in rem:
    start, end = idx

    epoch = rem[idx]
    epoch = filter_signal(epoch, targetFs, 'bandpass', (w1,w2), remove_edges=False)
    epoch = hilbert(epoch)

    inst_phase = np.angle(epoch)
    inst_amp = np.abs(epoch)

    # trough indices
    tridx = _detect_troughs(inst_phase, -3)

    # differences between troughs
    trdiff = np.diff(tridx)

    # smoothed trough differences
    sdiff_seq[idx] = np.convolve(trdiff, filt, 'same')

    # dict of trough differences for each REM period
    tridx_seq[idx] = tridx

    eeg_seq[idx] = inst_amp

    # differences between troughs
    trdiff_list += list(trdiff)

    # amplitude of the entire REM sleep
    rem_eeg = np.concatenate((rem_eeg, inst_amp)) 

trdiff = np.array(trdiff_list)
trdiff_sm = np.convolve(trdiff, filt, 'same')

# potential candidates for phasic REM:
# the smoothed difference between troughs is less than
# the 10th percentile:
thr1 = np.percentile(trdiff_sm, 10)
# the minimum smoothed difference in the candidate phREM is less than
# the 5th percentile
thr2 = np.percentile(trdiff_sm, 5)
# the peak amplitude is larger than the mean of the amplitude
# of the REM EEG.
thr3 = rem_eeg.mean()

phasicREM = {rem_idx:[] for rem_idx in rem.keys()}

for rem_idx in tridx_seq:
    rem_start, rem_end = rem_idx
    offset = rem_start * targetFs

    # trough indices
    tridx = tridx_seq[rem_idx]

    # smoothed trough interval
    sdiff = sdiff_seq[rem_idx]

    # amplitude of the REM epoch
    eegh = eeg_seq[rem_idx]

    # get the candidates for phREM
    cand_idx = np.where(sdiff <= thr1)[0]
    cand = get_sequences(cand_idx)

    for start, end in cand:
        # Duration of the candidate in milliseconds
        dur = ( (tridx[end]-tridx[start]+1)/targetFs ) * 1000
        if dur < thr_dur:
            continue # Failed Threshold 1
        
        min_sdiff = np.min(sdiff[start:end])
        if min_sdiff > thr2:
            continue # Failed Threshold 2
        
        mean_amp =  np.mean(eegh[tridx[start]:tridx[end]+1])
        if mean_amp < thr3:
            continue # Failed Threshold 3
        
        t_a = tridx[start] + offset
        t_b = np.min((tridx[end] + offset, rem_end * targetFs))
        
        ph_idx = (t_a, t_b+1)
        phasicREM[rem_idx].append(ph_idx)

phasic = []
for rem_idx in phasicREM:
    phasic += phasicREM[rem_idx]
    
phasicREM
#%% Plotting prep
p_t = create_hypnogram(phasic, len(data))

t_vec = np.arange(0, len(data)/targetFs, 1/targetFs)

nsr_seg = 1
perc_overlap = 0.8
vm = 3000

cmap = plt.cm.hot

from matplotlib.colors import LinearSegmentedColormap

# Define the custom colors
colors = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]]

# Create a custom colormap
my_map = LinearSegmentedColormap.from_list('brs', colors, N=5)

#my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 5)

freq, t, SP = spectrogram(data, fs=targetFs, window='hann', 
                          nperseg=int(nsr_seg * targetFs), 
                          noverlap=int(nsr_seg * targetFs * perc_overlap))
dt = t[1] - t[0]
ifreq = np.where(freq <= 20)[0]

gamma = (50, 90)
df = freq[1] - freq[0]
igamma = np.where((freq >= gamma[0]) & (freq <= gamma[1]))[0]
pow_gamma = SP[igamma,:].sum(axis=0) * df

#%% Plotting
fig = plt.figure(figsize=(12,6), layout='constrained')
fig.suptitle(name, fontsize=12)
axs = fig.subplot_mosaic([["states"],
                          ["lfp"],
                          ["phasic"],
                          ["iti"],
                          ["spectrogram"],
                          ["gamma"]], sharex=True,
                         gridspec_kw = {'height_ratios':[1, 8, 1, 8, 8, 8],
                                        'hspace':0.05}
                         )
tmp = axs["states"].pcolorfast(t_vec, [0, 1], np.array([hypno]), vmin=1, vmax=5)
tmp.set_cmap(my_map)
_despine_axes(axs["states"])    
    
axs["lfp"].plot(t_vec, data, color='k')

axs["spectrogram"].pcolorfast(t, freq[ifreq], SP[ifreq, :], vmin=0, vmax=vm, cmap='hot')
axs["spectrogram"].set_ylabel("Freq. (Hz)")

axs["phasic"].set_ylabel("Phasic")
axs["phasic"].step(t_vec, p_t, c='b')
_despine_axes(axs["phasic"])

for rem_idx in phasicREM:    
    rem_start, rem_end = rem_idx
    rem_start, rem_end = rem_start*targetFs, (rem_end+1)*targetFs
    
    tridx = tridx_seq[rem_idx] 
    sdiff = sdiff_seq[rem_idx]
    eegh = eeg_seq[rem_idx]
    
    tridx = (tridx + rem_start)/targetFs
    
    axs["lfp"].axvspan(rem_idx[0], rem_idx[1], facecolor=[0.8, 0.8, 0.8], alpha=0.3)
    for start, end in phasicREM[rem_idx]:
        axs["lfp"].plot(t_vec[start:end], data[start:end], color='r')
    
    axs["iti"].plot(tridx[:-1], sdiff, drawstyle="steps-pre", color='k')
    axs["lfp"].plot(t_vec[rem_start:rem_end], eegh, 'y', '--')
    axs["lfp"].plot([t_vec[rem_start], t_vec[rem_end]], [thr3, thr3], 'r', '--')

axs["iti"].axhline(y=thr1, color='r', linestyle='--')
axs["iti"].axhline(y=thr2, color='y', linestyle='--')
axs["iti"].set_ylabel("ITI")

axs["gamma"].plot(t, pow_gamma, '.-')
axs["gamma"].set_ylabel(r'$\gamma$')

axs["lfp"].set_ylabel("LFP")
#%%
from functions import phasic_rem_v3
phasic_rem_v3(data, hypno, targetFs, pplot=True)
#%%









