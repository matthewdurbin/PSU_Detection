# -*- coding: utf-8 -*-
"""
Load lab data from 2/17/21

Based on a Caen Digitizer with DPP-PSD
Running through COMPASS, saved as binary unfiltered data files

@author: matth
"""
import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.signal import savgol_filter
import glob
import sys
import gc

#%% 
def PHA(waveform):
    """
    Smooths single waveform, calculates baseline and PH
    Returns PH 
    """
    sf_window, sf_poly = 57, 3
    bl_window = 50
    smooth_wave = savgol_filter(waveform, sf_window, sf_poly)
    baseline = np.mean(waveform[:bl_window])
    max_height = np.min(smooth_wave)
    PH = np.round(baseline - max_height)
    
    return(PH)
#%%
def process(filename):
    """
    Proceses .bin file of filename.bin
    Loads one waveform, PHA's the wave form
    Returns vector of heights 
    
    wave_bytes, extra_data will depend on digitizer/firmware/DAQ settings!!
    """
    file = open(filename, mode='rb')
    fileContent = file.read()
    wave_bytes = 3016 # how many bytes are dedicated to a single waveform
    extra_data = 12 # how many bytes are non waveform samples
    wave_number = int(len(fileContent)/wave_bytes) # number of waveforms in data
    heights = np.zeros(wave_number)

    for i in range(wave_number):
        # unpacks a wave form (raw), and runs PHA on it
        raw = struct.unpack('h'*int(3016/2),fileContent[wave_bytes*i:wave_bytes*(i+1)])
        heights[i] = PHA(raw[extra_data:])
        # Uncomment to plot first 0 wave froms
        # if i < 50:
        #     plt.plot(raw[extra_data:])
            
        raw = None
        
    file.close()   
    
    return(heights)
 
#%% Parse
def parse(data):
    """
    Smooths data, averages baseline, calculates height
    for batch of waveforms. Returns PHS
    """
    data_smooth = np.zeros((data.shape))
    for i in range(len(data)):
        data_smooth[i] = savgol_filter(data[i], 57, 3)
    
    baseline = np.zeros((len(data)))
    max_height = np.zeros((len(data)))
    for i in range(len(data)):
        baseline[i] = np.mean(data[i,:50])
        max_height[i] = np.min(data_smooth[i])
        
    phs = np.round(baseline - max_height)
    counts, channel = np.histogram(phs, bins=np.arange(0,1024,1))
    data = None
    return(counts)

#%% Load a batch of Runs 

# List all .bin files in the folder
files = glob.glob('E:/CAENDATA/DAQ/**/*.bin', recursive=True) 
spec = np.zeros((len(files), 1023))
for i in range(len(files)):
    print("Loading ", i, "/", len(files)-1)
    heights = process(files[i])
    spec[i], a = np.histogram(heights, bins=np.arange(0,1024,1))
    del heights # for memory sake
    gc.collect() # also for memory sake
    print("Processed ", i, "/", len(files)-1)
    
np.save('spec_217.npy', spec)    
np.save('files_217.npy', files) # worth saving, to note what order the spec is in


