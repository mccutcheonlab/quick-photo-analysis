# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:38:23 2019

Helper functions for quick-photometry-analysis notebook


@author: James Rig
"""

import numpy as np
import timeit
import random
import xlrd
import csv
import os

import tdt

import scipy.signal as sig

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl

import JM_general_functions as jmf
import JM_custom_figs as jmfig
"""
this function makes 'snips' of a data file ('data' single scalar) aligned to an
event of interest ('event', list of times in seconds).

If a timelocked map is needed to align data precisely (e.g. with TDT equipment)
then it is necessary to pass a t2sMap to the function.

preTrial and trialLength are in seconds.

Will put data into bins if requested.

"""

class Session(object):
    
    def __init__(self, sessionID='Untitled session'):
        self.sessionID = sessionID
        self.left = {}
        self.right = {}
        self.both = {}
        
    def set_tick(self):
        try:
            self.tick = self.ttls['Tick'].onset
            print('Ticks set correctly.')
        except:
            print('Ticks not set - check that ticks are labeled correctly in tdt file.')    
        
    def time2samples(self):
        maxsamples = len(self.tick)*int(self.fs)
        if (len(self.data) - maxsamples) > 2*int(self.fs):
            print('Something may be wrong with conversion from time to samples')
            print(str(len(self.data) - maxsamples) + ' samples left over. This is more than double fs.')
            self.t2sMap = np.linspace(min(self.tick), max(self.tick), maxsamples)
        else:
            self.t2sMap = np.linspace(min(self.tick), max(self.tick), maxsamples)
            print('t2sMap made correctly.')

    def event2sample(self, EOI):
        idx = (np.abs(self.t2sMap - EOI)).argmin()   
        return idx
 

def sessionlicksFig(ax):
    if x.left['exist'] == True:
        licks = x.left['lickdata']['licks']
        ax.hist(licks, range(0, 3600, 60), color=x.left['color'], alpha=0.4)          
        yraster = [ax.get_ylim()[1]] * len(licks)
        ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=x.left['color'])

    if x.right['exist'] == True:
        licks = x.right['lickdata']['licks']
        ax.hist(licks, range(0, 3600, 60), color=x.right['color'], alpha=0.4)          
        yraster = [ax.get_ylim()[1]] * len(licks)
        ax.scatter(licks, yraster, s=50, facecolors='none', edgecolors=x.right['color'])           
    
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Licks per min')
        
def sessionFig(x, ax):
    ax.plot(x.data, color='blue', linewidth=0.1)
    try:
        ax.plot(x.dataUV, color='m', linewidth=0.1)
    except:
        print('No UV data.')
            
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60*x.fs))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    


def correctforbaseline(blue, uv):
    pt = len(blue)
    X = np.fft.rfft(uv, pt)
    Y = np.fft.rfft(blue, pt)
    Ynet = Y-X

    datafilt = np.fft.irfft(Ynet)

    datafilt = sig.detrend(datafilt)

    b, a = sig.butter(9, 0.012, 'low', analog=True)
    datafilt = sig.filtfilt(b, a, datafilt)
    
    return datafilt

def lerner_correction(blue, uv):
    x = np.array(uv)
    y = np.array(blue)
    bls = np.polyfit(x, y, 1)
    Y_fit_all = np.multiply(bls[0], x) + bls[1]
    Y_dF_all = y - Y_fit_all
    dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
    std_dFF = np.std(dFF)
    
    return [dFF, std_dFF]


def snipper(data, timelock, fs = 1, t2sMap = [], preTrial=10, trialLength=30,
                 adjustBaseline = True,
                 bins = 0):

    if len(timelock) == 0:
        print('No events to analyse! Quitting function.')
        raise Exception('no events')

    pps = int(fs) # points per sample
    pre = int(preTrial*pps) 
#    preABS = preTrial
    length = int(trialLength*pps)
# converts events into sample numbers
    event=[]
    if len(t2sMap) > 1:
        for x in timelock:
            event.append(np.searchsorted(t2sMap, x, side="left"))
    else:
        event = [x*fs for x in timelock]

    new_events = []
    for x in event:
        if int(x-pre) > 0:
            new_events.append(x)
    event = new_events

    nSnips = len(event)
    snips = np.empty([nSnips,length])
    avgBaseline = []

    for i, x in enumerate(event):
        start = int(x) - pre
        avgBaseline.append(np.mean(data[start : start + pre]))
        try:
            snips[i] = data[start : start+length]
        except ValueError: # Deals with recording arrays that do not have a full final trial
            snips = snips[:-1]
            avgBaseline = avgBaseline[:-1]
            nSnips = nSnips-1

    if adjustBaseline == True:
        snips = np.subtract(snips.transpose(), avgBaseline).transpose()
        snips = np.divide(snips.transpose(), avgBaseline).transpose()

    if bins > 0:
        if length % bins != 0:
            snips = snips[:,:-(length % bins)]
        totaltime = snips.shape[1] / int(fs)
        snips = np.mean(snips.reshape(nSnips,bins,-1), axis=2)
        pps = bins/totaltime
              
    return snips, pps

def mastersnipper(data, dataUV, data_filt,
                  t2sMap, fs, bgMAD,
                  events,
                  bins=300,
                  baselinebins=100,
                  preTrial=10,
                  trialLength=30,    
                  threshold=8,
                  peak_between_time=[0, 1],
                  latency_events=[],
                  latency_direction='pre',
                  max_latency=30,
                  verbose=True):
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        blueTrials, uvTrials, filtTrials, filtTrials_z, filtTrials_z_adjBL, filt_avg, filt_avg_z, noiseindex, peak, latency = ([] for i in range(10))
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))
        
        blueTrials,_ = snipper(data, events,
                                   t2sMap=t2sMap,
                                   fs=fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        uvTrials,_ = snipper(dataUV, events,
                                   t2sMap=t2sMap,
                                   fs=fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        filtTrials,_ = snipper(data_filt, events,
                                   t2sMap=t2sMap,
                                   fs=fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline=False)
        
        filtTrials_z = zscore(filtTrials, baseline_points=baselinebins)
        filtTrials_z_adjBL = zscore(filtTrials, baseline_points=50)

        sigSum = [np.sum(abs(i)) for i in filtTrials]
        sigSD = [np.std(i) for i in filtTrials]

        noiseindex = [i > bgMAD*threshold for i in sigSum]
                        
        # do i need to remove noise trials first before averages
        filt_avg = np.mean(removenoise(filtTrials, noiseindex), axis=0)
        if verbose: print('{} noise trials removed'.format(sum(noiseindex)))
        filt_avg_z = zscore(filt_avg)

    
        bin2s = bins/trialLength
        peakbins = [int((preTrial+peak_between_time[0])*bin2s),
                    int((preTrial+peak_between_time[1])*bin2s)]
        peak = [np.mean(trial[peakbins[0]:peakbins[1]]) for trial in filtTrials_z]
        
        latency = []

        if len(latency_events) > 1: 
            for event in events:
                if latency_direction == 'pre':
                    try:
                        latency.append(np.abs([lat-event for lat in latency_events if lat-event<0]).min())
                    except ValueError:
                        latency.append(np.NaN)
                
                elif latency_direction == 'post':
                    try:
                        latency.append(np.abs([lat-event for lat in latency_events if lat-event>0]).min())
                    except ValueError:
                        latency.append(np.NaN)

            latency = [x if (x<max_latency) else np.NaN for x in latency]
            if latency_direction == 'pre':
                latency = [-x for x in latency]
        else:
            print('No latency events found')

    output = {}
    output['blue'] = blueTrials
    output['uv'] = uvTrials
    output['filt'] = filtTrials
    output['filt_z'] = filtTrials_z
    output['filt_z_adjBL'] = filtTrials_z_adjBL
    output['filt_avg'] = filt_avg
    output['filt_avg_z'] = filt_avg_z
    output['noise'] = noiseindex
    output['peak'] = peak
    output['latency'] = latency
    
    return output

def zscore(snips, baseline_points=100):
    
    BL_range = range(baseline_points)
    z_snips = []
    try:
        for i in snips:
            mean = np.mean(i[BL_range])
            sd = np.std(i[BL_range])
            z_snips.append([(x-mean)/sd for x in i])
    except IndexError:
        mean = np.mean(snips[BL_range])
        sd = np.std(snips[BL_range])
        z_snips = [(x-mean)/sd for x in snips]

    return z_snips

"""
This function will check for traces that are outliers or contain a large amount
of noise, relative to other trials (or relative to the whole data file.
"""


def findnoise(data, background, t2sMap = [], fs = 1, bins=0, method='sd'):
    
    bgSnips, _ = snipper(data, background, t2sMap=t2sMap, fs=fs, bins=bins, 
                        adjustBaseline=False)
  
    if method == 'sum':
        bgSum = [np.sum(abs(i)) for i in bgSnips]
        bgMAD = med_abs_dev(bgSum)
    elif method == 'sd':
        bgSD = [np.std(i) for i in bgSnips]
        bgMAD = med_abs_dev(bgSD)
   
    return bgMAD

def removenoise(snipsIn, noiseindex):
    snipsOut = np.array([x for (x,v) in zip(snipsIn, noiseindex) if not v])   
    return snipsOut

def findphotodiff(blue, UV, noise):
    blueNoNoise = removenoise(blue, noise)
    UVNoNoise = removenoise(UV, noise)
    diffSig = blueNoNoise-UVNoNoise
    return diffSig

def makerandomevents(minTime, maxTime, spacing = 77, n=100):
    events = []
    total = maxTime-minTime
    start = 0
    for i in np.arange(0,n):
        if start > total:
            start = start - total
        events.append(start)
        start = start + spacing
    events = [i+minTime for i in events]
    return events

def med_abs_dev(data, b=1.4826):
    median = np.median(data)
    devs = [abs(i-median) for i in data]
    mad = np.median(devs)*b
                   
    return mad

print('Importing required functions...')

x = Session('Session to analyze')
print('Initializing session...')