# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:51:54 2019

@author: jmc010
"""

import numpy as np
import matplotlib.pyplot as plt

"""
this function makes 'snips' of a data file ('data' single scalar) aligned to an
event of interest ('event', list of times in seconds).

If a timelocked map is needed to align data precisely (e.g. with TDT equipment)
then it is necessary to pass a t2sMap to the function.

preTrial and trialLength are in seconds.

Will put data into bins if requested.

"""
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

def mastersnipper(x, events,
                  bins=300,
                  preTrial=10,
                  trialLength=30,
                  threshold=8,
                  peak_between_time=[0, 1],
                  output_as_dict=True,
                  latency_events=[],
                  latency_direction='pre',
                  max_latency=30,
                  verbose=True):
    
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        blueTrials, uvTrials, noiseindex, diffTrials, peak, latency = ([] for i in range(5))
    else:
        if verbose: print('{} events to analyze.'.format(len(events)))
        
        blueTrials,_ = snipper(x.data, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        uvTrials,_ = snipper(x.dataUV, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        filtTrials,_ = snipper(x.data_filt, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline=False)
        
        filtTrials_z = zscore(filtTrials)
        filtTrials_z_adjBL = zscore(filtTrials, baseline_points=50)
        
        # bgMAD = findnoise(x.data_filt, x.randomevents,
        #                       t2sMap=x.t2sMap, fs=x.fs, bins=bins,
        #                       method='sum')
        
        # print(bgMAD)

        sigSum = [np.sum(abs(i)) for i in filtTrials]
        sigSD = [np.std(i) for i in filtTrials]

        noiseindex = [i > x.bgMAD*threshold for i in sigSum]
                        
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

    if output_as_dict == True:
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
    else:
        return blueTrials, blueTrials_raw, blueTrials_z, blueTrials_z_adjBL, uvTrials, uvTrials_raw, uvTrials_z, noiseindex, diffTrials, peak, latency

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