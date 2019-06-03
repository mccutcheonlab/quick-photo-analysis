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
    
    def check4events(self, trialL, lickL, trialR, lickR):
        try:
            lt = getattr(self.ttls, trialL)
            self.left['exist'] = True
            self.left['sipper'] = lt.onset
            self.left['sipper_off'] = lt.offset
            ll = getattr(self.ttls, lickL)
            self.left['licks'] = np.array([i for i in ll.onset if i<max(self.left['sipper_off'])])
            self.left['licks_off'] = ll.offset[:len(self.left['licks'])]
            print('Events found on left')
        except AttributeError:
            self.left['exist'] = False
            self.left['sipper'] = []
            self.left['sipper_off'] = []
            self.left['licks'] = []
            self.left['licks_off'] = []
            print('No events found on left')
           
        try:
            rt = getattr(self.ttls, trialR)
            self.right['exist'] = True
            self.right['sipper'] = rt.onset
            self.right['sipper_off'] = rt.offset
            rl = getattr(self.ttls, lickR)
            self.right['licks'] = np.array([i for i in rl.onset if i<max(self.right['sipper_off'])])
            self.right['licks_off'] = rl.offset[:len(self.right['licks'])]
            print('Events found on right')
        except AttributeError:
            self.right['exist'] = False
            self.right['sipper'] = []
            self.right['sipper_off'] = []
            self.right['licks'] = []
            self.right['licks_off'] = []
            print('No events found on right')
            
        if self.left['exist'] == True and self.right['exist'] == True:
            print('Events on both sides detected to also separating into forced choice and free choice')
            try:
                first = findfreechoice(self.left['sipper'], self.right['sipper'])
                self.both['sipper'] = self.left['sipper'][first:]
                self.both['sipper_off'] = self.left['sipper_off'][first:]
                self.left['sipper'] = self.left['sipper'][:first-1]
                self.left['sipper_off'] = self.left['sipper_off'][:first-1]
                self.right['sipper'] = self.right['sipper'][:first-1]
                self.right['sipper_off'] = self.right['sipper_off'][:first-1]
                self.left['licks-forced'], self.left['licks-free'] = dividelicks(self.left['licks'], self.both['sipper'][0])
                self.right['licks-forced'], self.right['licks-free'] = dividelicks(self.right['licks'], self.both['sipper'][0])
                self.left['nlicks-forced'] = len(self.left['licks-forced'])
                self.right['nlicks-forced'] = len(self.right['licks-forced'])
                self.left['nlicks-free'] = len(self.left['licks-free'])
                self.right['nlicks-free'] = len(self.right['licks-free'])

            except IndexError:
                print('Problem separating out free choice trials')
        else:
            self.left['licks-forced'] = self.left['licks']
            self.right['licks-forced'] = self.right['licks']

def findfreechoice(left, right):
    first = [idx for idx, x in enumerate(left) if x in right][0]
    return first
        
def dividelicks(licks, time):
    before = [x for x in licks if x < time]
    after = [x for x in licks if x > time]
    
    return before, after   

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
#
def mastersnipper(x, events,
                  bins=300,
                  preTrial=10,
                  trialLength=30,
                  threshold=10,
                  peak_between_time=[0, 1],
                  output_as_dict=True,
                  latency_events=[],
                  latency_direction='pre'):
    if len(events) < 1:
        print('Cannot find any events. All outputs will be empty.')
        blueTrials, uvTrials, noiseindex, diffTrials, peak, latency = ([] for i in range(5))
    else:
        blueTrials,_ = snipper(x.data, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        blueTrials_raw,_ = snipper(x.data, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline = False)
        uvTrials,_ = snipper(x.dataUV, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength)
        uvTrials_raw,_ = snipper(x.dataUV, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline = False)
#        blueTrials_processed,_ = snipper(x.data_filt, events,
#                                   t2sMap=x.t2sMap,
#                                   fs=x.fs,
#                                   bins=bins,
#                                   preTrial=preTrial,
#                                   trialLength=trialLength,
#                                   adjustBaseline = False)
        
        blueTrials_z = zscore(blueTrials_raw)
        blueTrials_z_adjBL = zscore(blueTrials_raw, baseline_points=50)
        uvTrials_z = zscore(uvTrials_raw)
        
        bgMAD = findnoise(x.data, x.randomevents,
                              t2sMap=x.t2sMap, fs=x.fs, bins=bins,
                              method='sum')        
        sigSum = [np.sum(abs(i)) for i in blueTrials]
        sigSD = [np.std(i) for i in blueTrials]
        noiseindex = [i > bgMAD*threshold for i in sigSum]
    
        diffTrials = findphotodiff(blueTrials, uvTrials, noiseindex)
        bin2s = bins/trialLength
        peakbins = [int((preTrial+peak_between_time[0])*bin2s),
                    int((preTrial+peak_between_time[1])*bin2s)]
        peak = [np.mean(trial[peakbins[0]:peakbins[1]]) for trial in blueTrials_z]
        
        latency = []
        try:
            for event in events:
                if latency_direction == 'pre':
                    latency.append(np.abs([lat-event for lat in latency_events if lat-event<0]).min())
                elif latency_direction == 'post':
                    latency.append(np.abs([lat-event for lat in latency_events if lat-event>0]).min())
                else:
                    latency.append(np.abs([lat-event for lat in latency_events]).min())
#            latency = [x if (x<30) else None for x in latency]
            latency = np.asarray(latency)
            latency[latency>30] = np.nan
        except ValueError:
            print('No latency events found')

    if output_as_dict == True:
        output = {}
        output['blue'] = blueTrials
        output['blue_raw'] = blueTrials_raw
        output['blue_z'] = blueTrials_z
        output['blue_z_adj'] = blueTrials_z_adjBL
        output['uv'] = uvTrials
        output['uv_raw'] = uvTrials_raw
        output['uv_z'] = uvTrials_z
        output['noise'] = noiseindex
        output['diff'] = diffTrials
        output['peak'] = peak
        output['latency'] = latency
        return output
    else:
        return blueTrials, blueTrials_raw, blueTrials_z, blueTrials_z_adjBL, uvTrials, uvTrials_raw, uvTrials_z, noiseindex, diffTrials, peak, latency

def zscore(snips, baseline_points=100):
    
    BL_range = range(baseline_points)
    z_snips = []
    for i in snips:
        mean = np.mean(i[BL_range])
        sd = np.std(i[BL_range])
        z_snips.append([(x-mean)/sd for x in i])
        
    return z_snips

def findnoise(data, background, t2sMap = [], fs = 1, bins=0, method='sd'):
    
    bgSnips, _ = snipper(data, background, t2sMap=t2sMap, fs=fs, bins=bins)
    
    if method == 'sum':
        bgSum = [np.sum(abs(i)) for i in bgSnips]
        bgMAD = med_abs_dev(bgSum)
    elif method == 'sd':
        bgSD = [np.std(i) for i in bgSnips]
        bgMAD = med_abs_dev(bgSD)
   
    return(bgMAD)

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

"""
This function will calculate data for bursts from a train of licks. The threshold
for bursts and clusters can be set. It returns all data as a dictionary.
"""
def lickCalc(licks, offset = [], burstThreshold = 0.25, runThreshold = 10, 
             binsize=60, histDensity = False, adjustforlonglicks='none'):
    # makes dictionary of data relating to licks and bursts
    if type(licks) != np.ndarray or type(offset) != np.ndarray:
        try:
            licks = np.array(licks)
            offset = np.array(offset)
        except:
            print('Licks and offsets need to be arrays and unable to easily convert.')
            return

    lickData = {}

    if len(offset) > 0:        
        lickData['licklength'] = offset - licks[:len(offset)]
        lickData['longlicks'] = [x for x in lickData['licklength'] if x > 0.3]
    else:
        lickData['licklength'] = []
        lickData['longlicks'] = []
    
    if adjustforlonglicks != 'none':
        if len(lickData['longlicks']) == 0:
            print('No long licks to adjust for.')
        else:
            lickData['median_ll'] = np.median(lickData['licklength'])
            lickData['licks_adj'] = int(np.sum(lickData['licklength'])/lickData['median_ll'])
            if adjustforlonglicks == 'interpolate':
                licks_new = []
                for l, off in zip(licks, offset):
                    x = l
                    while x < off - lickData['median_ll']:
                        licks_new.append(x)
                        x = x + lickData['median_ll']
                licks = licks_new
        
    lickData['licks'] = licks
    lickData['ilis'] = np.diff(np.concatenate([[0], licks]))
    lickData['shilis'] = [x for x in lickData['ilis'] if x < burstThreshold]
    lickData['freq'] = 1/np.mean([x for x in lickData['ilis'] if x < burstThreshold])
    lickData['total'] = len(licks)
    
    # Calculates start, end, number of licks and time for each BURST 
    lickData['bStart'] = [val for i, val in enumerate(lickData['licks']) if (val - lickData['licks'][i-1] > burstThreshold)]  
    lickData['bInd'] = [i for i, val in enumerate(lickData['licks']) if (val - lickData['licks'][i-1] > burstThreshold)]
    lickData['bEnd'] = [lickData['licks'][i-1] for i in lickData['bInd'][1:]]
    lickData['bEnd'].append(lickData['licks'][-1])

    lickData['bLicks'] = np.diff(lickData['bInd'] + [len(lickData['licks'])])    
    lickData['bTime'] = np.subtract(lickData['bEnd'], lickData['bStart'])
    lickData['bNum'] = len(lickData['bStart'])
    if lickData['bNum'] > 0:
        lickData['bMean'] = np.nanmean(lickData['bLicks'])
        lickData['bMean-first3'] = np.nanmean(lickData['bLicks'][:3])
    else:
        lickData['bMean'] = 0
        lickData['bMean-first3'] = 0
    
    lickData['bILIs'] = [x for x in lickData['ilis'] if x > burstThreshold]

    # Calculates start, end, number of licks and time for each RUN
    lickData['rStart'] = [val for i, val in enumerate(lickData['licks']) if (val - lickData['licks'][i-1] > runThreshold)]  
    lickData['rInd'] = [i for i, val in enumerate(lickData['licks']) if (val - lickData['licks'][i-1] > runThreshold)]
    lickData['rEnd'] = [lickData['licks'][i-1] for i in lickData['rInd'][1:]]
    lickData['rEnd'].append(lickData['licks'][-1])

    lickData['rLicks'] = np.diff(lickData['rInd'] + [len(lickData['licks'])])    
    lickData['rTime'] = np.subtract(lickData['rEnd'], lickData['rStart'])
    lickData['rNum'] = len(lickData['rStart'])

    lickData['rILIs'] = [x for x in lickData['ilis'] if x > runThreshold]
    try:
        lickData['hist'] = np.histogram(lickData['licks'][1:], 
                                    range=(0, 3600), bins=int((3600/binsize)),
                                          density=histDensity)[0]
    except TypeError:
        print('Problem making histograms of lick data')
        
    return lickData

def calculate_lick_params(x, burstThreshold=0.5):
    
    print('Burst threshold set at', str(burstThreshold), 's')
    
    try:
        x.left['lickdata'] = lickCalc(x.left['licks'],
                          offset = x.left['licks_off'],
                          burstThreshold=burstThreshold)
        print('Parameters for left licks calculated...')
        
    except IndexError:
        x.left['lickdata'] = 'none'
        print('No left licks to calculate.')
    
    try:
        x.right['lickdata'] = lickCalc(x.right['licks'],
                  offset = x.right['licks_off'],
                  burstThreshold=burstThreshold)
        print('Parameters for right licks calculated...')
        
    except IndexError:
        x.right['lickdata'] = 'none'
        print('No right licks to calculate.')
        
def make_snips(x, bins=300):
    
    print('Number of bins=300')
    
    try:
        x.randomevents = makerandomevents(120, max(x.tick)-120)
        x.bgTrials, x.pps = snipper(x.data, x.randomevents,
                                t2sMap=x.t2sMap, fs=x.fs, bins=bins)
        print('Random events and background trials constructed (used for working out noise)')
    except:
        print('Unable to make rnadom events and/or background trials.')
        
    try:   
        for side in [x.left, x.right]:   
            if side['exist'] == True:
                side['snips_sipper'] = mastersnipper(x, side['sipper'], peak_between_time=[0, 5])
                side['snips_licks'] = mastersnipper(x, side['lickdata']['rStart'], peak_between_time=[0, 2])
                try:
                    timelock_events = [licks for licks in side['lickdata']['rStart'] if licks in side['licks-forced']]
                    latency_events = side['sipper']
                    side['snips_licks_forced'] = mastersnipper(x, timelock_events, peak_between_time=[0, 2],
                                                                    latency_events=latency_events, latency_direction='pre')
                except KeyError:
                    pass
                try:
                    side['lats'] = jmf.latencyCalc(side['lickdata']['licks'], side['sipper'], cueoff=side['sipper_off'], lag=0)
                except TypeError:
                    print('Cannot work out latencies as there are lick and/or sipper values missing.')
                    side['lats'] = []
        print('Snips made successfully (I think).')
    except:
        print('Problem making snips')

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
    
def behavFigsCol(gs1, col, side):
    ax = plt.subplot(gs1[1, col])
    jmfig.licklengthFig(ax, side['lickdata'], color=side['color'])
    
    ax = plt.subplot(gs1[2, col])
    jmfig.iliFig(ax, side['lickdata'], color=side['color'])
    
    ax = plt.subplot(gs1[3, col])
    jmfig.cuerasterFig(ax, side['sipper'], side['lickdata']['licks'])
    
def sessionFig(x, ax):
    ax.plot(x.data, color='blue', linewidth=0.1)
    try:
        ax.plot(x.dataUV, color='m', linewidth=0.1)
    except:
        print('No UV data.')
            
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60*x.fs))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    
def photoFigsCol(gs1, col, pps, snips_sipper, snips_licks):
    ax = plt.subplot(gs1[2, col])
    jmfig.trialsFig(ax, snips_sipper['blue'], pps, noiseindex = snips_sipper['noise'],
                    eventText = 'Sipper',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[3, col])
    jmfig.trialsMultShadedFig(ax, [snips_sipper['uv'], snips_sipper['blue']],
                              pps, noiseindex = snips_sipper['noise'],
                              eventText = 'Sipper')
    
    ax = plt.subplot(gs1[2, col+1])
    jmfig.shadedError(ax, snips_sipper['blue_z'])
    
    ax = plt.subplot(gs1[4, col])
    jmfig.trialsFig(ax, snips_licks['blue'], pps, noiseindex=snips_licks['noise'],
                    eventText = 'First Lick',
                    ylabel = 'Delta F / F0')
    
    ax = plt.subplot(gs1[5, col])
    jmfig.trialsMultShadedFig(ax, [snips_licks['uv'], snips_licks['blue']],
                              pps, noiseindex=snips_licks['noise'],
                              eventText = 'First Lick')
    
    ax = plt.subplot(gs1[4, col+1])
    jmfig.shadedError(ax, snips_licks['blue_z'])


print('Importing required functions...')

x = Session('Session to analyze')
print('Initializing session...')