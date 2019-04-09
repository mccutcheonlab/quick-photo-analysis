# -*- coding: utf-8 -*-
"""
This module will contain all of my useful functions

Created on Thu Apr 27 15:48:54 2017


@author: James Rig
"""

import numpy as np
import timeit
import random
import matplotlib.pyplot as plt
import xlrd
import csv
import os


"""
medfilereader takes the following arguments:
    filename - including path
    sessionToExtract - 1 is default, for situations in which more than one session is included in a single file
    varsToExtract - as strings, number of outputs must match
    verbose - False is default
    remove_var_header - False is default, removes first value in array, useful when negative numbers are used to signal array start
"""        
def medfilereader(filename, varsToExtract = 'all',
                  sessionToExtract = 1,
                  verbose = False,
                  remove_var_header = False):
    if varsToExtract == 'all':
        numVarsToExtract = np.arange(0,26)
    else:
        numVarsToExtract = [ord(x)-97 for x in varsToExtract]
    
    f = open(filename, 'r')
    f.seek(0)
    filerows = f.readlines()[8:]
    datarows = [isnumeric(x) for x in filerows]
    matches = [i for i,x in enumerate(datarows) if x == 0.3]
    if sessionToExtract > len(matches):
        print('Session ' + str(sessionToExtract) + ' does not exist.')
    if verbose == True:
        print('There are ' + str(len(matches)) + ' sessions in ' + filename)
        print('Analyzing session ' + str(sessionToExtract))
    
    varstart = matches[sessionToExtract - 1]
    medvars = [[] for n in range(26)]
    
    k = int(varstart + 27)
    for i in range(26):
        medvarsN = int(datarows[varstart + i + 1])
        
        medvars[i] = datarows[k:k + int(medvarsN)]
        k = k + medvarsN
        
    if remove_var_header == True:
        varsToReturn = [medvars[i][1:] for i in numVarsToExtract]
    else:
        varsToReturn = [medvars[i] for i in numVarsToExtract]

    if np.shape(varsToReturn)[0] == 1:
        varsToReturn = varsToReturn[0]
    return varsToReturn

def metafilemaker(xlfile, metafilename, sheetname='metafile', fileformat='csv'):
    with xlrd.open_workbook(xlfile) as wb:
        sh = wb.sheet_by_name(sheetname)  # or wb.sheet_by_name('name_of_the_sheet_here')
        
        if fileformat == 'csv':
            with open(metafilename+'.csv', 'w', newline="") as f:
                c = csv.writer(f)
                for r in range(sh.nrows):
                    c.writerow(sh.row_values(r))
        if fileformat == 'txt':
            with open(metafilename+'.txt', 'w', newline="") as f:
                c = csv.writer(f, delimiter="\t")
                for r in range(sh.nrows):
                    c.writerow(sh.row_values(r))
    
def metafilereader(filename):
    
    f = open(filename, 'r')
    f.seek(0)
    header = f.readlines()[0]
    f.seek(0)
    filerows = f.readlines()[1:]
    
    tablerows = []
    
    for i in filerows:
        tablerows.append(i.split('\t'))
        
    header = header.split('\t')
    # need to find a way to strip end of line \n from last column - work-around is to add extra dummy column at end of metafile
    return tablerows, header

def isnumeric(s):
    try:
        x = float(s)
        return x
    except ValueError:
        return float('nan')
    
def remcheck(val, range1, range2):
    # function checks whether value is within range of two decimels
    if (range1 < range2):
        if (val > range1) and (val < range2):
            return True
        else:
            return False
    else:
        if (val > range1) or (val < range2):
            return True
        else:
            return False
    
def random_array(dims,n, multiplier = 10):
    
    data = []
    import numpy as np
    try:
        if len(dims) == 2:
            data = np.empty((dims), dtype=np.object)        
            for i in range(np.shape(data)[0]):
                for j in range(np.shape(data)[1]):
                    data[i][j] = np.random.random((n))*multiplier
        elif len(dims) > 2:
            print('Too many dimensions!')
            return
        
        elif len(dims) == 1:
            data = np.empty((dims), dtype=np.object)
            for i,j in enumerate(data):
                data[i] = np.random.random((n))*multiplier
    except TypeError:
        print('Dimensions need to be in a list or matrix')
        return

    return data

def med_abs_dev(data, b=1.4826):
    median = np.median(data)
    devs = [abs(i-median) for i in data]
    mad = np.median(devs)*b
                   
    return mad

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

"""
This function gets extracts blue trace, uv trace, and noise index and
outputs the data as a dictionary by default. If no random events are given then
no noise index is produced.
"""

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
        blueTrials_processed,_ = snipper(x.data_filt, events,
                                   t2sMap=x.t2sMap,
                                   fs=x.fs,
                                   bins=bins,
                                   preTrial=preTrial,
                                   trialLength=trialLength,
                                   adjustBaseline = False)
        
        blueTrials_z = zscore(blueTrials_processed)
        blueTrials_z_adjBL = zscore(blueTrials_processed, baseline_points=50)
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

"""
This function will check for traces that are outliers or contain a large amount
of noise, relative to other trials (or relative to the whole data file.
"""


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
"""
This function will find nearby events to one of interest. It should output an
array with the same first dimension as the arrays produced by SNIPPER which
will allow them to be indexed appropriately.

evBefore and evAfter variables will allow only events before or only events
after to be returned.
"""
def nearestevents(timelock, events, preTrial=10, trialLength=30):
#    try:
#        nTrials = len(timelock)
#    except TypeError:
#        nTrials = 1
    data = []
    start = [x - preTrial for x in timelock]
    end = [x + trialLength for x in start]
    for start, end in zip(start, end):
        data.append([x for x in events if (x > start) & (x < end)])
    for i, x in enumerate(data):
        data[i] = x - timelock[i]      
    
    return data

def findfirst(events, afterEvent = True):

    if afterEvent == False:
        events = np.multiply(events, -1)
    
    first = []    
    for x in events:
        try:
            first.append([f for f in x if (f > 0)][0])
        except:
            first.append([])
    
    if afterEvent == False:
        first = [f*-1 for f in first]
    first = np.asarray(first, dtype=object)
    
    return first
    
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

def findphantomlicks(licks, sipper, delay=0, postsipper=1.5, verbose=True):
    phlicks=[]
    for i in sipper:
        try:
            phlicks.append([ind for ind, val in enumerate(licks) if (val > i+delay) and (val < i+delay+postsipper) and (licks[ind+1]-val > 0.5) and (val-licks[ind-1] > 0.5)])
        except IndexError:
            phlicks.append([ind for ind, val in enumerate(licks) if (val > i+delay) and (val < i+delay+postsipper)])
    phlicks = [item for sublist in phlicks for item in sublist]
    if verbose == True:
        print(str(len(phlicks)) + ' phantom licks removed')
    return phlicks

def latencyCalc(licks, cueon, cueoff=10, nolat=np.nan, lag=3):
    if type(cueoff) == int:
        cueoff = [i+cueoff for i in cueon]
    lats=[]
    for on,off in zip(cueon, cueoff): 
        try:
            currentlat = [i-(on+lag) for i in licks if (i>on) and (i<off)][0]
        except IndexError:
            currentlat = nolat
        lats.append(currentlat)

    return(lats)

def distractedOrNot(distractors, licks, delay=1):   
    firstlick = []
    distractedArray = []

    for d in distractors:               
        try:
            firstlick.append([i-d for i in licks if (i > d)][0])
        except IndexError:
            firstlick.append(np.NaN)

    distractedArray = np.array([i>delay for i in firstlick], dtype=bool)
    
    if np.isnan(firstlick)[-1] == 1: 
        distractedArray[-1] = True
    
    return firstlick, distractedArray

def calcDistractors(licks):
#    disStart = [licks[idx-2] for idx, val in enumerate(licks) if (val - licks[idx-2]) < 1]
#    disEnd = [val for idx, val in enumerate(licks) if (val - licks[idx-2]) < 1]
#    
#    distractors = [val for idx, val in enumerate(disEnd) if disStart[idx] - disEnd[idx-1] > 1]
    d = []

    i=2
    while i < len(licks):
        if licks[i] - licks[i-2] < 1:
            d.append(licks[i])
            i += 1
            try:
                while licks[i] - licks[i-1] < 1:
                    i += 1
            except IndexError:
                pass
        else:
            i += 1
    
    distractors = d
    
    return distractors

def distractionCalc(licks, post=1, pre=1):
    # here, n is the first lick in the burst that causes the distractor,
    # nminus1 is the lick preceding this burst (has to be over a second before)
    # and nplus2 is the third lick in the burst, the one that actually triggers
    # the distractor
    distimes = [nplus2 for n,nplus2,nminus1 in zip(licks[1:-2], licks[3:], licks[:-3]) if
                (nplus2-n < post) & (n-nminus1 > pre)]
    # print(distimes)
    return distimes

def distractionCalc2(licks, post=1, pre=1):
    # similar to above function but returns first lick of triplet
    distimes = [n for n,nplus2,nminus1 in zip(licks[1:-2], licks[3:], licks[:-3]) if
                (nplus2-n < post) & (n-nminus1 > pre)]
    # print(distimes)
    return distimes

def sidakcorr(pval, ncomps=3):
    corr_p = 1-((1-pval)**ncomps)   
    return corr_p

def sidakcorr_R(robj, ncomps=3):
    pval = (list(robj.rx('p.value'))[0])[0]
    corr_p = 1-((1-pval)**ncomps)   
    return corr_p

def discrete2continuous(onset, offset=[], nSamples=[], fs=[]):
    # this function takes timestamp data (e.g. licks) that can include offsets 
    # as well as onsets, and returns a digital on/off array (y) as well as the 
    # x output. The number of samples (nSamples) and sample frequency (fs) can 
    # be input or if they are not (default) it will attempt to calculate them 
    # based on the timestamp data. It has not been fully stress-tested yet.
    
    try:
        fs = int(fs)
    except TypeError:
        isis = np.diff(onset)
        fs = int(1 / (min(isis)/2)) 
    
    if len(nSamples) == 0:
        nSamples = int(fs*max(onset))    
    
    outputx = np.linspace(0, nSamples/fs, nSamples)
    outputy = np.zeros(len(outputx))
    
    if len(offset) == 0:
        for on in onset:
            idx = (np.abs(outputx - on)).argmin()
            outputy[idx] = 1
    else:
        for i, on in enumerate(onset):
            start = (np.abs(outputx - on)).argmin()
            stop = (np.abs(outputx - offset[i])).argmin()
            outputy[start:stop] = 1

    return outputx, outputy

def getuserhome():
    path = os.environ['USERPROFILE']
    return path

def data2obj2D(data):
    obj = np.empty((np.shape(data)[0], np.shape(data)[1]), dtype=np.object)
    for i,x in enumerate(data):
        for j,y in enumerate(x):
            obj[i][j] = np.array(y)
    return obj

def data2obj1D(data):
    obj = np.empty(len(data), dtype=np.object)
    for i,x in enumerate(data):
        obj[i] = np.array(x)  
    return obj

def flatten_list(listoflists):
    try:
        flat_list = [item for sublist in listoflists for item in sublist]
        return flat_list
    except:
        print('Cannot flatten list. Maybe is in the wrong format. Returning empty list.')
        return []
    
def findpercentilevalue(data, percentile):

    if (0 < percentile) and (percentile <= 1):
        position = int(percentile*len(data))
    else:
        print('Value for percentile must be between 0 and 1')
        return

    sorteddata = np.sort(data)
    value = sorteddata[position-1]
    
    return value
