# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:11:45 2017

@author: jaimeHP
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, count
from collections import Sequence
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab
import JM_general_functions as jmf
"""
This function will create bar+scatter plots when passed a 1 or 2 dimensional
array. Data needs to be passed in as a numpy object array, e.g.

data = np.empty((2), dtype=np.object)
data[0] = np.array(allData['nCasLicks'][index])
data[1] = np.array(allData['nMaltLicks'][index])

Various options allow specification of colors and paired/unpaired plotting.
It can return the figures, axes, bars, and scatters for further modification.

e.g.
fig1, ax1, barlist1, sc1 = jmf.barscatter(data)

for i in barlist1[1].get_children():
    i.set_color('g')

"""
def barscatter(data, transpose = False, unequal=False,
                groupwidth = .75,
                barwidth = .9,
                paired = False,
                spaced = False,
                yspace = 20,
                xspace = 0.1,
                barfacecoloroption = 'same', # other options 'between' or 'individual'
                barfacecolor = ['white'],
                baredgecoloroption = 'same',
                baredgecolor = ['black'],
                baralpha = 1,
                scatterfacecoloroption = 'same',
                scatterfacecolor = ['white'],
                scatteredgecoloroption = 'same',
                scatteredgecolor = ['grey'],
                scatterlinecolor = 'grey', # Don't put this value in a list
                scattersize = 80,
                scatteralpha = 1,
                spreadscatters = False,
                linewidth=1,
                xlim=[],
                ylim=[],
                ylabel = 'none',
                xlabel = 'none',
                grouplabel = 'auto',
                itemlabel = 'none',
                barlabels = [],
                barlabeloffset=0.025,
                grouplabeloffset=0,
                yaxisparams = 'auto',
                show_legend = 'none',
                legendloc='upper right',
                xfontsize=8,
                ax=[]):

    if unequal == True:
        dims = np.ndim(data)
        data_obj = np.ndarray((np.shape(data)), dtype=np.object)
        if dims == 1:
            for i, dim in enumerate(data):
                data_obj[i] = np.array(dim, dtype=np.object)
            data = data_obj
        elif dims == 2:            
            for i1, dim1 in enumerate(data):
                for i2, dim2 in enumerate(dim1):
                    data_obj[i1][i2] = np.array(dim2, dtype=np.object)
            data = data_obj
        else:
            print('Cannot convert that number of dimensions or data is in wrong format. Attmepting to make graph assuming equal groups.')
    
    if type(data) != np.ndarray or data.dtype != np.object:
        dims = np.shape(data)
        if len(dims) == 2 or len(dims) == 1:
            data = data2obj1D(data)

        elif len(dims) == 3:
            data = data2obj2D(data)
              
        else:
            print('Cannot interpret data shape. Should be 2 or 3 dimensional array. Exiting function.')
            return

    # Check if transpose = True
    if transpose == True:
        data = np.transpose(data)
        
    # Initialize arrays and calculate number of groups, bars, items, and means
    
    barMeans = np.zeros((np.shape(data)))
    items = np.zeros((np.shape(data)))
    
    nGroups = np.shape(data)[0]
    groupx = np.arange(1,nGroups+1)

    if len(np.shape(data)) > 1:
        grouped = True
        barspergroup = np.shape(data)[1]
        barwidth = (barwidth * groupwidth) / barspergroup
        
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                barMeans[i][j] = np.mean(data[i][j])
                items[i][j] = len(data[i][j])
        
    else:
        grouped = False
        barspergroup = 1
        
        for i in range(np.shape(data)[0]):
            barMeans[i] = np.mean(data[i])
            items[i] = len(data[i])
    
    # Calculate x values for bars and scatters
    
    xvals = np.zeros((np.shape(data)))
    barallocation = groupwidth / barspergroup
    k = (groupwidth/2) - (barallocation/2)
    
    if grouped == True:
        
        for i in range(np.shape(data)[0]):
            xrange = np.linspace(i+1-k, i+1+k, barspergroup)
            for j in range(barspergroup):
                xvals[i][j] = xrange[j]
    else:
        xvals = groupx
    
    # Set colors for bars and scatters
     
    barfacecolorArray = setcolors(barfacecoloroption, barfacecolor, barspergroup, nGroups, data)
    baredgecolorArray = setcolors(baredgecoloroption, baredgecolor, barspergroup, nGroups, data)
     
    scfacecolorArray = setcolors(scatterfacecoloroption, scatterfacecolor, barspergroup, nGroups, data, paired_scatter = paired)
    scedgecolorArray = setcolors(scatteredgecoloroption, scatteredgecolor, barspergroup, nGroups, data, paired_scatter = paired)
    
    # Initialize figure
    if ax == []:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    # Make bars
    barlist = []
    barx = []
    for x, y, bfc, bec in zip(xvals.flatten(), barMeans.flatten(),
                              barfacecolorArray, baredgecolorArray):
        barx.append(x)
        barlist.append(ax.bar(x, y, barwidth,
                         facecolor = bfc, edgecolor = bec,
                         zorder=-1))
    
    # Uncomment these lines to show method for changing bar colors outside of
    # function using barlist properties
    #for i in barlist[2].get_children():
    #    i.set_color('r')
    
    # Make scatters
    sclist = []
    if paired == False:
        for x, Yarray, scf, sce  in zip(xvals.flatten(), data.flatten(),
                                        scfacecolorArray, scedgecolorArray):
            if spaced == True:
                xVals, yVals = xyspacer(ax, x, Yarray, bindist=yspace, space=xspace)
                sclist.append(ax.scatter(xVals, yVals, s = scattersize,
                             c = scf,
                             edgecolors = sce,
                             zorder=20))           
            else:
                for y in Yarray:
                     sclist.append(ax.scatter(x, y, s = scattersize,
                                     c = scf,
                                     edgecolors = sce,
                                     zorder=20))

    elif grouped == True:
        for x, Yarray, scf, sce in zip(xvals, data, scfacecolorArray, scedgecolorArray):
            for y in np.transpose(Yarray.tolist()):
                sclist.append(ax.plot(x, y, '-o', markersize = scattersize/10,
                         color = scatterlinecolor,
                         linewidth=linewidth,
                         markerfacecolor = scf,
                         markeredgecolor = sce,
                         zorder=20))
    elif grouped == False:
        for n,_ in enumerate(data[0]):
            y = [y[n-1] for y in data]
            sclist.append(ax.plot(xvals, y, '-o', markersize = scattersize/10,
                         color = scatterlinecolor,
                         linewidth=linewidth,
                         markerfacecolor = scfacecolorArray[0],
                         markeredgecolor = scedgecolorArray[0],
                         zorder=20))
    
    # Label axes
    if ylabel != 'none':
        ax.set_ylabel(ylabel)
    
    if xlabel != 'none':
        ax.set_xlabel(xlabel)
    
    # Set range and tick values for Y axis
    if yaxisparams != 'auto':
        ax.set_ylim(yaxisparams[0])
        ax.set_yticks(yaxisparams[1])
       
    # X ticks
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off') # labels along the bottom edge are off

    if len(xlim) > 0:
        ax.set_xlim(xlim)
    if len(ylim) > 0:
        ax.set_ylim(ylim)
        
    xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        
    if grouplabel == 'auto':
        ax.tick_params(labelbottom='off')
    else:
        ax.tick_params(labelbottom='off')

        groupx = np.arange(1, len(grouplabel)+1)
        if len(xlim) > 0:
            groupx = [x for x in groupx]
        xpos = (groupx - ax.get_xlim()[0])/xrange

        for x, label in zip(xpos, grouplabel):
            ax.text(x, -0.05+grouplabeloffset, label, va='top', ha='center', fontsize=xfontsize, transform=ax.transAxes)
        
    if len(barlabels) > 0:
        if len(barlabels) != len(barx):
            print('Wrong number of bar labels for number of bars!')
        else:
            xpos = (barx - ax.get_xlim()[0])/xrange
            ypos = (-ax.get_ylim()[0]/yrange) - barlabeloffset

            for x, label in zip(xpos, barlabels):
                ax.text(x, ypos, label, va='top', ha='center', fontsize=xfontsize, transform=ax.transAxes)

    # Hide the right and top spines and set bottom to zero
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    
    if show_legend == 'within':
        if len(itemlabel) != barspergroup:
            print('Not enough item labels for legend!')
        else:
            legendbar = []
            legendtext = []
            for i in range(barspergroup):
                legendbar.append(barlist[i])
                legendtext.append(itemlabel[i])
            plt.legend(legendbar, legendtext, loc=legendloc)
    
    return ax, barx, barlist, sclist

#plt.savefig('foo.png')
        
# To do
# check if n's are the same for paired and if not default to unpaired
# add color options for scatters
# add alpha options etc
# add axis options
# remove white background
# work out how to export or save as pdf, tiff, eps etc
# work out how to return handles to scatters so can be altered outside of function
# make help doc
# make html file to show usage using ijupyter

      
def setcolors(coloroption, colors, barspergroup, nGroups, data, paired_scatter = False):
            
    nColors = len(colors)
    
    if (paired_scatter == True) & (coloroption == 'within'):
        print('Not possible to make a Paired scatter plot with Within setting.')
        coloroption = 'same'
        
    if coloroption == 'within':
        if nColors < barspergroup:
            print('Not enough colors for this option! Reverting to one color.')
            coloroption = 'same'
        elif nColors > barspergroup:
            colors = colors[:barspergroup]
        coloroutput = [colors for i in data]
        coloroutput = list(chain(*coloroutput))
        
    if coloroption == 'between':
        if nColors < nGroups:
            print('Not enough colors for this option! Reverting to one color.')
            coloroption = 'same'
        elif nColors > nGroups:
            colors = colors[:nGroups]
        if paired_scatter == False:
            coloroutput = [[c]*barspergroup for c in colors]
            coloroutput = list(chain(*coloroutput))
        else:
            coloroutput = colors
            
    if coloroption == 'individual':
        if nColors < nGroups*barspergroup:
            print('Not enough colors for this color option')
            coloroption = 'same'
        elif nColors > nGroups*barspergroup:
            coloroutput = colors[:nGroups*barspergroup]
        else: 
            coloroutput = colors
    
    if coloroption == 'same':
        coloroutput = [colors[0] for x in range(len(data.flatten()))]

    return coloroutput

def data2obj1D(data):
    obj = np.empty(len(data), dtype=np.object)
    for i,x in enumerate(data):
        obj[i] = np.array(x)  
    return obj

def data2obj2D(data):
    obj = np.empty((np.shape(data)[0], np.shape(data)[1]), dtype=np.object)
    for i,x in enumerate(data):
        for j,y in enumerate(x):
            obj[i][j] = np.array(y)
    return obj

def xyspacer(ax, x, yvals, bindist=20, space=0.1):
    
    histrange=[]
    histrange.append(min(ax.get_ylim()[0], min(yvals)))
    histrange.append(max(ax.get_ylim()[1], max(yvals)))

    yhist = np.histogram(yvals, bins=bindist, range=histrange)
    
    xvals=[]
    for ybin in yhist[0]:
        if ybin == 1:
            xvals.append(x)
        elif ybin > 1:          
            temp_vals = np.linspace(x-space, x+space, num=ybin)
            for val in temp_vals:
                xvals.append(val)
                
    yvals.sort()

    return xvals, yvals

def depth(seq):
    for level in count():
        if not seq:
            return level
        seq = list(chain.from_iterable(s for s in seq if isinstance(s, Sequence)))

        
# Arguments to include in function

def trialsFig(ax, trials, pps=1, preTrial=10, scale=5, noiseindex = [],
              plotnoise=True,
              eventText='event', 
              ylabel=''):

    if len(noiseindex) > 0:
        trialsNoise = np.array([i for (i,v) in zip(trials, noiseindex) if v])
        trials = np.array([i for (i,v) in zip(trials, noiseindex) if not v])
        if plotnoise == True:
            ax.plot(trialsNoise.transpose(), c='red', alpha=0.1)
        
    ax.plot(trials.transpose(), c='grey', alpha=0.4)
    ax.plot(np.mean(trials,axis=0), c='k', linewidth=2)
     
    ax.set(ylabel = ylabel)
    ax.xaxis.set_visible(False)
            
    scalebar = scale * pps

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    scalebary = (yrange / 10) + ax.get_ylim()[0]
    scalebarx = [ax.get_xlim()[1] - scalebar, ax.get_xlim()[1]]
    
    ax.plot(scalebarx, [scalebary, scalebary], c='k', linewidth=2)
    ax.text((scalebarx[0] + (scalebar/2)), scalebary-(yrange/50), str(scale) +' s', ha='center',va='top')
 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    xevent = pps * preTrial  
    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    
    return ax

def trialsMultFig(ax, trials, pps=1, preTrial=10, scale=5,
              eventText='event', 
              ylabel='',
              linecolor=['m', 'b'],):
    
    for i in [0, 1]:
        y = trials[i].transpose()   
        ax.plot(y, c=linecolor[i], linewidth=1)
     
    ax.set(ylabel = ylabel)
    ax.xaxis.set_visible(False)
            
    scalebar = scale * pps

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    scalebary = (yrange / 10) + ax.get_ylim()[0]
    scalebarx = [ax.get_xlim()[1] - scalebar, ax.get_xlim()[1]]
    
    ax.plot(scalebarx, [scalebary, scalebary], c='k', linewidth=2)
    ax.text((scalebarx[0] + (scalebar/2)), scalebary-(yrange/50), str(scale) +' s', ha='center',va='top')
 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    xevent = pps * preTrial
    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    
    return ax

def trialsShadedFig(ax, trials, pps = 1, scale = 5, preTrial = 10,
                    noiseindex=[],
                    eventText = 'event', ylabel = '',
                    linecolor='k', errorcolor='grey'):
    
    if len(noiseindex) > 0:
        trials = np.array([i for (i,v) in zip(trials, noiseindex) if not v])
    
    yerror = [np.std(i)/np.sqrt(len(i)) for i in trials.T]
    y = np.mean(trials,axis=0)
    x = np.arange(0,len(y))
    
    ax.plot(x, y, c=linecolor, linewidth=2)

    errorpatch = ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    ax.set(ylabel = ylabel)
    ax.xaxis.set_visible(False)
            
    scalebar = scale * pps

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    scalebary = (yrange / 10) + ax.get_ylim()[0]
    scalebarx = [ax.get_xlim()[1] - scalebar, ax.get_xlim()[1]]
    
    ax.plot(scalebarx, [scalebary, scalebary], c='k', linewidth=2)
    ax.text((scalebarx[0] + (scalebar/2)), scalebary-(yrange/50), '5 s', ha='center',va='top')
 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    xevent = pps * preTrial
    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    
    return ax, errorpatch

def trialsMultShadedFig(ax, trials, pps = 1, scale = 5, preTrial = 10,
                        noiseindex=[],
                        eventText = 'event', ylabel = '',
                        linecolor=['m', 'b'], errorcolor=['r', 'g'],
                        title=''):

    for i in [0, 1]:
        if len(noiseindex) > 0:
            trials[i] = np.array([i for (i,v) in zip(trials[i], noiseindex) if not v])
        yerror = [np.std(i)/np.sqrt(len(i)) for i in trials[i].T]
        y = np.mean(trials[i],axis=0)
        x = np.arange(0,len(y))
    
        ax.plot(x, y, c=linecolor[i], linewidth=2)

        errorpatch = ax.fill_between(x, y-yerror, y+yerror, color=errorcolor[i], alpha=0.4)
    
    ax.set(ylabel = ylabel)
    ax.xaxis.set_visible(False)
            
    scalebar = scale * pps

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    scalebary = (yrange / 10) + ax.get_ylim()[0]
    scalebarx = [ax.get_xlim()[1] - scalebar, ax.get_xlim()[1]]
    
    ax.plot(scalebarx, [scalebary, scalebary], c='k', linewidth=2)
    ax.text((scalebarx[0] + (scalebar/2)), scalebary-(yrange/50), '5 s', ha='center',va='top')
 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    xevent = pps * preTrial
    ax.plot([xevent, xevent],[ax.get_ylim()[0], ax.get_ylim()[1] - yrange/20],'--')
    ax.text(xevent, ax.get_ylim()[1], eventText, ha='center',va='bottom')
    ax.set_title(title)
    
    return ax, errorpatch

def trialstiledFig(gs, trials, pps = 1, preTrial = 10):
    for i in np.arange(np.shape(trials)[0]):
        col=int(i/7)
        row=(i%7)+1

        ax = plt.subplot(gs[row, col])
        ax.plot(trials[i,:].transpose(), c='grey', alpha=0.4)   

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        

def licklengthFig(ax, data, contents = '', color='grey'):          
    if len(data['longlicks']) > 0:
        longlicklabel = str(len(data['longlicks'])) + ' long licks,\n' +'max = ' + '%.2f' % max(data['longlicks']) + ' s.'        
    else:
        longlicklabel = 'No long licks.'
    
    figlabel = str(len(data['licklength'])) + ' total licks.\n' + longlicklabel

    ax.hist(data['licklength'], np.arange(0, 0.3, 0.01), color=color)
    ax.text(0.9, 0.9, figlabel, ha='right', va='top', transform = ax.transAxes)
    ax.set_xlabel('Lick length (s)')
    ax.set_ylabel('Frequency')
    ax.set_title(contents)
    
def iliFig(ax, data, contents = '', color='grey'):
    ax.hist(data['ilis'], np.arange(0, 0.5, 0.02), color=color)
    
    figlabel = '%.2f Hz' % data['freq']
    ax.text(0.9, 0.9, figlabel, ha='right', va='top', transform = ax.transAxes)
    
    ax.set_xlabel('Interlick interval (s)')
    ax.set_ylabel('Frequency')
    ax.set_title(contents)
    
def burstlengthFig(ax, data, contents='', color3rdbar=False):
    
    figlabel = (str(data['bNum']) + ' total bursts\n' +
                str('%.2f' % data['bMean']) + ' licks/burst.')
                                                
    n, bins, patches = ax.hist(data['bLicks'], range(0, 20), normed=1)
    ax.set_xticks(range(1,20))
    ax.set_xlabel('Licks per burst')
    ax.set_ylabel('Frequency')
    ax.set_xticks([1,2,3,4,5,10,15])
#        ax.text(0.9, 0.9, figlabel1, ha='right', va='center', transform = ax.transAxes)
    ax.text(0.9, 0.8, figlabel, ha='right', va='center', transform = ax.transAxes)
    
    if color3rdbar == True:
        patches[3].set_fc('r')
    
def ibiFig(ax, data, contents = ''):
    ax.hist(data['bILIs'], range(0, 20), normed=1)
    ax.set_xlabel('Interburst intervals')
    ax.set_ylabel('Frequency')
    
def heatmapFig(ax, data, events = [], pps = 1, sortEvs = [], sortDir = 'asc'):
    if len(sortEvs)> 0:
        sortOrder = np.argsort(sortEvs)
        data = [data[i, :] for i in sortOrder] 

    ntrials,_ = np.shape(data)
    ax.pcolormesh(data, shading = 'flat')
    ax.set_ylabel('Trials')
    ax.set_yticks([0, ntrials-1])
    ax.invert_yaxis()
#    
def addevent2heatmap(ax, events, pps = 1, color='w'):
    for i, x in enumerate(events):
        try:
            ax.plot([(x+10)*pps, (x+10)*pps], [i, i+1], c=color)
        except ZeroDivisionError:
            pass
        
def sessionlicksFig(ax, data):
    ax.hist(data['licks'], range(0, 3600, 60))
      
    yraster = [ax.get_ylim()[1]] * len(data['licks'])
    ax.scatter(data['licks'], yraster, s=50, facecolors='none', edgecolors='grey')
    
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Licks per min')
#    ax.set_title('Rat ' + self.rat + ': Session ' + self.session)

def sessioncueslicksFig(ax, cues1, licks1, cues2=[], licks2=[]):
    ax.hist(data['licks'], range(0, 3600, 60))
      
    yraster = [ax.get_ylim()[1]] * len(data['licks'])
    ax.scatter(data['licks'], yraster, s=50, facecolors='none', edgecolors='grey')
    
    ax.set_xticks(np.multiply([0, 10, 20, 30, 40, 50, 60],60))
    ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Licks per min')
#    ax.set_title('Rat ' + self.rat + ': Session ' + self.session)

def distractionrasterFig(ax, timelock, events,
                         pre = 1, post = 1,
                         sortevents=None, sortdirection='ascending'):

    if sortevents != None:
        if len(timelock) != len(sortevents):
            print('Length of sort events does not match timelock events; no sorting')
        else:
            if sortdirection == 'ascending':
                sortOrder = np.argsort(sortevents)
            else:
                sortOrder = np.argsort(sortevents)[::-1]
                
            timelock = [timelock[i] for i in sortOrder]
    
    rasterData = [[] for i in timelock]
    
    for i,x in enumerate(timelock):
        rasterData[i] = [j-x for j in events if (j > x-pre) & (j < x+post)]

    for ith, trial in enumerate(rasterData):
        ax.vlines(trial, ith + .5, ith + 1.5)
        
def cuerasterFig(ax, timelock, events,
                         pre = 10, post = 30,
                         sortevents=None, sortdirection='ascending'):

    if sortevents != None:
        if len(timelock) != len(sortevents):
            print('Length of sort events does not match timelock events; no sorting')
        else:
            if sortdirection == 'ascending':
                sortOrder = np.argsort(sortevents)
            else:
                sortOrder = np.argsort(sortevents)[::-1]
                
            timelock = [timelock[i] for i in sortOrder]
    
    rasterData = [[] for i in timelock]
    
    for i,x in enumerate(timelock):
        rasterData[i] = [j-x for j in events if (j > x-pre) & (j < x+post)]

    for ith, trial in enumerate(rasterData):
        ax.vlines(trial, ith + .5, ith + 1.5)   
        
def firstVsphotoFig(ax, photodata, lickdata):
    
    avgphoto = [np.mean(i[100:110]) for i in photodata]
    
    ax.scatter(avgphoto, lickdata)
    
def cumulativelickFig(ax, firstlick, normed=True, color='g'):
    sorted_data = np.sort(firstlick)
    yvals = np.arange(len(sorted_data)+1)
    
    if normed == True:
        nlicks = len(sorted_data)
        yvals =yvals/nlicks
        
    a = ax.step(np.concatenate([sorted_data, sorted_data[[-1]]]),
             yvals, color=color)
    ax.set_xscale('log', basex=10)
    ax.set_xlim([0.1, 1000])
    
    return ax, a

def latencyFig(ax, x):
    lats = []
    latlabels = []
    colors = []
    try:
        lats.append(x.left['lats'])
        latlabels.append(x.left['subs'][:3])
        colors.append(x.left['color'])
    except:
        pass
    
    try:
        lats.append(x.right['lats'])
        latlabels.append(x.right['subs'][:3])
        colors.append(x.right['color'])
    except:
        pass
    
    for x, vals in enumerate(lats):
        ax.bar(x+1, np.nanmean(vals), facecolor=colors[x], edgecolor='none',zorder=-1)
        ax.scatter((x+1)*np.ones([len(vals)]), vals, c='none', edgecolors='k', zorder=1)
   
    ax.set_xlim([0, 3])
    ax.set_xticks(np.arange(1, len(lats)+1))
    ax.set_xticklabels(latlabels)
    ax.set_ylabel('Latency (s)')
    
def setsameaxislimits(axes, axis='y'):
    axmin = []
    axmax = []
    for ax in axes:
        axmin.append(ax.get_ylim()[0])
        axmax.append(ax.get_ylim()[1])
    min_axmin = np.min(axmin)
    max_axmax = np.max(axmax)
    
    for ax in axes:
        ax.set_ylim([min_axmin, max_axmax])
#                sclist.append(ax.plot(x, y, '-o', markersize = scattersize/10,
#                         color = scatterlinecolor,
#                         markerfacecolor = scf,
#                         markeredgecolor = sce))
        
def invisible_axes(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for sp in ['left', 'right', 'top', 'bottom']:
        ax.spines[sp].set_visible(False)

def shadedError(ax, yarray, linecolor='black', errorcolor = 'xkcd:silver', linewidth=1):
    yarray = np.array(yarray)
    y = np.mean(yarray, axis=0)
    yerror = np.std(yarray)/np.sqrt(len(yarray))
    x = np.arange(0, len(y))
    ax.plot(x, y, color=linecolor, linewidth=1)
    ax.fill_between(x, y-yerror, y+yerror, color=errorcolor, alpha=0.4)
    
    return ax

def ax2prop(axlims, n):
    """
    axlims should be provided as result of ax.get_xlims()
    n should be a list of values to convert
    """
    axrange = axlims[1] - axlims[0]
    output = []
    for n in n:
        output.append((n - axlims[0])/axrange)
        
    return output

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_violinstats(dataset, points=100, bw_method=None):

    def _kde_method(X, coords):
        # fallback gracefully if the vector contains only one value
        if np.all(X[0] == X):
            return (X[0] == coords).astype(float)
        kde = mlab.GaussianKDE(X, bw_method)
        return kde.evaluate(coords)

    vpstats = cbook.violin_stats(dataset, _kde_method, points=points)
    return vpstats