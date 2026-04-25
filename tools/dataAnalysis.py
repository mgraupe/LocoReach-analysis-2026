import time
import numpy as np
import sys
import os
import scipy, scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
from scipy import io
import pdb
import scipy.ndimage
import itertools
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import cv2
from scipy import signal
from scipy.signal import find_peaks
import pickle
import random
from statsmodels.stats.anova import anova_single
import scikits.bootstrap as boot
from scipy import ndimage
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib
import multiprocessing as mp
from joblib import Parallel, delayed
import scipy.stats as stats
from tools.pyqtgraph.Qt import QtGui, QtCore
import tools.pyqtgraph as pg
matplotlib.use('TkAgg') # WxAgg
from array import array
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import array as arr
from numpy import trapz
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
numcores = mp.cpu_count()-1
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy.stats import vonmises_line

def getSpeed(angles,times,circumsphere,minSpacing):
    angleJumps = angles[np.concatenate((([True]),np.diff(angles)!=0.))] # find angles at the points where the angle value changes
    timePoints = times[np.concatenate((([True]),np.diff(angles)!=0.))]  # find times at the points where the angle value changes
    #

    #
    dt = np.diff(timePoints) # delta t values of the time array
    dtMultipleSpace = dt//minSpacing # how many times does the minSpacing fit in the gaps
    dtMultipleSpace = dtMultipleSpace[dtMultipleSpace>1] # gap must be as big as 2 times the spacing to add a point
    # note that gap must be larger than 2 times the spacing
    startGap = np.arange(len(timePoints))[np.hstack(((dt>2.*minSpacing),np.array(False)))] # index values of the start of the gap
    endGap = np.arange(len(timePoints))[np.hstack((np.array(False),(dt>2.*minSpacing)))] # index values of the end of the gap
    newSpacingValue = (timePoints[endGap] - timePoints[startGap]) / dtMultipleSpace
    newTvalues = []
    newAvalues = []
    for i in range(len(dtMultipleSpace)):
        newTvalues.extend(timePoints[startGap][i] + newSpacingValue[i] * np.arange(1, dtMultipleSpace[i]))
        newAvalues.extend(np.repeat(angleJumps[startGap][i], (dtMultipleSpace[i] - 1)))
    timesNew = np.hstack((timePoints,np.asarray(newTvalues)))
    anglesNew = np.hstack((angleJumps,np.asarray(newAvalues)))
    both = np.row_stack((timesNew,anglesNew))
    bothSorted = both[:,both[0].argsort()]
    angularSpeed = np.diff(bothSorted[1])/np.diff(bothSorted[0])
    angularSpeedM = (angularSpeed[1:]+angularSpeed[:-1])/2.

    linearSpeed = angularSpeedM*circumsphere/360.
    speedTimes = bothSorted[0][1:-1]
    #pdb.set_trace()
    angularSpeedSmooth = scipy.signal.medfilt(angularSpeed,kernel_size=9)
    linearSpeedSmooth = scipy.signal.medfilt(linearSpeed,kernel_size=9)
    return (angularSpeedSmooth,linearSpeedSmooth,speedTimes,angularSpeed,linearSpeed)

def crosscorr(deltat, y0, y1, correlationRange=1.5, fast=False):
    """
            home-written routine to calcualte cross-correlation between two contiuous traces
            new version from February 9th, 2011
    """

    if len(y0) != len(y1):
        print('Data to be correlated has different dimensions!')
        sys.exit(1)

    y0mean = y0.mean()
    y1mean = y1.mean()
    y0sd = y0.std()
    y1sd = y1.std()

    if y0sd != 0 and y1sd != 0:
        y0norm = (y0 - y0mean) / y0sd
        y1norm = (y1 - y1mean) / y1sd
    else:
        y0norm = y0 - y0mean
        y1norm = y1 - y1mean

    # defined range calculation of cross-correlation
    # value is specified in main routine
    # deltat = 0.9

    pointnumber1 = len(y0)

    ncorrrange = np.ceil(correlationRange / deltat)
    corrrange = np.arange(2 * ncorrrange + 1) - ncorrrange
    ycorr = np.zeros(len(corrrange))

    # print corrrange
    if fast:
        pass
    else:
        for n in corrrange:
            corrpairs = pointnumber1 - abs(n)
            # ccc = arange(corrpairs)
            # print n
            if n < 0:
                y1mod = np.hstack((y1norm[int(-abs(n)):], y1norm[:-int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n > -10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1)
            elif n == 0:
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1norm)) / (float(pointnumber1))
                # print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], (float(pointnumber1-1))
            elif n > 0:
                y1mod = np.hstack((y1norm[int(abs(n)):], y1norm[:int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n < 10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1-1)
            else:
                print('Problem!')
                exit(1)
            # print n , ycorr[n+ncorrrange]

    float_corrrange = np.array([float(i) for i in corrrange])

    xcorr = float_corrrange * deltat

    normcorr = np.column_stack((xcorr, ycorr))
    return normcorr

    ############################################################
    ## high-pass filter from http://nullege.com/codes/show/src@obspy.signal-0.3.3@obspy@signal@filter.py
    ############################################################

def highpass(data, freq, df=200, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency freq using corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz; Default 200.
    :param corners: Filter corners. Note: This is twice the value of PITSA's
            filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
            This results in twice the number of corners but zero phase shift in
            the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='highpass', ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)

############################################################
## high-pass filter from http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
############################################################
def butter_highpass(interval, sampling_rate, cutoff, order=5):

    nyq = sampling_rate * 0.5

    stopfreq = float(cutoff)
    cornerfreq = 0.4 * stopfreq  # (?)

    ws = cornerfreq / nyq
    wp = stopfreq / nyq

    # for bandpass:
    # wp = [0.2, 0.5], ws = [0.1, 0.6]

    N, wn = scipy.signal.buttord(wp, ws, 3, 16)  # (?)

    # for hardcoded order:
    # N = order

    b, a = scipy.signal.butter(N, wn, btype='high')  # should 'high' be here for bandpass?
    sf = scipy.signal.lfilter(b, a, interval)
    return sf

##################################################################
## high-pass filters the ephys recording and extracts spikes through thresholding
##################################################################
def extractSpikes(eData, eTime, stim=False):

    highpassfreq = 150.  # Hz
    spikecountwindow = 0.05  # in sec
    binWidth = 1.E-3  # in sec
    stimRinging = 0.002

    dt = np.mean(eTime[1:] - eTime[:-1])
    rate = 1. / dt

    # set binned array for convolution
    binWidth = 1.E-3  # in sec
    tbins = np.linspace(0., len(eData) * dt, int(len(eData) * dt / binWidth) + 1)
    nspikecountwindow = spikecountwindow / binWidth

    ############################################
    # create new group in hdf5 file
    #grp_spikes = self.analyzed_data.require_group('spiking_data')

    detectSpikes = True
    #if ('spikeTreshold' in grp_spikes.keys()) and ('artifactTreshold' in grp_spikes.keys()):
    #    input_ = raw_input('Spike and Artifact detection thresholds exist already. Do you want to re-detect spikes? (\'y\', or any other key for no) : ')
    #    if input_ != 'y':
    #        detectSpikes = False

    if detectSpikes:
        # get time of the stimuls
        if stim:
            # in case of external stimulation: exclude period of stimuli
            stimuli = self.analyzed_data['stimulation_data/stimulus_times'].value
            startStim = np.array(stimuli / dt, dtype=int)
            endStim = int(stimuli[-1] / dt) + int(stimRinging / dt)  # eDataReplaced = copy(eData)

        # high-pass filter recording #################################
        # eDataHP = self.analysisTools.highpass(eData,highpassfreq,rate,corners=4,zerophase=True)
        eDataHP = butter_highpass(eData, rate, highpassfreq, order=4)
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'ephys_data_high-pass', eDataHP)

        # detect spikes ################################################
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow(title="Data plotting")
        win.resize(1800, 600)
        win.setWindowTitle('high-pass filtered recording')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        pg.setConfigOptions(antialias=True)
        x2 = np.linspace(-100, 100, 1000)
        data2 = np.sin(x2) / x2
        p8 = win.addPlot(row=1, col=0, title="set threshold for spike detection with mouse click")
        p8.plot(eDataHP, pen=(255, 255, 255, 200))
        # lr = pg.LinearRegionItem([400,700])
        # lr.setZValue(-10)
        # p8.addItem(lr)
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        hLineSpikes = pg.InfiniteLine(angle=0, pen=pg.mkPen(0, 255, 0), movable=False)
        hLineArtifacts = pg.InfiniteLine(angle=0, pen=pg.mkPen(255, 0, 0), movable=False)
        p8.addItem(vLine, ignoreBounds=True)
        p8.addItem(hLine, ignoreBounds=True)
        p8.addItem(hLineSpikes, ignoreBounds=True)
        p8.addItem(hLineArtifacts, ignoreBounds=True)
        vb = p8.vb

        # detectionTreshold = empty(0)

        def detectSpikeTimes(tresh):
            global detectionTreshold
            excursion = eDataHP < tresh  # threshold ephys trace
            excursionInt = np.array(excursion, dtype=int)  # convert boolean array into array of zeros and ones
            diff = excursionInt[1:] - excursionInt[:-1]  # calculate difference
            spikeStart = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == 1))]  # a difference of one is the start of a spike
            spikeEnd = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == -1))]  # a difference of -1 is the spike end
            if (spikeEnd[0] - spikeStart[0]) < 0.:  # if trace starts below threshold
                spikeEnd = spikeEnd[1:]
            if (spikeEnd[-1] - spikeStart[-1]) < 0.:  # if trace ends below threshold
                spikeStart = spikeStart[:-1]
            if len(spikeStart) != len(spikeEnd):  # unequal lenght of starts and ends is a problem of course
                print('problem in length of spikeStart and spikeEnd')
                sys.exit(1)
            spikeT = []
            for i in range(len(spikeStart)):
                if (spikeEnd[i] - spikeStart[i]) > 10:  # ignore if difference between end and start is smaller than 15 points
                    nMin = np.argmin(eDataHP[spikeStart[i]:spikeEnd[i]]) + spikeStart[i]
                    spikeT.append(nMin)
            # detectionTreshold = tresh
            return spikeT

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if p8.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                if index > 0 and index < len(eDataHP):
                    label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='font-size: 12pt'>y=%s</span>" % (mousePoint.x(), eDataHP[index]))
                vLine.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())

        pointSpikes = [0]
        pointArtifacts = [0]
        sSpikes = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0))
        sSpikes.addPoints(x=pointSpikes, y=len(pointSpikes) * [0])
        p8.addItem(sSpikes)

        sArtifacts = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
        sArtifacts.addPoints(x=pointArtifacts, y=len(pointArtifacts) * [0])
        p8.addItem(sArtifacts)

        def mouseClickedSpikes(evt):
            posClick = evt.pos()
            if p8.sceneBoundingRect().contains(posClick):
                mousePointC = vb.mapSceneToView(posClick)
                hLineSpikes.setPos(mousePointC.y())
                threshold = mousePointC.y()
                pointSpikes = detectSpikeTimes(threshold)
                # print 'spikes clicked:',pointSpikes
                sSpikes.setData(x=pointSpikes, y=eDataHP[pointSpikes])

        def mouseClickedArtifacts(evt):
            posClick = evt.pos()
            if p8.sceneBoundingRect().contains(posClick):
                mousePointC = vb.mapSceneToView(posClick)
                hLineArtifacts.setPos(mousePointC.y())
                # print type(evt)
                threshold = mousePointC.y()
                pointArtifacts = detectSpikeTimes(threshold)
                # sArtifacts.clear()
                sSpikes.setData(x=spikesRaw[0], y=spikesRaw[1])
                sArtifacts.setData(x=pointArtifacts, y=eDataHP[pointArtifacts])

        # first graphical dialog to set spike treshold
        proxy = pg.SignalProxy(p8.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
        p8.scene().sigMouseClicked.connect(mouseClickedSpikes)

        pdb.set_trace()  # input_ = input("Chose spike treshold in graphical window. Press any button to continue.")
        spikesRaw = sSpikes.getData()
        spikeTreshold = hLineSpikes.getPos()[1]  # copy(detectionTreshold)

        # second graphical dialog to set treshold for artifacts
        p8.scene().sigMouseClicked.disconnect(mouseClickedSpikes)
        p8.scene().sigMouseClicked.connect(mouseClickedArtifacts)

        pdb.set_trace()
        # input_ = input("Chose artifact treshold in graphical window. Press any button to continue.")

        falseSpikes = sArtifacts.getData()
        artifactTreshold = hLineArtifacts.getPos()[1]

        while True:
            input_ = input("Enter pairs of indicies of regions to excluce from spike detection (e.g. [[0,700],[5450,5560]]). Press any number if None. : ")
            try:
                aaa = len(input_)
            except:
                print('No regions to exclude specified.')
                exclusionBorders = None
                break
            else:
                print('recorded')
                exclusionBorders = input_
                break
        while True:
            input2_ = input("Enter steps to exclude after stimulus onset - length of stimulus artifact (e.g. 200 corresponding to 2 ms). Press any key if None. : ")
            try:
                type(input2_)
            except:
                print('No regions to exclude specified.')
                artifactLength = None
                break
            else:
                artifactLength = input2_
                break

        #
        # pdb.set_trace()
        lspikes = spikesRaw[0].tolist()
        lartif = falseSpikes[0].tolist()
        spikes0 = [x for x in lspikes if x not in lartif]
        # add spike artifacts to regions to remove
        if artifactLength:
            if exclusionBorders == None:
                exclusionBorders = []
            if stim:
                for i in range(len(startStim)):
                    exclusionBorders.append([startStim[i], startStim[i] + artifactLength])
        # remove spikes which fall in to regions to exclude
        if exclusionBorders:
            spikes1 = list(spikes0)
            for n in range(len(exclusionBorders)):
                spikes1 = [x for x in spikes1 if not (x > exclusionBorders[n][0] and x < exclusionBorders[n][1])]

        spikeTimes = eTime.value[np.array(spikes1, dtype=int)]
        #firingRate = brian.firing_rate(spikeTimes)
        #cv = brian.CV(spikeTimes)

        # pdb.set_trace()
        ######################################################
        # convolv original spike trains with Gaussian kernels
        binnedspikes, _ = np.histogram(spikeTimes, tbins)
        spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
        # convert the convolved spike trains to units of spikes/sec
        spikesconv *= 1. / binWidth

        # save data
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'spikeTreshold', array([spikeTreshold]))
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'artifactTreshold', array([artifactTreshold]))
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'spikes', spikeTimes)
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'firing_rate_evolution', spikesconv, ['dt', binWidth])
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'firing_rate', array([firingRate]))
        #self.h5pyTools.createOverwriteDS(grp_spikes, 'CV', array([cv]))

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
#################################################################################
# detect spikes in ephys trace
#################################################################################
def detectSpikeTimes(tresh,eDataHP,ephysTimes,positive=True,plot=False):
    #global detectionTreshold
    while True:
        if positive :
            excursion = eDataHP > tresh  # threshold ephys trace
        else:
            excursion = eDataHP < tresh
        excursionInt = np.array(excursion, dtype=int)  # convert boolean array into array of zeros and ones
        diff = excursionInt[1:] - excursionInt[:-1]  # calculate difference
        spikeStart = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == 1))]  # a difference of one is the start of a spike
        spikeEnd = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == -1))]  # a difference of -1 is the spike end
        if len(spikeEnd)>0 and len(spikeStart)>0:
            if (spikeEnd[0] - spikeStart[0]) < 0.:  # if trace starts below threshold
                spikeEnd = spikeEnd[1:]
            if (spikeEnd[-1] - spikeStart[-1]) < 0.:  # if trace ends below threshold
                spikeStart = spikeStart[:-1]
            if len(spikeStart) != len(spikeEnd):  # unequal lenght of starts and ends is a problem of course
                print('problem in length of spikeStart and spikeEnd')
                sys.exit(1)
        spikeT = []
        spikeStart = spikeStart[spikeStart>100]
        #for i in range(len(spikeStart)):
        #    #if (spikeEnd[i] - spikeStart[i]) > 10:  # ignore if difference between end and start is smaller than 15 points
        #    nMin = spikeStart[i] #np.argmin(eDataHP[spikeStart[i]:spikeEnd[i]]) + spikeStart[i]
        #    spikeT.append(nMin)
        # detectionTreshold = tresh
        #pdb.set_trace()
        spikeTimes = ephysTimes[spikeStart]
        if plot:
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            ax.plot(ephysTimes,eDataHP)
            ax.plot(ephysTimes[spikeStart], eDataHP[spikeStart],'.')
            ax.axhline(y=tresh, ls='--', c='0.5')
            plt.show()
        if plot:
            print('Is threshold ok? ->No : type new threshold value  ->Yes : press Enter')
            recInput = input()
            if recInput == "":
                break
            else:
                tresh = float(recInput)
            print('new threshold : ', tresh)
        else:
            break
        #recInputIdx = [int(i) for i in recInput.split(',')]
    return (spikeTimes,spikeStart,tresh)

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
def mapToXbit(inputArray,xBitEncoding):
    oldMin = np.min(inputArray)
    oldMax = np.max(inputArray)
    newMin = 0.
    newMax = 2**xBitEncoding-1.
    normXBit = newMin + (inputArray - oldMin) * newMax / (oldMax - oldMin)
    normXBitInt = np.array(normXBit, dtype=int)
    return normXBitInt

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# dataAnalysis.determineFrameTimes(exposureArray[0],arrayTimes,frames)
def determineFrameTimes(exposureArray,arrayTimes,frames,rec=None):
    display = False
    #pdb.set_trace()
    numberOfFrames = len(frames)
    exposure = exposureArray > 20                # threshold trace
    exposureInt = np.array(exposure, dtype=int)  # convert boolean array into array of zeros and ones
    difference = np.diff(exposureInt)            # calculate difference
    expStart = np.arange(len(exposureArray))[np.concatenate((np.array([False]), difference == 1))]  # a difference of one is the start of a spike
    expEnd = np.arange(len(exposureArray))[np.concatenate((np.array([False]), difference == -1))]  # a difference of -1 is the spike end
    if (expEnd[0] - expStart[0]) < 0.:  # if trace starts above threshold
        expEnd = expEnd[1:]
    if (expEnd[-1] - expStart[-1]) < 0.:  # if trace ends above threshold
        expStart = expStart[:-1]
    frameDuration = expEnd - expStart
    midExposure = (expStart + expEnd)/2
    expStartTime = arrayTimes[expStart.astype(int)]
    expEndTime   = arrayTimes[expEnd.astype(int)]
    #framesIdxDuringRec = np.array(len(softFrameTimes))[(arrayTimes[expEnd[0]]+0.002) < softFrameTimes]
    #framesIdxDuringRec = framesIdxDuringRec[:len(expStart)]

    if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
        recordedFrames = frames[3:(len(midExposure) + 3)]
    elif arrayTimes[int(midExposure[0])]<0.003:
        recordedFrames = frames[2:(len(midExposure) + 2)]
    else:
        recordedFrames = frames[:len(midExposure)]
    print('number of tot. frames, recorded frames, exposures start, end :',numberOfFrames,len(recordedFrames), len(expStart), len(expEnd))
    if display:
        ledON = np.zeros(len(exposureArray))
        for i in range(11):
            ledON[((i*1.)<=arrayTimes) & ((i*1.+0.2)>arrayTimes)] = 1.
        ledON[29.<=arrayTimes] = 1.
        data = np.loadtxt('/home/mgraupe/2019.04.01_000-%s.csv' % (rec[-3:]),delimiter=',',skiprows=1,usecols=(0,1))
        print(len(data))
        plt.plot(arrayTimes,exposureArray/32.)
        plt.plot(arrayTimes,ledON)
        print('fist frame at %s sec' % arrayTimes[int(midExposure[0])],end='')
        if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[3:(len(midExposure) + 3), 1] - 148.6) / 105.4, 'o-')
            print(3)
        elif arrayTimes[int(midExposure[0])]<0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[2:(len(midExposure) + 2), 1] - 148.6) / 105.4, 'o-')
            print(2)
        else:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[:len(midExposure), 1] - 148.6) / 105.4, 'o-')
            print(0)
        #plt.plot(softFrameTimes[data[:-6,0].astype(np.int)+6],(data[:-6,1]-148.6)/105.4)
        #plt.plot(softFrameTimes,np.ones(len(softFrameTimes)),'|')
        plt.show()

        pdb.set_trace()

    return (expStartTime,expEndTime,recordedFrames)

#################################################################################
def generatePlotWithSTD(data,std=[2,3,4],names = None):
    matplotlib.use('TkAgg')
    nData = len(data)
    fig = plt.figure(figsize=(15,15))
    for i in range(nData):
        ax0 = fig.add_subplot(nData,1,i+1)
        if names is not None:
            ax0.set_title('%s' % names[i])
        STD = np.std(data[i])
        MM  = np.mean(data[i])
        for n in range(len(std)):
            ax0.axhline(y=MM+std[n]*STD,ls='--',c=plt.cm.RdYlBu(n/len(std)),label='%s STD' % std[n])
            ax0.axhline(y=MM-std[n]*STD, ls='--', c=plt.cm.RdYlBu(n/len(std)))
        ax0.axhline(y=MM,c='C1',label='mean')
        ax0.plot(data[i],c='C0')
        if i==2:
            ax0.plot(np.abs(data[i]), c='C4')
        ax0.legend()
    plt.show()

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
def determineFramesToExclude(frames,probIdx):
    listOfFramesToExclude = []
    canBeUsed = True
    # first let's decide on how many LED's (if any) are present in the FOV
    for i in range(len(probIdx)):
        currIdx = probIdx[i]
        continueDetectLoop = True
        while continueDetectLoop:
            print('checking idx, started at :', currIdx, probIdx[i])
            #frame8bit = np.array(np.transpose(frames[currIdx]), dtype=np.uint8)
            img = cv2.cvtColor(frames[currIdx], cv2.COLOR_GRAY2BGR)
            # rungs = []
            #imgPure = img.copy()
            cv2.imshow("PureImage", img)
            print('e if to exclude; r to remove from exclude; left right arrows to go back-forward one frame; o to specify another idx; f to move to next; x if recording contains too many errors and cannot be used :')
            PressedKey = cv2.waitKey(0)
            print(PressedKey)
            if PressedKey == 81: # left arrow key
                currIdx -=1
            elif PressedKey == 83: # right arrow key
                currIdx +=1
            elif PressedKey == 101: # y key
                print('%s added to exclude list' %currIdx)
                listOfFramesToExclude.append(currIdx)
            elif PressedKey == 114: # e key
                print('%s removed from exclude list' % currIdx)
                listOfFramesToExclude.remove(currIdx)
            elif PressedKey == 111 : # o key
                nIdx = input('specify a new idx to check :')
                currIdx = int(nIdx)
            elif PressedKey == 102: # f key
                continueDetectLoop = False
            elif PressedKey == 120: # x key
                canBeUsed = False
                break
            else:
                print('Key not recognized, try again.')
            print('current exclude list :',listOfFramesToExclude)
        if not canBeUsed:
            break
    cv2.destroyWindow("PureImage") # only destroy window at the end of the exploration
    lofEx = list(dict.fromkeys(listOfFramesToExclude)) # removes duplicates
    lofEx.sort()
    print('starting list of indexes :', probIdx)
    print('indexes to exclude :', lofEx)
    lofEx = np.asarray(lofEx,dtype=int)
    #pdb.set_trace()
    return (lofEx, canBeUsed)


#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# ([ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],verbose=True)
def determineErroneousFrames(frames):

    # first threshold metrics of the movie to detect and exclude erroneous frames with horizontal lines, flash-back frames #########################
    frameDiff = []
    lineDiff = []
    print('calculating frame and line diffs ... ',end='')
    for i in range(len(frames)):
        if i>0:
            frameDiffAllPix = cv2.absdiff(frames[i],frames[i-1])
            fD = np.average(frameDiffAllPix)
            frameDiff.append(fD)
        lineDiffAllLines = cv2.absdiff(frames[i][:,1:],frames[i][:,:-1])
        lD = np.average(lineDiffAllLines,axis=0)
        lineDiff.append(lD)
        #pdb.set_trace()
    print('done!')
    frameDiff = np.asarray(frameDiff)
    lineDiff = np.asarray(lineDiff)
    lineDiffSum = np.sum(lineDiff,axis=1)
    frameDiffDiff = np.diff(frameDiff)
    generatePlotWithSTD([lineDiffSum,frameDiff,frameDiffDiff],std=[3,3.5,4],names=['lineDiffSum','frameDiff','diff of FrameDiff'])
    # trick to display the above image
    #frame8bit = np.array(np.transpose(frames[0]), dtype=np.uint8)
    img = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
    cv2.imshow('HoldImage',img)
    cv2.waitKey(0) #cv2.imshow()
    cv2.destroyWindow('HoldImage')
    thresholdingInput = input("Specify which trace to use (lineDiffSum - 1, frameDiff - 2, diff of frameDiff - 3; and which multiple of the STD (e.g. 1 3.5); type '4 0' if recording cannote be used (too many errors); '5 0' for recordings without errors : ")
    threshold = [float(i) for i in thresholdingInput.split()]
    print('choice :', threshold)
    #pdb.set_trace()
    if threshold[0] == 5.:
        idxToExclude = np.array([], dtype=np.int64)
        canBeUsed = True
        plt.close('all')
        return(idxToExclude,canBeUsed)
    elif threshold[0] == 1.:
        thresholded = lineDiffSum > np.mean(lineDiffSum) + np.std(lineDiffSum)*threshold[1]
        outlierIdx = np.arange(len(lineDiffSum))[thresholded]  # use indices taking into account missed frames
    elif threshold[0] == 2.:
        thresholded = frameDiff > np.mean(frameDiff) + np.std(frameDiff) * threshold[1]
        outlierIdx = np.arange(len(frameDiff))[thresholded]  # use indices taking into account missed frames
        outlierIdx += 1 # this is since the difference trace does not start at at the first frame but at the difference between first and second frame
    elif threshold[0] == 3.:
        thresholded = np.abs(frameDiffDiff) > np.mean(frameDiffDiff) + np.std(frameDiffDiff)*threshold[1]
        outlierIdx = np.arange(len(frameDiffDiff))[thresholded]  # use indices taking into account missed frames
        outlierIdx += 2  # this is since the difference trace does not start at at the first frame but at the difference between first and second frame
    elif threshold[0] == 4.:
        canBeUsed = False
        idxToExclude = np.array([], dtype=np.int64)
        plt.close('all')
        return (idxToExclude, canBeUsed)
    print('length and identity of possible erronous frames :' , len(outlierIdx), outlierIdx)
    (idxExclude,canBeUsed) = determineFramesToExclude(frames,outlierIdx)
    #excludeMask = np.ones(len(ledVideoRoi[2]),dtype=bool)
    # add indicies for equivalent frames
    sameFrames = (frameDiff == 0)
    sameFrameIdx = np.arange(len(frameDiff))[sameFrames]
    sameFrameIdx += 1
    print('same frames were recorded here :', sameFrameIdx)
    idxToExclude = np.sort(np.concatenate((sameFrameIdx, idxExclude)))
    #pdb.set_trace()
    #excludeMask[idxToExclude] = False
    plt.close('all')
    return (idxToExclude,canBeUsed)

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# ([ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],verbose=True)
def determineFrameTimesBasedOnLED(ledVideoRoi, cameraExposure, ledDAQc, pc, verbose=False, tail=False,manualThreshold=False):
    ##############################################################################################################
    # auxiliary function to convert bimodal trace into boolean array
    def traceToBinary(trace,threshold=None):
        rescaledTrace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
        if threshold is None:
            rescaledTraceBin = rescaledTrace > 0.3
        else:
            rescaledTraceBin = rescaledTrace > threshold
        return (rescaledTrace,rescaledTraceBin)
    ##############################################################################################################

    def traceToBinaryForChangingMaxMin(trace,threshold=None):
        maxTrace = ndimage.maximum_filter(trace, size=5*2)
        minTrace = ndimage.minimum_filter(trace, size=5*2)
        rescaledTrace = (trace - minTrace) / (maxTrace - minTrace)
        if threshold is None:
            rescaledTraceBin = rescaledTrace > 0.3
        else:
            rescaledTraceBin = rescaledTrace > threshold
        return (rescaledTrace,rescaledTraceBin)
    ##############################################################################################################

    # maps LED daq control trace to boolean array ################################################################
    # TODO this number is zero on the behavior setup and 4 here
    if pc == 'behaviorPC':
        LEDcontrolIdx = 0 # which trace of the DAQ recording is linked to the  !!!
    elif pc == '2photonPC':
        LEDcontrolIdx = 4
    else:
        print('Make sure the computer of the recording is specified.')
    ledDAQcontrolBin = traceToBinary(ledDAQc[0][LEDcontrolIdx])[1]  # here the threshold is not important as the trace is binary to start out with
    # convert LED roi traces from video to boolean arrays
    ledVideoRoiBins = []
    ledVideoRoiRescaled = []
    allLEDVideoRoiValues = []
    # determine threshold   [ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude]
    # tail covering the LEDs for some
    if tail:
        matplotlib.use('TkAgg')
        print(' in tail ...')
        anticipateCorrectValues = True
        for i in range(ledVideoRoi[1][0]):
             plt.plot(ledVideoRoi[0][i],'o-',ms=2,label='%s' % i)
        plt.legend(loc=1)
        plt.show()
        if anticipateCorrectValues:
            inputA = input('Index until which the recording is not affected by the tail (integer; type 0 if recording is ok) :')
            #inputA=350
            untilOKidx = int(inputA)
            if untilOKidx != 0:
                period = [7, 7, 7, 5]
                for i in range(ledVideoRoi[1][0]):
                    maxVal = np.max(ledVideoRoi[0][i][20:untilOKidx])
                    minVal = np.min(ledVideoRoi[0][i][20:untilOKidx])
                    for n in range(period[i]):
                        # repeatValue(ledVideoRoi[0][i][(untilOKidx+n):],7)
                        isHigh = [True if abs(ledVideoRoi[0][i][(untilOKidx + n)] - maxVal) < abs(ledVideoRoi[0][i][(untilOKidx + n)] - minVal) else False]
                        if isHigh:
                            ledVideoRoi[0][i][(untilOKidx + n):][::period[i]] = ledVideoRoi[0][i][(untilOKidx + n)]
                        else:
                            ledVideoRoi[0][i][(untilOKidx + n):][::period[i]] = ledVideoRoi[0][i][(untilOKidx + n)]  # ledVideoRoi[0][i][]
        else:
            maxV = [254,251,250,213]
            minV = [200,174,217,147]
            idxMaxV = [[8580,8582,8583,8585],
                       [],
                       [8589],
                       []]
            idxMinV = [[8581,8584,8586,8588],
                       [8579,8580],
                       [8590,8591],
                       [8584,8585,8586,8589,8590,8591]]
            for i in range(4):
                for n in idxMaxV[i]:
                    ledVideoRoi[0][i][n] = maxV[i]
                for m in idxMinV[i]:
                    ledVideoRoi[0][i][m] = minV[i]

        # fig = plt.figure()
        # for i in range(ledVideoRoi[1][0]):
        #     #ax = fig.add_subplot(3,1,i)
        #     plt.plot(ledVideoRoi[0][i],'o-',ms=2,label='%s' % i)
        #     #ax.set_xlim(8)
        # plt.legend(loc=1)
        # plt.show()
        # pdb.set_trace()
    ###########
    for i in range(ledVideoRoi[1][0]):
        allLEDVideoRoiValues.extend(traceToBinaryForChangingMaxMin(ledVideoRoi[0][i])[0]) # rescale all values to [0,1] and stack them
    allLEDVideoRoiValues = np.sort(np.asarray(allLEDVideoRoiValues)) # convert to array and sort
    luminocityDifferences = np.diff(allLEDVideoRoiValues)
    idxMaxDiff = np.argmax(luminocityDifferences)
    LEDVideoThreshold =  allLEDVideoRoiValues[idxMaxDiff] + (allLEDVideoRoiValues[idxMaxDiff+1] - allLEDVideoRoiValues[idxMaxDiff])/2.
    if pc == 'behaviorPC':
        #illumLEDcontrolThreshold = LEDVideoThreshold**4.49185827 # mapping, i.e. exponent, from tools/fitOfIlluminationValues
        illumLEDcontrolThreshold = LEDVideoThreshold**18.37008924
    elif pc == '2photonPC':
        illumLEDcontrolThreshold = LEDVideoThreshold**2.61290794 # 2pinvivo

    # if (illumLEDcontrolThreshold) < 0.08 or manualThreshold:
    #     print('thresholds before: ',LEDVideoThreshold, illumLEDcontrolThreshold)
    #     #print('LED threshold extremly low! Fixed by setting both threshold to 0.8.')
    #     fig = plt.figure(figsize=(10,10))
    #     ax = fig.add_subplot(111)
    #     #ax.plot(np.ones(len(allLEDVideoRoiValues)),allLEDVideoRoiValues,'.',ms=0.5)
    #     ax.axhline(y=LEDVideoThreshold,ls='--',c='C0')
    #     ax.plot(allLEDVideoRoiValues,'.',ms=0.5,c='C0')
    #     #plt.plot(np.ones(len(ledDAQcontrolBin)),ledDAQcontrolBin,'.')
    #     #ax.plot(ledDAQcontrolBin,'.',ms=0.5)
    #     plt.show()
    #     # thresholdInput = '0.8,0.8'
    #     # thresholdInput = ''
    #     thresholdInput = input('Provide alternative thresholds (e.g. 0.8,0.7), otherwise press enter to keep current thresholds : ')
    #     if not thresholdInput == '':
    #         newThresholds = [float(i) for i in thresholdInput.split(',')]
    #         LEDVideoThreshold = newThresholds[0]
    #         illumLEDcontrolThreshold = newThresholds[1]
        


    # find start and end of camera exposure period ################################################################
    exposureInt = np.array(cameraExposure[0][0], dtype=int)  # convert boolean array into array of zeros and ones
    difference = np.diff(exposureInt)            # calculate difference
    expStart = np.arange(len(exposureInt))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
    expEnd = np.arange(len(exposureInt))[np.concatenate((np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
    #pdb.set_trace()
    if (expEnd[0] - expStart[0]) < 0.:  # if trace starts above threshold
        print('exposure at start of recording')
        expEnd = expEnd[1:]
        exposureAtStart = True
    else:
        exposureAtStart = False
    if (expEnd[-1] - expStart[-1]) < 0.:  # if trace ends above threshold
        print('exposure during end of recording')
        exposureAtEnd = True
        expStart = expStart[:-1]
    else:
        exposureAtEnd = False
    expStart = expStart.astype(int)
    expEnd   = expEnd.astype(int)
    expStartTime = cameraExposure[1][expStart] # everything was based on indicies up to this point : here indicies -> time
    expEndTime   = cameraExposure[1][expEnd]   # everything was based on indicies up to this point : here indicies -> time
    frameDuration = expEndTime - expStartTime
    print('first frame started at ', expStartTime[0]*1000., 'ms' )

    ## based on exposure start-stop, how bright should the DAQ LED signal be ##########################################
    startEndExposureTime = np.column_stack((expStartTime, expEndTime))
    startEndExposurepIdx = np.column_stack((expStart,expEnd)) # create a 2-column array with 1st column containing start and 2nd column containing end index
    illumLEDcontrol = [np.mean(ledDAQcontrolBin[b[0]:b[1]]) for b in startEndExposurepIdx]  # extract MEAN illumination value - from LED control trace - during exposure period
    illumLEDcontrol = np.asarray(illumLEDcontrol)
    adjusted = False
    if manualThreshold:
        while True:
            if (illumLEDcontrolThreshold) < 0.08 or manualThreshold:
                sortedIllumLEDcontrol = np.sort(illumLEDcontrol)
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                #ax.plot(np.ones(len(allLEDVideoRoiValues)),allLEDVideoRoiValues,'.',ms=0.5)
                print('thresholds before: ',LEDVideoThreshold, illumLEDcontrolThreshold)
                if (illumLEDcontrolThreshold < 0.08) and not (adjusted):
                    illumLEDcontrolThreshold = 0.2
                    adjusted = True # do it only once
                print('adjusted thresholds: ', LEDVideoThreshold, illumLEDcontrolThreshold)
                print('video (up, down, down fraction) : ', np.sum(sortedIllumLEDcontrol > LEDVideoThreshold), np.sum(sortedIllumLEDcontrol < LEDVideoThreshold), np.sum(sortedIllumLEDcontrol < LEDVideoThreshold)/len(sortedIllumLEDcontrol))
                print('illum (up, down, down fraction): ', np.sum(allLEDVideoRoiValues>illumLEDcontrolThreshold), np.sum(allLEDVideoRoiValues<illumLEDcontrolThreshold), np.sum(allLEDVideoRoiValues<illumLEDcontrolThreshold)/len(allLEDVideoRoiValues))
                ax.axhline(y=illumLEDcontrolThreshold,ls='--',c='C0')
                ax.plot(np.linspace(0,1,len(sortedIllumLEDcontrol)),sortedIllumLEDcontrol,'.',ms=0.5,c='C0')
                ax.axhline(y=LEDVideoThreshold, ls='--', c='C1')
                ax.plot(np.linspace(0,1,len(allLEDVideoRoiValues)),allLEDVideoRoiValues, '.', ms=0.5,c='C1')
                #plt.plot(np.ones(len(ledDAQcontrolBin)),ledDAQcontrolBin,'.')
                #ax.plot(ledDAQcontrolBin,'.',ms=0.5)
                plt.show()
                # thresholdInput = '0.8,0.8'
                # thresholdInput = ''
                thresholdInput = input('Provide alternative thresholds (e.g. 0.8,0.2), otherwise press enter exit loop : ')
                if not thresholdInput == '':
                    newThresholds = [float(i) for i in thresholdInput.split(',')]
                    LEDVideoThreshold = newThresholds[0]
                    illumLEDcontrolThreshold = newThresholds[1]
                else:
                    break
    else:
        LEDVideoThreshold = LEDVideoThreshold
        illumLEDcontrolThreshold = illumLEDcontrolThreshold
        # print('press any key to check/redefine thresholds; exit loop with space or enter:')
        # PressedKey = cv2.waitKey(0)
        # if PressedKey == 13 or PressedKey == 32: # Enter or Space
        #     break
        # else:
        #     pass

    print('thresholds final: ',LEDVideoThreshold, illumLEDcontrolThreshold)
    # pdb.set_trace()
    # LEDVideoThreshold = 0.8
    # illumLEDcontrolThreshold = 0.8
    #print('adjusted thresholds : ', LEDVideoThreshold, illumLEDcontrolThreshold)

    (illumLEDcontrolrescaled, illumLEDcontrolBin) = traceToBinary(illumLEDcontrol, threshold=illumLEDcontrolThreshold)  # 0.2 and 0.15 before
    # pdb.set_trace()
    # threshold and convert to binary
    for i in range(ledVideoRoi[1][0]):
        ledVideoRoiBins.append(traceToBinary(ledVideoRoi[0][i],threshold=LEDVideoThreshold)[1])  # 0.6 before 0.4
        ledVideoRoiRescaled.append(traceToBinary(ledVideoRoi[0][i])[0])

    #plt.plot(allLEDVideoRoiValues,illumLEDcontrol, '.', ms=0.5)
    #plt.show()
    #plt.plot(illumLEDcontrol)
    ## loop over frame numbers and extract binary number shown by leds ################################################
    nFrames = len(ledVideoRoiBins[0])
    binNumbers = np.array([[False,False,False],[True,False,False],[False,True,False],[True,True,False],[False,False,True],[True,False,True],[False,True,True],[True,True,True]])
    recordedFrames = 0
    frameCount = []
    binNumberInFrame = np.column_stack((ledVideoRoiBins[0],ledVideoRoiBins[1],ledVideoRoiBins[2]))
    frameNBefore = 0
    oldI = -1
    exceptionsInFrameCount = []
    idxToExclude = ledVideoRoi[5]
    for i in range(nFrames):
        if i not in idxToExclude:
            matchBool = np.all(np.equal(binNumberInFrame[i],binNumbers),axis=1) # which of the boolean number corresponds to the current frame pattern : return is a boolean list from 0 to 8 with one TRUE entry
            matchFrameN = np.arange(len(binNumbers))[matchBool][0]   # converts the boolean list into the index corresponding to the match
            frameDiff = matchFrameN - frameNBefore  # difference in count to previous frame
            if frameDiff < 0: # else : negative difference indicates that the counter restarted
                frameDiff+=7
            if (frameDiff != 1) and (frameDiff != -6):
                print(i,oldI,i-oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1])
                exceptionsInFrameCount.append([i,oldI,i-oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1]])
            if matchFrameN == 0: # counter will start at 0 and possibly go back to zero after end of recording
                if (i>70) and (i<(nFrames-10)):
                    print(i,oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1])
                    print('strange, zero frame in the middle of recording')
                    pdb.set_trace()
                #frameDiff = 0
            frameCount.append([i,matchFrameN,frameDiff,int(ledVideoRoiBins[3][i]),oldI])
            frameNBefore = matchFrameN
            oldI = i
    frameCount = np.asarray(frameCount,dtype=int) # convert list to integer array
    idxRecordedFrames = np.cumsum(frameCount[:,2]) # use the frame differences to generate new index corresponding to video recording
    idxCounting = np.argwhere(idxRecordedFrames>0) # start and end index with first and last frame recording the counter
    idxFramesDuringRecording = idxRecordedFrames[idxCounting[0,0]:(idxCounting[-1,0]+1)] - 1  # remove leading and trailing zeros, and remove one to have the new index start with zero, cumsum makes first index to be 1
    #pdb.set_trace()
    if exposureAtStart:   # remove first frame if exposure was active during start of recording, i.e., at t = 0 s
        idxFramesDuringRecording = idxFramesDuringRecording[1:] - 1
    if exposureAtEnd:
        idxFramesDuringRecording = idxFramesDuringRecording[:-1]
    #pdb.set_trace()
    idxMissingFrames = np.delete(np.arange(idxFramesDuringRecording[-1]+1),idxFramesDuringRecording)

    #idxTestMask = idxFramesDuringRecording < len(illumLEDcontrolBin) # index should not exceed length of array
    #illum = illumLEDcontrolBin[idxFramesDuringRecording[idxTestMask]]
    ##  the excluded frames - based on distortions - need to be removed from the video sequence
    videoRoi = ledVideoRoiBins[3]
    mask = np.ones(len(videoRoi),dtype=bool)
    mask[idxToExclude]  = False
    #pdb.set_trace()
    ## first loop to align the START of the frame recording - in illumLEDcontrolBin - with the video recording
    if any(idxToExclude < 20):
        print('Early frames to exclude. Problem!')
        pdb.set_trace()
    else:
        tmpIdx = np.where(videoRoi==True) # Index of the first ON frame for the 4th LED
        idxFirstFrameRec = tmpIdx[0][0] # extract index of first frame during recording
        if exposureAtStart:
            idxFirstFrameRec+=1         # increase that
        for j in range(70):
            videoRoiWOEX = videoRoi[mask][j:]
            if np.all(videoRoiWOEX[:20] == illumLEDcontrolBin[:20]): # note that illumLEDcontrolBin already accounts for a recording during stat of rec, this frame is removed
                missedFramesBegin = j
                break
    try:
        a=missedFramesBegin
        #a = lllll
    except:
        print("bad alignement !!!!!!!!!!!!!!!!!!!")
        print(videoRoi[mask][:20],illumLEDcontrolBin[:20])
        plt.plot(illumLEDcontrolrescaled,'.',ms=0.5)
        plt.plot(ledVideoRoiRescaled[3],'.',ms=0.5)

        plt.show()
        pdb.set_trace()
    if (idxFirstFrameRec == missedFramesBegin) or (missedFramesBegin == 0):
        print('Number of frames recorded before first full exposed frame during recording :', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20] )
        videoRoiWOEX = videoRoi[mask][missedFramesBegin:]
    elif idxFirstFrameRec == (missedFramesBegin+1):
        missedFramesBegin+=1
        print('Number of frames recorded before first full exposed frame during recording (increased by one):', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20])
        videoRoiWOEX = videoRoi[mask][missedFramesBegin:]
    else:
        print('Number of frames recorded before first full exposed frame during recording :', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20] )
        print('Problem with determining index of first recorded frame.')
        pdb.set_trace()
    #pdb.set_trace()
    ## second loop in order to align the
    shiftDifference = []
    lengthOfIllumLEDcontrol = len(illumLEDcontrolBin)
    lengthOfROIinVideo = len(videoRoiWOEX)
    lengthOfIdxCount = idxFramesDuringRecording[-1] + 1
    print('length of illumLEDcontrolBin and videoRoiWOEX and idxFramesDuringRecording[-1] : ', lengthOfIllumLEDcontrol, lengthOfROIinVideo, lengthOfIdxCount)
    for i in range(-10,11,1): # loop to shift the mask over
        idxTemp = idxFramesDuringRecording + i # shift the array by increasing the indicies by a certain number
        idxIllum = idxTemp[(idxTemp>=0)&(idxTemp<lengthOfIllumLEDcontrol)] # indicies have to be larger than zero and should not be larger than the length of the illumLEDcontrolBin array
        #idxIllum = idx[idx<lengthOfIllumLEDcontrol]  # indicies should not be larger than the length of the illumLEDcontrolBin array
        illum = illumLEDcontrolBin[idxIllum]  # illumination at these indicies

        # idxMissing = np.delete(np.arange(idxIllum[-1]), idxIllum) #[i:]
        # idxMissing = np.delete(np.arange(idxFramesDuringRecording[-1]), idxIllum)  # [i:]
        idxMissing = np.delete(np.arange(lengthOfIllumLEDcontrol), idxIllum)
        NidxRemovedAtExtremities = idxIllum[0] + ((lengthOfIllumLEDcontrol-1) - idxIllum[-1]) # counts number of frames missing in the beginning and end
        NidxRemovedAtExtremities -= np.sum((idxMissingFrames<idxIllum[0]) | (idxMissingFrames>idxIllum[-1])) # reduce if missing frames are in the extrimities
        #if i<0:
        #    frameOverlap = [0 if ((lengthOfIllumLEDcontrol+np.abs(i)+1)<(lengthOfROIinVideo+len(idxMissingFrames))) else ((lengthOfROIinVideo+len(idxMissingFrames)) - (lengthOfIllumLEDcontrol+np.abs(i)+1))]
        #elif i>=0:
        #    frameOverlap = [i if (lengthOfIllumLEDcontrol<(lengthOfROIinVideo+len(idxMissingFrames))) else (lengthOfIllumLEDcontrol-(lengthOfROIinVideo+len(idxMissingFrames)+i+1))]
        #print('overlap :',frameOverlap[0])
        #print('test',len(np.intersect1d(idxMissing,idxMissingFrames)), idxMissingFrames, idxMissing,(-len(idxMissing)),NidxRemovedAtExtremities)
        compareIdx = len(np.intersect1d(idxMissing,idxMissingFrames)) - len(idxMissing)  + np.abs(NidxRemovedAtExtremities) # abs(i)

        len0 = len(illum)
        len1 = lengthOfROIinVideo
        if len0 < len1:
            compare = np.sum(np.equal(illum,videoRoiWOEX[:len0]))
            versch = compare - len0
            totLength = len0
            largeLength = len1
        else:
            compare = np.sum(np.equal(illum[:len1],videoRoiWOEX))
            versch = compare - len1
            totLength = len1
            largeLength = len0

        #pdb.set_trace()
        shiftDifference.append([i, versch, totLength, compareIdx,NidxRemovedAtExtremities])
        print(i, versch, totLength, compareIdx, NidxRemovedAtExtremities, idxMissing, idxMissingFrames)
        #if i >=0 :
    #pdb.set_trace()
    #compare = np.equal()
    shiftDifference = np.asarray(shiftDifference)
    if len(idxToExclude)==0:   # without erronous frames ...
        shiftDifference = shiftDifference[shiftDifference[:,0]>=0]   # ... use only the shifts which are larger than zero
    shiftToZero = shiftDifference[:,0][(shiftDifference[:,1]==0) & (shiftDifference[:,3]==0)]
    finalLength = shiftDifference[:,2][(shiftDifference[:,1]==0) & (shiftDifference[:,3]==0)]
    if len(shiftToZero)>1 or len(shiftToZero)==0:
        if len(shiftToZero)==0:
            idxTemp = idxFramesDuringRecording + 0
            idx = idxTemp[idxTemp >= 0]
            idxIllum = idx[idx < len(illumLEDcontrolBin)]
            # pdb.set_trace()
            shortest = [len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) if len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) < len(illumLEDcontrolrescaled[idxIllum]) else len(
                illumLEDcontrolrescaled[idxIllum])]
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]], illumLEDcontrolrescaled[idxIllum][:shortest[0]], 'o', ms=1)
            plt.show()
            pdb.set_trace()
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            print('difference at :',)
            ax.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]], 'o-',ms=0.5,lw=0.3)
            ax.plot(illumLEDcontrolrescaled[idxIllum][:shortest[0]], 'o-',ms=0.5,lw=0.3)
            plt.show()
            #plt.clf()
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111)
            ax.axhline(y=LEDVideoThreshold,c='C0',ls='--')
            ax.axhline(y=illumLEDcontrolThreshold,c='C1',ls='--')
            ledVideo = ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]]
            ledIllumDAQ = illumLEDcontrolrescaled[idxIllum][:shortest[0]]
            ax.plot(np.arange(len(ledVideo))[ledVideo > LEDVideoThreshold],ledVideo[ledVideo > LEDVideoThreshold], 'v', c='C0', ms=2)
            ax.plot(np.arange(len(ledVideo))[ledVideo < LEDVideoThreshold],ledVideo[ledVideo < LEDVideoThreshold], 'o', c='C0', ms=2)
            ax.plot(np.arange(len(ledIllumDAQ))[ledIllumDAQ > illumLEDcontrolThreshold],ledIllumDAQ[ledIllumDAQ>illumLEDcontrolThreshold], 'v',c='C1', ms=2)
            ax.plot(np.arange(len(ledIllumDAQ))[ledIllumDAQ < illumLEDcontrolThreshold],ledIllumDAQ[ledIllumDAQ<illumLEDcontrolThreshold], 'o', c='C1', ms=2)
            plt.show()
            pdb.set_trace()
        elif shiftToZero[1] == (shiftToZero[0]+5):
            print('Multiple shifts to zero, so multiple perfect overlays exist. First overlay with shift %s will be used.' % shiftToZero[0])
            pass
        else:
            print('Problem! More than one shift led to perfect overlay!')
            #np.arange(np.diff(idxRecordedFrames)>1)
            #
            idxTemp = idxFramesDuringRecording + 0
            idx = idxTemp[idxTemp>=0]
            idxIllum = idx[idx<len(illumLEDcontrolBin)]
            #pdb.set_trace()
            shortest = [len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) if len(ledVideoRoiRescaled[3][mask][missedFramesBegin:])<len(illumLEDcontrolrescaled[idxIllum]) else len(illumLEDcontrolrescaled[idxIllum])]
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]],illumLEDcontrolrescaled[idxIllum][:shortest[0]],'o',ms=1)
            plt.show()
            pdb.set_trace()
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]],'o-')
            plt.plot(illumLEDcontrolrescaled[idxIllum][:shortest[0]],'o-')
            plt.show()
            pdb.set_trace()
    finalShiftToZero = shiftToZero[0]
    print('final shift to zero and final length : ', finalShiftToZero,finalLength[0])
    idxTemp = idxFramesDuringRecording + finalShiftToZero
    idx = idxTemp[idxTemp>=0]
    idxIllumFinal = idx[idx<len(illumLEDcontrolBin)][:finalLength[0]]
    compareIllumination = False
    if compareIllumination:
        illum = illumLEDcontrolrescaled[idxIllumFinal]
        videoROI = ledVideoRoiRescaled[3][mask][missedFramesBegin:]
        shortest = [len(videoROI) if len(videoROI) < len(illum) else len(illum)]
        bothCombined = np.column_stack((videoROI[:shortest[0]],illum[:shortest[0]]))
        plt.plot(videoROI[:shortest[0]], illum[:shortest[0]], 'o')
        plt.show()
        pdb.set_trace()
        try :
            illumValues = pickle.load( open('illuminatoinValues.p', 'rb' ) )
        except :
            illumValues = bothCombined
        else:
            illumValues = np.row_stack((illumValues,bothCombined))
        pickle.dump(illumValues, open('illuminatoinValues.p', 'wb'))
    frameTimes = startEndExposureTime[idxIllumFinal]
    frameStartStopIdx = startEndExposurepIdx[idxIllumFinal]
    videoIdx = np.arange(len(ledVideoRoiBins[3]))[mask][missedFramesBegin:][:finalLength[0]]
    #recFrames = ledVideoRoi[2][videoIdx]
    ddd = np.diff(idxIllumFinal)
    print('Total number of dropped and excluded frames : ', np.sum(ddd-1), 'out of',len(ledVideoRoi[2]),'frame in total.')
    print('Excluded frames :', len(idxToExclude))
    print('Dropped frames :', np.sum(ddd-1)-len(idxToExclude))
    frameSummary = np.array([len(ledVideoRoi[2]),np.sum(ddd-1),len(idxToExclude), np.sum(ddd-1)-len(idxToExclude)])
    #pdb.set_trace()
    return (idxIllumFinal,frameTimes,frameStartStopIdx,videoIdx,frameSummary)
    ##############################################################################################################################
    #pdb.set_trace()

    for i in range(10):
        #print(i)
        #idxTest = idxRecordedFramesCleaned[1:-3] - 1
        #compare = ledVideoRoiBins[3][2:-3] == illumLEDcontrolBin[idxTest]
        #shortestLength = [len(illum) if (0<(len(ledVideoRoiBins)-(len(illum)+i))) else ]

        videoRoi = ledVideoRoiBins[3][i]
        if len(videoRoi) > len(illum):
            #shortestLength = len(illum)
            #else:
            #shortestLength = len(videoRoi)
            print('problem in length relations')
            pdb.set_trace()
        compare = illum[:len(videoRoi)] == videoRoi
        differences =  np.sum(np.invert(compare))
        print('number of differences :', i, differences,i,len(illum)+i,len(videoRoi))
        shifting.append([i,differences,i,len(illum)+i,len(videoRoi)])
    shifting = np.asarray(shifting)
    correctShift = np.argwhere(shifting[:,1]==0)
    if len(correctShift) == 0:
        print('No perfect overlay has been found')
        print(shifting)
        pdb.set_trace()
    elif len(correctShift)>1:
        print('Multiple corret overlays have been found. Suspicious!')
        print(shifting)
        pdb.set_trace()
    elif len(correctShift) == 1:
        rightShift = shifting[correctShift[0][0]]
        print('The correct shift is ', rightShift)
        print('Number of recorded videos :', len(ledVideoRoi[2][rightShift[2]:rightShift[3]]))
        print('Number of associated time points :', len(startEndExposurepIdx[idxRecordedFramesCleaned[idxTestMask]][:rightShift[4]]))
        ddd = np.diff(idxRecordedFramesCleaned[idxTestMask][:rightShift[4]])
        print('Number of gaps, number of lost frames, and size of gaps :', len(ddd[ddd>1]),np.sum(ddd[ddd>1]) - len(ddd[ddd>1]), ddd[ddd>1])
        idxVideo = np.arange(rightShift[2],rightShift[3])
        idxTimePoints = idxRecordedFramesCleaned[idxTestMask][:rightShift[4]]
        #pdb.set_trace()
        return (idxVideo,idxTimePoints,startEndExposureTime,startEndExposurepIdx,rightShift)


    if compare == 0:
        #plt.plot(ledVideoRoi[0][3][2:-3], 'o-', label='ledVideoRoi')
        ii = 3
        plt.plot(ledVideoRoiRescaled[3][ii:len(illum)+ii], 'o-', label='ledVideoRoiRescaled')
        plt.plot(ledVideoRoiBins[3][ii:len(illum)+ii],'o-',label='ledVideoRoiBins')
        plt.plot(illumLEDcontrol[idxRecordedFramesCleaned[idxTestMask]],'o-',label='illumLEDcontrol')
        plt.plot(illumLEDcontrolBin[idxRecordedFramesCleaned[idxTestMask]], 'o-', label='illumLEDcontrolBin')
        plt.legend()

        plt.show()
        pdb.set_trace()
        #compare = illumLEDcontrol[idxTest] ==
        #idxTest = idxRecordedFramesCleaned[1:-1]-1
        totLength = len(illumLEDcontrol[idxTest])
        ret = np.array_equal(illumLEDcontrol[idxRecordedFramesCleaned],ledVideoRoiBins[3][i:(totLength+i)])
        print(i,ret)
        pdb.set_trace()
    pdb.set_trace()



    if len(illumLEDcontrol) <= len(ledVIDEOroi):
        illuminationLonger = True
        ledVIDEOroiMask = np.arange(len(ledVIDEOroi)) < len(illumination)
        illuminationMask = np.arange(len(illumination)) < len(illumination)
        cc = crosscorr(1,illumination,ledVIDEOroi[ledVIDEOroiMask],20) # calculate cross-correlation between LED in video and LED from DAQ array
    else:
        illumniationLonger = False
        ledVIDEOroiMask = np.arange(len(ledVIDEOroi)) < len(ledVIDEOroi)
        illuminationMask = np.arange(len(illumination)) < len(ledVIDEOroi)
        cc = crosscorr(1, illumination[illuminationMask], ledVIDEOroi, 3)  # calculate cross-correlation between LED in video and LED from DAQ array
    peaks = find_peaks(cc[:,1],height=0)
    if len(peaks[0]) > 1:
        print('MULTIPLE peaks found in cross-correlogram between LED brigthness and DAQ array')
        pdb.set_trace()
    elif len(peaks[0]) == 0:
        print('NO peaks were found in cross-correlogram between LED brigthness and DAQ array')
        pdb.set_trace()
    else:
        pdb.set_trace()
        shift = cc[:,0][peaks[0][0]]
        shiftInt = int(shift)
        print('video trace has to be shifted by (float and int number) ', shift, shiftInt)
    #print(len(ledVIDEOroi),len(illumination))
    #pdb.set_trace()
    if verbose:
        if shiftInt >= 0:
            plt.plot(ledVIDEOroi[ledVIDEOroiMask][shiftInt:],'o-',ms=1,label='Video roi (shifted)')
        else:
            plt.plot(ledVIDEOroi[ledVIDEOroiMask][:shiftInt], 'o-', ms=1, label='Video roi (shifted)')
        plt.plot(illumination[illuminationMask],'o-',ms=1,label='from LED daq control')
        plt.legend()
        plt.show()

    frameIdx = np.arange(len(ledVIDEOroi))
    recordedFramesIdx = frameIdx[shiftInt:(len(illumination)+shiftInt)]
    #pdb.set_trace()
    return (startEndExpTime,startEndExpIdx,recordedFramesIdx)
    #####  end of current implementation ##############################################################################################
    #framesIdxDuringRec = np.array(len(softFrameTimes))[(arrayTimes[expEnd[0]]+0.002) < softFrameTimes]
    #framesIdxDuringRec = framesIdxDuringRec[:len(expStart)]

    if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
        recordedFrames = frames[3:(len(midExposure) + 3)]
    elif arrayTimes[int(midExposure[0])]<0.003:
        recordedFrames = frames[2:(len(midExposure) + 2)]
    else:
        recordedFrames = frames[:len(midExposure)]
    print('number of tot. frames, recorded frames, exposures start, end :',numberOfFrames,len(recordedFrames), len(expStart), len(expEnd))
    if display:
        ledON = np.zeros(len(exposureArray))
        for i in range(11):
            ledON[((i*1.)<=arrayTimes) & ((i*1.+0.2)>arrayTimes)] = 1.
        ledON[29.<=arrayTimes] = 1.
        data = np.loadtxt('/home/mgraupe/2019.04.01_000-%s.csv' % (rec[-3:]),delimiter=',',skiprows=1,usecols=(0,1))
        print(len(data))
        plt.plot(arrayTimes,exposureArray/32.)
        plt.plot(arrayTimes,ledON)
        print('fist frame at %s sec' % arrayTimes[int(midExposure[0])],end='')
        if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[3:(len(midExposure) + 3), 1] - 148.6) / 105.4, 'o-')
            print(3)
        elif arrayTimes[int(midExposure[0])]<0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[2:(len(midExposure) + 2), 1] - 148.6) / 105.4, 'o-')
            print(2)
        else:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[:len(midExposure), 1] - 148.6) / 105.4, 'o-')
            print(0)
        #plt.plot(softFrameTimes[data[:-6,0].astype(np.int)+6],(data[:-6,1]-148.6)/105.4)
        #plt.plot(softFrameTimes,np.ones(len(softFrameTimes)),'|')
        plt.show()

        pdb.set_trace()

    return (expStartTime,expEndTime,recordedFrames)


#################################################################################
# detect spikes in ephys trace
#################################################################################
def applyImageNormalizationMask(frames,imageMetaInfo,normFrame,normImageMetaInfo,mouse, date, rec):
    print(imageMetaInfo, normImageMetaInfo)

    pixelRange = 10
    print('small, large frame : ', np.shape(frames), np.shape(normFrame))
    print('pixel-ratio, x-ratio, y-ratio',  imageMetaInfo[4]/normImageMetaInfo[4],end='')
    fig = plt.figure()
    rect1 = patches.Rectangle(normImageMetaInfo[:2], normImageMetaInfo[2], normImageMetaInfo[3],linewidth=1,edgecolor='C0',facecolor='none')
    rect2 = patches.Rectangle(imageMetaInfo[:2],imageMetaInfo[2],imageMetaInfo[3],linewidth=1,edgecolor='C1',facecolor='none')

    framesF = np.array(frames,dtype=float)
    avgFrame = np.average(frames[:,:,:,0],axis=0)
    # rescale image stack to the resolution of the normalization image
    framesRescaled = scipy.ndimage.zoom(framesF, [1,imageMetaInfo[4]/normImageMetaInfo[4],imageMetaInfo[4]/normImageMetaInfo[4],1], order=3)

    # average across all time points of image stack
    #avgFrameZ = np.average(framesRescaled[:,:,:,0],axis=0)
    # rescale the average image to match pixel-size of normalization image, the re-scaling factor of the ratio of the pixel-sizes : stack/norm
    avgFrameZ = scipy.ndimage.zoom(avgFrame, imageMetaInfo[4]/normImageMetaInfo[4], order=3)
    # x,y location in pixel indices of the stack in the normalization image
    xLoc = int(np.round((imageMetaInfo[0] - normImageMetaInfo[0]) / normImageMetaInfo[4]))
    yLoc = int(np.round((imageMetaInfo[1] - normImageMetaInfo[1]) / normImageMetaInfo[4]))

    # dimensions of the rescaled image
    xDim = np.shape(avgFrameZ)[0]
    yDim = np.shape(avgFrameZ)[1]

    #####################
    ax0 = fig.add_subplot(2,3,4)
    ax0.set_title('avg. of image stack',size=7)
    ax0.imshow(np.transpose(avgFrame))

    ax1 = fig.add_subplot(2,3,5)
    ax1.set_title('avg. of image stack : rescaled to norm. image pixel size',size=7)
    ax1.imshow(np.transpose(avgFrameZ))

    print('image stack : ', np.shape(framesRescaled))
    scipy.io.savemat('%s_%s_%s_imageStackBeforeRescaling.mat' % (mouse, date, rec), mdict={'dataArray': framesF})
    scipy.io.savemat('%s_%s_%s_imageStack.mat' % (mouse, date, rec), mdict={'dataArray': framesRescaled})

    #img_stack_uint8 = mapToXbit(avgFrameZ,8)
    #pdb.set_trace()
    #tiff.imsave('avg_imageStack_scaled.tif', np.array(img_stack_uint8, dtype=np.uint8))

    ax2 = fig.add_subplot(2,3,1)
    ax2.set_title('normalization image with image stack rectangle',size=7)
    ret = patches.Rectangle([(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4],(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4]],imageMetaInfo[2]/normImageMetaInfo[4],imageMetaInfo[3]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(normFrame[0,:,:,0]))
    ax2.add_patch(ret)
    scipy.io.savemat('%s_%s_%s_registrationImage.mat' % (mouse, date, rec), mdict={'dataArray': normFrame[0,:,:,0]})


    ax2 = fig.add_subplot(2,3,6)
    ax2.set_title('area of image stack from normalization image',size=7)
    #ret = patches.Rectangle([(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4],(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4]],imageMetaInfo[3]/normImageMetaInfo[4],imageMetaInfo[2]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]))
    #ax2.add_patch(ret)
    print('norm. image : ', np.shape(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]))
    scipy.io.savemat('%s_%s_%s_normalizationImage.mat' % (mouse, date, rec), mdict={'dataArray': normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]})




    normFrameF = np.array(normFrame, dtype=float)
    test1 = scipy.ndimage.gaussian_filter1d(normFrameF[:,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),:], 2, axis=1)
    test2 = scipy.ndimage.gaussian_filter1d(test1, 2, axis=2)

    #test1 = scipy.ndimage.gaussian_filter1d(framesF, 2, axis=1)
    #test2 = scipy.ndimage.gaussian_filter1d(test1, 2, axis=2)

    #norm8bit = mapToXbit(test2,8)

    filter1D = scipy.ndimage.gaussian_filter1d(framesRescaled, 2, axis=1)
    filter2D = scipy.ndimage.gaussian_filter1d(filter1D, 2, axis=2)

    norm = filter2D / test2
    norm8bit = mapToXbit(norm,8)

    ax2 = fig.add_subplot(2,3,2)
    ax2.set_title('normalized average image',size=7)
    #ret = patches.Rectangle([(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4],(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4]],imageMetaInfo[3]/normImageMetaInfo[4],imageMetaInfo[2]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(np.average(norm[:,:,:,0],axis=0)))

    plt.show()
    #pdb.set_trace()

    errMatrix = np.zeros((pixelRange*2+1,pixelRange*2+1))
    # #row, col = np.indices(err)
    xRange = np.arange(pixelRange*2+1)
    yRange = np.copy(xRange)
    # #avgFrameZ = np.copy(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0])
    for xy in itertools.product(xRange, yRange):

         #err = np.abs((normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0] - avgFrameZ) ** 2).sum() / (xDim*yDim)
         xStart = xLoc + xy[0] - pixelRange
         yStart = yLoc + xy[1] - pixelRange
         normImg = normFrameF[0,xStart:(xStart+xDim),yStart:(yStart+yDim),0]
         normImgNorm = mapToXbit(normImg,8) #normImg - np.average(normImg)
         avgFrameZNorm = mapToXbit(np.average(framesRescaled[:,:,:,0],axis=0),8) #avgFrameZ - np.average(avgFrameZ)
         errMatrix[xy[0],xy[1]] = ((normImgNorm - avgFrameZNorm) ** 2).sum() / (xDim*yDim)

    minimumIndices = np.argwhere(errMatrix == np.min(errMatrix))

    print('MI :', minimumIndices)
    #pdb.set_trace()


    #xNorm = np.linspace(normImageMetaInfo[0],normImageMetaInfo[0]+
    #ax.add_patch(rect1)
    #ax.add_patch(rect2)
    #ax.set_ylim(normImageMetaInfo[1]-10,normImageMetaInfo[1]+normImageMetaInfo[3]+10)
    #ax.set_xlim(normImageMetaInfo[0]-10,normImageMetaInfo[0]+normImageMetaInfo[2]+10)
    #plt.patches.Rectangle(normImageMetaInfo[:2],normImageMetaInfo[2],normImageMetaInfo[3])
    #plt.patches.Rectangle(imageMetaInfo[:2],imageMetaInfo[2],imageMetaInfo[3])
    #plt.show()
    #pdb.set_trace()
    return norm8bit


#################################################################################
# detect spikes in ephys trace
#################################################################################
def detectPawTrackingOutlies(pawTraces,pawMetaData):
    jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
    threshold = 60


    def findOutliersBasedOnMaxSpeed(onePawData,jointName,i): # should be an 3 column array frame#, x, y

        frDisplOrig = np.sqrt((np.diff(onePawData[:, 1])) ** 2 + (np.diff(onePawData[:, 2])) ** 2) / np.diff(onePawData[:, 0])
        onePawDataTmp = np.copy(onePawData)
        onePawIndicies = np.arange(len(onePawData))
        # excursionsBoolOld = np.zeros(len(pawDataTmp)-1,dtype=bool)
        nIt = 0
        while True:  # cycle as long as there are large displacements
            frDispl = np.sqrt((np.diff(onePawDataTmp[:,1])) ** 2 + (np.diff(onePawDataTmp[:,2])) ** 2) / np.diff(onePawDataTmp[:, 0])  # calculate displacement
            excursionsBoolTmp = frDispl > threshold  # threshold displacement
            print(nIt, sum(excursionsBoolTmp))
            nIt += 1
            if sum(excursionsBoolTmp) == 0:  # no excursions above threshold are found anymore -> exit loop
                break
            else:
                onePawDataTmp = onePawDataTmp[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]
                onePawIndicies = onePawIndicies[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]

        print('%s # of positions, # of detected mis-trackings, fraction : ' % (jointName), len(onePawData), len(onePawData) - len(onePawDataTmp), (len(onePawData) - len(onePawDataTmp)) / len(onePawData))
        if jointName=='tail_base_bottom':
            pdb.set_trace()

        return (len(onePawData),len(onePawDataTmp),onePawIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig)

    pawTrackingOutliers = []
    for i in range(len(jointNames)):
        (tot,correct,correctIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig) = findOutliersBasedOnMaxSpeed(np.column_stack((pawTraces[:,0],pawTraces[:,(i*3+1)],pawTraces[:,(i*3+2)])),jointNames[i],i)
        pawTrackingOutliers.append([i,tot,correct,correctIndicies,jointNames[i],onePawData,onePawDataTmp,frDispl,frDisplOrig])
    return pawTrackingOutliers
    #pdb.set_trace()

#################################################################################
def detectPawTrackingOutliersObstacle(pawTraces,pawMetaData):
    jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
    threshold = 70
    print(jointNames)
    def findOutliersBasedOnMaxSpeedObstacle(onePawData,jointName,i): # should be an 3 column array frame#, x, y
        # if jointName=='obstacle':
        #     threshold=100
        # else:
        #     threshold = 80
        frDisplOrig = np.sqrt((np.diff(onePawData[:, 1])) ** 2 + (np.diff(onePawData[:, 2])) ** 2) / np.diff(onePawData[:, 0])
        onePawDataTmp = np.copy(onePawData)
        onePawIndicies = np.arange(len(onePawData))
        # excursionsBoolOld = np.zeros(len(pawDataTmp)-1,dtype=bool)
        nIt = 0
        while True:  # cycle as long as there are large displacements
            frDispl = np.sqrt((np.diff(onePawDataTmp[:,1])) ** 2 + (np.diff(onePawDataTmp[:,2])) ** 2) / np.diff(onePawDataTmp[:, 0])  # calculate displacement


            excursionsBoolTmp = frDispl > threshold  # threshold displacement
            print(nIt, sum(excursionsBoolTmp))
            nIt += 1
            if sum(excursionsBoolTmp) == 0:  # no excursions above threshold are found anymore -> exit loop
                break
            else:
                onePawDataTmp = onePawDataTmp[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]
                onePawIndicies = onePawIndicies[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]

        print('%s # of positions, # of detected mis-trackings, fraction : ' % (jointName), len(onePawData), len(onePawData) - len(onePawDataTmp), (len(onePawData) - len(onePawDataTmp)) / len(onePawData))
        # if jointName=='tail_base_bottom':
        #     # pdb.set_trace()

        return (len(onePawData),len(onePawDataTmp),onePawIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig)

    pawTrackingOutliersDic = {}
    pawTrackingOutliersList=[]
    pawTrackingOutliersBot_paw=[]
    b=0
    bot_paw = ['front_left_bottom', 'front_right_bottom', 'hind_left_bottom', 'hind_right_bottom']
    for i in range(len(jointNames)):
        pawTrackingOutliersDic[jointNames[i]] = {}
        (tot,correct,correctIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig) = findOutliersBasedOnMaxSpeedObstacle(np.column_stack((pawTraces[:,0],pawTraces[:,(i*3+1)],pawTraces[:,(i*3+2)])),jointNames[i],i)
        pawTrackingOutliersList.append([i,tot,correct,correctIndicies,jointNames[i],onePawData,onePawDataTmp,frDispl,frDisplOrig]) #all pf these are parameters that we stock in each jointName
        parameters = [i,tot,correct,correctIndicies,jointNames[i],onePawData,onePawDataTmp,frDispl,frDisplOrig]
        parameters_string = ['i','tot','correct','correctIndicies','jointName','onePawData','onePawDataTmp','frDispl','frDisplOrig']
        if  any([x in jointNames[i] for x in bot_paw]):
            pawTrackingOutliersBot_paw.append([b,tot,correct,correctIndicies,jointNames[i],onePawData,onePawDataTmp,frDispl,frDisplOrig])
            b+=1
        for j in range(len(parameters)):
            pawTrackingOutliersDic[jointNames[i]][parameters_string[j]] = parameters[j]

    return pawTrackingOutliersDic,pawTrackingOutliersList,pawTrackingOutliersBot_paw

    #################################################################################
def detectPawTrackingOutliersObstacleVids(pawTraces, pawMetaData):
    jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
    threshold = 70
    print(jointNames)

    def findOutliersBasedOnMaxSpeedObstacle(onePawData, jointName, i):  # should be an 3 column array frame#, x, y
        # if jointName=='obstacle':
        #     threshold=100
        # else:
        #     threshold = 80
        frDisplOrig = np.sqrt((np.diff(onePawData[:, 1])) ** 2 + (np.diff(onePawData[:, 2])) ** 2) / np.diff(
            onePawData[:, 0])
        onePawDataTmp = np.copy(onePawData)
        onePawIndicies = np.arange(len(onePawData))
        # excursionsBoolOld = np.zeros(len(pawDataTmp)-1,dtype=bool)
        nIt = 0
        while True:  # cycle as long as there are large displacements
            frDispl = np.sqrt((np.diff(onePawDataTmp[:, 1])) ** 2 + (np.diff(onePawDataTmp[:, 2])) ** 2) / np.diff(
                onePawDataTmp[:, 0])  # calculate displacement

            excursionsBoolTmp = frDispl > threshold  # threshold displacement
            print(nIt, sum(excursionsBoolTmp))
            nIt += 1
            if sum(excursionsBoolTmp) == 0:  # no excursions above threshold are found anymore -> exit loop
                break
            else:
                onePawDataTmp = onePawDataTmp[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]
                onePawIndicies = onePawIndicies[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]

        print('%s # of positions, # of detected mis-trackings, fraction : ' % (jointName), len(onePawData),
              len(onePawData) - len(onePawDataTmp), (len(onePawData) - len(onePawDataTmp)) / len(onePawData))
        # if jointName=='tail_base_bottom':
        #     # pdb.set_trace()

        return (len(onePawData), len(onePawDataTmp), onePawIndicies, onePawData, onePawDataTmp, frDispl, frDisplOrig)

    pawTrackingOutliersDic = {}
    pawTrackingOutliersList = []
    pawTrackingOutliersBot_paw = []
    b = 0
    bot_paw = ['front_left_bottom', 'front_right_bottom', 'hind_left_bottom', 'hind_right_bottom']
    for i in range(len(jointNames)):

        pawTrackingOutliersDic[jointNames[i]] = {}
        tot=[]
        correct=[]
        correctIndicies=np.array([])
        onePawData=np.empty((3))
        onePawDataTmp=np.empty((3))
        frDispl=np.array([])
        frDisplOrig=np.array([])
        for v in np.unique(pawMetaData['obs_number']):
            # print('obstacle ids', np.unique(pawMetaData['obs_number']))
            vmask=pawMetaData['obs_number']==v
            try:
                (tot_v, correct_v, correctIndicies_v, onePawData_v, onePawDataTmp_v, frDispl_v,frDisplOrig_v) = findOutliersBasedOnMaxSpeedObstacle(np.column_stack((pawTraces[:, 0][vmask], pawTraces[:, (i * 3 + 1)][vmask], pawTraces[:, (i * 3 + 2)][vmask])), jointNames[i],i)
            except:
                print('missmatch between frame labeled and obstacle frame numbers for label', jointNames[i], len(pawTraces[:, 0]), len(pawMetaData['obs_number']), 'please regenerate video with proper angle range and analyze with DLC')
                pdb.set_trace()
            # print(len(onePawDataTmp_v), len(frDispl_v), len(onePawData_v), len(frDisplOrig_v))
            correctIndicies=np.concatenate((correctIndicies,correctIndicies_v))
            onePawData=np.vstack((onePawData,onePawData_v))
            onePawDataTmp=np.vstack((onePawDataTmp,onePawDataTmp_v))
            frDispl = np.concatenate((frDispl, frDispl_v))
            frDisplOrig=np.concatenate((frDisplOrig, frDisplOrig_v))
        onePawData=onePawData[1:]
        onePawDataTmp=onePawDataTmp[1:]
        # pdb.set_trace()
        pawTrackingOutliersList.append([i, tot, correct, correctIndicies, jointNames[i], onePawData, onePawDataTmp, frDispl,frDisplOrig])  # all pf these are parameters that we stock in each jointName
        parameters = [i, tot, correct, correctIndicies, jointNames[i], onePawData, onePawDataTmp, frDispl,frDisplOrig]
        parameters_string = ['i', 'tot', 'correct', 'correctIndicies', 'jointName', 'onePawData', 'onePawDataTmp','frDispl', 'frDisplOrig']
        if any([x in jointNames[i] for x in bot_paw]):
            pawTrackingOutliersBot_paw.append(
                [b, tot, correct, correctIndicies, jointNames[i], onePawData, onePawDataTmp, frDispl, frDisplOrig])
            b += 1
        for j in range(len(parameters)):
            pawTrackingOutliersDic[jointNames[i]][parameters_string[j]] = parameters[j]

    return pawTrackingOutliersDic, pawTrackingOutliersList, pawTrackingOutliersBot_paw

#################################################################################
#################################################################################
# convert ca traces in easily usable numpy array
#################################################################################
def getCaWheelPawInterpolatedDictsPerDay(nSess,allCorrDataPerSession,allStepData,showFig = False):

    baselineTime = 5.
    # calcium traces ##############################################################
    trialStartUnixTimes = []

    fTraces = allCorrDataPerSession[nSess]['caImg']['Fluo'] #[3][0][0]
    timeStamps = allCorrDataPerSession[nSess]['caImg']['timeStamps'] # [3][0][3] # the array containing the time-stamp array
    recordings = np.unique(timeStamps[:, 1]) # determine how many recordings where performed
    caTracesDict = {}
    for n in range(len(recordings)):
        mask = (timeStamps[:, 1] == recordings[n])
        triggerStart = timeStamps[:, 5][mask]
        trialStartUnixTimes.append(timeStamps[:, 3][mask][0])
        if n > 0:
            if oldTriggerStart > triggerStart[0]:
                print('problem in trial order')
                sys.exit(1)
        # for i in range(len(fTraces)):
        # triggerstart - time of the acq start trigger for the current acquisition
        # timeStamps[:, 4][mask] - time of the first pixel in the frame passed since acqModeEpoch
        caTracesTime = (timeStamps[:, 4][mask] - triggerStart) # triggerStart is negative
        #pdb.set_trace()
        caTracesFluo = fTraces[:, mask]
        # pdb.set_trace()
        # caTraces.append(np.column_stack((caTracesTime,caTracesFluo)))
        caTracesDict[n] = np.row_stack((caTracesTime, caTracesFluo))
        #print(np.shape(np.row_stack((caTracesTime, caTracesFluo))))
        oldTriggerStart = triggerStart[0]

    # wheel speed  ######################################################
    # also find calmest pre-motorization period
    minPreMotorMeanV = 1000.
    wheelTracks = allCorrDataPerSession[nSess]['wheel'] #[1]
    nRec = 0
    # print(len(wheelTracks))
    wheelSpeedDict = {}
    for n in range(len(wheelTracks)):
        wheelRecStartTime = wheelTracks[n]['timeStamp']#[3]
        if (trialStartUnixTimes[nRec] - wheelRecStartTime) < 1.:
            # if not wheelTracks[n][4]:
            # recStartTime = wheelTracks[0][3]
            if nRec > 0:
                if oldRecStartTime > wheelRecStartTime:
                    print('problem in trial order')
                    sys.exit(1)
            wheelTime = wheelTracks[n]['sTimes']#[2]
            wheelSpeed = wheelTracks[n]['linearSpeed']#[1] # linear wheel speed in cm/s
            angleSpeed = wheelTracks[n]['angluarSpeed']#[0]
            angleTime  = wheelTracks[n]['angleTimes']#[5]

            wheelSpeedDict[nRec] = np.row_stack((wheelTime, wheelSpeed))
            #pdb.set_trace()
            preMMask = (wheelTime < baselineTime)
            preMmeanV = np.mean(np.abs(wheelSpeed[preMMask]))
            #print(nSess, nRec, preMmeanV)
            if preMmeanV < minPreMotorMeanV:
                slowestRec = nRec
                minPreMotorMeanV = np.copy(preMmeanV)

            nRec += 1
            oldRecStartTime = wheelRecStartTime
    print('trials with slowest baseline period :', slowestRec)
    # normalize ca-traces by baseline fluorescence : fluorescence during the first baselineTime seconds in the least active recording #############################################
    mask = (caTracesDict[slowestRec][0] < baselineTime)
    F0 = np.mean(caTracesDict[slowestRec][1:][:, mask], axis=1)
    #pdb.set_trace()
    for n in range(len(recordings)):
        normalizedCaTraces = (caTracesDict[n][1:] - F0[:, np.newaxis]) / F0[:, np.newaxis]
        caTracesDict[n][1:] = np.copy(normalizedCaTraces)

    # pdb.set_trace()
    # paw speed  ######################################################
    pawTracks = allCorrDataPerSession[nSess]['paws']#[2]
    nRec = 0
    pawTracksDict = {}
    pawID = []
    for n in range(len(pawTracks)):
        # if not wheelTracks[n][4]:
        pawRecStartTime = pawTracks[n]['recStartTime']#[4]
        if (trialStartUnixTimes[nRec] - pawRecStartTime) < 1.:
            if nRec > 0:
                if oldRecStartTime > pawRecStartTime:
                    print('problem in trial order')
                    sys.exit(1)
            pawTracksDict[nRec] = {}
            for i in range(4):
                # pdb.set_trace()
                if nRec == 0:
                    pawID.append(pawTracks[n]['jointNamesFramesInfo'][i][0])
                # pawTracksDict[nFig][i]['pawID'] = pawTracks[n][2][i][0]
                pawSpeedTime = pawTracks[n]['pawSpeed'][i][:,0] # times of cleared paw speed
                pawSpeed = pawTracks[n]['pawSpeed'][i][:,1]    # 1 is combined speed in 2-d plane of the camera view from below,
                pawTracksDict[nRec][i] = np.row_stack((pawSpeedTime,pawSpeed))  # interp = interp1d(pawSpeedTime, pawSpeed)  # newPawSpeedAtCaTimes = interp(caTracesTime[nFig])  # pawTracksDict[i]['pawSpeed'].extend(newPawSpeedAtCaTimes)

            oldRecStartTime = pawRecStartTime
            nRec += 1
    # interpolation #############################################################################
    # interp = interp1d(wheelTime, wheelSpeed)
    # interpMask = (caTracesTime[nFig] >= wheelTime[0]) & (caTracesTime[nFig] <= wheelTime[-1])
    # newWheelSpeedAtCaTimes = interp(caTracesTime[nFig][interpMask])
    # wheelSpeedAll.extend(newWheelSpeedAtCaTimes)

    wheelSpeedDictInterp = wheelSpeedDict.copy()
    pawTracksDictInterp = pawTracksDict.copy()
    caTracesDictInterp = caTracesDict.copy()

    for nrec in range(len(caTracesDict)):
        # determine interpolation range
        startInterpTime = np.max((caTracesDict[nrec][0, 0], wheelSpeedDict[nrec][0, 0], pawTracksDict[nrec][0][0, 0], pawTracksDict[nrec][1][0, 0], pawTracksDict[nrec][2][0, 0], pawTracksDict[nrec][3][0, 0]))
        endInterpTime = np.min((caTracesDict[nrec][0, -1], wheelSpeedDict[nrec][0, -1], pawTracksDict[nrec][0][0, -1], pawTracksDict[nrec][1][0, -1], pawTracksDict[nrec][2][0, -1], pawTracksDict[nrec][3][0, -1]))

        interpMask = (caTracesDict[nrec][0] >= startInterpTime) & (caTracesDict[nrec][0] <= endInterpTime)

        # restrict ca-traces to interpolation range
        #pdb.set_trace()
        #matrix = np.copy(caTracesDict[nrec])
        caTracesDictInterp[nrec] = caTracesDict[nrec][:,interpMask]

        # interpolate wheel speed
        interpWheel = interp1d(wheelSpeedDict[nrec][0], wheelSpeedDict[nrec][1])#,kind='cubic')
        newWheelSpeedAtCaTimes = interpWheel(caTracesDict[nrec][0][interpMask])
        wheelSpeedDictInterp[nrec] = np.row_stack((caTracesDict[nrec][0][interpMask], newWheelSpeedAtCaTimes))

        # interpolate paw speed
        for i in range(4):
            interpPaw = interp1d(pawTracksDict[nrec][i][0], pawTracksDict[nrec][i][1])#,kind='cubic')
            newPawSpeedAtCaTimes = interpPaw(caTracesDict[nrec][0][interpMask])
            pawTracksDictInterp[nrec][i] = np.row_stack((caTracesDict[nrec][0][interpMask], newPawSpeedAtCaTimes))
            #pawAll[i].extend(newPawSpeedAtCaTimes)
        if showFig:
            cc = ['C0','C1']
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(2):
                ax.plot(pawTracksDictInterp[nrec][i][0],pawTracksDictInterp[nrec][i][1]/np.max(pawTracksDictInterp[nrec][i][1]),c=cc[i])

                idxSwings = allStepData[nSess][4][nrec][3][i][1]
                # print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nSess][4][nrec][4][i][2]
                # pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                for k in range(len(idxSwings)):  # loop over all swings
                    startSwingTime = recTimes[idxSwings[k, 0]]
                    endSwingTime = recTimes[idxSwings[k, 1]]
                    ax.fill_between((startSwingTime,endSwingTime), 0, 1, color='0.6', alpha=0.5, transform=ax.get_xaxis_transform())
                    ax.fill_between((startSwingTime,endSwingTime), 0, 1, color='0.4', alpha=0.5, transform=ax.get_xaxis_transform())

            #ax.plot(caTracesDictInterp[nrec][0],caTracesDictInterp[nrec][1]/np.max(caTracesDictInterp[nrec][1]),c='black')
            plt.show()
            pdb.set_trace()

    return (wheelSpeedDictInterp,pawTracksDictInterp,caTracesDictInterp,wheelSpeedDict,pawTracksDict,caTracesDict,slowestRec)

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doRegressionAnalysis(mouse,allCorrDataPerSession,allStepData,borders=None,figShow=False):
    matplotlib.use('TkAgg')  # WxAgg
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    #SVR(kernel='rbf', C=1e3, gamma=0.1)
    #from sklearn.ensemble import RandomForestRegressor
    regressionN = 6
    Rvalues = []
    #for nSess in range(len(allCorrDataPerSession)):
    for nDay in range(len(allCorrDataPerSession)):
        print(nDay,allCorrDataPerSession[nDay]['folder'])
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP, wheelSpeedDict, pawTracksDict, caTracesDict, slowestTrial) = getCaWheelPawInterpolatedDictsPerDay(nDay,allCorrDataPerSession,allStepData)
        # ATTENTION : all of the arrays also contain a time array
        # dims of wheelSpeedDictInterP : [nSessions][2][valuesOverTimeSame]
        # dims of pawTracksDictInterP :  [nSessions][nPaw][2][valuesOverTimeSame]
        # dims of caTracesDictInterP :  [nSessions][nRois+1][valuesOverTimeSame]
        # dims of wheelSpeedDict : [nSessions][2][valuesOverTime]
        # dims of pawTracksDict : [nSessions][nPaw][2][valuesOverTime]
        # dims of caTracesDict : [nSessions][nRois+1][valuesOverTime]
        #(wheelSpeedDict, pawTracksDict, caTracesDict,aa,bb,cc) = getCaWheelPawInterpolatedDictsPerDay(nSess, allCorrDataPerSession)
        nRecWheel = len(wheelSpeedDictInterP)
        nRecPaw = len(pawTracksDictInterP)
        nRecCa = len(caTracesDictInterP)
        print('Recording length :', nRecWheel, nRecPaw, nRecCa)
        if (nRecWheel != nRecPaw ) or (nRecWheel != nRecCa):
            print('problem in number of recordings listed in dictionaries')
        # loop over 5 different regressions, each using a different combination of test and train samples
        recs = range(nRecCa)
        RTempValues = []
        for reg in range(nRecCa): # loop over all recordings
            recsForTraining = recs.copy()
            recsForTraining.remove(reg)
            recsForTest = [reg]
            Rval1 = []
            for d in range(regressionN):  # loop over wheel speed, the four paw speeds and the combined speed
                #print('recording iteration %s, variable %s' %(reg,d))
                #pdb.set_trace()
                # concatenate data
                if borders is not None:
                    timeMaskTrain = (caTracesDictInterP[recsForTraining[0]][0]>=borders[0])&(caTracesDictInterP[recsForTraining[0]][0]<=borders[1])
                    timeMaskTest = (caTracesDictInterP[recsForTest[0]][0] >= borders[0]) & (caTracesDictInterP[recsForTest[0]][0] <= borders[1])
                else:
                    timeMaskTrain = (caTracesDictInterP[recsForTraining[0]][0]>=0)&(caTracesDictInterP[recsForTraining[0]][0]<=1000.)
                    timeMaskTest  = (caTracesDictInterP[recsForTest[0]][0]>=0)&(caTracesDictInterP[recsForTest[0]][0]<=1000.)
                X = np.copy(caTracesDictInterP[recsForTraining[0]][1:][:,timeMaskTrain])
                Xtest = np.copy(caTracesDictInterP[recsForTest[0]][1:][:,timeMaskTest])
                #pdb.set_trace()
                if d == 0:
                    Y = np.copy(wheelSpeedDictInterP[recsForTraining[0]][1:][:,timeMaskTrain])
                    Ytest = np.copy(wheelSpeedDictInterP[recsForTest[0]][1:][:,timeMaskTest])
                    YtestTime = np.copy(wheelSpeedDictInterP[recsForTest[0]][0][timeMaskTest])
                elif (d>0) and (d<5):
                    pawId = d-1
                    Y = np.copy(pawTracksDictInterP[recsForTraining[0]][pawId][1:][:,timeMaskTrain])
                    Ytest = np.copy(pawTracksDictInterP[recsForTest[0]][pawId][1:][:,timeMaskTest])
                    YtestTime = np.copy(pawTracksDictInterP[recsForTest[0]][pawId][0][timeMaskTest])
                elif d==5: # case where all four paw speeds are added together
                    pawSpeedTrain = []
                    pawSpeedTest = []
                    for i in range(4):
                        pawSpeedTrain.append(np.copy(pawTracksDictInterP[recsForTraining[0]][i][1:][:, timeMaskTrain]))
                        pawSpeedTest.append(np.copy(pawTracksDictInterP[recsForTest[0]][i][1:][:, timeMaskTest]))
                    Y = pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                    Ytest = pawSpeedTest[0] + pawSpeedTest[1] + pawSpeedTest[2] + pawSpeedTest[3]
                    #pdb.set_trace()
                for t in recsForTraining[1:]:
                    if borders is not None:
                        timeMaskTrain = (caTracesDictInterP[t][0] >= borders[0]) & (caTracesDictInterP[t][0] <= borders[1])
                    else:
                        timeMaskTrain = (caTracesDictInterP[t][0] >= 0) & (caTracesDictInterP[t][0] <= 1000.)
                    X = np.column_stack((X,caTracesDictInterP[t][1:][:,timeMaskTrain]))
                    if d == 0:
                        Y = np.column_stack((Y,wheelSpeedDictInterP[t][1:][:,timeMaskTrain]))
                    elif (d>0) and (d<5):
                        Y = np.column_stack((Y, pawTracksDictInterP[t][pawId][1:][:,timeMaskTrain]))
                    elif d==5:
                        pawSpeedTrain = []
                        for i in range(4):
                            pawSpeedTrain.append(np.copy(pawTracksDictInterP[t][i][1:][:, timeMaskTrain]))
                        speedTemp =  pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                        Y = np.column_stack((Y, speedTemp))
                #pdb.set_trace()
                Y = Y[0]
                X = np.transpose(X)
                Ytest = Ytest[0]
                Xtest = np.transpose(Xtest)
                # linear regression  ########################################
                linReg = LinearRegression()
                linReg.fit(X,Y)
                #svm_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
                #svm_rbf.fit(X,Y)
                YTrainPred = linReg.predict(X)
                YTestPred  = linReg.predict(Xtest)
                R2trainLR = linReg.score(X, Y)
                R2testLR = linReg.score(Xtest, Ytest) # 1. - np.sum((Ytest-YTestPred)**2)/np.sum((Ytest - np.mean(Ytest))**2)#linReg.score(Xtest, Ytest)
                #print(linReg.coef_)
                #print(linReg.intercept_)
                #yPred = linReg.predict(np.transpose(X))
                # random forest ##############################################
                #randForestReg = RandomForestRegressor(n_estimators=20)
                #randForestReg.fit(X, Y)
                #R2trainRF = randForestReg.score(X, Y)
                #R2testRF= randForestReg.score(Xtest, Ytest)
                #
                if figShow :
                    print('R2 test :',R2testLR)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(YtestTime,Ytest,lw=2)
                    ax.plot(YtestTime,YTestPred,lw=2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_position(('outward', 10))
                    ax.spines['left'].set_position(('outward', 10))
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    plt.show()

                Rval1.extend([R2trainLR,R2testLR])
            RTempValues.append(Rval1)
        #pdb.set_trace()
        Rs = np.zeros(regressionN*2)
        for reg in range(5):
            Rs += RTempValues[reg]
        Rs /=5.
        Rvalues.append(Rs)

    return Rvalues

#################################################################################
# perform linear regression between spiking activity and behavioral measures
#################################################################################
def crossValidatedRegression(regModels,X,y,t,fold,visualize=False):
    cols = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
    regOutput = {}
    ###########
    if visualize:
        plt.plot(t,y,color='black')
    # get coeffss : apply the regression models consecutively
    for j in range(len(regModels)):
        regOutput[j] = {}
        print('applying', regModels[j][0])
        regModel = regModels[j][1]
        regModel.fit(X, y)
        # print(regModel.coef_)
        regOutput[j]['name'] = regModels[j][0]
        regOutput[j]['coefficients'] = regModel.coef_
        regOutput[j]['fitScore'] = regModel.score(X, y)
        regOutput[j]['scores'] = np.zeros(fold*5)
        if visualize:
            plt.plot(t,regModel.predict(X),c=cols[j],label=regModels[j][0]+': %s ' % (np.round(regOutput[j]['fitScore'],3)))
        del regModel
    if visualize:
        plt.xlabel('time (s)')
        plt.ylabel('firing rate')
        plt.legend(frameon=False)
        plt.show()
    ########

    if fold>0:
        print('performing cross-validation')
        # to k-fold cross-validated regression to access the score
        for j in range(len(regModels)):
            #kf = KFold(n_splits=fold)
            i = 0
            #scores = cross_val_score(regModels[j][1], X, y, cv=fold)
            #print(regModels[j][0],scores)
            rkf = RepeatedKFold(n_splits=fold, n_repeats=5, random_state=42)
            #for train_index, test_index in kf.split(X):
            for train_index, test_index in rkf.split(X):
                #print(i,len(train_index),len(test_index))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                regModel = regModels[j][1]
                regModel.fit(X_train, y_train)
                regOutput[j]['scores'][i] = regModel.score(X_test, y_test)
                #if visualize:
                #    plt.plot(t[test_index],regModel.predict(X_test),label=(regModels[j][0] if i==0 else None))
                del regModel
                i+=1
        for j in range(len(regModels)):
            print(regOutput[j]['name'],regOutput[j]['fitScore'],np.mean(regOutput[j]['scores']),np.std(regOutput[j]['scores']))#,regOutput[j]['scores'])

    return regOutput



        # # print(regModel.intercept_)
        # sc = regModel.score(Xregressors_scaled[tmask], YspikeCount[tmask])
        # print('score:', sc)
        # Ypred = regModel.predict(Xregressors_scaled[tmask])
        # del regModel
        # plt.plot(tbinCenters[tmask], Ypred, label=regs[j][0] + ' %s' % np.round(sc, 3))
        # plt.xlabel('time (s)')
        # plt.ylabel('firing rate')  # pass


#################################################################################
# shuffles variable within chunks
#################################################################################
def shuffleVariable(y,ttime,dt,chunkSize):
    nChunk = int(chunkSize/dt)
    shuffleIterations = int(np.ceil(len(ttime)/nChunk))
    for i in range(shuffleIterations):
        np.random.shuffle(y[(i*nChunk):((i+1)*nChunk)])
    return y


#################################################################################
# perform linear regression between spiking activity and behavioral measures
#################################################################################
#  cPawPos,pawSpeed,ephys,swingStanceDict,sTimes,linearSpeed)
def performGLManalysis(date, rec, pawPos,pawSpeed,ephys,swingStanceD,sTimes,linearSpeed):
    matplotlib.use('TkAgg')
    # create spike-count vector
    spikeTimes = ephys[0]
    print('firing rate :', 1./np.mean(np.diff(spikeTimes)))
    dt = 0.01
    shiftRange = 0.2 # binary columns are shifted back- and forth in time by this delay in s
    nShift = int(shiftRange/dt)
    tbins = np.linspace(0., 60., int(60 / dt) + 1, endpoint=True)
    tbinCenters = (tbins[1:]+tbins[:-1])/2
    binnedspikes, _ = np.histogram(spikeTimes, tbins)
    spikecountwindow = 0.02
    nspikecountwindow = int(spikecountwindow / dt) # + 0.5)
    #YspikeCount = np.convolve(binnedspikes, np.ones(nspikecountwindow), 'same')
    binnedspikes=np.array(binnedspikes,dtype=float)
    YspikeCount = scipy.ndimage.gaussian_filter1d(binnedspikes, nspikecountwindow,axis=0) # convolve with Gaussian kernel
    print(len(YspikeCount),len(binnedspikes),nspikecountwindow)
    # create regressor matrix
    #Xregressors = np.zeros((len(tbinCenters),1+4*4))
    Xregressors = np.zeros((len(tbinCenters), 1))
    # interpolate wheel speed
    interpWheel = interp1d(sTimes, linearSpeed,fill_value='extrapolate')#,kind='cubic')
    Xregressors[:,0] = interpWheel(tbinCenters)
    # interpolate paw position and paw speed
    for i in range(4):
        interpPawPos = interp1d(pawPos[i][:,0],pawPos[i][:,1],fill_value='extrapolate')
        #Xregressors[:,1+i] = interpPawPos(tbinCenters)
        Xregressors = np.column_stack((Xregressors,interpPawPos(tbinCenters)))
    for i in range(4):
        interpPawSpeed = interp1d(pawSpeed[i][:,0],pawSpeed[i][:,2],fill_value='extrapolate') # pawSpeed[i][1] is total speed, 2 is x, 3 is y speed
        #Xregressors[:,5+i] = interpPawSpeed(tbinCenters)
        Xregressors = np.column_stack((Xregressors, interpPawSpeed(tbinCenters)))
    for i in range(4):
        idxSwings = swingStanceD['swingP'][i][1]
        recTimes = swingStanceD['forFit'][i][2]
        idxSwings = np.asarray(idxSwings)
        binnedSwingStartTimes, _ = np.histogram(recTimes[idxSwings[:,0]], tbins)
        binnedSwingEndTimes, _ = np.histogram(recTimes[idxSwings[:, 1]], tbins)
        startTshift = np.zeros((len(binnedSwingStartTimes),2*nShift+1))
        endTshift = np.zeros((len(binnedSwingEndTimes),2*nShift+1))
        n = 0
        for j in range(-nShift,nShift+1):
            startTshift[:,n] = np.roll(binnedSwingStartTimes,j)
            endTshift[:,n] = np.roll(binnedSwingEndTimes, j)
            #plt.plot(startTshift[:,n])
            n+=1
        #plt.show()
        Xregressors = np.column_stack((Xregressors, startTshift))
        Xregressors = np.column_stack((Xregressors, endTshift))
        #Xregressors[:,9+i] = binnedSwingStartTimes
        #Xregressors[:,13+i] = binnedSwingEndTimes
    #pdb.set_trace()
    print('shape or regressor matrix :', np.shape(Xregressors))
    ##  preprocessing of the data
    #scaler = preprocessing.MinMaxScaler().fit(Xregressors) # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
    #Xregressors_scaled = scaler.transform(Xregressors)
    #pdb.set_trace()
    Xregressors_scaled = np.copy(Xregressors)
    Xregressors_scaled[:,:9] = scipy.stats.zscore(Xregressors[:,:9],axis=0)
    #Xregressors_scaled = np.copy(Xregressors)
    #pdb.set_trace()
    #Xregressors_scaled[:8] = Xregressors[:8] # preserve the sparse data
    timeLimits = [10,50]
    tmask = (tbinCenters>timeLimits[0]) & (tbinCenters<timeLimits[1])
    # generate list of regression models
    # alpha multiplies the penalty terms : for alpha=0 is equivalent to an ordinary least square
    # ridge regression : l2 regularization
    # elastic net : The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    regs = [('Linear Regression',linear_model.LinearRegression()),('Ridge regression',linear_model.Ridge(alpha=100.))]#,('Elastic Net Regression',linear_model.ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=0))]#,('GLM with log link function',linear_model.PoissonRegressor(alpha=1e-6/len(YspikeCount[tmask])))]
    #regs = [('Elastic Net Regression',linear_model.ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=0))]\
    #               ('Ridge regression', linear_model.Ridge(alpha=1.)),
    #        ('Elastic Net Regression', linear_model.ElasticNet(alpha=0.01, random_state=0))]
    regResults = crossValidatedRegression(regs,Xregressors_scaled[tmask],YspikeCount[tmask],tbinCenters[tmask],fold=0,visualize=False)
    print(len(Xregressors_scaled[tmask]))
    #plt.plot(tbinCenters[tmask],YspikeCount[tmask])
    #coeffss = []
    #pdb.set_trace()
    #plt.legend(frameon=False)
    #plt.show()
    return regResults
    plt.clf()
    fig = plt.figure(figsize=(12,4))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.1)
    cols = ['C0','C1','C2','C3','C4']
    pawID = ['FL','FR','HL','HR']
    ax = fig.add_subplot(1,5,1)
    for i in range(len(regs)):
        ax.plot(regResults[i]['coefficients'][:9],'o-',label=regs[i][0])
    ax.set_ylabel('beta-weight')
    plt.xticks(np.arange(9),['wheel speed','x-pos FL','x-pos FR','x-pos HL','x-pos HR','v FL','v FR','v HL','v HR'],rotation=45, ha='right',fontsize=8)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.legend(frameon=False)
    tVector = np.linspace(-nShift,nShift,2*nShift+1,endpoint=True)*dt
    shifts = 2*nShift + 1
    for j in range(4):
        ax = fig.add_subplot(1,5,j+2)
        for i in range(len(regs)):
            ax.set_title(pawID[j])
            ax.plot(tVector,regResults[i]['coefficients'][(9+(2*j)*shifts):(9+(2*j+1)*shifts)],c=cols[i],ls=':',label=(None if j<3 else 'swingStart '+regs[i][0]))
            ax.plot(tVector,regResults[i]['coefficients'][(9+(2*j+1)*shifts):(9+(2*j+2)*shifts)],c=cols[i],ls='-',label=(None if j<3 else 'swingEnd '+regs[i][0]))
            #ax.set_ylim(-0.7,1.6)
        ax.axvline(x=0,ls=':',c='0.4')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('beta-weight')
        plt.legend(frameon=False)
    plt.show()
    pdb.set_trace()
    # perform linear regression : no regularization
    print('Linear regression')
    Lreg = linear_model.LinearRegression()
    Lreg.fit(Xregressors_scaled, YspikeCount)
    print(Lreg.coef_)
    print(Lreg.intercept_)
    print('score:',Lreg.score(Xregressors_scaled,YspikeCount))
    Ypred = Lreg.predict(Xregressors_scaled)
    plt.title('Linear reg.')
    plt.plot(YspikeCount)
    plt.plot(Ypred)
    plt.show()
    def constructDesignMatrix(pawID=0,variable='wheelSpeed',chunkSize=3,shuffle=False):
        Xregressors = np.zeros((len(tbinCenters), 1))
        # interpolate wheel speed
        interpWheel = interp1d(sTimes, linearSpeed, fill_value='extrapolate')  # ,kind='cubic')
        newWheelSpeed =  interpWheel(tbinCenters)
        if (shuffle) and ('wheelSpeed' in variable):
            newWheelSpeed = shuffleVariable(newWheelSpeed,tbinCenters,dt,chunkSize)
        Xregressors[:, 0] = newWheelSpeed
        # interpolate paw position and paw speed
        for i in range(4):
            interpPawPos = interp1d(pawPos[i][:, 0], pawPos[i][:, 1], fill_value='extrapolate')
            # Xregressors[:,1+i] = interpPawPos(tbinCenters)
            newPawPos = interpPawPos(tbinCenters)
            if (shuffle) and (i in pawID) and ('pawPosition' in variable):
                newPawPos = shuffleVariable(newPawPos,tbinCenters,dt,chunkSize)
            Xregressors = np.column_stack((Xregressors, newPawPos))
        for i in range(4):
            interpPawSpeed = interp1d(pawSpeed[i][:, 0], pawSpeed[i][:, 2], fill_value='extrapolate')  # pawSpeed[i][1] is total speed, 2 is x, 3 is y speed
            # Xregressors[:,5+i] = interpPawSpeed(tbinCenters)
            newPawSpeed = interpPawSpeed(tbinCenters)
            if (shuffle) and (i in pawID) and ('pawSpeed' in variable):
                newPawSpeed = shuffleVariable(newPawSpeed, tbinCenters, dt, chunkSize)
            Xregressors = np.column_stack((Xregressors, newPawSpeed))
        for i in range(4):
            idxSwings = swingStanceD['swingP'][i][1]
            recTimes = swingStanceD['forFit'][i][2]
            idxSwings = np.asarray(idxSwings)
            binnedSwingStartTimes, _ = np.histogram(recTimes[idxSwings[:, 0]], tbins)
            binnedSwingEndTimes, _ = np.histogram(recTimes[idxSwings[:, 1]], tbins)
            if (shuffle) and (i in pawID) and ('swingStart' in variable):
                binnedSwingStartTimes = shuffleVariable(binnedSwingStartTimes, tbinCenters, dt, chunkSize)
            if (shuffle) and (i in pawID) and ('stanceStart' in variable):
                binnedSwingEndTimes = shuffleVariable(binnedSwingEndTimes, tbinCenters, dt, chunkSize)
            startTshift = np.zeros((len(binnedSwingStartTimes), 2 * nShift + 1))
            endTshift = np.zeros((len(binnedSwingEndTimes), 2 * nShift + 1))
            n = 0
            for j in range(-nShift, nShift + 1):
                startTshift[:, n] = np.roll(binnedSwingStartTimes, j)
                endTshift[:, n] = np.roll(binnedSwingEndTimes, j)
                n += 1
            Xregressors = np.column_stack((Xregressors, startTshift))
            Xregressors = np.column_stack((Xregressors, endTshift))  # Xregressors[:,9+i] = binnedSwingStartTimes  # Xregressors[:,13+i] = binnedSwingEndTimes
        return Xregressors

    matplotlib.use('TkAgg')
    # create spike-count vector
    spikeTimes = ephys[0]
    print('firing rate :', 1./np.mean(np.diff(spikeTimes)))
    dt = 0.01
    shiftRange = 0.2 # binary columns are shifted back- and forth in time by this delay in s
    nShift = int(shiftRange/dt)
    tbins = np.linspace(0., 60., int(60 / dt) + 1, endpoint=True)
    tbinCenters = (tbins[1:]+tbins[:-1])/2
    binnedspikes, _ = np.histogram(spikeTimes, tbins)
    spikecountwindow = 0.03 # sigma of the Gaussian kernel
    nspikecountwindow = int(spikecountwindow / dt + 0.5)
    binnedspikes = np.array(binnedspikes,dtype=float)
    #YspikeCount = scipy.ndimage.gaussian_filter1d(binnedspikes, nspikecountwindow,axis=0) # convolve with Gaussian kernel
    YspikeCount = np.convolve(binnedspikes, np.ones(nspikecountwindow), 'same')  # convolve spike-count with square kernel
    #print(len(YspikeCount),len(binnedspikes),nspikecountwindow)
    #plt.hist(YspikeCount,bins=30)
    #plt.show()
    #pdb.set_trace()
    # create regressor matrix
    #Xregressors = np.zeros((len(tbinCenters),1+4*4))
    timeLimits = [10,50]
    tmask = (tbinCenters>timeLimits[0]) & (tbinCenters<timeLimits[1])

    regs = [('Ridge Regression', linear_model.Ridge(alpha=1))]

    Xregressors = constructDesignMatrix()
    #pdb.set_trace()
    print('shape or regressor matrix :', np.shape(Xregressors))
    ##  preprocessing of the data
    scaler = preprocessing.MinMaxScaler().fit(Xregressors) # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
    Xregressors_scaled = scaler.transform(Xregressors)
    regResultsFullModel = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=10, visualize=False)
    coffs = regResultsFullModel[0]['coefficients']
    shuffleForWeights = False
    if shuffleForWeights:
        shuffCoeffsWheelSpeed = []
        for i in range(100):
            # (pawID=0,variable='wheelSpeed',chunkSize=3,shuffle=False):
            Xregressors = constructDesignMatrix(pawID=[0,1,2,3],variable=['wheelSpeed'],chunkSize=2.,shuffle=True)
            # pdb.set_trace()
            #print('shape or regressor matrix :', np.shape(Xregressors))
            ##  preprocessing of the data
            scaler = preprocessing.MinMaxScaler().fit(Xregressors)  # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
            Xregressors_scaled = scaler.transform(Xregressors)
            regResultsShuffle = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=0, visualize=False)
            shuffCoeffsWheelSpeed.append(regResultsShuffle[0]['coefficients'])
        shuffCoeffsPawPos = []
        for i in range(100):
            # (pawID=0,variable='wheelSpeed',chunkSize=3,shuffle=False):
            Xregressors = constructDesignMatrix(pawID=[0,1,2,3],variable=['pawPosition'],chunkSize=2.,shuffle=True)
            # pdb.set_trace()
            #print('shape or regressor matrix :', np.shape(Xregressors))
            ##  preprocessing of the data
            scaler = preprocessing.MinMaxScaler().fit(Xregressors)  # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
            Xregressors_scaled = scaler.transform(Xregressors)
            regResultsShuffle = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=0, visualize=False)
            shuffCoeffsPawPos.append(regResultsShuffle[0]['coefficients'])
        #
        shuffCoeffsPawSpeed = []
        for i in range(100):
            # (pawID=0,variable='wheelSpeed',chunkSize=3,shuffle=False):
            Xregressors = constructDesignMatrix(pawID=[0,1,2,3],variable=['pawSpeed'],chunkSize=2.,shuffle=True)
            # pdb.set_trace()
            #print('shape or regressor matrix :', np.shape(Xregressors))
            ##  preprocessing of the data
            scaler = preprocessing.MinMaxScaler().fit(Xregressors)  # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
            Xregressors_scaled = scaler.transform(Xregressors)
            regResultsShuffle = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=0, visualize=False)
            shuffCoeffsPawSpeed.append(regResultsShuffle[0]['coefficients'])
        shuffCoeffsSwingStance = []
        for i in range(100):
            # (pawID=0,variable='wheelSpeed',chunkSize=3,shuffle=False):
            Xregressors = constructDesignMatrix(pawID=[0,1,2,3],variable=['swingStart','stanceStart'],chunkSize=2.,shuffle=True)
            # pdb.set_trace()
            #print('shape or regressor matrix :', np.shape(Xregressors))
            ##  preprocessing of the data
            scaler = preprocessing.MinMaxScaler().fit(Xregressors)  # This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.
            Xregressors_scaled = scaler.transform(Xregressors)
            regResultsShuffle = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=0, visualize=False)
            shuffCoeffsSwingStance.append(regResultsShuffle[0]['coefficients'])
        coffs = np.asarray(coffs)
        shuffCoeffsWheelSpeed =  np.asarray(shuffCoeffsWheelSpeed)
        shuffCoeffsPawPos =  np.asarray(shuffCoeffsPawPos)
        shuffCoeffsPawSpeed =  np.asarray(shuffCoeffsPawSpeed)
        shuffCoeffsSwingStance =  np.asarray(shuffCoeffsSwingStance)
        shuffCoeffs = np.copy(shuffCoeffsWheelSpeed)
        shuffCoeffs[:,1:5] = shuffCoeffsPawPos[:,1:5]
        shuffCoeffs[:,5:10] = shuffCoeffsPawSpeed[:,5:10]
        shuffCoeffs[:, 10:]  = shuffCoeffsSwingStance[:,10:]
        #pdb.set_trace()
        fig = plt.figure(figsize=(12,4))
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.1)
        cols = ['C0','C1','C2','C3','C4']
        #pawID = ['FL','FR','HL','HR']
        ax0 = fig.add_subplot(1,5,1)
        ax0.axhline(y=0,ls='--',c='0.5')
        ax0.plot(coffs[:9],'o-')
        ax0.plot(np.mean(shuffCoeffs,axis=0)[:9])
        ax0.fill_between(np.arange(9),np.percentile(shuffCoeffs,5,axis=0)[:9],np.percentile(shuffCoeffs,95,axis=0)[:9],alpha=0.5)
        #ax1 = fig.add_subplot(1,2,2)
        tVector = np.linspace(-nShift,nShift,2*nShift+1,endpoint=True)*dt
        shifts = 2*nShift + 1
        for j in range(4):
            ax1 = fig.add_subplot(1,5,j+2)
            for i in range(len(regs)):
                #ax.set_title(pawID[j])
                ax1.plot(tVector,coffs[(9+(2*j)*shifts):(9+(2*j+1)*shifts)])
                ax1.plot(tVector,np.mean(shuffCoeffs[:,(9+(2*j)*shifts):(9+(2*j+1)*shifts)],axis=0)) #label=(None if j<3 else 'swingStart '+regs[i][0]))
                ax1.fill_between(tVector, np.percentile(shuffCoeffs[:,(9+(2*j)*shifts):(9+(2*j+1)*shifts)], 5, axis=0), np.percentile(shuffCoeffs[:,(9+(2*j)*shifts):(9+(2*j+1)*shifts)], 95, axis=0), alpha=0.5)
        plt.show()
        pdb.set_trace()
    #Xregressors_scaled = np.copy(Xregressors)
    #pdb.set_trace()
    #Xregressors_scaled[:8] = Xregressors[:8] # preserve the sparse data
    # generate list of regression models
    # alpha multiplies the penalty terms : for alpha=0 is equivalent to an ordinary least square
    # ridge regression : l2 regularization
    # elastic net : The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    regs = [('Linear Regression',linear_model.LinearRegression()),('Ridge regression',linear_model.Ridge(alpha=1.)),('Elastic Net Regression',linear_model.ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=0)),('GLM with log link function',linear_model.PoissonRegressor(alpha=1e-6/len(YspikeCount[tmask])))]
    #regs = [('Ridge Regression',linear_model.Ridge(alpha=1.))]#,('GLM with log link function',linear_model.PoissonRegressor(alpha=1e-6/len(YspikeCount[tmask])))]
    #               ('Ridge regression', linear_model.Ridge(alpha=1.)),
    #        ('Elastic Net Regression', linear_model.ElasticNet(alpha=0.01, random_state=0))]
    RRidge = []
    #import statsmodels.api as sm
    #gamma_model = sm.GLM(YspikeCount[tmask],Xregressors_scaled[tmask], family=sm.families.Poisson())
    #gamma_results = gamma_model.fit()
    #print(gamma_results.summary())
    #pdb.set_trace()
    checkAlphaVariable = False
    if checkAlphaVariable :
        for i in range(100):
            al = float(i) #1./(1.1220184543019633**i)
            #regs = [('GLM with log link function',linear_model.PoissonRegressor(alpha=al))]
            regs = [('Ridge Regression', linear_model.Ridge(alpha=al))]
            regResults = crossValidatedRegression(regs,Xregressors_scaled[tmask],YspikeCount[tmask],tbinCenters[tmask],fold=10,visualize=False)
            RRidge.append([i,al,regResults[0]['fitScore'],np.mean(regResults[0]['scores'])])
        RRidge = np.asarray(RRidge)
        plt.title('Ridge Regression with log-link function')
        plt.plot(RRidge[:,1], RRidge[:,2], 'o-',label='fit score')
        plt.plot(RRidge[:,1],RRidge[:,3],'o-',label='cross validated score')
        plt.xlabel('alpha (penalty weight)')
        plt.ylabel('R^2')
        #plt.xscale('log')
        #plt.legend(frameon=False)
        plt.show()
        pdb.set_trace()
        print(len(Xregressors_scaled[tmask]))
    #plt.plot(tbinCenters[tmask],YspikeCount[tmask])
    #coeffss = []
    #pdb.set_trace()
    #plt.legend(frameon=False)
    #plt.show()
    regs = [('Linear Regression',linear_model.LinearRegression()),('Ridge regression',linear_model.Ridge(alpha=1.)),('Elastic Net Regression',linear_model.ElasticNet(alpha=0.01,l1_ratio=0.5,random_state=0)),('GLM with log link function',linear_model.PoissonRegressor(alpha=1e-6/len(YspikeCount[tmask])))]
    #regResultsFullModel = {}
    #for i in range(len(regs)):
    regResultsFullModel = crossValidatedRegression(regs, Xregressors_scaled[tmask], YspikeCount[tmask], tbinCenters[tmask], fold=0, visualize=False)
    plt.clf()
    fig = plt.figure(figsize=(12,4))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.1)
    cols = ['C0','C1','C2','C3','C4']
    pawID = ['FL','FR','HL','HR']
    ax = fig.add_subplot(1,5,1)
    for i in range(len(regs)):
        ax.plot(regResultsFullModel[i]['coefficients'][:9],'o-',label=regs[i][0])
    ax.set_ylabel('beta-weight')
    plt.xticks(np.arange(9),['wheel speed','x-pos FL','x-pos FR','x-pos HL','x-pos HR','v FL','v FR','v HL','v HR'],rotation=45, ha='right',fontsize=8)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.legend(frameon=False)
    tVector = np.linspace(-nShift,nShift,2*nShift+1,endpoint=True)*dt
    shifts = 2*nShift + 1
    for j in range(4):
        ax = fig.add_subplot(1,5,j+2)
        for i in range(len(regs)):
            ax.set_title(pawID[j])
            ax.plot(tVector,regResultsFullModel[i]['coefficients'][(9+(2*j)*shifts):(9+(2*j+1)*shifts)],c=cols[i],ls='-',label=(None if j<3 else 'swingStart '+regs[i][0]))
            ax.plot(tVector,regResultsFullModel[i]['coefficients'][(9+(2*j+1)*shifts):(9+(2*j+2)*shifts)],c=cols[i],ls=':',label=(None if j<3 else 'swingEnd '+regs[i][0]))
            ax.set_ylim(-0.7,1.6)
        ax.axvline(x=0,ls=':',c='0.4')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('beta-weight')
        plt.legend(frameon=False)
    plt.show()
    pdb.set_trace()
    # perform linear regression : no regularization
    print('Linear regression')
    Lreg = linear_model.LinearRegression()
    Lreg.fit(Xregressors_scaled, YspikeCount)
    print(Lreg.coef_)
    print(Lreg.intercept_)
    print('score:',Lreg.score(Xregressors_scaled,YspikeCount))
    Ypred = Lreg.predict(Xregressors_scaled)
    plt.title('Linear reg.')
    plt.plot(YspikeCount)
    plt.plot(Ypred)
    plt.show()
    # perform elastic net regression : Linear regression with combined L1 and L2 priors as regularizer
    # Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
    print('Elastic net regression')
    ENreg = linear_model.ElasticNet(alpha=0.1,random_state=0)
    ENreg.fit(Xregressors_scaled, YspikeCount)
    print(ENreg.coef_)
    print(ENreg.intercept_)
    print('score:',ENreg.score(Xregressors_scaled,YspikeCount))
    Ypred = ENreg.predict(Xregressors_scaled)
    plt.title('Elastic net reg.')
    plt.plot(YspikeCount)
    plt.plot(Ypred)
    plt.show()
    #pdb.set_trace()
    # perform Generalized Linear Model with a Poisson distribution. This regressor uses the ‘log’ link function.
    # from sklearn.ensemble import HistGradientBoostingRegressor
    print('GLM with log link function')
    #Preg = HistGradientBoostingRegressor(loss="poisson",l2_regularization=1, max_leaf_nodes=128) #
    Preg = linear_model.PoissonRegressor(alpha=0)
    Preg.fit(Xregressors_scaled, YspikeCount)
    print(Preg.coef_)
    print(Preg.intercept_)
    print('score:',Preg.score(Xregressors_scaled, YspikeCount))
    Ypred = Preg.predict(Xregressors_scaled)
    plt.title('GLM with log link function')
    plt.plot(YspikeCount)
    plt.plot(Ypred)
    plt.show()
    pdb.set_trace()

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doContinuousRegressionAnalysis(mouse,allCorrDataPerSession,allStepData,borders=None,figShow=False):
    matplotlib.use('TkAgg')  # WxAgg
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    #SVR(kernel='rbf', C=1e3, gamma=0.1)
    #from sklearn.ensemble import RandomForestRegressor
    regressionN = 6
    Rvalues = []
    #for nSess in range(len(allCorrDataPerSession)):
    for nDay in range(len(allCorrDataPerSession)):
        print(nDay,allCorrDataPerSession[nDay]['folder'])
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP, wheelSpeedDict, pawTracksDict, caTracesDict, slowestTrial) = getCaWheelPawInterpolatedDictsPerDay(nDay,allCorrDataPerSession,allStepData)
        # ATTENTION : all of the arrays also contain a time array
        # dims of wheelSpeedDictInterP : [nSessions][2][valuesOverTimeSame]
        # dims of pawTracksDictInterP :  [nSessions][nPaw][2][valuesOverTimeSame]
        # dims of caTracesDictInterP :  [nSessions][nRois+1][valuesOverTimeSame]
        # dims of wheelSpeedDict : [nSessions][2][valuesOverTime]
        # dims of pawTracksDict : [nSessions][nPaw][2][valuesOverTime]
        # dims of caTracesDict : [nSessions][nRois+1][valuesOverTime]
        #(wheelSpeedDict, pawTracksDict, caTracesDict,aa,bb,cc) = getCaWheelPawInterpolatedDictsPerDay(nSess, allCorrDataPerSession)
        nRecWheel = len(wheelSpeedDictInterP)
        nRecPaw = len(pawTracksDictInterP)
        nRecCa = len(caTracesDictInterP)
        print('Recording length :', nRecWheel, nRecPaw, nRecCa)
        if (nRecWheel != nRecPaw ) or (nRecWheel != nRecCa):
            print('problem in number of recordings listed in dictionaries')
        # loop over 5 different regressions, each using a different combination of test and train samples
        recs = range(nRecCa)
        RTempValues = []
        #for reg in range(nRecCa): # loop over all recordings
        #recsForTraining = recs.copy()
        #recsForTraining.remove(reg)
        #recsForTest = [reg]
        varr = ['wheel speed','paw speed 0', 'paw speed 1','paw speed 2','paw speed 3','paw speed 0+1+2+3']
        for d in range(regressionN):  # loop over wheel speed, the four paw speeds and the combined speed
            print('regressing %s' %varr[d])
            #pdb.set_trace()
            # concatenate data
            if borders is not None:
                timeMaskTrain = (caTracesDictInterP[0][0]>=borders[0])&(caTracesDictInterP[0][0]<=borders[1])
                timeMaskTest = (caTracesDictInterP[0][0] >= borders[0]) & (caTracesDictInterP[0][0] <= borders[1])
            else:
                timeMaskTrain = (caTracesDictInterP[0][0]>=0)&(caTracesDictInterP[0][0]<=1000.)
                timeMaskTest  = (caTracesDictInterP[0][0]>=0)&(caTracesDictInterP[0][0]<=1000.)
            X = np.copy(caTracesDictInterP[0][1:][:,timeMaskTrain])
            #Xtest = np.copy(caTracesDictInterP[0][1:][:,timeMaskTest])
            #pdb.set_trace()
            if d == 0:
                Y = np.copy(wheelSpeedDictInterP[0][1:][:,timeMaskTrain])
                #Ytest = np.copy(wheelSpeedDictInterP[0][1:][:,timeMaskTest])
                #YtestTime = np.copy(wheelSpeedDictInterP[0][0][timeMaskTest])
            elif (d>0) and (d<5):
                pawId = d-1
                Y = np.copy(pawTracksDictInterP[0][pawId][1:][:,timeMaskTrain])
                #Ytest = np.copy(pawTracksDictInterP[recsForTest[0]][pawId][1:][:,timeMaskTest])
                #YtestTime = np.copy(pawTracksDictInterP[recsForTest[0]][pawId][0][timeMaskTest])
            elif d==5: # case where all four paw speeds are added together
                pawSpeedTrain = []
                #pawSpeedTest = []
                for i in range(4):
                    pawSpeedTrain.append(np.copy(pawTracksDictInterP[0][i][1:][:, timeMaskTrain]))
                    #pawSpeedTest.append(np.copy(pawTracksDictInterP[recsForTest[0]][i][1:][:, timeMaskTest]))
                Y = pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                #Ytest = pawSpeedTest[0] + pawSpeedTest[1] + pawSpeedTest[2] + pawSpeedTest[3]
                #pdb.set_trace()
            for t in recs[1:]:
                if borders is not None:
                    timeMaskTrain = (caTracesDictInterP[t][0] >= borders[0]) & (caTracesDictInterP[t][0] <= borders[1])
                else:
                    timeMaskTrain = (caTracesDictInterP[t][0] >= 0) & (caTracesDictInterP[t][0] <= 1000.)
                X = np.column_stack((X,caTracesDictInterP[t][1:][:,timeMaskTrain]))
                if d == 0:
                    Y = np.column_stack((Y,wheelSpeedDictInterP[t][1:][:,timeMaskTrain]))
                elif (d>0) and (d<5):
                    Y = np.column_stack((Y, pawTracksDictInterP[t][pawId][1:][:,timeMaskTrain]))
                elif d==5:
                    pawSpeedTrain = []
                    for i in range(4):
                        pawSpeedTrain.append(np.copy(pawTracksDictInterP[t][i][1:][:, timeMaskTrain]))
                    speedTemp =  pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                    Y = np.column_stack((Y, speedTemp))
            #pdb.set_trace()
            Y = Y[0]
            X = np.transpose(X)
            #Ytest = Ytest[0]
            nRegressionIterations = 10
            Rval1 = np.zeros(2)
            for n in range(nRegressionIterations):
                #print(n,end='')
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
                #Xtest = np.transpose(Xtest)
                # linear regression  ########################################
                linReg = LinearRegression()
                linReg.fit(X_train,y_train)
                #svm_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
                #svm_rbf.fit(X,Y)
                #YTrainPred = linReg.predict(X)
                y_test_pred  = linReg.predict(X_test)
                R2trainLR = linReg.score(X_train, y_train)
                R2testLR = linReg.score(X_test, y_test) # 1. - np.sum((Ytest-YTestPred)**2)/np.sum((Ytest - np.mean(Ytest))**2)#linReg.score(Xtest, Ytest)
                #print(linReg.coef_)
                #print(linReg.intercept_)
                #yPred = linReg.predict(np.transpose(X))
                # random forest ##############################################
                #randForestReg = RandomForestRegressor(n_estimators=20)
                #randForestReg.fit(X, Y)
                #R2trainRF = randForestReg.score(X, Y)
                #R2testRF= randForestReg.score(Xtest, Ytest)
                #
                if figShow :
                    print('R2 test :',R2testLR)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(y_test,lw=2)
                    ax.plot(y_test_pred,lw=2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_position(('outward', 10))
                    ax.spines['left'].set_position(('outward', 10))
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    plt.show()

                Rval1+= np.array([R2trainLR,R2testLR])
            RTempValues.extend(Rval1/nRegressionIterations)
        #pdb.set_trace()
        #Rs = np.zeros(regressionN*2)
        #for reg in range(nRegressionIterations):
        #    Rs += RTempValues[reg]
        #Rs /=nRegressionIterations
        Rvalues.append([nDay,allCorrDataPerSession[nDay]['folder'],RTempValues])

    return Rvalues
#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateStepTriggeredCaTraces(mouse,allCorrDataPerSession,allStepData,trigger='swingOnset',calculateFast=False):  # swingOnset or swingOffset
    matplotlib.use('TkAgg')
    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    timeAxis = np.linspace(-0.4,0.6,int((0.4+0.6)/0.02)+1)
    timeAxisRescaled = np.linspace(-1.,2.,int((1.+2.)/0.02)+1)
    preStanceMask = timeAxis<-0.1
    preStanceRescaledMask = timeAxisRescaled<-0.2
    K = len(timeAxis)
    KRescaled = len(timeAxisRescaled)
    caTraces = []
    maxTimeDelay = 1.
    for nDay in range(len(allCorrDataPerSession)):
        print(allCorrDataPerSession[nDay]['folder'], allStepData[nDay][0], nDay)
        # consistency check
        if not (allCorrDataPerSession[nDay]['folder'] ==  allStepData[nDay][0]):
            print('All animal data and swing data not from the same day!')
        #
        caRecTime = allCorrDataPerSession[nDay]['caImg']['timeStamps'][0, 3]
        wheelRecTime = allStepData[nDay][1][0][3] # check recording start of first recording allStepData[nDay][1][3]
        pawRecTime = allStepData[nDay][2][0][4] # again, third indices picks first recording
        #pdb.set_trace()
        timeDiffCaWheel = np.abs(caRecTime-wheelRecTime)
        timeDiffCaPaw = np.abs(caRecTime-pawRecTime)
        timeDiffWheelPaw = np.abs(wheelRecTime-pawRecTime)
        if any(np.array([timeDiffCaWheel,timeDiffCaPaw,timeDiffWheelPaw]) > maxTimeDelay):
            print('PROBLEM in data consistency!')
            print('recordings are separated by %s - %s - %s s min' % (timeDiffCaWheel,timeDiffCaPaw,timeDiffWheelPaw))
            pdb.set_trace()
        else:
            print('Delay between recordings is :', timeDiffCaWheel,timeDiffCaPaw,timeDiffWheelPaw, 's')
        #print(allCorrDataPerSession[nDay][0],nDay)  getCaWheelPawInterpolatedDictsPerDay
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict,slowestTrial) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession,allStepData)
        #if len(allStepData[nDay-1][4])==6:
        #    print('more recordings :',len(allStepData[nDay][4]))
        #    addIdx = 1
        #else:
        #    addIdx = 0
        #pdb.set_trace()
        N = len(caTracesDict[0][1:]) # number of ROIs
        caSnippets = [[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)]] #np.zeros((4,N,K))
        caSnippetsRescaled = [[[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)],[[] for i in range(N)]]
        recordingID = [[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)]]
        recordingIDRescaled = [[[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]]

        NRecs=len(allStepData[nDay][4])
        for nrec in range(NRecs): # loop over the five recordings of a day
            for i in range(4): # loop over the four paws
                #pdb.set_trace()
                idxSwings = allStepData[nDay][4][nrec][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay][4][nrec][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                if trigger == 'swingOnset':
                    NstepCycles = len(idxSwings)
                elif trigger == 'swingOffset':
                    NstepCyles = len(idxSwings)-1
                for k in range(NstepCyles): # loop over all swings
                    startSwingTime = recTimes[idxSwings[k, 0]]
                    endSwingTime = recTimes[idxSwings[k, 1]]
                    if trigger == 'swingOnset':
                        triggerTime = startSwingTime
                        duration = (endSwingTime-startSwingTime)
                    elif trigger == 'swingOffset':
                        triggerTime = endSwingTime
                        duration = recTimes[idxSwings[k+1, 0]] - endSwingTime
                    if len(caTracesDict[nrec][1:])!=N:
                        print('problem in number of ROIs')
                        pdb.set_trace(0)
                    for l in range(len(caTracesDict[nrec][1:])): # loop over all ROIs
                        interpCa = interp1d(caTracesDict[nrec][0]-triggerTime, caTracesDict[nrec][l+1])#,kind='cubic')
                        interpCaRescaled = interp1d((caTracesDict[nrec][0]-triggerTime)/(duration), caTracesDict[nrec][l+1])#,kind='cubic')
                        ############
                        try:
                            newCaTraceAtSwing = interpCa(timeAxis)
                        except ValueError:
                            pass
                        else:
                            #caSnippets[i,l,:] += newCaTraceAtSwing
                            caSnippets[i][l].append(newCaTraceAtSwing)
                            recordingID[i][l].append(nrec)
                        ############
                        try:
                            newCaTraceAtSwingRescaled = interpCaRescaled(timeAxisRescaled)
                        except ValueError:
                            #print('error')
                            pass
                        else:
                            #caSnippets[i,l,:] += newCaTraceAtSwing
                            caSnippetsRescaled[i][l].append(newCaTraceAtSwingRescaled)
                            recordingIDRescaled[i][l].append(nrec)
        #pdb.set_trace()
        # 4 paws
        # N number of ROIS
        # 2 mean and std
        # K number of time points during average
        alpha = 0.05
        def get_CI(dat, alpha=.05):
            return np.array(list(map(lambda x: boot.ci(x, alpha=alpha), dat)))
        caSnippetsArray = np.zeros((4, N, 3 + 3*NRecs, K))
        caSnippetsRescaledArray = np.zeros((4, N, 3 + 3*NRecs, KRescaled))
        for i in range(4): # loop over four paws
            for l in range(N): # loop over all ROIs
                caTempArray = np.asarray(caSnippets[i][l])
                #pdb.set_trace()
                caSnippetsZscores = (caTempArray - np.mean(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
                if calculateFast:
                    CI = np.zeros((len(timeAxis),2))
                else:
                    print('calculating CIs of global average ...')
                    CI = np.array(list(map(lambda x: boot.ci(x, alpha=.05), caSnippetsZscores.T)))
                    print('done')
                caTemp = np.mean(caSnippetsZscores,axis=0)
                #caTempSTD = np.std(caSnippetsZscores,axis=0)
                caSnippetsArray[i,l,0,:] = caTemp
                caSnippetsArray[i,l,1,:] = CI[:,0]
                caSnippetsArray[i,l,2,:] = CI[:,1]
                caTempRec = []
                for nrec in range(NRecs):
                    recMask =  (np.asarray(recordingID[i][l]) == nrec)
                    caTempRec.append(caSnippetsZscores[recMask])
                if calculateFast:
                    ci_cell_trial = np.zeros((NRecs, len(timeAxis), 2))
                else:
                    print('calculating CIs of recording average for paw %s and roi %s ...' % (i, l))
                    ci_cell_trial = np.array(Parallel(n_jobs=numcores)(delayed(get_CI)(c.T, alpha) for c in caTempRec))
                    print('done')
                for nrec in range(NRecs):
                    caSnippetsArray[i, l, 3 + nrec*3, :] = np.mean(caTempRec[nrec], axis=0)
                    caSnippetsArray[i, l, 4 + nrec*3, :] = ci_cell_trial[nrec][:,0]
                    caSnippetsArray[i, l, 5 + nrec*3, :] = ci_cell_trial[nrec][:,1]
                #pdb.set_trace()
                caTempRescaledArray = np.asarray(caSnippetsRescaled[i][l])
                #pdb.set_trace()
                caSnippetsRescaledZscores = (caTempRescaledArray - np.mean(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis])# /np.std(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis]
                caTempRe = np.mean(caSnippetsRescaledZscores,axis=0)

                if calculateFast:
                    CI = np.zeros((len(timeAxisRescaled),2))
                else:
                    print('calculating CIs of global rescaled recording average')
                    CI = np.array(list(map(lambda x: boot.ci(x, alpha=.05), caSnippetsRescaledZscores.T)))
                    print('done')
                #caTempReSTD = np.std(caSnippetsRescaledZscores,axis=0)
                caSnippetsRescaledArray[i,l,0,:] = caTempRe
                caSnippetsRescaledArray[i,l,1,:] = CI[:,0]
                caSnippetsRescaledArray[i,l,2,:] = CI[:,1]
                caTempRecRescaled = []
                for nrec in range(NRecs):
                    recMask =  (np.asarray(recordingIDRescaled[i][l]) == nrec)
                    caTempRecRescaled.append(caSnippetsRescaledZscores[recMask])
                    #caTemp = np.mean(caSnippetsRescaledZscores[recMask], axis=0)
                    #caTempSTD = np.std(caSnippetsRescaledZscores[recMask], axis=0)
                    #CI = np.array(list(map(lambda x: boot.ci(x, alpha=.05), caSnippetsRescaledZscores[recMask].T)))
                if calculateFast :
                    ci_cell_trial_rescaled =  np.zeros((NRecs,len(timeAxisRescaled),2))
                else:
                    print('calculating CIs of rescaled recording average for paw %s and roi %s ...' % (i, l))
                    ci_cell_trial_rescaled = np.array(Parallel(n_jobs=numcores)(delayed(get_CI)(c.T, alpha) for c in caTempRecRescaled))
                    print('done')
                for nrec in range(NRecs):
                    caSnippetsRescaledArray[i, l, 3 + nrec*3, :] = np.mean(caTempRecRescaled[nrec], axis=0)
                    caSnippetsRescaledArray[i, l, 4 + nrec*3, :] = ci_cell_trial_rescaled[nrec][:,0]
                    caSnippetsRescaledArray[i, l, 5 + nrec*3, :] = ci_cell_trial_rescaled[nrec][:,1]
        caTraces.append([allCorrDataPerSession[nDay]['folder'],allStepData[nDay][0],nDay,timeAxis,caSnippetsArray,timeAxisRescaled,caSnippetsRescaledArray])

    return caTraces

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateStepTriggeredCaTracesAllPaws(mouse,allCorrDataPerSession,allStepData):
    maxSeparation = 0.# min separation between swings in sec

    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    timeAxis = np.linspace(-0.4,0.6,(0.6+0.4)/0.02+1)
    timeAxisRescaled = np.linspace(-1.,2.,(2+1)/0.02+1)
    preStanceMask = timeAxis<-0.1
    preStanceRescaledMask = timeAxisRescaled<-0.2
    K = len(timeAxis)
    KRescaled = len(timeAxisRescaled)
    caTraces = []
    swingT = []
    for nDay in range(len(allCorrDataPerSession)):
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession)
        if len(allStepData[nDay-1][4])==6:
            print('more recordings :',len(allStepData[nDay-1][4]))
            addIdx = 1
        else:
            addIdx = 0
        #pdb.set_trace()
        N = len(caTracesDict[0][1:])
        print(allCorrDataPerSession[nDay][0], allStepData[nDay - 1][0], nDay, N)
        caSnippets = [[] for i in range(N)] #np.zeros((4,N,K))
        caSnippetsRescaled = [[] for i in range(N)]
        caSnippetsArray = np.zeros((N,2,K))
        caSnippetsRescaledArray = np.zeros((N, 2, KRescaled))
        swingSnippets = [[] for i in range(5)]
        for nrec in range(5): # loop over the five recordings of a day
            swingTimes = np.zeros(3)
            for i in range(4): # loop over the four paws and lump all swing times together
                idxSwings = allStepData[nDay-1][4][nrec+addIdx][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay-1][4][nrec+addIdx][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                startSwingTimes = recTimes[idxSwings[:,0]]
                endSwingTimes = recTimes[idxSwings[:,1]]
                swingTimes = np.vstack((swingTimes,np.column_stack((startSwingTimes, endSwingTimes,np.repeat(i,len(startSwingTimes))))))
            swingTimes = swingTimes[1:] # remove first element which was zeros only
            # sort swing times according to swing start
            swingTimes = swingTimes[swingTimes[:,0].argsort()]
            # remove swings with fall within the minimum separation betweeen swings
            diffSwings = np.diff(swingTimes[:,0]) # calculate inter-swing intervals
            swingTimesSparse = swingTimes[np.concatenate((diffSwings>maxSeparation,np.array([True])))] # only use swings which fall above separation time
            swingSnippets[nrec] = swingTimes
            for k in range(len(swingTimesSparse)): # loop over all swings
                startSwingTime = swingTimesSparse[k, 0]
                endSwingTime = swingTimesSparse[k, 1]

                if len(caTracesDict[nrec][1:])!=N: print('problem in number of ROIs')
                for l in range(len(caTracesDict[nrec][1:])): # loop over all ROIs
                    interpCa = interp1d(caTracesDict[nrec][0]-startSwingTime, caTracesDict[nrec][l+1])#,kind='cubic')
                    interpCaRescaled = interp1d((caTracesDict[nrec][0]-startSwingTime)/(endSwingTime-startSwingTime), caTracesDict[nrec][l+1])#,kind='cubic')
                    ############
                    try:
                        newCaTraceAtSwing = interpCa(timeAxis)
                    except ValueError:
                        pass
                    else:
                        #caSnippets[i,l,:] += newCaTraceAtSwing
                        caSnippets[l].append(newCaTraceAtSwing)
                    ############
                    try:
                        newCaTraceAtSwingRescaled = interpCaRescaled(timeAxisRescaled)
                    except ValueError:
                        #print('error')
                        pass
                    else:
                        #caSnippets[i,l,:] += newCaTraceAtSwing
                        caSnippetsRescaled[l].append(newCaTraceAtSwingRescaled)
        #pdb.set_trace()
        #for i in range(4):
        for l in range(N):
            caTempArray = np.asarray(caSnippets[l])
            caSnippetsZscores = (caTempArray - np.mean(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
            caTemp = np.mean(caSnippetsZscores,axis=0)
            caTempSTD = np.std(caSnippetsZscores,axis=0)
            caSnippetsArray[l,0,:] = caTemp
            caSnippetsArray[l,1,:] = caTempSTD
            #
            caTempRescaledArray = np.asarray(caSnippetsRescaled[l])
            #pdb.set_trace()
            caSnippetsRescaledZscores = (caTempRescaledArray - np.mean(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
            caTempRe = np.mean(caSnippetsRescaledZscores,axis=0)
            caTempReSTD = np.std(caSnippetsRescaledZscores,axis=0)
            caSnippetsRescaledArray[l,0,:] = caTempRe
            caSnippetsRescaledArray[l,1,:] = caTempReSTD
        swingT.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,swingSnippets])
        caTraces.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,caSnippetsArray,caSnippetsRescaledArray])

    return caTraces


#################################################################################
def calcualteAllDeltaTValues(t1,t2):
    l1 = len(t1)
    l2 = len(t2)

    fannedOut1 = np.tile(t1,(l2,1))
    fannedOut2 = np.tile(t2,(l1,1))
    transposedFannedOut2 = np.transpose(fannedOut2)
    #print l1, l2
    #print shape(fannedOut1), shape(transposedFannedOut2)
    #pdb.set_trace()
    differences = fannedOut1 - transposedFannedOut2 # subtract(fannedOut1,transposedFannedOut2)
    return differences.flatten()

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateInterstepTimeHistogram(mouse,allCorrDataPerSession,allStepData):
    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    #timeAxis = np.linspace(-0.4,0.6,(0.6+0.4)/0.02+1)
    #timeAxisRescaled = np.linspace(-1.,2.,(2+1)/0.02+1)
    #preStanceMask = timeAxis<-0.1
    #preStanceRescaledMask = timeAxisRescaled<-0.2
    #K = len(timeAxis)
    #KRescaled = len(timeAxisRescaled)
    pawSwingTimes = []
    for nDay in range(1,len(allCorrDataPerSession)):
        print(allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay)
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession)
        if len(allStepData[nDay-1][4])==6:
            print('more recordings :',len(allStepData[nDay-1][4]))
            addIdx = 1
        else:
            addIdx = 0
        #pdb.set_trace()
        #N = len(caTracesDict[0][1:])
        #caSnippets = [[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)]] #np.zeros((4,N,K))
        #caSnippetsRescaled = [[[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]]
        #caSnippetsArray = np.zeros((4,N,2,K))
        #caSnippetsRescaledArray = np.zeros((4, N, 2, KRescaled))
        allData = []
        for nrec in range(5): # loop over the five recordings of a day
            startSwingTimes = [[] for i in range(4)]
            endSwingTimes = [[] for i in range(4)]
            for i in range(4): # loop over the four paws
                idxSwings = allStepData[nDay-1][4][nrec+addIdx][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay-1][4][nrec+addIdx][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                startSwingT = recTimes[idxSwings[:,0]]
                endSwingT = recTimes[idxSwings[:,1]]
                startSwingTimes[i].append(startSwingT)
                endSwingTimes[i].append(endSwingT)
            allData.append([startSwingTimes,endSwingTimes])
        interPawSwingTimes = [[] for i in range(4)]
        interStepTimes = []
        stepLengths = [[] for i in range(4)]
        #pdb.set_trace()
        for nrec in range(5):
            for i in range(4):
                interPawSwingTimes[i].extend(calcualteAllDeltaTValues(allData[nrec][0][i][0],allData[nrec][0][i][0]))
                stepLengths[i].extend(allData[nrec][1][i][0]-allData[nrec][0][i][0])
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0],allData[nrec][0][1][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0], allData[nrec][0][2][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0], allData[nrec][0][3][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][1][0], allData[nrec][0][2][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][1][0], allData[nrec][0][3][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][2][0], allData[nrec][0][3][0]))
        #pdb.set_trace()
        pawSwingTimes.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,interPawSwingTimes,stepLengths,interStepTimes])

    return pawSwingTimes
#################################################################################
# remove empty columns and row - from the image registration routine
#################################################################################
def removeEmptyColumnAndRows(img):
    # hmask = np.invert(np.sum(img, axis=0) == 0) # looking for zeros does not work as the boundary values are not zeros all the time
    # vmask = np.invert(np.sum(img, axis=1) == 0)
    htemp = (img == img[0,:]) # look instead for same values in row
    hmask = np.invert(htemp.all(axis=0))
    vtemp = (img == img[:,0])
    vmask = np.invert(vtemp.all(axis=1))
    idxH = np.arange(len(hmask))[np.hstack((False,np.diff(hmask)>0))]
    idxV = np.arange(len(vmask))[np.hstack((False, np.diff(vmask)>0))]
    #pdb.set_trace()
    if len(idxV) == 0:
        idxV = np.array([0,np.shape(img)[0]])
    if len(idxH) == 0:
        idxH = np.array([0,np.shape(img)[1]])

    croppedImg = img[idxV[0]:idxV[1],idxH[0]:idxH[1]]

    cutLengths = np.vstack((idxV,idxH))
    #pdb.set_trace()
    return cutLengths


#################################################################################
# remove empty columns and row - from the image registration routine
#################################################################################
def alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,refDate,otherDate,movementValues,figSave=False,figDir=''):
    #matplotlib.use('TkAgg')
    column1 = np.maximum(cutLengthsA[:,0],cutLengthsB[:,0])
    column2 = np.minimum(cutLengthsA[:,1],cutLengthsB[:,1])
    cutLenghts = np.column_stack((column1,column2))

    imgA = imgA[cutLenghts[0,0]:cutLenghts[0,1],cutLenghts[1,0]:cutLenghts[1,1]]
    imgB = imgB[cutLenghts[0,0]:cutLenghts[0,1],cutLenghts[1,0]:cutLenghts[1,1]]
    # Find size of ref image
    sz = imgA.shape

    corr = signal.correlate(imgA - imgA.mean(), imgB - imgB.mean(), mode='same', method='fft')
    maxIdx = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    shifty = np.shape(imgA)[0]/2. - maxIdx[0]
    shiftx = np.shape(imgA)[1]/2. - maxIdx[1]
    print('max of cross-correlation : ', shiftx, shifty )
    #pdb.set_trace()
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION  #cv2.MOTION_EUCLIDEAN # cv2.MOTION_TRANSLATION  # MOTION_EUCLIDEAN
    warp_mode = cv2.MOTION_AFFINE #EUCLIDEAN #HOMOGRAPHY
    warp_modes = [cv2.MOTION_AFFINE,cv2.MOTION_EUCLIDEAN,cv2.MOTION_TRANSLATION]

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    #try:
    imA_u8 = (((imgA-np.min(imgA))/(np.max(imgA)-np.min(imgA)))*255).astype(np.uint8) #cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
    imB_u8 = (((imgB-np.min(imgB))/(np.max(imgB)-np.min(imgB)))*255).astype(np.uint8)
    #pdb.set_trace()
    warpResults = []
    corrMax = []
    for w in range(len(warp_modes)):
        print('testing ', warp_modes[w])
        # warp_matrix1[0, 2] = aS.xOffset
        # warp_matrix1[1, 2] = aS.yOffset
        if (movementValues[0] != 0) and (movementValues[1] != 0):
            warp_matrix[0, 2] = movementValues[0]  # -20.
            warp_matrix[1, 2] = movementValues[1]  # -40.
        else:
            warp_matrix[0, 2] = shiftx  # -20.
            warp_matrix[1, 2] = shifty  # -40.
        #try:
        #    (cc, warp_matrixRet) = cv2.findTransformECC(imA_u8, imB_u8, warp_matrix, warp_modes[w], criteria, inputMask = None, gaussFiltSize=5)
        #except TypeError:
        try :
            (cc, warp_matrixRet) = cv2.findTransformECC(imA_u8, imB_u8, warp_matrix, warp_modes[w], criteria, inputMask = None)
        except:
            print('findTransformECC did not converge')
            cc = -1
            warp_matrixRet = warp_matrix
        #print(warp_matrixRet,warp_matrix)
        warpResults.append([w,warp_modes[w],np.copy(warp_matrixRet),np.copy(cc)])
        corrMax.append(cc)
        #if cc>0.8:
        #    break
    print(warpResults)
    corrMax = np.asarray(corrMax)
    maxCorr = np.argmax(corrMax)
    cc_max = warpResults[maxCorr][3]
    warp_matrix_max = warpResults[maxCorr][2]
    #except:
    #print('findTransformECC output : ',cc,warp_matrixRet)
    #print('find image transformation did not converge')
    #warp_matrixRet = np.copy(warp_matrix_max)
    #cc = None
    #else:
        #pass
    #(cc2, warp_matrix2Ret) = cv2.findTransformECC(imBD, im820, warp_matrix2, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        imgB_aligned = cv2.warpPerspective(imgB, warp_matrix_max, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        imgB_aligned = cv2.warpAffine(imgB, warp_matrix_max, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    print('result of image alignment-> warp-matrix  and correlation coefficient : ', warp_matrix_max, cc_max)

    if figSave :
        ##################################################################
        # Show final results
        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 11, 'axes.titlesize': 11, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 2  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.06)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],hspace=0.2)
        # ax0 = plt.subplot(gssub[0])

        # fig = plt.figure(figsize=(10,10))

        #plt.figtext(0.1, 0.95, '%s ' % (aS.animalID), clip_on=False, color='black', size=14)

        ax0 = plt.subplot(gs[0])
        ax0.set_title('reference image %s' % refDate)
        ax0.imshow(imgA)

        ax0 = plt.subplot(gs[1])
        ax0.set_title('to-be-aligned image %s' % otherDate )
        ax0.imshow(imgB)

        ax0 = plt.subplot(gs[2])
        ax0.set_title('overlay of both images')
        overlayBefore = cv2.addWeighted(imgA/np.max(imgA), 1, imgB/np.max(imgB), 1, 0)
        ax0.imshow(overlayBefore)

        ax0 = plt.subplot(gs[3])
        ax0.set_title('overlay after alignement c = %s \nof BD-AD images' % np.round(cc_max,4), fontsize=10)
        overlayAfter = cv2.addWeighted(imgA/np.max(imgA), 1, imgB_aligned/np.max(imgB_aligned), 1, 0)
        ax0.imshow(overlayAfter)

        #plt.show()
        plt.savefig(figDir + 'ImageAlignment_%s-%s.pdf' % (refDate,otherDate))  # plt.savefig(figOutDir+'ImageAlignment_%s.png' % aS.animalID)  # plt.show()
        plt.close()

    return (warp_matrix_max,cc_max)

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def alignROIsCheckOverlap(statRef,opsRef,statAlign,opsAlign,warp_matrix,refDate,otherDate,figSave=False,figDir=''):
    ncellsRef= len(statRef)
    ncellsAlign = len(statAlign)

    imMaskRef   = np.zeros((opsRef['Ly'], opsRef['Lx']))
    imMaskAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

    intersectionROIs = []
    intersectionROIsA = []
    for n in range(0,ncellsRef):
        imMaskRef[:] = 0
        #if iscellBD[n][0]==1:
        #pdb.set_trace()
        ypixRef = statRef[n]['ypix']
        xpixRef = statRef[n]['xpix']
        imMaskRef[ypixRef,xpixRef] = 1
        for m in range(0,ncellsAlign):
            imMaskAlign[:] = 0
            #if iscellAD[m][0]==1:
            ypixAl = statAlign[m]['ypix']
            xpixAl = statAlign[m]['xpix']
            # perform homographic transform : rotation + translation
            #pdb.set_trace()
            points = np.column_stack((xpixAl,ypixAl))
            newPoints = np.copy(points)
            #pdb.set_trace()
            warp_matrix_inverse = np.copy(warp_matrix)
            cv2.invertAffineTransform(warp_matrix,warp_matrix_inverse)
            #newPoints = cv2.transform(points,warp_matrix_inverse)
            xpixAlPrime = np.rint(xpixAl*warp_matrix_inverse[0,0] + ypixAl*warp_matrix_inverse[0,1] + warp_matrix_inverse[0,2])
            ypixAlPrime = np.rint(xpixAl*warp_matrix_inverse[1,0] + ypixAl*warp_matrix_inverse[1,1] + warp_matrix_inverse[1,2]) # - np.rint(warp_matrix[1,2])
            xpixAlPrime = np.array(xpixAlPrime,dtype=int)
            ypixAlPrime = np.array(ypixAlPrime,dtype=int)
            #pdb.set_trace()
            # make sure pixels remain within
            xpixAlPrime2 = xpixAlPrime[(xpixAlPrime<opsAlign['Lx'])&(ypixAlPrime<opsAlign['Ly'])]
            ypixAlPrime2 = ypixAlPrime[(xpixAlPrime<opsAlign['Lx'])&(ypixAlPrime<opsAlign['Ly'])]
            imMaskAlign[ypixAlPrime2,xpixAlPrime2] = 1
            #imMaskAlign[xpixAlPrime2,ypixAlPrime2] = 1
            intersection = np.sum(np.logical_and(imMaskRef,imMaskAlign))
            eitherOr = np.sum(np.logical_or(imMaskRef,imMaskAlign))
            if intersection>0.2:
                #print(n,m,intersection,eitherOr,intersection/eitherOr)
                intersectionROIs.append([n,m,xpixRef,ypixRef,xpixAlPrime2,ypixAlPrime2,intersection,eitherOr,intersection/eitherOr])
                intersectionROIsA.append([n,m,intersection,eitherOr,intersection/eitherOr])
    # clean up intersection ROIs; each ROI should only overlap once
    def removeDoubleCellOccurrences(interROIs,column):
        uniquePerColumn = np.unique(interROIs[:,column],return_counts=True) # find unique occurrences
        multipleCells = uniquePerColumn[0][uniquePerColumn[1]>1]  # which cells occur more than once in the first column
        indiciesToRemove = []
        for i in multipleCells:
            indicies = np.argwhere(interROIs[:,column]==i)
            maxIdx = np.argmax(interROIs[indicies[:,0]][:,4])
            delIndicies = np.delete(indicies,maxIdx)
            indiciesToRemove.extend(delIndicies)
        return indiciesToRemove
    intersectionROIsA = np.asarray(intersectionROIsA)
    removeIdicies0 = removeDoubleCellOccurrences(intersectionROIsA,0)
    removeIdicies1 = removeDoubleCellOccurrences(intersectionROIsA,1)
    removeIndicies = np.asarray(removeIdicies0 + removeIdicies1)
    removeIndicies = np.unique(removeIndicies)
    cleanedIntersectionROIs = []
    for i in range(len(intersectionROIs)):
        if i not in removeIndicies:
            cleanedIntersectionROIs.append(intersectionROIs[i])
    #pdb.set_trace()
    if len(removeIndicies)>0:
        intersectionROIsA = np.delete(intersectionROIsA,removeIndicies,axis=0)
    #pdb.set_trace()
    if figSave:
        imRef = opsRef['meanImg']
        imAlign = opsAlign['meanImg']
        ##################################################################
        # Show final results
        fig = plt.figure(figsize=(15, 15))  ########################

        plt.figtext(0.1, 0.95, '%s and %s' % (refDate,otherDate), clip_on=False, color='black', size=14)

        ax0 = fig.add_subplot(3, 2, 1)  #############################
        ax0.set_title('reference img')
        ax0.imshow(imRef)

        ax0 = fig.add_subplot(3, 2, 2)  #############################
        ax0.set_title('image to be aligned')
        ax0.imshow(imAlign)


        ax0 = fig.add_subplot(3, 2, 3)  #############################
        ax0.set_title('ROIs in reference image')
        imRef = np.zeros((opsRef['Ly'], opsRef['Lx']))
        imRefB = np.zeros((opsRef['Ly'], opsRef['Lx']))

        for n in range(0, ncellsRef):
            ypixR = statRef[n]['ypix']
            xpixR = statRef[n]['xpix']
            imRef[ypixR, xpixR] = n + 1
            imRefB[ypixR, xpixR] = 1

        ax0.imshow(imRef, cmap='gist_ncar')

        ax0 = fig.add_subplot(3, 2, 4)  #############################
        ax0.set_title('ROIs in aligned image')
        imAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))
        imAlignB = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

        for n in range(0, ncellsAlign):
            ypixA = statAlign[n]['ypix']
            xpixA = statAlign[n]['xpix']
            imAlign[ypixA, xpixA] = n + 1
            imAlignB[ypixA, xpixA] = 2

        ax0.imshow(imAlign, cmap='gist_ncar')


        ax0 = fig.add_subplot(3, 2, 5)  #############################
        ax0.set_title('overlapping ROIs Ref-Aligned')
        imRef = np.zeros((opsRef['Ly'], opsRef['Lx']))
        imAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

        for n in range(0, len(cleanedIntersectionROIs)):
            ypixR = cleanedIntersectionROIs[n][3]
            xpixR = cleanedIntersectionROIs[n][2]
            ypixA = cleanedIntersectionROIs[n][5]
            xpixA = cleanedIntersectionROIs[n][4]
            imRef[ypixR, xpixR] = 1
            imAlign[ypixA, xpixA] = 2

        overlayBothROIs1 = cv2.addWeighted(imRef, 1, imAlign, 1, 0)
        #overlayBothROIs1B = cv2.addWeighted(imRefB, 1, imAlignB, 1, 0)
        ax0.imshow(overlayBothROIs1)

        ax0 = fig.add_subplot(3, 2, 6)  #############################
        ax0.set_title('fraction of ROI overlap Ref-Aligned')
        interFractions1 = []
        for n in range(0, len(cleanedIntersectionROIs)):
            interFractions1.append(cleanedIntersectionROIs[n][8])

        ax0.hist(interFractions1, bins=15)

        plt.savefig(figDir + 'ROIalignment_%s-%s.pdf' % (refDate, otherDate))
        #plt.show()
        plt.close()

    return (cleanedIntersectionROIs,intersectionROIsA)
    #pickle.dump(intersectionROIs, open( dataOutDir + 'ROIintersections_%s.p' % aS.animalID, 'wb' ) )

#################################################################################
# find ROI recorded on ref day and on any other given day
#################################################################################
def findMatchingRois(mouse,allCorrDataPerSession,analysisLocation,refDate=0):
    # check for sanity
    nDays = len(allCorrDataPerSession)

    refDay = allCorrDataPerSession[refDate][0]
    print('fluo images will be aligned to recordings of :', refDay)
    refDayCaData = allCorrDataPerSession[refDate][3][0]
    refImg = refDayCaData[2]['meanImgE']
    refImgCutLengths = removeEmptyColumnAndRows(refImg)
    opsRef = refDayCaData[2]
    statRef = refDayCaData[4]

    # create list of recoridng day indicies
    recDaysList = [i for i in range(nDays)]
    movementValuesPreset = np.zeros((len(recDaysList), 2))
    # movementValuesPreset[0] = np.array([-20,-47])
    movementValuesPreset[1] = np.array([143, 153])
    # movementValuesPreset[3] = np.array([-1,15])

    # remove day used for referencing
    recDaysList.remove(refDate)
    if os.path.exists(analysisLocation+'/alignmentData.p'):
        allDataRead = pickle.load(open(analysisLocation+'/alignmentData.p'))
    else:
        allDataRead = None
    allData = []
    for nDay in recDaysList:
        print(allCorrDataPerSession[nDay][0],nDay)
        #imgE = allCorrDataPerSession[nDay][3][0][2]['meanImgE']
        img = allCorrDataPerSession[nDay][3][0][2]['meanImgE']
        cutLengths = removeEmptyColumnAndRows(img)
        if allDataRead is not None:
            warp_matrix = allDataRead[nDay][3]
        else:
            (warp_matrix,cc) = alignTwoImages(refImg,refImgCutLengths,img,cutLengths,allCorrDataPerSession[refDate][0],allCorrDataPerSession[nDay][0],movementValuesPreset[nDay],figShow=True,)
        opsAlign  = allCorrDataPerSession[nDay][3][0][2]
        statAlign = allCorrDataPerSession[nDay][3][0][4]
        (cleanedIntersectionROIs,intersectionROIsA) = alignROIsCheckOverlap(statRef,opsRef,statAlign,opsAlign,warp_matrix,allCorrDataPerSession[refDate][0],allCorrDataPerSession[nDay][0],showFig=True)
        print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statRef), len(statAlign), len(cleanedIntersectionROIs))
        allData.append([allCorrDataPerSession[nDay][0],nDay,cutLengths,warp_matrix,cc,cleanedIntersectionROIs,intersectionROIsA])

    intersectingCellsInRefRecording = np.arange(len(statRef))
    for nDay in recDaysList:
        intersectingCellsInRefRecording = np.intersect1d(intersectingCellsInRefRecording,allData[nDay][5][:,0])
        print(nDay,allCorrDataPerSession[nDay][0],intersectingCellsInRefRecording)

    pdb.set_trace()

    return 0


#################################################################################
# correlates mean fluo images recorded all possible recording day combinations
#################################################################################
def findOverlayMatchingRoisAllDayCombinations(allCorrDataPerSession, figLocation, allDataRead=None,saveFigure=True):
    nDays = len(allCorrDataPerSession)
    movementValuesPreset = np.zeros((nDays*nDays, 2))
    allOverlayData = {}
    #corrMatrix = np.zeros((nDays,nDays))
    nPair = 0
    for nDayA in range(nDays):
        for nDayB in range(nDays):
            if nDayA != nDayB :
                print(nDayA,nDayB, allCorrDataPerSession[nDayA]['folder'], allCorrDataPerSession[nDayB]['folder'])

                #imgA = allCorrDataPerSession[nDayA][3][0][2]['meanImg']
                imgA = allCorrDataPerSession[nDayA]['caImg']['ops']['meanImg']
                cutLengthsA = removeEmptyColumnAndRows(imgA)
                opsA = allCorrDataPerSession[nDayA]['caImg']['ops']  # allCorrDataPerSession[nDayA][3][0][2]
                statA = allCorrDataPerSession[nDayA]['caImg']['stat'] # allCorrDataPerSession[nDayA][3][0][4]

                imgB =  allCorrDataPerSession[nDayB]['caImg']['ops']['meanImg'] # allCorrDataPerSession[nDayB][3][0][2]['meanImg']
                cutLengthsB = removeEmptyColumnAndRows(imgB)
                opsB =  allCorrDataPerSession[nDayB]['caImg']['ops'] # allCorrDataPerSession[nDayB][3][0][2]
                statB = allCorrDataPerSession[nDayB]['caImg']['stat'] # allCorrDataPerSession[nDayB][3][0][4]

                if (allDataRead is not None) and (allDataRead[nPair][0]==allCorrDataPerSession[nDayA]['folder']) and (allDataRead[nPair][1]==allCorrDataPerSession[nDayB]['folder']):
                    print('warp_matrix for current pair of recordings exists and will be used')
                    warp_matrix = allDataRead[nPair][6]
                    cc = allDataRead[nPair][7]
                else:
                    (warp_matrix,cc) = alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,allCorrDataPerSession[nDayA]['folder'],allCorrDataPerSession[nDayB]['folder'],movementValuesPreset[nPair],figSave=saveFigure,figDir=figLocation)
                #corrMatrix[nDayA,nDayB] = cc
                (cleanedIntersectionROIs, intersectionROIsA) = alignROIsCheckOverlap(statA, opsA, statB, opsB, warp_matrix, allCorrDataPerSession[nDayA]['folder'], allCorrDataPerSession[nDayB]['folder'],figSave=saveFigure, figDir=figLocation)
                print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statA), len(statB), len(cleanedIntersectionROIs))
                #allOverlayData.append([allCorrDataPerSession[nDayA]['folder'], allCorrDataPerSession[nDayB]['folder'], nDayA, nDayB, cutLengthsA, cutLengthsB, warp_matrix, cc,cleanedIntersectionROIs, intersectionROIsA,statA,statB])
                allOverlayData[nPair] = {}
                allOverlayData[nPair]['folderA'] = allCorrDataPerSession[nDayA]['folder']  #0
                allOverlayData[nPair]['folderB'] = allCorrDataPerSession[nDayB]['folder']  #1
                allOverlayData[nPair]['nDayA'] = nDayA #2
                allOverlayData[nPair]['nDayB'] = nDayB  #3
                allOverlayData[nPair]['cutLengthsA'] = cutLengthsA #4
                allOverlayData[nPair]['cutLengthsB'] = cutLengthsB  #5
                allOverlayData[nPair]['warp_matrix'] = warp_matrix #6
                allOverlayData[nPair]['cc'] = cc #7
                allOverlayData[nPair]['cleanedIntersectionROIs'] = cleanedIntersectionROIs  #8
                allOverlayData[nPair]['intersectionROIsA'] = intersectionROIsA  #9
                allOverlayData[nPair]['statA'] = statA  #10
                allOverlayData[nPair]['statB'] = statB  #11
                nPair+=1
        #pdb.set_trace()
    return allOverlayData

#################################################################################
# correlates mean fluo images of one day recorded at 910 and 820 nm
#################################################################################
def findOverlayMatchingRoisDuringOneDay(allCorrDataPerSession910,allCorrDataPerSession820, figLocation, allDataRead=None,saveFigure=True):

    nDays910 = len(allCorrDataPerSession910)
    nDays820 = len(allCorrDataPerSession820)
    movementValuesPreset = np.zeros((nDays910, 2))
    allAlignData = {}
    #corrMatrix = np.zeros((nDays,nDays))
    maxTimeDelay = 45*60 # maximal 40 min difference btw. 910 and 820 recording
    nPair = 0
    for nDay910 in range(nDays910):
        for nDay820 in range(nDays820):
            if allCorrDataPerSession910[nDay910]['folder'][:-4] ==  allCorrDataPerSession820[nDay820]['folder'][:-4]:
                #pdb.set_trace()
                timeDiff = np.abs(allCorrDataPerSession910[nDay910]['caImg']['timeStamps'][0,3] - allCorrDataPerSession820[nDay820]['caImg']['timeStamps'][0,3])
                if timeDiff>maxTimeDelay:
                    print('PROBLEM in data consistency!')
                    print('910 and 820 recordings are separated by %s min' % str(timeDiff/60.))
                    pdb.set_trace()
                else:
                    print('Delay between 910 and 820 imaging sessions is :', timeDiff/60.,'min')
                print(nDay910,nDay820, allCorrDataPerSession910[nDay910]['folder'], allCorrDataPerSession820[nDay820]['folder'])

                img910 = allCorrDataPerSession910[nDay910]['caImg']['ops']['meanImg'] #allCorrDataPerSession910[nDay910][3][0][2]['meanImg']
                cutLengths910 = removeEmptyColumnAndRows(img910)
                ops910 = allCorrDataPerSession910[nDay910]['caImg']['ops'] # allCorrDataPerSession910[nDay910][3][0][2]
                stat910 = allCorrDataPerSession910[nDay910]['caImg']['stat']

                img820 = allCorrDataPerSession820[nDay820]['caImg']['ops']['meanImg'] #[3][0][2]['meanImg']
                cutLengths820 = removeEmptyColumnAndRows(img820)
                ops820 = allCorrDataPerSession820[nDay820]['caImg']['ops'] #[3][0][2]
                stat820 = allCorrDataPerSession820[nDay820]['caImg']['stat'] #[3][0][4]

                if (allDataRead is not None) and (allDataRead[nPair][0]==allCorrDataPerSession910[nDay910]['folder']) and (allDataRead[nPair][1]==allCorrDataPerSession820[nDay820]['folder']):
                    print('warp_matrix for current pair of recordings exists and will be used')
                    warp_matrix = allDataRead[nPair][6]
                    cc = allDataRead[nPair][7]
                else:
                    (warp_matrix,cc) = alignTwoImages(img910,cutLengths910,img820,cutLengths820,allCorrDataPerSession910[nDay910]['folder'],allCorrDataPerSession820[nDay820]['folder'],movementValuesPreset[nPair],figSave=saveFigure,figDir=figLocation)
                #corrMatrix[nDayA,nDayB] = cc
                (cleanedIntersectionROIs, intersectionROIsA) = alignROIsCheckOverlap(stat910, ops910, stat820, ops820, warp_matrix, allCorrDataPerSession910[nDay910]['folder'], allCorrDataPerSession820[nDay820]['folder'],figSave=saveFigure, figDir=figLocation)
                print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(stat910), len(stat820), len(cleanedIntersectionROIs))
                #allAlignData.append([allCorrDataPerSession910[nDay910][0], allCorrDataPerSession820[nDay820][0], nDay910, nDay820, cutLengths910, cutLengths820, warp_matrix, cc,cleanedIntersectionROIs, intersectionROIsA,stat910,stat820])
                allAlignData[nPair] = {}
                allAlignData[nPair]['folder910'] = allCorrDataPerSession910[nDay910]['folder'] #0
                allAlignData[nPair]['folder820'] = allCorrDataPerSession820[nDay820]['folder'] #1
                allAlignData[nPair]['nDay910'] = nDay910 #2
                allAlignData[nPair]['nDay820'] = nDay820 #3
                allAlignData[nPair]['cutLengths910'] = cutLengths910 #4
                allAlignData[nPair]['cutLengths820'] = cutLengths820 #5
                allAlignData[nPair]['warp_matrix'] = warp_matrix  #6
                allAlignData[nPair]['cc'] = cc   # 7
                allAlignData[nPair]['cleanedIntersectionROIs'] = cleanedIntersectionROIs  #8
                allAlignData[nPair]['intersectionROIsA'] = intersectionROIsA #9
                allAlignData[nPair]['statA'] = stat910 #10
                allAlignData[nPair]['statB'] = stat820 #11
                nPair+=1
    return allAlignData


#################################################################################
# find ROIs recorded across successive recording days
#################################################################################
def findMatchingRoisSuccessivDays(mouse,allCorrDataPerSession,analysisLocation,expDate,figLocation, allDataRead=None):
    # check for sanity
    nDays = len(allCorrDataPerSession)

    # create list of recoridng day indicies
    #recDaysList = [i for i in range(nDays)]
    movementValuesPreset = np.zeros((nDays, 2))
    # movementValuesPreset[0] = np.array([-20,-47])
    # movementValuesPreset[1] = np.array([143, 153])
    # movementValuesPreset[3] = np.array([-1,15])

    allDataStore = []
    for nPair in range(nDays-1):
        nDayA = nPair
        nDayB = nPair + 1
        print(nPair, allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0])

        imgA = allCorrDataPerSession[nDayA][3][0][2]['meanImg']
        cutLengthsA = removeEmptyColumnAndRows(imgA)
        opsA = allCorrDataPerSession[nDayA][3][0][2]
        statA = allCorrDataPerSession[nDayA][3][0][4]

        imgB = allCorrDataPerSession[nDayB][3][0][2]['meanImg']
        cutLengthsB = removeEmptyColumnAndRows(imgB)
        opsB = allCorrDataPerSession[nDayB][3][0][2]
        statB = allCorrDataPerSession[nDayB][3][0][4]

        if (allDataRead is not None) and (allDataRead[nPair][0]==allCorrDataPerSession[nDayA][0]) and (allDataRead[nPair][1]==allCorrDataPerSession[nDayB][0]):
            print('warp_matrix for current pair of recordings exists and will be used')
            warp_matrix = allDataRead[nPair][6]
            cc = allDataRead[nPair][7]
        else:
            (warp_matrix,cc) = alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],movementValuesPreset[nPair],figShow=True,figDir=figLocation)

        (cleanedIntersectionROIs,intersectionROIsA) = alignROIsCheckOverlap(statA,opsA,statB,opsB,warp_matrix,allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],showFig=True,figDir=figLocation)
        print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statA), len(statB), len(cleanedIntersectionROIs))
        allDataStore.append([allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],nDayA,nDayB,cutLengthsA,cutLengthsB,warp_matrix,cc,cleanedIntersectionROIs,intersectionROIsA])

    #intersectingCellsInRefRecording = np.arange(len(statA))
    #for nDay in recDaysList:
    #    intersectingCellsInRefRecording = np.intersect1d(intersectingCellsInRefRecording,allData[nDay][5][:,0])
    #    print(nDay,allCorrDataPerSession[nDay][0],intersectingCellsInRefRecording)
    #pdb.set_trace()
    return allDataStore

#################################################################################
# check which ROIs were recoreded across recordings days
#################################################################################
def roisRecordedAllDays(allCorrDataPerSession,allAlignData,alignData910And820,correlationThres):
    def getMatchingPairs(aD):
        ROIpairs = []
        for i in range(len(aD)):
            ROIpairs.append(aD[i][:2])
        ROIpairs = np.asarray(ROIpairs)
        return ROIpairs

    #pdb.set_trace()
    #allCombis = list(itertools.combinations((1,2,3,4,5,6,7,8,9),2))
    nDays = len(allCorrDataPerSession)
    correlationThreshold = correlationThres
    intersectData = {}
    nPair = 0
    for nDayA in range(nDays):
        nRef = 0
        corrDays = []
        # first loop to find indicies which are present in all recordings with good match; note that the idxRemaining will get smaller in each interation over days
        for nDayB in range(nDays):
            if nDayA != nDayB :
                #print(nDayA,nDayB, allCorrDataPerSession[nDayA][0], allCorrDataPerSession[nDayB][0], nRef, nPair)
                if not (nDayA == allAlignData[nPair]['nDayA'] and nDayB == allAlignData[nPair]['nDayB']):
                    print(nDayA, nDayB, allAlignData[nPair][2], allAlignData[nPair][3])
                    print('sanity check failed! The pairing doesn\'t correspond to the day-pair')
                if nRef==0:
                    matchingRoisBefore = getMatchingPairs(allAlignData[nPair]['cleanedIntersectionROIs'])  # [8]
                    if allAlignData[nPair]['cc']>correlationThreshold:
                        corrDays.append([nDayB,allCorrDataPerSession[nDayB]['folder'],nPair])   #[0]
                else:
                    matchingRoisAfter = getMatchingPairs(allAlignData[nPair]['cleanedIntersectionROIs'])  # [8]
                    if allAlignData[nPair]['cc']>correlationThreshold:
                        #print(nDayA, nDayB, allCorrDataPerSession[nDayA][0], allCorrDataPerSession[nDayB][0], nRef, nPair)
                        idxRemaining = np.intersect1d(matchingRoisBefore[:,0], matchingRoisAfter[:,0]) #
                        idxRemainingBefore = [key for key,val in enumerate(matchingRoisBefore[:,0]) if val in idxRemaining]
                        idxRemainingAfter = [key for key,val in enumerate(matchingRoisAfter[:,0]) if val in idxRemaining]
                        BeforeAlsoAfter = matchingRoisBefore[idxRemainingBefore]
                        AfterAlsoBefore = matchingRoisAfter[idxRemainingAfter]
                        print('ROIS remaining before and after : ', len(idxRemaining), nDayB, allCorrDataPerSession[nDayB]['folder'])
                        matchingRoisBefore = np.copy(BeforeAlsoAfter)
                        corrDays.append([nDayB,allCorrDataPerSession[nDayB]['folder'],nPair])
                        idxRemainingGood = np.copy(idxRemaining)
                    #intersectData.append([nPair,matchingRoisBefore,matchingRoisAfter,idxRemaining,BeforeAlsoAfter,AfterAlsoBefore])
                    #print(nPair,matchingRoisBefore,matchingRoisAfter,idxRemaining,BeforeAlsoAfter,AfterAlsoBefore)
                    #pdb.set_trace()
                nRef+=1
                nPair+=1
        remainingRoisExists = (True if 'idxRemainingGood' in locals() else False)
        print(nDayA, allCorrDataPerSession[nDayA]['folder'], len(corrDays), (len(idxRemainingGood) if remainingRoisExists else 0), (idxRemainingGood if remainingRoisExists else None), corrDays)
        #intersectData.append([nDayA, allCorrDataPerSession[nDayA]['folder'], len(corrDays), (len(idxRemainingGood) if remainingRoisExists else 0), (idxRemainingGood if remainingRoisExists else None), corrDays])
        intersectData[nDayA] = {}
        intersectData[nDayA]['nDayA'] = nDayA # 0
        intersectData[nDayA]['folder'] = allCorrDataPerSession[nDayA]['folder']  # 1
        intersectData[nDayA]['lenCorrDays'] = len(corrDays)  # 2
        intersectData[nDayA]['lenIdxRemainingGood'] = (len(idxRemainingGood) if remainingRoisExists else 0) # 3
        intersectData[nDayA]['idxRemainingGood'] = (idxRemainingGood if remainingRoisExists else None) # 4
        intersectData[nDayA]['corrDays'] = corrDays #5

    # second loop to determine identity of the remaining ROIs in all recordings
    #idxDay = intersectData[idxRef][5][r][0]
    #idxRoi = intersectData[idxRef][5][r][3 + n][1]
    #dayID = intersectData[idxRef][5][r][1]
    #nPairB = 0
    for i in range(len(intersectData)):
        print(i, intersectData[i]['folder'], intersectData[i]['lenCorrDays'],intersectData[i]['lenIdxRemainingGood'])
        idxRemainingGood = intersectData[i]['idxRemainingGood']
        for nDay in range(intersectData[i]['lenCorrDays']):
            #print(allAlignData[intersectData[i][5][2]][2])
            matchingRois =  getMatchingPairs(allAlignData[intersectData[i]['corrDays'][nDay][2]]['cleanedIntersectionROIs'])
            #print('match :',matchingRois,idxRemainingGood)
            idxRemaining = [key for key,val in enumerate(matchingRois[:,0]) if val in idxRemainingGood]
            RemainingRois = matchingRois[idxRemaining]
            intersectData[i]['corrDays'][nDay].append(RemainingRois)
            #roiData.append([intersectData[i][0],intersectData[i][1],])
    #pdb.set_trace()

    # another loop to append 820 identities of remaining ROIs
    for i in range(len(intersectData)):
        #print(i, intersectData[i][1], intersectData[i][2], intersectData[i][3])
        idxRemainingGood = intersectData[i]['idxRemainingGood']
        for n in range(len(alignData910And820)):
            if (intersectData[i]['folder'] == alignData910And820[n]['folder820']) and (idxRemainingGood is not None):
                print(i,n,intersectData[i]['folder'],alignData910And820[n]['folder820'])
                matchingRois =  getMatchingPairs(alignData910And820[n]['cleanedIntersectionROIs'])
                idxRemaining = [key for key,val in enumerate(matchingRois[:,0]) if val in idxRemainingGood]
                RemainingRois = matchingRois[idxRemaining]
                #intersectData[i].append(RemainingRois)
                intersectData[i]['RemainingRois'] = RemainingRois

    return intersectData

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doCorrelationAnalysisLocomotionPeriod(allCorrDataPerSession,allStepData):
    #
    matplotlib.use('TkAgg')
    xPixToUm = 0.79
    yPixToUm = 0.8
    motorizationCorrelations = []
    varExplainedMotorization = []
    maxMotorization = [12, 53] # in sec

    for nDay in range(len(allCorrDataPerSession)):
        print(nDay,allCorrDataPerSession[nDay]['folder'])
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP, wheelSpeedDict, pawTracksDict, caTracesDict, slowestTrial) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession,allStepData)
        # ATTENTION : all of the arrays also contain a time array
        # dims of wheelSpeedDictInterP : [nSessions][2][valuesOverTimeSame]
        # dims of pawTracksDictInterP :  [nSessions][nPaw][2][valuesOverTimeSame]
        # dims of caTracesDictInterP :  [nSessions][nRois+1][valuesOverTimeSame]
        # dims of wheelSpeedDictInterP : [nSessions][2][valuesOverTime]
        # dims of pawTracksDict : [nSessions][nPaw][2][valuesOverTime]
        # dims of caTracesDict : [nSessions][nRois+1][valuesOverTime]
        # correlations between calcium traces ######################################################################
        stat = allCorrDataPerSession[nDay]['caImg']['stat']#[3][0][4]
        nTrials = len(caTracesDict)
        nRois = np.shape(caTracesDict[0])[0] - 1

        allCoords = []
        for i in range(nRois):
            allCoords.append([i, stat[i]['med'][0], stat[i]['med'][1]])  # first extract the coordinates of all ROIs
        allCoordsSorted = sorted(allCoords, key=lambda x: (x[1], x[2]))  # list according to increasing x and y coordinates
        allCoordsSorted = np.asarray(allCoordsSorted)
        combis = list(itertools.combinations(np.array(allCoordsSorted[:, 0], dtype=int), 2))  # use the sorted list to create the combinations
        corrCaTraces = np.zeros((len(combis), nTrials, 9))
        # for moving average windows
        tWindow = 1.  # removing changes on the order of 1 s and longer
        ttime = caTracesDict[0][0]
        dt = np.mean(np.diff(ttime))
        Nwindow = int(tWindow / dt + 0.5)

        for i in range(len(combis)):
            # get location information of both ROIs
            xy0 = stat[combis[i][0]]['med']
            xy1 = stat[combis[i][1]]['med']
            # calculate eucleadian distance and x, y distance between cells
            euclDist = np.sqrt(((xy0[1]-xy1[1])*xPixToUm)**2 + ((xy0[0]-xy1[0])*yPixToUm)**2)
            xyDist = ([(xy1[1]-xy0[1])*xPixToUm,(xy0[0]-xy1[0])*yPixToUm]) # note that the the y-value is inverted on purpose; cell order from the top
            #pdb.set_trace()
            for t in range(nTrials):
                mask = (caTracesDict[t][0] >= maxMotorization[0]) & (caTracesDict[t][0] < maxMotorization[1])
                corrTemp = scipy.stats.pearsonr(caTracesDict[t][combis[i][0]+1][mask],caTracesDict[t][combis[i][1]+1][mask])
                caTemp0 = np.convolve(caTracesDict[t][combis[i][0]+1][mask], np.ones((Nwindow,))/Nwindow, mode='same')
                caTemp1 = np.convolve(caTracesDict[t][combis[i][1]+1][mask], np.ones((Nwindow,))/Nwindow, mode='same')
                shortCorrTemp = scipy.stats.pearsonr(caTracesDict[t][combis[i][0]+1][mask]-caTemp0,caTracesDict[t][combis[i][1]+1][mask]-caTemp1)
                corrCaTraces[i,t] = np.array([combis[i][0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1],shortCorrTemp[0],shortCorrTemp[1]])

        # activity measures of ca traces before vs during walking ######################################################################
        tBaseline = 5
        tMotorization = [12,53]
        activityCaTraces = np.zeros((nRois,nTrials,4))
        for n in range(nRois):
            for t in range(nTrials):
                baselineMask = caTracesDict[t][0]<tBaseline
                activityMask = (caTracesDict[t][0]>=tMotorization[0]) & (caTracesDict[t][0]<tMotorization[1])
                (baseLMean,baseLSTD) = (np.mean(caTracesDict[t][n+1][baselineMask]),np.std(caTracesDict[t][n+1][baselineMask]))
                (actMean, actSTD) = (np.mean(caTracesDict[t][n + 1][activityMask]), np.std(caTracesDict[t][n + 1][activityMask]))
                activityCaTraces[n,t] = np.array([baseLMean,baseLSTD,actMean, actSTD])

        ###################################################################
        # correlation between calcium and paw as well as paw speed
        corrCaWheel = np.zeros((nRois,nTrials, 3))
        corrCaPawTraces = np.zeros((nRois,nTrials, 9))
        for i in range(nRois):
            for t in range(nTrials):
                caMask  = (caTracesDictInterP[t][0]>= maxMotorization[0]) & (caTracesDictInterP[t][0] < maxMotorization[1])
                wheelMask = (wheelSpeedDictInterP[t][0]>= maxMotorization[0]) & (wheelSpeedDictInterP[t][0] < maxMotorization[1])
                paw0Mask = (pawTracksDictInterP[t][0][0]>=maxMotorization[0]) & (pawTracksDictInterP[t][0][0] < maxMotorization[1])
                paw1Mask = (pawTracksDictInterP[t][1][0]>=maxMotorization[0]) & (pawTracksDictInterP[t][1][0] < maxMotorization[1])
                paw2Mask = (pawTracksDictInterP[t][2][0]>=maxMotorization[0]) & (pawTracksDictInterP[t][2][0] < maxMotorization[1])
                paw3Mask = (pawTracksDictInterP[t][3][0]>=maxMotorization[0]) & (pawTracksDictInterP[t][3][0] < maxMotorization[1])
                corrWheelTemp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1][caMask], wheelSpeedDictInterP[t][1][wheelMask])
                corrCaWheel[i][t] = np.array([i,corrWheelTemp[0],corrWheelTemp[1]])
                corrPaw0Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1][caMask], pawTracksDictInterP[t][0][1][paw0Mask])
                corrPaw1Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1][caMask], pawTracksDictInterP[t][1][1][paw1Mask])
                corrPaw2Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1][caMask], pawTracksDictInterP[t][2][1][paw2Mask])
                corrPaw3Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1][caMask], pawTracksDictInterP[t][3][1][paw3Mask])
                corrCaPawTraces[i][t] = np.array([i,corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])

        ###################################################################
        # correlation btw. PCA components and wheel, paw speeds
        # concatenate calcium, wheel speed and paw speed for PCA
        pawAll = {}
        pawMask = {}
        for t in range(nTrials):
            caMask = (caTracesDictInterP[t][0] >= maxMotorization[0]) & (caTracesDictInterP[t][0] < maxMotorization[1])
            wheelMask = (wheelSpeedDictInterP[t][0] >= maxMotorization[0]) & (wheelSpeedDictInterP[t][0] < maxMotorization[1])
            for i in range(4):
                pawMask[i] = (pawTracksDictInterP[t][i][0] >= maxMotorization[0]) & (pawTracksDictInterP[t][i][0] < maxMotorization[1])
            if t == 0:
                caAll = caTracesDictInterP[t][:,caMask]
                wheelAll = wheelSpeedDictInterP[t][:,wheelMask]
                for i in range(4):
                    pawAll[i] = pawTracksDictInterP[t][i][:,pawMask[i]]
            else:
                caAll = np.column_stack((caAll,caTracesDictInterP[t][:,caMask]))
                wheelAll = np.column_stack((wheelAll,wheelSpeedDictInterP[t][:,wheelMask]))
                for i in range(4):
                    pawAll[i] = np.column_stack((pawAll[i],pawTracksDictInterP[t][i][:,pawMask[i]]))
        #pdb.set_trace()
        pcaComponents = 5
        print('doing PCA ...')
        X = np.transpose(caAll[1:])
        pca = PCA(n_components=pcaComponents)
        pca.fit(X)
        X_pca = pca.transform(X)
        #print(pca.components_)
        pcaCorrs = np.zeros((pcaComponents,11))
        varExplainedMotorization.append(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_)
        for i in range(pcaComponents):
            corrWheelTemp = scipy.stats.pearsonr(X_pca[:,i], wheelAll[1])
            corrPaw0Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[0][1])
            corrPaw1Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[1][1])
            corrPaw2Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[2][1])
            corrPaw3Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[3][1])
            pcaCorrs[i] = ([i,corrWheelTemp[0],corrWheelTemp[1],corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])

        ###################################################################
        motorizationCorrelations.append([nTrials,corrCaTraces,corrCaWheel,corrCaPawTraces,pcaCorrs,activityCaTraces])

    return (motorizationCorrelations,varExplainedMotorization)
#################################################################################
# chose day of reference from FOV alignment data
#################################################################################
def getRefIdxPerMouse(mouse):
    refIdxMouseDict = {'210214_m12':7,
                       '210214_m13':4,
                       '210214_m14':5,
                       '210214_m15':6,
                       '210214_m17':6,
                       '210214_m18':9,
                       '210214_m19':5,
                       '210214_m20':3,
                       '210122_f83':2,
                       '210122_f84':6,
                       '210120_m85':1,
                       '210120_m86':0,
                       }

    idxRef = refIdxMouseDict[mouse]
    return idxRef

#################################################################################
# chose day of reference from FOV alignment data
#################################################################################
#(intersectData,nDays, nRois, refDay, dayList) = dataAnalysis.choseAlignmentReferenceDay(self.mouse,intersectData)
def  choseAlignmentReferenceDay(mouse,intersectData):

    idxRef = getRefIdxPerMouse(mouse)

    for i in range(len(intersectData)):
        #print(i, intersectData[i][0], intersectData[i][1], intersectData[i][2], intersectData[i][3], int(intersectData[i][2])*int(intersectData[i][3]))
        print(i,intersectData[i]['nDayA'],intersectData[i]['folder'],intersectData[i]['lenCorrDays'],intersectData[i]['lenIdxRemainingGood'],int(intersectData[i]['lenCorrDays'])*int(intersectData[i]['lenIdxRemainingGood']))
    #x = input('Enter the reference day of choice (e.g. 0, 1, 3) :')
    #idxRef = int(x)
    print('reference idx for mouse %s is %s' % (mouse,idxRef))
    # find position to insert entry for reference day itself
    pos = 0
    #pdb.set_trace()
    alreadyInserted = False
    for i in range(len(intersectData[idxRef]['corrDays'])):
        if intersectData[idxRef]['corrDays'][i][0] < idxRef:
            pos+=1
        elif intersectData[idxRef]['corrDays'][i][0] == idxRef:
            alreadyInserted = True
        else:
            break
    print('insert postions : %s already inserted %s ' % (pos,alreadyInserted))
    #pdb.set_trace()
    # add entry for reference day itself
    if not alreadyInserted:
        llInsert = [idxRef,intersectData[idxRef]['folder'],0]
        #llInsert.append([np.array([a,a]) for a in intersectData[idxRef][4]])
        llInsert.append(np.column_stack((intersectData[idxRef]['idxRemainingGood'],intersectData[idxRef]['idxRemainingGood'])))
        intersectData[idxRef]['corrDays'].insert(pos,llInsert)
    #pdb.set_trace()
    # remove specific recordings
    if mouse == '210214_m15': # remove 2021.05.12_003 for 210214_m15
        delPos = -1
        for i in range(len(intersectData[idxRef]['corrDays'])):
            if intersectData[idxRef]['corrDays'][i][1] == '2021.05.12_003':
                delPos = i
                break
        if delPos!=-1: # only delete entry if date still exists
            intersectData[idxRef]['corrDays'].pop(delPos)

    nDays = len(intersectData[idxRef]['corrDays']) # reference day itself is added
    nRois = intersectData[idxRef]['lenIdxRemainingGood']
    refDay = intersectData[idxRef]['folder']

    dayList = []
    for i in range(len(intersectData[idxRef]['corrDays'])):
        dayList.extend([intersectData[idxRef]['corrDays'][i][0]])
    print('dayList',dayList)
    return (intersectData,nDays, nRois, idxRef, refDay, dayList)

#################################################################################
# calculate null distribution of activity increases using shuffling
#################################################################################
#calculateNullDistribution(allActivity,caTracesDict[slowestTrial][0],shuffleInterval,tBaseline,tMotorization)
def calculateNullDistribution(allActivity,timeVector,shuffleInterval,tBaseline,tMotorization,nShuffle,trials):

    shuffleLength = sum(timeVector<shuffleInterval)
    baseLineN = round(tBaseline/shuffleInterval)
    motorizationN = round((tMotorization[1]-tMotorization[0])/shuffleInterval)
    allActivityN = int(np.floor(len(allActivity)/shuffleLength))

    diffs = []
    for n in range(nShuffle):
        intervalIdx = random.sample(range(0, allActivityN), (baseLineN + motorizationN * trials))
        #pdb.set_trace()
        baseLine = np.concatenate(([allActivity[(i*shuffleLength):(i*shuffleLength+shuffleLength)] for i in intervalIdx[:5]]))
        motorizationActivity = np.concatenate(([allActivity[(i*shuffleLength):(i*shuffleLength+shuffleLength)] for i in intervalIdx[5:]]))
        diffs.append(np.mean(motorizationActivity)-np.mean(baseLine))
        #pdb.set_trace()
    diffs = np.asarray(diffs)
    return diffs


#################################################################################
# chose day of reference from FOV alignment data
#################################################################################
#(intersectData,nDays, nRois, refDay, dayList) = dataAnalysis.choseAlignmentReferenceDay(self.mouse,intersectData)
def getDayListFor820Recs(mouse,intersectData,allCorrDataPerSession):

    idxRef = getRefIdxPerMouse(mouse)
    dayList910 = []
    for i in range(len(intersectData[idxRef]['corrDays'])):
        dayList910.extend([intersectData[idxRef]['corrDays'][i][1]])
    print('%s 910 dayList' % mouse,dayList910)

    dayList820 = []
    for nDay in range(len(allCorrDataPerSession)):
        print(nDay,allCorrDataPerSession[nDay]['folder'])
        if allCorrDataPerSession[nDay]['folder'] in dayList910:
            dayList820.append(nDay)
    print('%s 820 dayList' % mouse,dayList820)

    return dayList820

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doCorrelationAnalysis(allCorrDataPerSession,allStepData):
    #
    matplotlib.use('TkAgg')
    xPixToUm = 0.79
    yPixToUm = 0.8
    #itertools.combinations(arr,2)
    sessionCorrelations = []
    varExplained = []

    for nDay in range(len(allCorrDataPerSession)):
        print(nDay,allCorrDataPerSession[nDay]['folder'])
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict,slowestTrial) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession, allStepData)
        # ATTENTION : all of the arrays also contain a time array
        # dims of wheelSpeedDictInterP : [nSessions][2][valuesOverTimeSame]
        # dims of pawTracksDictInterP :  [nSessions][nPaw][2][valuesOverTimeSame]
        # dims of caTracesDictInterP :  [nSessions][nRois+1][valuesOverTimeSame]
        # dims of wheelSpeedDict : [nSessions][2][valuesOverTime]
        # dims of pawTracksDict : [nSessions][nPaw][2][valuesOverTime]
        # dims of caTracesDict : [nSessions][nRois+1][valuesOverTime]
        # correlations between calcium traces ######################################################################
        stat = allCorrDataPerSession[nDay]['caImg']['stat']  #allCorrDataPerSession[nDay][3][0][4]
        nTrials = len(caTracesDict)
        nRois = np.shape(caTracesDict[0])[0]-1

        allCoords = []
        for i in range(nRois):
            allCoords.append([i,stat[i]['med'][0],stat[i]['med'][1]])  # first extract the coordinates of all ROIs
        allCoordsSorted = sorted(allCoords,key=lambda x: (x[1],x[2]))  # list according to increasing x and y coordinates
        allCoordsSorted = np.asarray(allCoordsSorted)
        combis = list(itertools.combinations(np.array(allCoordsSorted[:,0],dtype=int), 2)) # use the sorted list to create the combinations
        corrCaTraces = np.zeros((len(combis),nTrials,9))
        # for moving average windows
        tWindow = 1. # removing changes on the order of 1 s and longer
        ttime = caTracesDict[0][0]
        dt = np.mean(np.diff(ttime))
        Nwindow = int(tWindow / dt + 0.5)

        for i in range(len(combis)):
            # get location information of both ROIs
            #print(combis[i][0],combis[i][1])
            xy0 = stat[combis[i][0]]['med']
            xy1 = stat[combis[i][1]]['med']
            # calculate eucleadian distance and x, y distance between cells
            euclDist = np.sqrt(((xy0[1]-xy1[1])*xPixToUm)**2 + ((xy0[0]-xy1[0])*yPixToUm)**2)
            xyDist = ([(xy1[1]-xy0[1])*xPixToUm,(xy0[0]-xy1[0])*yPixToUm]) # note that the the y-value is inverted on purpose; cell order from the top
            #pdb.set_trace()
            for t in range(nTrials):
                corrTemp = scipy.stats.pearsonr(caTracesDict[t][combis[i][0]+1],caTracesDict[t][combis[i][1]+1])
                caTemp0 = np.convolve(caTracesDict[t][combis[i][0] + 1], np.ones((Nwindow,)) / Nwindow, mode='same')
                caTemp1 = np.convolve(caTracesDict[t][combis[i][1] + 1], np.ones((Nwindow,)) / Nwindow, mode='same')
                shortCorrTemp = scipy.stats.pearsonr(caTracesDict[t][combis[i][0] + 1] - caTemp0, caTracesDict[t][combis[i][1] + 1] - caTemp1)
                corrCaTraces[i,t] = np.array([combis[i][0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1],shortCorrTemp[0],shortCorrTemp[1]])

        # activity measures of ca traces before vs during walking ######################################################################
        tBaseline = 5 # in sec
        tMotorization = [12,53] # in sec
        shuffleInterval = 1 # in sec
        nShuffle = 1000
        activityCaTraces = np.zeros((nRois,6))
        for n in range(nRois):
            baselineMask = caTracesDict[slowestTrial][0] < tBaseline
            (baseLMean, baseLSTD) = (np.mean(caTracesDict[slowestTrial][n + 1][baselineMask]), np.std(caTracesDict[slowestTrial][n + 1][baselineMask]))
            motorizationActivity = []
            allActivity = []
            allTime = []
            for t in range(nTrials): # concatenate all motorization periods
                #baselineMask = caTracesDict[t][0]<tBaseline
                activityMask = (caTracesDict[t][0]>=tMotorization[0]) & (caTracesDict[t][0]<tMotorization[1])
                motorizationActivity.extend(caTracesDict[t][n+1][activityMask])
                allActivity.extend(caTracesDict[t][n+1])
            change_dFF_shuff = calculateNullDistribution(allActivity,caTracesDict[slowestTrial][0],shuffleInterval,tBaseline,tMotorization,nShuffle,nTrials)
            (actMean, actSTD) = (np.mean(motorizationActivity), np.std(motorizationActivity))
            change_dFF = (actMean-baseLMean)
            # p(k) = sum(change_dFF_shuff(:,k) > abs(change_dFF(k)) |  change_dFF_shuff(:,k) < -abs(change_dFF(k)))/num_reps;
            #pdb.set_trace()
            pActivityDifference = np.sum((change_dFF_shuff > np.abs(change_dFF)) |  (change_dFF_shuff < -np.abs(change_dFF)))/nShuffle
            #twoSidedPValues = stats.ttest_1samp(nullDifferences,(astMean-baseLmean)[1]
            activityCaTraces[n] = np.array([baseLMean,baseLSTD,actMean, actSTD,change_dFF,pActivityDifference])
        #pdb.set_trace()
        ###################################################################
        # correlation between calcium and paw as well as paw speed
        corrCaWheel = np.zeros((nRois,nTrials, 3))
        corrCaPawTraces = np.zeros((nRois,nTrials, 9))
        for i in range(nRois):
            for t in range(nTrials):
                corrWheelTemp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1], wheelSpeedDictInterP[t][1])
                corrCaWheel[i][t] = np.array([i,corrWheelTemp[0],corrWheelTemp[1]])
                corrPaw0Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1], pawTracksDictInterP[t][0][1])
                corrPaw1Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1], pawTracksDictInterP[t][1][1])
                corrPaw2Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1], pawTracksDictInterP[t][2][1])
                corrPaw3Temp = scipy.stats.pearsonr(caTracesDictInterP[t][i+1], pawTracksDictInterP[t][3][1])
                corrCaPawTraces[i][t] = np.array([i,corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])

        ###################################################################
        # correlation btw. PCA components and wheel, paw speeds
        # concatenate calcium, wheel speed and paw speed for PCA
        pawAll = {}
        for t in range(nTrials):
            if t == 0:
                caAll = caTracesDictInterP[t]
                wheelAll = wheelSpeedDictInterP[t]
                for i in range(4):
                    pawAll[i] = pawTracksDictInterP[t][i]
            else:
                caAll = np.column_stack((caAll,caTracesDictInterP[t]))
                wheelAll = np.column_stack((wheelAll,wheelSpeedDictInterP[t]))
                for i in range(4):
                    pawAll[i] = np.column_stack((pawAll[i],pawTracksDictInterP[t][i]))
        #pdb.set_trace()
        pcaComponents = 5
        print('doing PCA ...')
        X = np.transpose(caAll[1:])
        pca = PCA(n_components=pcaComponents)
        pca.fit(X)
        X_pca = pca.transform(X)
        #print(pca.components_)
        pcaCorrs = np.zeros((pcaComponents,11))
        varExplained.append(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_)
        for i in range(pcaComponents):
            corrWheelTemp = scipy.stats.pearsonr(X_pca[:,i], wheelAll[1])
            corrPaw0Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[0][1])
            corrPaw1Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[1][1])
            corrPaw2Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[2][1])
            corrPaw3Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[3][1])
            pcaCorrs[i] = ([i,corrWheelTemp[0],corrWheelTemp[1],corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])

        ###################################################################
        sessionCorrelations.append([nTrials,corrCaTraces,corrCaWheel,corrCaPawTraces,pcaCorrs,activityCaTraces])

    return (sessionCorrelations,varExplained)

##########################################################
# Get absolute speed of paws
def getPawSpeed(recordingsM, mouse_tracks, showFig=True):
    mouse_speed = []
    for d in range(len(recordingsM)):
        day_speed = []
        for s in range(len(recordingsM[d])):
            sess_speed = []
            for p in range(len(recordingsM[d][s])):
                paw_speed = np.diff(recordingsM[d][s][p][:, 0]) / np.diff(recordingsM[d][s][p][:, 2])
                sess_speed.append(paw_speed)
            day_speed.append(sess_speed)
        mouse_speed.append(day_speed)
    mouse_time = []
    for d in range(len(recordingsM)):
        day_time = []
        for s in range(len(recordingsM[d])):
            sess_time = []
            for p in range(len(recordingsM[d][s])):
                mean_time = np.mean([mouse_tracks[d][s][2], mouse_tracks[d][s][3]], axis=0)
                frames_time = mean_time[recordingsM[d][s][p][:, 2].astype(int)]
                sess_time.append(np.delete(frames_time, 0))
            day_time.append(sess_time)
        mouse_time.append(day_time)

    return mouse_speed, mouse_time
##########################################################
# Get average stride length and min/max values
def calculateDistanceBtwLineAndPoint(x1,y1,x2,y2,x0,y0):
    nenner   = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    zaehler  = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1
    dist = zaehler/nenner
    return dist

##########################################################
# Get average stride length and min/max values
def verifyJointID(expectedID, actualID):
    assert len(expectedID) == len(actualID), print('expected and actual paw labels do not have the same length (expected, actual):', expectedID, actualID)
    for i in range(len(expectedID)):
        if not (expectedID[i] in actualID[i]):
            print('Expected paw ID does not match the acutal paw tracking label (expected : actual) :', expectedID[i], ' : ', actualID[i])
            pdb.set_trace()


##########################################################
def findSwingPhasesFlat(tracks, pawTracks,analysisConfig,pawTracksAll,showFigFit=False,showFigPaw=False,verbose=False, obstacle=False,jointID=None) :

    matplotlib.use('TkAgg')
    speedDiffThresh = 10  # cm/s Speed threshold, determine with variance
    pawList = ['front_left', 'front_right', 'hind_left', 'hind_right']
    #pdb.set_trace()
    # This checks if 'b' is bytes before trying to decode it.
    # If it's already a string, it leaves it alone.
    jointID = [b.decode("utf-8") if isinstance(b, bytes) else b for b in jointID]
    # jointID = [b.decode("utf-8") for b in jointID] # since each element in jointID is bytes not str, e.g. b'front_left'
    verifyJointID(pawList,jointID)
    minimalLengthOfSwing = 3 # number of frames @ 200 Hz
    minimalLengthOfStance = 4
    zIdx = [2,3,4,5]

    trailingStart = 1
    trailingEnd = 1
    bounds = [1400, 5320]
    wheelCircumsphere = 80.4 # in cm

    errfunc = lambda p, x1, y1, x2, y2, x3, y3, x4, y4: np.sum(1./np.abs(x1-p*y1)) + np.sum(1./np.abs(x2-p*y2))+ np.sum(1./np.abs(x3-p*y3)) + np.sum(1./np.abs(x4-p*y4))
    # guess some fit parameters
    p0 = 0.025
    # calculate wheel speed at the frame times : requires interpolation of the wheel speed
    interpAngle = interp1d(tracks[5][:,0],tracks[5][:,1])

    interp = interp1d(tracks[2], tracks[1])

    forFit = []
    for i in range(4):

        mask = ((pawTracks[3][i][:,0])>=min(tracks[2])) & ((pawTracks[3][i][:,0])<=max(tracks[2]))
        maskAngle = ((pawTracks[5][i][:,0])>=min(tracks[5][:,0])) & ((pawTracks[5][i][:,0])<=max(tracks[5][:,0]))
        if not obstacle:
            newWheelSpeedAtPawTimes = -interp(pawTracks[3][i][:,0][mask])
            newWheelAngleAtPawTimes = -interpAngle(pawTracks[5][i][:, 0][maskAngle])
        else:
            newWheelSpeedAtPawTimes = interp(pawTracks[3][i][:,0][mask])
            newWheelAngleAtPawTimes = interpAngle(pawTracks[5][i][:, 0][maskAngle])



        newX  = (pawTracks[5][i][:,1][maskAngle])*0.025 + (newWheelAngleAtPawTimes*80./360.) #- (pawTracks[5][i][:,1][maskAngle][0])*0.025                                  #   pawPos[i][:,0][maskAngle]
        #pdb.set_trace()
        forFit.append([newWheelSpeedAtPawTimes, pawTracks[3][i][:,2][mask],pawTracks[3][i][:,0][mask],mask,np.array(pawTracks[3][i][:,4][mask],dtype=int),np.column_stack((pawTracks[5][i][:,0][maskAngle],newX))])
        # newWheelSpeedAtPawTimes, pawSpeedX, pawSpeedTime,
    (p1, success) = scipy.optimize.leastsq(errfunc, p0 ,args=(forFit[0][0],forFit[0][1],forFit[1][0],forFit[1][1],forFit[2][0],forFit[2][1],forFit[3][0],forFit[3][1]))
    print('fit parameter : ', p1)
    if (np.abs(p1) - 0.025) > 0.005:
        print('PROBLEM : fit of paw speed to wheel speed didn\'t deliver expected results (~0.025) instead : ',p1)
        # pdb.set_trace()
    if showFigFit :
        plt.plot(forFit[0][1]*p1, 'red')
        plt.plot(forFit[0][0])
        plt.show()
    pawPos = pawTracks[5]
    swingPhases = []
    for i in range(4):
        #pdb.set_trace()
        print(pawTracks[2][i])
        speedDiff = (forFit[i][0] - forFit[i][1]*p1)
        thresholded = np.abs(speedDiff) > speedDiffThresh
        startStop = np.diff(forFit[i][4][thresholded]) > 1 # use indices taking into account missed frames
        mmmStart = np.hstack((([True]), startStop))
        mmmStop = np.hstack((startStop, ([True])))
        startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
        stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
        swingIndices = np.column_stack((startIdx, stopIdx))
        nIdx = 0
        cleanedSwingIndicies = []
        #pdb.set_trace()
        while True:
            # if verbose: print(nIdx,forFit[i][4][swingIndices[nIdx,0]],forFit[i][2][swingIndices[nIdx,0]],forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]],end=' ')
            if forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]]>0:
                #if swingIndices[nIdx,1]-swingIndices[nIdx,0]>2:
                #cleanedSwingIndicies.append([forFit[i][4][swingIndices[nIdx,0]]-trailingStart,forFit[i][4][swingIndices[nIdx,1]]+trailingEnd])
                sttart = (swingIndices[nIdx,0]-trailingStart) if (swingIndices[nIdx,0]-trailingStart)>0 else 0
                ennd   = (swingIndices[nIdx,1]+trailingEnd) if (swingIndices[nIdx,1]+trailingEnd)<len(forFit[i][4]) else (len(forFit[i][4])-1)
                cleanedSwingIndicies.append([sttart,ennd])
                if (cleanedSwingIndicies[-1][1]-cleanedSwingIndicies[-1][0])< minimalLengthOfSwing :# remove short swing phases
                    del cleanedSwingIndicies[-1]
                if len(cleanedSwingIndicies)>1:
                    if cleanedSwingIndicies[-2][1] > cleanedSwingIndicies[-1][0]: # remove overlapping swing phases
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]
            nIdx += 1
            if nIdx==(len(swingIndices)-1):
                break

        maskAngle = ((pawPos[i][:, 0]) >= min(tracks[5][:, 0])) & ((pawPos[i][:, 0]) <= max(tracks[5][:, 0]))
        newWheelAngleAtPawTimes = interpAngle(pawPos[i][:, 0][maskAngle])
        newLinearXposition = (pawPos[i][:, 1][maskAngle]) * p1 + (newWheelAngleAtPawTimes * wheelCircumsphere / 360.)
        newXtime = pawPos[i][:,0][maskAngle]
        speedTimes = forFit[i][2]

        #pawPos = pawTracks[5]
        stepCharacter = []

        for n in range(len(cleanedSwingIndicies)):
            idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][0]]))
            idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][1]]))
            stepDuration = (pawPos[i][:, 0][idxStanceStart] - pawPos[i][:, 0][idxSwingStart])
            stepLength = (pawPos[i][:, 1][idxStanceStart] - pawPos[i][:, 1][idxSwingStart])

            #
            idxSwingStartNew = np.argmin(np.abs(newXtime - speedTimes[cleanedSwingIndicies[n][0]]))
            idxStanceStartNew = np.argmin(np.abs(newXtime - speedTimes[cleanedSwingIndicies[n][1]]))
            stepLengthLinear = (newLinearXposition[idxStanceStartNew] - newLinearXposition[idxSwingStartNew])
            #
            speedDuringStep = speedDiff[cleanedSwingIndicies[n][0]:cleanedSwingIndicies[n][1]]
            thresholded = np.abs(speedDuringStep) < 10. # use different, larger threshold
            stepCharacter.append([n, cleanedSwingIndicies[n][0], cleanedSwingIndicies[n][1], stepDuration, stepLength, stepLengthLinear])
        swingPhases.append([i, cleanedSwingIndicies, stepCharacter])

        if showFigPaw:

            fig = plt.figure(figsize=(30, 14))
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 2)
            # ax1.fill_between(forFit[i][2],stanceDistances[i][0],stanceDistances[i][1],color='0.8')
            ax1.plot(forFit[i][2], forFit[i][0])
            ax1.plot(forFit[i][2], forFit[i][1] * p1)

            # ax2.fill_between(forFit[i][2], stanceDistances[i][0], stanceDistances[i][1], color='0.8')
            ax2.plot(forFit[i][2], forFit[i][0])
            ax2.plot(forFit[i][2], forFit[i][1] * p1)
            ax3 = fig.add_subplot(2, 3, 4)
            ax3.plot(pawPos[i][:, 0], pawPos[i][:, 1], c='C1')

            if obstacle:
                ax4 = fig.add_subplot(2, 3, 5)
                ax4.plot(pawPos[zIdx[i]][:, 0], 600 - pawPos[zIdx[i]][:, 2], c='C1')
                ax5 = fig.add_subplot(2, 3, 6)
                ax5.plot(pawPos[zIdx[i]][:, 0], 600 - pawPos[zIdx[i]][:, 2], c='C1')


            for n in range(len(cleanedSwingIndicies)):
                idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][0]]))
                idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][1]])) + 1

                startI = int(cleanedSwingIndicies[n][0])
                endI = int(cleanedSwingIndicies[n][1]) + 1
                ax1.fill_between(forFit[i][2][range(startI, endI)], 0, 1, color='0.5', alpha=0.5,
                                 transform=ax1.get_xaxis_transform())
                ax2.fill_between(forFit[i][2][range(startI, endI)], 0, 1, color='0.5', alpha=0.5,
                                 transform=ax2.get_xaxis_transform())
                # ax3.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                linearPawPos = np.array(forFit[i][5])

                swingLength = (linearPawPos[endI, 1] - linearPawPos[startI, 1])

                ax1.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1,
                         c=('C3'))
                ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1,
                         c=('C3'))
                ax3.plot(pawPos[i][:, 0][range(idxSwingStart, idxStanceStart)],
                         pawPos[i][:, 1][idxSwingStart:idxStanceStart], c=('C3'))
                # ax3.text(pawPos[i][:, 0][range(idxSwingStart, idxStanceStart)][-1],
                #          pawPos[i][:, 1][idxSwingStart:idxStanceStart][-1], fontsize=6)

                ax1.set_xlim(40, 45)
                ax1.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                ax1.set_xlabel('time (s)')
                ax1.set_ylabel('speed (cm)')
                # ax3.set_xlim(40,45)
                ax3.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                ax3.set_xlabel('time (s)')
                ax3.set_ylabel('x-position (pixel)')
            plt.show()
    print('swings found',len(swingPhases[0][1]))
    return (swingPhases, forFit)
# (tracks,pawTracks,stanceSwingsParams)
######################################################################################
def findStancePhases(tracks, pawTracks,rungMotion,analysisConfig,pawTracksAll,showFigFit=False,showFigPaw=False,verbose=False, obstacle=False, redefineStanceDistances=False) :
    matplotlib.use('TkAgg')
    speedDiffThresh = 10  # cm/s Speed threshold, determine with variance
    # speedDiffThresh=20
    minimalLengthOfSwing = 3 # number of frames @ 200 Hz
    minimalLengthOfStance = 4
    zIdx = [2,3,4,5]
    #thStance = 10
    #thSwing = 2
    trailingStart = 1
    trailingEnd = 1
    bounds = [1400, 5320]
    wheelCircumsphere = 80.4 # in cm
    # if not obstacle:
    #     pawIdx=[0,1,2,3]
    # else:
    #     #indicies of bottom paw tracks
    #     pawIdx=[8,9,10,11]
    # error function : difference betweeen paw and wheel speed ; the inverse of the absolute difference is used to emphasize small values which would be the stance phases
    errfunc = lambda p, x1, y1, x2, y2, x3, y3, x4, y4: np.sum(1./np.abs(x1-p*y1)) + np.sum(1./np.abs(x2-p*y2))+ np.sum(1./np.abs(x3-p*y3)) + np.sum(1./np.abs(x4-p*y4))
    # guess some fit parameters
    p0 = 0.025
    # calculate wheel speed at the frame times : requires interpolation of the wheel speed

    interpAngle = interp1d(tracks[5][:,0],tracks[5][:,1])
    if not obstacle:
        interp = interp1d(tracks[2], tracks[1])
    else:
        interp = interp1d(-tracks[2], -tracks[1])
    forFit = []
    for i in range(4):

        mask = ((pawTracks[3][i][:,0])>=min(tracks[2])) & ((pawTracks[3][i][:,0])<=max(tracks[2]))
        maskAngle = ((pawTracks[5][i][:,0])>=min(tracks[5][:,0])) & ((pawTracks[5][i][:,0])<=max(tracks[5][:,0]))
        if not obstacle:
            newWheelSpeedAtPawTimes = -interp(pawTracks[3][i][:,0][mask])
            newWheelAngleAtPawTimes = -interpAngle(pawTracks[5][i][:, 0][maskAngle])
        else:
            newWheelSpeedAtPawTimes = interp(pawTracks[3][i][:,0][mask])
            newWheelAngleAtPawTimes = interpAngle(pawTracks[5][i][:, 0][maskAngle])



        newX  = (pawTracks[5][i][:,1][maskAngle])*0.025 + (newWheelAngleAtPawTimes*80./360.) #- (pawTracks[5][i][:,1][maskAngle][0])*0.025                                  #   pawPos[i][:,0][maskAngle]
        #pdb.set_trace()
        forFit.append([newWheelSpeedAtPawTimes, pawTracks[3][i][:,2][mask],pawTracks[3][i][:,0][mask],mask,np.array(pawTracks[3][i][:,4][mask],dtype=int),np.column_stack((pawTracks[5][i][:,0][maskAngle],newX))])
        # newWheelSpeedAtPawTimes, pawSpeedX, pawSpeedTime,
    (p1, success) = scipy.optimize.leastsq(errfunc, p0 ,args=(forFit[0][0],forFit[0][1],forFit[1][0],forFit[1][1],forFit[2][0],forFit[2][1],forFit[3][0],forFit[3][1]))
    print('fit parameter : ', p1)
    if (np.abs(p1) - 0.025) > 0.005:
        print('PROBLEM : fit of paw speed to wheel speed didn\'t deliver expected results (~0.025) instead : ',p1)
        # pdb.set_trace()
    if showFigFit :
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,1]*p1)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,2]*p0)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,2]*p1)
        #plt.plot(pawTracks[0][3][1][:,0],pawTracks[0][3][1][:,2]*p1)
        #plt.plot(pawTracks[0][3][2][:,0],pawTracks[0][3][2][:,2]*p1)
        #plt.plot(pawTracks[0][3][3][:,0],pawTracks[0][3][3][:,2]*p1)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,3]*p1)
        #plt.plot(pawTracks[0][3][0][:,0][mask],newSpeedAtPawTimes)
        #plt.plot(tracks[0][2], -tracks[0][1])
        plt.plot(forFit[0][1]*p1, 'red')
        plt.plot(forFit[0][0])
        plt.show()
    #pdb.set_trace()
    ##############################################################################################################
    # calculate paw-rung distance
    pawRungDistances = []
    obsRungs = []
    for i in range(4):
        rungInd = []

        for n in forFit[i][4]:

            #pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameters])
            # (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5))

            #if
            if obstacle:
                obsRungs.extend(rungMotion[3][n][2][rungMotion[3][n][6]>0])

            rungLocs = rungMotion[3][n][3]
            #xPaw = pawTracks[6][0] + pawTracks[0][n,(i*3+1)]
            #yPaw = pawTracks[6][2] + pawTracks[0][n,(i*3+2)]
            #size of array dont match!!!!!!!!!!!!!!!!
            try:
                xPaw =  pawTracks[5][i][n,1]
                yPaw = pawTracks[5][i][n,2]
            except:
                pass

            distances = calculateDistanceBtwLineAndPoint(rungLocs[:,0],rungLocs[:,1],rungLocs[:,2],rungLocs[:,3],xPaw,yPaw)
            sortedArguments  = np.argsort(np.abs(distances))

            #closestRungIdx = np.argmin(np.abs(distances))
            closestRungNumber = rungMotion[3][n][2][sortedArguments[0]]
            closestDist = distances[sortedArguments[0]]

            secondClosestRungNumber = rungMotion[3][n][2][sortedArguments[1]]
            secondClosestDist = distances[sortedArguments[1]]
            #pdb.set_trace()
            if obstacle:
                closestRungObsIdentity=int(rungMotion[3][n][6][sortedArguments[0]])
                # if closestRungObsIdentity >0:
                #     # print(closestRungObsIdentity)
                secondClosestRungObsIdentity=int(rungMotion[3][n][6][sortedArguments[1]])
                rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw,closestRungObsIdentity,secondClosestRungObsIdentity])
            else:
                rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw])
            #pdb.set_trace()

        rungInd = np.asarray(rungInd)
        pawRungDistances.append([i,rungInd])
        #pdb.set_trace()
    obsRungs = np.unique(obsRungs)
    ##############################################################################################################
    # determine regions during which the speed is different for more than xLength values #########################
    #stanceDistances = [[10, 40],[10,40],[-4,40],[-4,40]]
    #pdb.set_trace()
    if (analysisConfig['swingStanceExtraction']['stanceDistances'] is None) or (redefineStanceDistances):
        stanceDistances =[[-10, 15], [-10, 15], [-30, 20], [-30, 15]]
        analysisConfig['swingStanceExtraction']['stanceDistances'] = stanceDistances
    else:
        stanceDistances = analysisConfig['swingStanceExtraction']['stanceDistances']
    print('stanceDistances used : ', stanceDistances)
    swingPhases = []

    for i in range(4):
        #pdb.set_trace()
        print(pawTracks[2][i],pawTracks[7][i])
        # thresholded = speedDiff > speedDiffThresh
        # startStop = np.diff(np.arange(len(speedDiff))[thresholded]) > 1
        # mmmStart = np.hstack((([True]), startStop))  # np.logical_or(np.hstack((([True]),startStop)),np.hstack((startStop,([True]))))
        # mmmStop = np.hstack((startStop, ([True])))
        # startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
        # stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
        # minLengthThres = (stopIdx - startIdx) > minLength
        # startStep = startIdx[minLengthThres] - trailingStart
        # endStep = stopIdx[minLengthThres] + trailingEnd
        # return np.column_stack((startStep, endStep))
        ##
        speedDiff = (forFit[i][0] - forFit[i][1]*p1)
        thresholded = np.abs(speedDiff) > speedDiffThresh
        startStop = np.diff(forFit[i][4][thresholded]) > 1 # use indices taking into account missed frames
        mmmStart = np.hstack((([True]), startStop))
        mmmStop = np.hstack((startStop, ([True])))
        startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
        stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
        swingIndices = np.column_stack((startIdx, stopIdx))
        nIdx = 0
        cleanedSwingIndicies = []
        #pdb.set_trace()
        while True:
            if verbose: print(nIdx,forFit[i][4][swingIndices[nIdx,0]],forFit[i][2][swingIndices[nIdx,0]],forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]],end=' ')
            if forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]]>0:
                #if swingIndices[nIdx,1]-swingIndices[nIdx,0]>2:
                #cleanedSwingIndicies.append([forFit[i][4][swingIndices[nIdx,0]]-trailingStart,forFit[i][4][swingIndices[nIdx,1]]+trailingEnd])
                sttart = (swingIndices[nIdx,0]-trailingStart) if (swingIndices[nIdx,0]-trailingStart)>0 else 0
                ennd   = (swingIndices[nIdx,1]+trailingEnd) if (swingIndices[nIdx,1]+trailingEnd)<len(forFit[i][4]) else (len(forFit[i][4])-1)
                cleanedSwingIndicies.append([sttart,ennd])
                if (cleanedSwingIndicies[-1][1]-cleanedSwingIndicies[-1][0])< minimalLengthOfSwing :# remove short swing phases
                    if verbose : print('short swing phase')
                    del cleanedSwingIndicies[-1]
                if len(cleanedSwingIndicies)>1:
                    if cleanedSwingIndicies[-2][1] > cleanedSwingIndicies[-1][0]: # remove overlapping swing phases
                        if verbose : print('overlapping')
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]
                #startIdx = forFit[i][4][cleanedSwingIndicies[-1][0]]
                #endIdx   = forFit[i][4][cleanedSwingIndicies[-1][1]]
                #meanSpeedDiff = np.mean(p1*forFit[i][1][startIdx:endIdx])
                #if meanSpeedDiff < speedDiffThresh : # remove osciallatory phases
                #    del cleanedSwingIndicies[-1]
                if len(cleanedSwingIndicies) > 1:
                    if (cleanedSwingIndicies[-1][0]-cleanedSwingIndicies[-2][1])<minimalLengthOfStance: # remove very short stance phases
                        if verbose : print('short stance phase')
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]

                    #pdb.set_trace()
                    #print(cleanedSwingIndicies)
                    #print(cleanedSwingIndicies[-2][1],cleanedSwingIndicies[-1][0],len(forFit[i][4]))
                if len(cleanedSwingIndicies) > 1: # remove stance phase if distance to rung is too large
                    mask = (pawRungDistances[i][1][:,0]>=forFit[i][4][cleanedSwingIndicies[-2][1]]) & (pawRungDistances[i][1][:,0]<=forFit[i][4][cleanedSwingIndicies[-1][0]])
                    meanDist = np.mean(pawRungDistances[i][1][:,1][mask])
                    stdDist  = np.std(pawRungDistances[i][1][:,1][mask])
                    #print(forFit[i][2][cleanedSwingIndicies[-1][0]],stdDist)
                    if  (meanDist < stanceDistances[i][0]) or (meanDist>stanceDistances[i][1]):
                        if verbose: print('distance to rung too large during stance :', meanDist)
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]

                if len(cleanedSwingIndicies) > 0: # remove swing phases for which there is no change in paw-rung distance
                    mask = (pawRungDistances[i][1][:, 0] >= forFit[i][4][cleanedSwingIndicies[-1][0]]) & (pawRungDistances[i][1][:, 0] <= forFit[i][4][cleanedSwingIndicies[-1][1]])
                    if np.std(pawRungDistances[i][1][:,1][mask])< 2. :
                        if verbose: print('no change in paw-rung distance during swing')
                        del cleanedSwingIndicies[-1]


            nIdx += 1
            if nIdx==(len(swingIndices)-1):
                break
        #pdb.set_trace()
        cSIA = np.asarray(cleanedSwingIndicies)
        # extract rung number of stance phase #####################################
        stanceRungIdentity = []

        for n in range(len(cleanedSwingIndicies) + 1):
                #if cleanedSwingIndicies[(n if n<len(cleanedSwingIndicies) else (n-1))][0]>0: # index for the last count needs to be reduced since loop runs for len(cleanedSwingIndicies) + 1
                if n==0:
                    startStanceI = 0
                else:
                    startStanceI = int(cleanedSwingIndicies[n-1][1])
                if n==len(cleanedSwingIndicies):
                    endStanceI = len(pawRungDistances[i][1][:,3])

                else:
                    endStanceI   = int(cleanedSwingIndicies[n][0])+1
                #print(cleanedSwingIndicies[n],startStanceI,endStanceI)
                ###pdb.set_trace()
                #if not obstacle:
                (values,counts) = np.unique(pawRungDistances[i][1][:,3][startStanceI:endStanceI],return_counts=True)
                stanceRungIdentity.append(values[np.argmax(counts)])


                #else:
                #    (values,counts) = np.unique(pawRungDistances[i][1][:,3][startStanceI:endStanceI],return_counts=True)
                #    stanceRungIdentity.append(values[np.argmax(counts)])

                    #(values1, counts1) = np.unique(pawRungDistances[i][1][:,-2][startStanceI:endStanceI], return_counts=True)
                    #stanceObsIdentity.append(values1[np.argmax(counts1)])
                    #pdb.set_trace()
                    # print('obsStance',np.unique(pawRungDistances[i][1][:, -2][startStanceI:endStanceI], return_counts=True))
                    # if sum(values1)>0:
                    #     obsIdxStance=np.where(values1>0)[0][0]
                    #     stanceObsIdentity.append(values1[obsIdxStance])
                    #     pdb.set_trace()
                    # else:
                    #     stanceObsIdentity.append(values1[np.argmax(counts1)])
        if len(cleanedSwingIndicies)+1 != len(stanceRungIdentity):
            print('Swing phases : ', len(cleanedSwingIndicies))
            print('Stance rung ID : ',len(stanceRungIdentity))
            print('Problem, there should be one more swing than stance IDs.')
            pdb.set_trace()

        obsRungIdentity = [element in obsRungs for element in stanceRungIdentity]
        #r2 = [np.any(x==obsRungs) for x in stanceRungIdentity]
        # print('obsRungs : ', obsRungs)
        #pdb.set_trace()

        # find indecisive steps - steps with bimodal speed profile ################
        pawPos = pawTracks[5]
        if obstacle:
            allPawPos = pawTracksAll[5]
        #pdb.set_trace()
        maskAngle = ((pawPos[i][:, 0]) >= min(tracks[5][:, 0])) & ((pawPos[i][:, 0]) <= max(tracks[5][:, 0]))
        newWheelAngleAtPawTimes = interpAngle(pawPos[i][:, 0][maskAngle])
        newLinearXposition = (pawPos[i][:, 1][maskAngle]) * p1 + (newWheelAngleAtPawTimes * wheelCircumsphere / 360.)
        newXtime = pawPos[i][:,0][maskAngle]
        speedTimes = forFit[i][2]

        #pawPos = pawTracks[5]
        stepCharacter = []
        #pdb.set_trace()

        for n in range(len(cleanedSwingIndicies)):
            idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][0]]))
            idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][1]]))
            stepDuration = (pawPos[i][:, 0][idxStanceStart] - pawPos[i][:, 0][idxSwingStart])
            stepLength = (pawPos[i][:, 1][idxStanceStart] - pawPos[i][:, 1][idxSwingStart])
            stepLengthRungNumber = stanceRungIdentity[n+1] - stanceRungIdentity[n]
            #
            idxSwingStartNew = np.argmin(np.abs(newXtime - speedTimes[cleanedSwingIndicies[n][0]]))
            idxStanceStartNew = np.argmin(np.abs(newXtime - speedTimes[cleanedSwingIndicies[n][1]]))
            stepLengthLinear = (newLinearXposition[idxStanceStartNew] - newLinearXposition[idxSwingStartNew])
            #
            speedDuringStep = speedDiff[cleanedSwingIndicies[n][0]:cleanedSwingIndicies[n][1]]
            thresholded = np.abs(speedDuringStep) < 16. # use different, larger threshold #10.
            if sum(thresholded)==0:
                #pdb.set_trace()
                indecisiveStep = False
                closeIndicies = None
            else:
                startStop = np.diff(np.arange(len(speedDuringStep))[thresholded]) > 2 # use indices taking into account missed frames
                mmmStart = np.hstack((([True]), startStop))
                mmmStop = np.hstack((startStop, ([True])))
                #print(len(speedDuringStep),thresholded,mmmStart,speedDuringStep)
                startIdx = (np.arange(len(speedDuringStep))[thresholded])[mmmStart]
                stopIdx = (np.arange(len(speedDuringStep))[thresholded])[mmmStop]
                closeIndicies = np.column_stack((startIdx, stopIdx))
                # do not consider periods at the beginning or the end of the step
                mask = (startIdx > 3) & (stopIdx < (len(speedDuringStep)-3))
                closeIndicies = closeIndicies[mask]
                if any((closeIndicies[:,1]-closeIndicies[:,0])>=3): # if there are long periods below threshold
                    indecisiveStep = True
                elif len(closeIndicies)>=3: # if there are many values below threshold
                    indecisiveStep = True
                else:
                    indecisiveStep = False
                if indecisiveStep:
                    firstExpectedStancePeriod = np.where((closeIndicies[:,1]-closeIndicies[:,0])>=2)[0]
                    if len(firstExpectedStancePeriod)>0:
                        expectedStanceIdx = closeIndicies[firstExpectedStancePeriod[0],0]
                    else:
                        expectedStanceIdx = closeIndicies[0,0]
                else:
                    expectedStanceIdx = idxStanceStart - idxSwingStart
                #if indecisiveStep:
                #    np.where((closeIndicies[:,1]-closeIndicies[:,0])>=3)[0]
                #    expectedImpact =
            # check whether obstacle approach or cross step
            if stanceRungIdentity[n+1] in obsRungs: # step on obstacle
                stepObsCharacter = 1
            elif any((stanceRungIdentity[n]<obsRungs)&(stanceRungIdentity[n+1]>obsRungs)): # step crossing obstacle
                stepObsCharacter = 2
            else:
                stepObsCharacter = 0
            stepCharacter.append([n, cleanedSwingIndicies[n][0], cleanedSwingIndicies[n][1], indecisiveStep, closeIndicies, stepDuration, stepLength, stepLengthLinear,stepLengthRungNumber,stepObsCharacter,expectedStanceIdx])
        # pdb.set_trace()
        ###########################################################################
        #if i ==0:
        #    print(forFit[i][2][cSIA])
        #pdb.set_trace()
        #swingIndices[:, 0] = swingIndices[:, 0] - 0
        #swingIndices[:, 1] = swingIndices[:, 1] + 0
        if showFigPaw :

            fig = plt.figure(figsize=(30,14))
            ax = fig.add_subplot(2,3,1)
            ax.axvline(x=stanceDistances[i][0],color='0.6')
            ax.axvline(x=stanceDistances[i][1],color='0.6')
            ax.hist(pawRungDistances[i][1][:, 1],bins=100)
            plt.xlabel('paw rung distance (cm)')
            plt.ylabel('occurrence')
            #plt.show()

            ax1 = fig.add_subplot(2,3,2)
            ax2 = fig.add_subplot(2,3,3)
            #ax1.fill_between(forFit[i][2],stanceDistances[i][0],stanceDistances[i][1],color='0.8')
            ax1.plot(forFit[i][2], forFit[i][0])
            ax1.plot(forFit[i][2], forFit[i][1] * p1)

            #ax2.fill_between(forFit[i][2], stanceDistances[i][0], stanceDistances[i][1], color='0.8')
            ax2.plot(forFit[i][2], forFit[i][0])
            ax2.plot(forFit[i][2], forFit[i][1] * p1)
            ax3= fig.add_subplot(2,1,2)
            ax3.plot(pawPos[i][:,0], pawPos[i][:,1],c='C1')

            if obstacle:
                ax4 = fig.add_subplot(2, 3, 5)
                ax4.plot(allPawPos[zIdx[i]][:, 0], 600-allPawPos[zIdx[i]][:, 2], c='C1')
                ax5 = fig.add_subplot(2, 3, 6)
                ax5.plot(allPawPos[zIdx[i]][:, 0], 600-allPawPos[zIdx[i]][:, 2], c='C1')

            pawList = ['FL', 'FR', 'HL', 'HR']
            for n in range(len(cleanedSwingIndicies)):
                idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][0]]))
                idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][1]])) + 1

                startI = int(cleanedSwingIndicies[n][0])
                endI   = int(cleanedSwingIndicies[n][1]) + 1
                ax1.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                ax2.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                # ax3.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                linearPawPos=np.array(forFit[i][5])

                swingLength =(linearPawPos[endI,1] - linearPawPos[startI, 1])

                ax1.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c=('C3' if stepCharacter[n][9] else 'C2'))
                ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1, c=('C3' if stepCharacter[n][9] else 'C2'))
                ax3.plot(pawPos[i][:, 0][range(idxSwingStart, idxStanceStart)],pawPos[i][:, 1][idxSwingStart:idxStanceStart], c=('C3' if stepCharacter[n][9] else 'C2'))
                ax3.text(pawPos[i][:, 0][range(idxSwingStart, idxStanceStart)][-1],pawPos[i][:, 1][idxSwingStart:idxStanceStart][-1],str(stanceRungIdentity[n+1]),fontsize=6)
                if obstacle:
                    idxSwingStartTop = np.argmin(np.abs(allPawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][0]]))
                    idxStanceStartTop = np.argmin(np.abs(allPawPos[i][:, 0] - speedTimes[cleanedSwingIndicies[n][1]])) + 1
                    ax4.plot(allPawPos[zIdx[i]][:,0][range(idxSwingStartTop, idxStanceStartTop)], 600-allPawPos[zIdx[i]][:, 2][idxSwingStartTop:idxStanceStartTop], c=('C3' if stepCharacter[n][9] else 'C2'))
                    ax5.plot(allPawPos[zIdx[i]][:, 0][range(idxSwingStartTop, idxStanceStartTop)], 600-allPawPos[zIdx[i]][:, 2][idxSwingStartTop:idxStanceStartTop],c=('C3' if stepCharacter[n][9] else 'C2'))
                # if (swingLength<1) :
                #     ax1.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='C3')
                #     ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1, c='C3')
                # elif (swingLength<1) and stepCharacter[n][3]:
                #     ax1.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='C6')
                #     ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1, c='C6')

                #plt.fill_between()
                #plt.plot(range(startI,endI),forFit[i][1][startI:endI] * p1,c='C2')
                ax1.set_xlim(40,45)
                ax1.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                ax1.set_xlabel('time (s)')
                ax1.set_ylabel('speed (cm)')
                #ax3.set_xlim(40,45)
                ax3.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                ax3.set_xlabel('time (s)')
                ax3.set_ylabel('x-position (pixel)')
                if obstacle:
                    ax4.set_xlabel('time (s)')
                    ax4.set_ylabel('z-position (pixel)')
                    ax5.set_xlabel('time (s)')
                    ax5.set_ylabel('z-position (pixel)')
                    ax5.set_xlim(40, 45)
            #plt.xlim(4610, 4720)
            #plt.savefig('')
            plt.show()


        swingPhases.append([i,cleanedSwingIndicies,stanceRungIdentity,stepCharacter,obsRungIdentity])
    return (swingPhases,forFit)

##########################################################################
def findOverlayMatching_910_820_Rois(caimg910,caimg820, figureLocation, foldersRecordings910, foldersRecordings820,saveFigure=True):

    nDays910 = len(caimg910)
    nDays820 = len(caimg820)
    movementValuesPreset = np.zeros((nDays910, 2))
    allAlignData = {}
    #corrMatrix = np.zeros((nDays,nDays))
    maxTimeDelay = 30*60 # maximal 40 min difference btw. 910 and 820 recording
    nPair = 0

    print(nDays910,nDays820, foldersRecordings910, foldersRecordings820)

    img910 = caimg910['ops']['meanImg'] #allCorrDataPerSession910[nDay910][3][0][2]['meanImg']
    cutLengths910 = removeEmptyColumnAndRows(img910)
    ops910 = caimg910['ops'] # allCorrDataPerSession910[nDay910][3][0][2]
    stat910 = caimg910['stat']

    img820 = caimg820['ops']['meanImg'] #[3][0][2]['meanImg']
    cutLengths820 = removeEmptyColumnAndRows(img820)
    ops820 = caimg820['ops'] #[3][0][2]
    stat820 = caimg820['stat'] #[3][0][4]


    (warp_matrix,cc) = alignTwoImages(img910,cutLengths910,img820,cutLengths820,foldersRecordings910,foldersRecordings820,movementValuesPreset[nPair],figSave=saveFigure,figDir=figureLocation)
    #corrMatrix[nDayA,nDayB] = cc
    (cleanedIntersectionROIs, intersectionROIsA) = alignROIsCheckOverlap(stat910, ops910, stat820, ops820, warp_matrix, foldersRecordings910, foldersRecordings820,figSave=saveFigure, figDir=figureLocation)
    print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(stat910), len(stat820), len(cleanedIntersectionROIs))
    #allAlignData.append([allCorrDataPerSession910[nDay910][0], allCorrDataPerSession820[nDay820][0], nDay910, nDay820, cutLengths910, cutLengths820, warp_matrix, cc,cleanedIntersectionROIs, intersectionROIsA,stat910,stat820])
    allAlignData[nPair] = {}
    allAlignData[nPair]['folder910'] = foldersRecordings910 #0
    allAlignData[nPair]['folder820'] = foldersRecordings820 #1
    allAlignData[nPair]['nDay910'] = nDays910 #2
    allAlignData[nPair]['nDay820'] = nDays820 #3
    allAlignData[nPair]['cutLengths910'] = cutLengths910 #4
    allAlignData[nPair]['cutLengths820'] = cutLengths820 #5
    allAlignData[nPair]['warp_matrix'] = warp_matrix  #6
    allAlignData[nPair]['cc'] = cc   # 7
    allAlignData[nPair]['cleanedIntersectionROIs'] = cleanedIntersectionROIs  #8
    allAlignData[nPair]['intersectionROIsA'] = intersectionROIsA #9
    allAlignData[nPair]['statA'] = stat910 #10
    allAlignData[nPair]['statB'] = stat820 #11
    nPair+=1
    return (allAlignData,cleanedIntersectionROIs)

##########################################################################
def analyzeCalciumStim (Fluo910, Fluo820,timeStamps910, timeStamps820,cleanedIntersectionROIs):
    nRois910 = len(Fluo910)
    nRois820 = len(Fluo820)
    fTraces910 = Fluo910
    fTraces820 = Fluo820
    recordings910 = np.unique(timeStamps910[:, 1])  # determine how many recordings where performed
    recordings820 = np.unique(timeStamps820[:, 1])  # determine how many recordings where performed
    caTracesDict910 = {}
    caTracesDict820 = {}
    caTracesTrials820 = []
    caTracesDict={}
    # Slicing by recordings and getting timeStamps for 910nm
    for n in range(len(recordings910)):
        mask = (timeStamps910[:, 1] == recordings910[n])
        triggerStart910 = timeStamps910[:, 5][mask]
        # trialStartUnixTimes.append(timeStamps[:, 3][mask][0])
        caTracesTime910 = (timeStamps910[:, 4][mask] - triggerStart910)  # triggerStart is negative
        caTracesFluo910raw = fTraces910[:, mask]
        F0_910 = np.mean(caTracesFluo910raw[:, 15:150], axis=1)
        caTracesFluo910 = ((caTracesFluo910raw - F0_910[:, np.newaxis])) / F0_910[:, np.newaxis]
        caTracesDict910[n] = [caTracesTime910, caTracesFluo910]

        mask = (timeStamps820[:, 1] == recordings820[n])
        triggerStart820 = timeStamps820[:, 5][mask]
        # trialStartUnixTimes.append(timeStamps[:, 3][mask][0])
        caTracesTime820 = (timeStamps820[:, 4][mask] - triggerStart820)  # triggerStart is negative
        # pdb.set_trace()
        caTracesFluo820raw = fTraces820[:, mask]
        F0_820 = np.mean(caTracesFluo820raw[:, 15:150], axis=1)
        caTracesFluo820 = ((caTracesFluo820raw - F0_820[:, np.newaxis])) / F0_820[:, np.newaxis]
        caTracesDict820[n] = [caTracesTime820, caTracesFluo820]
        caTracesTrials820.append(caTracesFluo820)
        caTracesDict[n]={'caTime910':caTracesTime910, 'caTraces910':caTracesFluo910, 'caTime820':caTracesTime820, 'caTraces820':caTracesFluo820}


    newCaTime = np.linspace(0.1, 15, 15 * 30 + 1)
    avgArray910 = np.zeros((nRois910, len(newCaTime)))
    avgArray820 = np.zeros((nRois820, len(newCaTime)))
    nTrials = len(recordings910)
    TimedTrialCaTracesDict={}
    trialTraces820 = []
    trialTraces910 = []
    for i in range(len(recordings910)):
        # Averaging and interpolating time for 910nm
        interpCa910 = interp1d(caTracesDict910[i][0], caTracesDict910[i][1], bounds_error=False)
        newFluo910 = interpCa910(newCaTime)
        avgArray910 += newFluo910 / nTrials
        trialTraces910.append(newFluo910)
        # Averaging and interpolating time for 820nm
        interpCa820 = interp1d(caTracesDict820[i][0], caTracesDict820[i][1], bounds_error=False)
        newFluo820 = interpCa820(newCaTime)
        avgArray820 += newFluo820 / nTrials
        trialTraces820.append(newFluo820)
        TimedTrialCaTracesDict[i] = {'caTime': newCaTime, 'caTraces910': newFluo910, 'caTraces820': newFluo820}
    avgTracesDict={'caTime':newCaTime,'avgCaTraces910':avgArray910,'avgCaTraces820':avgArray820}
    print('number of individual trial ROIs', len(trialTraces820[0]))

    roiCount = 0
    alignedAvgCaTraces = []
    alignedTrialCaTraces = []
    amplitude910 = []
    amplitude820 = []
    amplitude = []

    #tresholding


    #calculate amplitude, std, std ratio, mean amplitude
    (treshAvgTraces,treshTimedTrialCaTraces,AvgStdMeanDict,avgTrialsStd820,tTestResults)=getTreshAmpAndStdFromAlignedROIs(nTrials, TimedTrialCaTracesDict,avgTracesDict,cleanedIntersectionROIs)


    return (nTrials,newCaTime, treshAvgTraces,treshTimedTrialCaTraces,AvgStdMeanDict,avgTrialsStd820,tTestResults)
def getTreshAmpAndStdFromAlignedROIs(nTrials, TimedTrialCaTracesDict, avgTracesDict,cleanedIntersectionROIs):
    treshTimedTrialCaTraces=[]
    treshAvgTraces=[]
    stimMask= (avgTracesDict['caTime'] > 5) & (avgTracesDict['caTime']  < 6)
    preStimMask= (avgTracesDict['caTime'] > 0) & (avgTracesDict['caTime']  < 5)
    postStimMask = (avgTracesDict['caTime'] > 6) & (avgTracesDict['caTime'] < 11)
    for c in range(len(cleanedIntersectionROIs)):

        roiIndex910 = cleanedIntersectionROIs[c][0]
        roiIndex820 = cleanedIntersectionROIs[c][1]

        maxAmpMask = (avgTracesDict['caTime'] > 5.5) & (avgTracesDict['caTime']  < 6)
        maxAmp = np.mean(avgTracesDict['avgCaTraces910'][:, maxAmpMask], axis=1)
        ampMask = (maxAmp >= 0.40)  # & (minAmp<0.5)

        # pdb.set_trace()
        if ampMask[roiIndex910]:
            treshAvgTraces.append([roiIndex910, avgTracesDict['avgCaTraces910'][roiIndex910], roiIndex820, avgTracesDict['avgCaTraces820'][roiIndex820]])
            #pdb.set_trace()
            for j in range(nTrials):
                treshTimedTrialCaTraces.append([roiIndex910, TimedTrialCaTracesDict[j]['caTraces910'][roiIndex910],roiIndex820,TimedTrialCaTracesDict[j]['caTraces820'][roiIndex820]])
    AvgStdMeanDict={}
    avgTrialsStd820 = []
    roiStdBeforeStimTrial820Array = []
    roiStdStimTrial820Array = []
    roiStdAfterStimTrial820Array = []

    for t in range(len(treshAvgTraces)):
        roiAmplitude910 = np.mean(treshAvgTraces[t][1][maxAmpMask], axis=0)
        roiAmplitude820 =np.mean(treshAvgTraces[t][3][maxAmpMask], axis=0)
        roiAmplitude=[roiAmplitude910,roiAmplitude820]

        roiStdBeforeStim820 = (np.std(treshAvgTraces[t][3][preStimMask], axis=0))
        roiStdStim820 = (np.std(treshAvgTraces[t][3][stimMask], axis=0))
        roiStdAfterStim820 = (np.std(treshAvgTraces[t][3][postStimMask], axis=0))
        roiStd820=[roiStdBeforeStim820,roiStdStim820,roiStdAfterStim820]

        roiStdBeforeStim910 =  (np.std(treshAvgTraces[t][1][preStimMask], axis=0))
        roiStdStim910 =  (np.std(treshAvgTraces[t][1][stimMask], axis=0))
        roiStdAfterStim910 =  (np.std(treshAvgTraces[t][1][postStimMask], axis=0))
        roiStd910=[roiStdBeforeStim910,roiStdStim910,roiStdAfterStim910]

        roiMeanBeforeStim820 = (np.mean(treshAvgTraces[t][3][preStimMask], axis=0))
        roiMeanStim820 = (np.mean(treshAvgTraces[t][3][stimMask], axis=0))
        roiMeanAfterStim820 = (np.mean(treshAvgTraces[t][3][postStimMask], axis=0))
        roiMean820 = [roiMeanBeforeStim820, roiMeanStim820, roiMeanAfterStim820]

        roiMeanBeforeStim910 =  (np.mean(treshAvgTraces[t][1][preStimMask], axis=0))
        roiMeanStim910 =  (np.mean(treshAvgTraces[t][1][stimMask], axis=0))
        roiMeanAfterStim910 =  (np.mean(treshAvgTraces[t][1][postStimMask], axis=0))
        roiMean910 = [roiMeanBeforeStim910, roiMeanStim910, roiMeanAfterStim910]

        # AvgStdMeanDict[t]={'amplitude910':roiAmplitude910,'amplitude820':roiAmplitude820,'std820pre':roiStdBeforeStim820,'std820stim':roiStdStim820,'std820post':roiStdAfterStim820,'std910pre':roiStdBeforeStim910,'std910stim':roiStdStim910,'std910post':roiStdAfterStim910
        #                    ,'mean820pre':roiMeanBeforeStim820,'mean820stim':roiMeanStim820, 'mean820post':roiMeanAfterStim820, 'mean910pre':roiMeanBeforeStim910,'mean910stim':roiMeanStim910, 'mean910post':roiMeanAfterStim910}
        AvgStdMeanDict[t] = {'amplitude': roiAmplitude,
                             'std820': roiStd820, 'std910': roiStd910, 'mean820': roiMean820, 'mean910': roiMean910}

        for r in range(nTrials):
            roiStdBeforeStimTrial820 = np.abs(np.std(treshTimedTrialCaTraces[r][3][preStimMask], axis=0))
            roiStdStimTrial820 = np.abs(np.std(treshTimedTrialCaTraces[r][3][stimMask], axis=0))
            roiStdAfterStimTrial820 = np.abs(np.std(treshTimedTrialCaTraces[r][3][postStimMask], axis=0))
            roiStdBeforeStimTrial820Array.append(roiStdBeforeStimTrial820)
            roiStdStimTrial820Array.append(roiStdStimTrial820)
            roiStdAfterStimTrial820Array.append(roiStdAfterStimTrial820)
            avgRoiStdBeforeStimTrial820=(np.mean(roiStdBeforeStimTrial820Array, axis=0))
            avgRoiStdStimTrial820=(np.mean(roiStdStimTrial820Array, axis=0))
            avgRoiStdAfterStimTrial820=(np.mean(roiStdAfterStimTrial820Array, axis=0))

            avgTrialsStd820.append([avgRoiStdBeforeStimTrial820, avgRoiStdStimTrial820, avgRoiStdAfterStimTrial820])

    #pdb.set_trace()
    #perform t-test
    avgTrialsStd_pd = pd.DataFrame(avgTrialsStd820, columns=['pre-stim', 'stim', 'post-stim'])
    Ttest_avgTrialsStd_preStim = stats.ttest_ind(avgTrialsStd_pd['pre-stim'], avgTrialsStd_pd['stim'])
    Ttest_avgTrialsStd_postStim = stats.ttest_ind(avgTrialsStd_pd['post-stim'], avgTrialsStd_pd['stim'])

    # std820_pd = pd.DataFrame(std820, columns=['pre-stim', 'stim', 'post-stim'])
    # Ttest_std820_preStim = stats.ttest_ind(std820_pd['pre-stim'], std820_pd['stim'])
    # Ttest_std820_postStim = stats.ttest_ind(std820_pd['post-stim'], std820_pd['stim'])

    tTestResults={'avgStd_preStim':Ttest_avgTrialsStd_preStim,'avgStd_postStim':Ttest_avgTrialsStd_postStim}

    return (treshAvgTraces,treshTimedTrialCaTraces,AvgStdMeanDict,avgTrialsStd820,tTestResults)



def analyzeComplexSpikes(F, Fneu, spks, iscell,figureLocation,timeStamps):
    Fi = (F - 0.7 * Fneu)
    cells = []
    notcells = []
    my_matrix = []
    for i in range(len(iscell[:, 0])):
        if iscell[i][0] == 1:
            cells.append(i)
        elif iscell[i][0] == 0:
            notcells.append(i)
    nFrames = np.shape(Fi)[1]
    for i in range(len(notcells)):
        my_matrix.append([])
        # getting the intersections of the threshold line & the spks
        x1 = np.arange(0, nFrames)
        y1 = spks[notcells[i], :nFrames] # spks aka deconvolved trace
        x2 = np.arange(0, nFrames)
        y2 = np.repeat(10, nFrames) # threshold line at 10
        opts = {'fill_value': 'extrapolate'}
        f1 = interpolate.interp1d(x1, y1, **opts)
        f2 = interpolate.interp1d(x2, y2, **opts)
        xmin = np.min((x1, x2))
        xmax = np.max((x1, x2))
        xuniq = np.unique((x1, x2))
        xvals = xuniq[(xmin <= xuniq) & (xuniq <= xmax)]
        intersects = []
        for xval in xvals:
            x0, = optimize.fsolve(lambda x: f1(x) - f2(x), xval)
            if (xmin <= x0 <= xmax
                    and np.isclose(f1(x0), f2(x0))
                    and not any(np.isclose(x0, intersects))):
                    intersects.append(x0)
        if len(intersects) == 0:
            continue
        # sorting the intersections to have 1 intersection per spike
        intersects.sort()
        intersects_1pt = []
        for l in range(len(intersects)):
            if l % 2 == 0:
                intersects_1pt.append(intersects[l])
        print(f"intersections for ROI {notcells[i]}:",intersects_1pt)

        # creating xMask: 10 frames below & 30 frames above each CS occurence
        y_values = []
        try:
            for n in range(len(intersects_1pt)):
                if intersects_1pt[n] - 10 > 0 and intersects_1pt[n] + 30 < nFrames:
                    xMask = (np.arange(0, nFrames) > (intersects_1pt[n] - 10)) & (np.arange(0, nFrames) < (intersects_1pt[n] + 30))
                    y_values.append(Fi[notcells[i], :][xMask])
                    #pdb.set_trace()
                elif intersects_1pt[n] - 10 < 0 or intersects_1pt[n] + 30 < nFrames:
                    intersects_1pt.remove(intersects_1pt[n])
                    continue
        except:
            continue
        print(f"y values for ROI {notcells[i]}:", y_values)
        # stop loop here
        #pdb.set_trace()
        y_values_array = np.asarray(y_values)
        averages = []
        # averaging the transients; in case of errors we move on to the next ROI
        try:
            averages = np.mean(y_values_array, axis=0)
        except:
            continue
        # mean substraction of the average transient
        pre_transient_avg = sum(averages[0:8]) / 8
        avg = averages - pre_transient_avg

        x = np.arange(0, 40) - 10
        y = avg
        # implementing the double exponential func to the transient decay time
        fitfunc = lambda p, x: np.where(x > p[2], p[0] * (np.exp(-(x - p[2]) / p[1]) - np.exp(-(x - p[2]) / p[3])), 0)
        errfunc = lambda p, x, y: (fitfunc(p, x) - y) ** 2
        # guess some fit parameters
        p0 = np.array([20, 10, -1, 1])
        # fit a gaussian to the correlation function
        p1, success = scipy.optimize.leastsq(errfunc, p0, args=(x, y))
        # compute the best fit function from the best fit parameters
        print(p1)
        yfit = fitfunc(p1, x)
        r = round(p1[1], 2)

        # adding ROI FR and transient distinguishing features to matrix
        my_matrix[i].append(f'{notcells[i]}') # ROI number
        my_matrix[i].append(len(intersects_1pt) / ((x1[-1] + 1) / 30))  # firing rate
        my_matrix[i].append(p1[1])  # transient decay time (tau2)
        my_matrix[i].append(trapz(avg, dx=5)) # AUC

    my_new_matrix = [x for x in my_matrix if x]
    # conversion of the matrix to a DataFrame
    df = pd.DataFrame(my_new_matrix)
    # pdb.set_trace()
    # naming the columns
    df.columns = ['Roi number', 'FiringRate', 'DecayTime', 'Area']
    df1 = df.query('DecayTime < 11')
    # pdb.set_trace()
    # performing cluster analysis with k = 2
    kmeans = KMeans(n_clusters=2)
    y = kmeans.fit_predict(df1[['DecayTime', 'Area']])
    df1['Cluster'] = y
    with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.precision', 3,):
        print(df1.sort_values(by='Cluster'))
    X = df1[['DecayTime', 'Area']]
    K = 2
    # Select random observation as centroids
    Centroids = (X.sample(n=K))
    diff = 1
    j = 0

    while (diff != 0):
        XD = X
        i = 1
        for index1, row_c in Centroids.iterrows():
            ED = []
            for index2, row_d in XD.iterrows():
                d1 = (row_c['DecayTime']-row_d['DecayTime']) ** 2
                d2 = (row_c['Area']-row_d['Area']) ** 2
                d = np.sqrt(d1 + d2)
                ED.append(d)
            X[i] = ED
            i = i + 1
        C = []
        for index, row in X.iterrows():
            min_dist = row[1]
            pos = 1
            for i in range(K):
                if row[i + 1] < min_dist:
                    min_dist = row[i + 1]
                    pos = i + 1
            C.append(pos)
        X['Cluster']=C
        Centroids_new = X.groupby(['Cluster']).mean()[['Area', 'DecayTime']]
        if j == 0:
            diff = 1
            j = j + 1
        else:
            diff = (Centroids_new['Area'] - Centroids['Area']).sum() + (Centroids_new[
                'DecayTime'] - Centroids['DecayTime']).sum()
            print(diff.sum())
        Centroids = X.groupby(['Cluster']).mean()[['Area', 'DecayTime']]

    color = ['blue', 'green']
    fig, ax1 = plt.subplots()
    for k in range(K):
        data = X[X['Cluster'] == k + 1]
        ax1.scatter(data['DecayTime'], data['Area'], c = color[k])
    ax1.scatter(Centroids['DecayTime'], Centroids['Area'], c ='red')
    ax1.set_xlabel('DecayTime')
    ax1.set_ylabel('Area')
    # generating cluster analysis output figure
    fig.savefig(figureLocation + 'CS_Clusters.pdf')

    df3 = df1.query('Cluster == 0')  # ROIs in Cluster 0
    df4 = df1.query('Cluster == 1')  # ROIs in Cluster 1

    n = 0

    #real_CS contains the indices of the ROIs with CS activity aka "Active ROIs"

    real_CS=[]
    for i in range(len(df3)):
        if n == 3:
            real_CS.append(df1.query('Cluster == 0')['Roi number'].values.tolist())
        if df3.iloc[i]['Area'] > 720:
            n += 1
    for i in range(len(df4)):
        if n == 3:
            real_CS.append(df1.query('Cluster == 1')['Roi number'].values.tolist())
        if df4.iloc[i]['Area'] > 720:
            n += 1

    print('indices of CS',real_CS)

    return(real_CS, my_new_matrix)

def getComplexSpikesStats(real_CS, timeStamps,spks, figureLocation):

    timeStamps = np.load('timeStamps.npy', allow_pickle=True)

    timeStamps_1ch = []  # timestamps from 1 channel
    for i in range(len(timeStamps)):
        if i % 2 == 0:
            timeStamps_1ch.append(timeStamps[i])

    timeStamps_s = []

    for i in range(len(timeStamps_1ch)):
        timeStamps_s.append(timeStamps_1ch[i][0] / 30)  # timestamps in seconds

    all_trials = [[], [], [], [], []]  # generating absolute time of each trial in seconds
    trial_number = [1.0, 2.0, 3.0, 4.0, 5.0]

    # seperation by trial

    for j in range(len(trial_number)):
        for n in range(len(timeStamps_s)):
            if timeStamps_1ch[n][1] == trial_number[j]:
                all_trials[j].append(timeStamps_s[n])

    len_trial1 = len(all_trials[0])
    len_trial2 = len_trial1 + len(all_trials[1])
    len_trial3 = len_trial2 + len(all_trials[2])
    len_trial4 = len_trial3 + len(all_trials[3])
    len_trial5 = len_trial4 + len(all_trials[4])

    from matplotlib.pyplot import figure

    allSpikes = []

    cs_locomotion = []
    cs_stable = []

    fr_locomotion = []
    fr_stable = []

    all_trials_fr_loco = [[], [], [], [], []]
    all_trials_fr_stable = [[], [], [], [], []]

    for m in range(len(all_trials)):
        print(f"CS of good ROIs during trial# {m + 1}")
        for i in range(len(real_CS[0])):
            # choosing appropriate frames based on trial number
            if m == 0:
                x1 = all_trials[m]
                y1 = spks[int(real_CS[0][i]),:len_trial1]
                x2 = all_trials[m]
                y2 = np.repeat(10, len(all_trials[m]))
            elif m == 1:
                x1 = all_trials[m]
                y1 = spks[int(real_CS[0][i]), len_trial1:len_trial2]
                x2 = all_trials[m]
                y2 = np.repeat(10, len(all_trials[m]))
            elif m == 2:
                x1 = all_trials[m]
                y1 = spks[int(real_CS[0][i]), len_trial2:len_trial3]
                x2 = all_trials[m]
                y2 = np.repeat(10, len(all_trials[m]))
            elif m == 3:
                x1 = all_trials[m]
                y1 = spks[int(real_CS[0][i]), len_trial3:len_trial4]
                x2 = all_trials[m]
                y2 = np.repeat(10, len(all_trials[m]))
            elif m == 4:
                x1 = all_trials[m]
                y1 = spks[int(real_CS[0][i]), len_trial4:len_trial5]
                x2 = all_trials[m]
                y2 = np.repeat(10, len(all_trials[m]))

            # getting the intersection between threshold line & spks

            opts = {'fill_value': 'extrapolate'}
            f1 = interpolate.interp1d(x1, y1, **opts)
            f2 = interpolate.interp1d(x2, y2, **opts)

            xmin = np.min((x1, x2))
            xmax = np.max((x1, x2))

            xuniq = np.unique((x1, x2))
            xvals = xuniq[(xmin <= xuniq) & (xuniq <= xmax)]

            intersects = []
            for xval in xvals:
                x0, = optimize.fsolve(lambda x: f1(x) - f2(x), xval)
                if (xmin <= x0 <= xmax and np.isclose(f1(x0), f2(x0)) and not any(np.isclose(x0, intersects))):
                    intersects.append(x0)

            if len(intersects) == 0:
                continue

            # sorting the intersections to have 1 intersection per spike

            intersects.sort()
            intersects_1pt = []
            for l in range(len(intersects)):
                if l % 2 == 0:
                    intersects_1pt.append(intersects[l])

            # baseline period (0-5s); locomotion period (10-50s)

            for k in range(len(intersects_1pt)):
                if 10 < intersects_1pt[k] < 50:
                    cs_locomotion.append(intersects_1pt[k])
            fr_locomotion.append(len(cs_locomotion) / 40)

            cs_locomotion[:] = []
            print("FR during locomotion", fr_locomotion)
            # pdb.set_trace()
            for g in range(len(intersects_1pt)):

                if 0 < intersects_1pt[g] < 5:
                    cs_stable.append(intersects_1pt[g])
            fr_stable.append(len(cs_stable) / 5)
            cs_stable[:] = []
            print("FR when stable", fr_stable)
            allSpikes.append(intersects_1pt)

        all_trials_fr_loco[m].extend((fr_locomotion)) # FR during locomotion
        all_trials_fr_stable[m].extend((fr_stable)) # FR during baseline period

        matplotlib.rcParams['font.size'] = 8.0

        np.random.seed(789680)

        colors1 = ['C{}'.format(i) for i in range(len(real_CS[0]))]
        #     lineoffsets1 = np.array([-9, -13, 1,
        #                              15, 6, 10])
        # linelengths1 = [5, 2, 9, 11, 3, 5]
        # fig, axes = plt.subplots(len(all_trials, figsize=(7,9))

        # generating the raster plot

        plt.figure(figsize=(20, 10))
        plt.eventplot(allSpikes, colors=colors1, linelength=0.3)
        plt.xlabel('Time(s)', fontsize=12)
        plt.ylabel('Active ROIs', fontsize=12)

        y1 = 10
        y2 = 50
        plt.axvspan(y1, y2, color='green', alpha=0.1, lw=0)
        y3 = 0
        y4 = 5
        plt.axvspan(y3, y4, color='red', alpha=0.1, lw=0)

        print("ROI CS activity all trials (LOCOMOTION)", all_trials_fr_loco)
        print("ROI CS activity all trials (BASELINE)", all_trials_fr_stable)

        # plt.set_title('matplotlib.axes.Axes.eventplot Example')
        # plt.show()

        del allSpikes[:]
        del fr_locomotion[:]
        del fr_stable[:]

    # getting the average CS activity of individual ROIs
    # on each recording day during baseline and locomotion periods

    loco_array = np.asarray(all_trials_fr_loco)
    stable_array = np.asarray(all_trials_fr_stable)

    avg_locofr_per_ROI = np.mean(loco_array, axis=0)
    avg_stablefr_per_ROI = np.mean(stable_array, axis=0)

    fig, ax = plt.subplots(1, figsize = (4,7))
    y2 = avg_locofr_per_ROI
    y1 = avg_stablefr_per_ROI
    x1 = ["Baseline"] * len(y2)
    x2 = ["Locomotion"] * len(y2)

    for i in range(len(x1)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], marker='o', markersize=3)
    fig.savefig(figureLocation + 'CS_loco_stable_allROIs.pdf')

    # getting the ratio of the FR during locomotion vs baseline

    ratios = []
    for i in range(len(y2)):
        ratios.append(y2[i] / y1[i])

    print("FR locomotion/FR baseline (Ratio:)", ratios)


    return(avg_locofr_per_ROI, avg_stablefr_per_ROI, ratios)

###################################
def getAmpAndTroughToPeakDelay(spike,timeAxis):
    if len(np.shape(spike))>1:
        idxMin = np.argmin(spike,axis=1) # if len(np.shape(spike))>1 else 0))
        idxMax = np.argmax(spike,axis=1) # only look to the left of the minimum (1 if len(np.shape(spike))>1 else 0))
        minTimes = np.asarray([timeAxis[x,y] for x,y in zip(range(len(timeAxis[:,0])),idxMin)])
        maxTimes = np.asarray([timeAxis[x,y] for x,y in zip(range(len(timeAxis[:,0])),idxMax)])
        minAmp = np.asarray([spike[x,y] for x,y in zip(range(len(spike[:,0])),idxMin)])
        maxAmp = np.asarray([spike[x,y] for x,y in zip(range(len(spike[:,0])),idxMax)])

    else:
        idxMin = np.argmin(spike, axis=0)  # if len(np.shape(spike))>1 else 0))
        idxMax = np.argmax(spike[idxMin:], axis=0) + idxMin # only look to the left of the minimum  # (1 if len(np.shape(spike))>1 else 0))
        minTimes = timeAxis[idxMin]
        maxTimes = timeAxis[idxMax]
        minAmp = spike[idxMin]
        maxAmp = spike[idxMax]
    #print(idxMin,idxMax,timeAxis[idxMin],timeAxis[idxMax],timeAxis[idxMax]-timeAxis[idxMin])
    return ((maxTimes-minTimes),(maxAmp-minAmp),idxMin)

###################################
def analyzeSpikingActivity(ephys,sTimes,linearSpeed,mouseRec):
    spikecountwindow = 0.02 # in sec
    binWidth = 1.E-3  # in sec
    tbins = np.linspace(0., 60., int(60./ binWidth) + 1)
    nspikecountwindow = spikecountwindow / binWidth


    matplotlib.use('TkAgg')
    baselineTime = 6
    walkingTime = [10,52]
    #  ([ss_time,cs_time,self._workingDataBase,grandDataBase])
    spike_times = [[],[]]
    spike_times[0] = ephys[0]
    spike_times[1] = ephys[1]
    workingDataBase = ephys[2]
    grandDataBase = ephys[3]
    labels = ['ss','cs']
    ephysDict = {}
    for i in range(len(labels)):
        if len(spike_times[i])>0:
            print('%s spikes exist, total number: %s' %(labels[i],len(spike_times[i])))
            ephysDict['%s_exist' % labels[i]] = np.array([True])
            isi = np.diff(spike_times[i])
            isiBaseline = np.diff(spike_times[i][spike_times[i]<baselineTime])
            isiWalking = np.diff(spike_times[i][(spike_times[i]>walkingTime[0])&(spike_times[i]<walkingTime[1])])
            ephysDict['%s_spikeTimes' % labels[i]] = spike_times[i]
            binnedspikes, _ = np.histogram(spike_times[i], tbins)
            spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
            # convert the convolved spike trains to units of spikes/sec
            spikesconv *= 1. / binWidth
            binTimes = (tbins[1:]+tbins[:-1])/2.
            ephysDict['%s_instFiringRate' % labels[i]] = np.column_stack((binTimes,spikesconv))
            interp = interp1d(binTimes, spikesconv,fill_value="extrapolate")
            #interpMask = (caTracesTime[nFig] >= wheelTime[0]) & (caTracesTime[nFig] <= wheelTime[-1])
            newFRatWheelTimes = interp(sTimes)
            (rBL,pBL) = scipy.stats.pearsonr(linearSpeed,newFRatWheelTimes)
            walkingMask = (sTimes>10.)&(sTimes<=52.)
            (rWP, pWP) = scipy.stats.pearsonr(linearSpeed[walkingMask], newFRatWheelTimes[walkingMask])
            numReps = 300
            rBLshuffle = np.zeros(numReps)
            rWPshuffle = np.zeros(numReps)
            for n in range(numReps):
                shift = np.random.randint(0,len(newFRatWheelTimes))
                (rBLshuffle[n],pBL) = scipy.stats.pearsonr(linearSpeed,np.roll(newFRatWheelTimes,shift))
                (rWPshuffle[n],pBL) = scipy.stats.pearsonr(linearSpeed[walkingMask], (np.roll(newFRatWheelTimes, shift))[walkingMask])
            #pdb.set_trace()
            pBLShuffle = np.sum((rBLshuffle > np.abs(rBL)) | (rBLshuffle < -np.abs(rBL)))/numReps
            pWPShuffle = np.sum((rWPshuffle > np.abs(rWP)) | (rWPshuffle < -np.abs(rWP)))/numReps
            print('significance of baseline pearson : ', pBLShuffle)
            print('significance of walking period pearson : ', pWPShuffle)
            ephysDict['%s_pearsonR' % labels[i]] = np.array([rBL, pBL, rWP, pWP,pBLShuffle,np.mean(rBLshuffle),pWPShuffle,np.mean(rWPshuffle)])
            #plt.plot(sTimes,linearSpeed)
            #plt.plot(sTimes,newFRatWheelTimes)
            #plt.plot(binTimes,spikesconv)
            ## calculation of different spike-count variabilities
            binSizes = [0.05,0.5,1.,5.] # in sec
            for j in range(len(binSizes)):
                tbins = np.linspace(0., 60., int(60./ binSizes[j]) + 1)
                binnedspikes, _ = np.histogram(spike_times[i], tbins)
                cv = np.std(binnedspikes) / np.mean(binnedspikes)
                ephysDict['%s_spike-count_cv_%s'  % (labels[i],binSizes[j])] = np.array([cv])
                #pdb.set_trace()
            #plt.show()
            ephysDict['%s_firingRate'  % labels[i]] = np.array([0 if len(isi)==0 else 1./np.mean(isi)])
            ephysDict['%s_cv'  % labels[i]] = np.array([0 if len(isi) == 0 else np.std(isi)/np.mean(isi)])
            ephysDict['%s_BaselineFiringRate' % labels[i]] = np.array([0. if len(isiBaseline) == 0 else 1./np.mean(isiBaseline)])
            ephysDict['%s_BaselineCV'  % labels[i]] = np.array([0. if len(isiBaseline) == 0 else np.std(isiBaseline)/np.mean(isiBaseline)])
            ephysDict['%s_WalkingFiringRate' % labels[i]] = np.array([0. if len(isiWalking) == 0 else 1./np.mean(isiWalking)])
            ephysDict['%s_WalkingCV'  % labels[i]] = np.array([0. if len(isiWalking) == 0 else np.std(isiWalking)/np.mean(isiWalking)])
            spikesWF = workingDataBase['%s_wave'% labels[i]]
            spikesTB = workingDataBase['%s_wave_span'% labels[i]]
            spikeletWaveform = np.average(spikesWF,axis=0)
            if ((mouseRec[0] == '220525_m28') and (mouseRec[1] == '2022.08.05_001')): # spike waveform is upside down for this one recording
                print('spike waveform multiplied with -1 for ', mouseRec)
                spikeletWaveform = spikeletWaveform*(-1.)
            #spikeletTimeBase = workingDataBase['%s_wave_span'% labels[i]][0] # all the time bases should be the same which is why using the 1st one suffices
            ephysDict['%s_spikeWaveForm'  % labels[i]] = np.row_stack((spikesTB[0],spikeletWaveform))
            (delay,amp,tminPeak) = getAmpAndTroughToPeakDelay(spikeletWaveform,spikesTB[0])
            (iDelay, iAmp, itminPeak) = getAmpAndTroughToPeakDelay(spikesWF, spikesTB)
            #print(delay,amp)
            #plt.plot(spikesTB[0],spikeletWaveform)
            #plt.show()
            ephysDict['%s_avgSpikeParams' % labels[i]] = np.array([delay,amp,tminPeak])
            ephysDict['%s_iSpikeParams' % labels[i]] = np.row_stack((iDelay, iAmp, itminPeak))
            # auto-correlation
            autoCorr = workingDataBase['%s_xprob'% labels[i]]
            autoCorrTB = workingDataBase['%s_xprob_span'% labels[i]]
            ephysDict['%s_autoCorr' % labels[i]] = np.row_stack((autoCorrTB,autoCorr))
            #
            dt = autoCorrTB[1]-autoCorrTB[0]
            bins = np.concatenate((autoCorrTB - dt/2.,np.array([autoCorrTB[-1]+dt/2.])))
            binEdges = np.linspace(-50,50,21,endpoint=True)
            binCenters = (binEdges[1:]+binEdges[:-1])/2.
            if i == 0:
                ttprev = ephys[2]['%s_time_to_prev_%s' % (labels[i],labels[i+1])]
                ttnext = ephys[2]['%s_time_to_next_%s' % (labels[i],labels[i+1])]
                hist, bin_edges = np.histogram(np.concatenate((-ttprev,ttnext)),binEdges)
                ephysDict['%s_crossCorr_%s' % (labels[i], labels[i + 1])] = np.row_stack((binCenters, hist))
            elif i ==1:
                ttprev = ephys[2]['%s_time_to_prev_%s' % (labels[i], labels[i -1])]
                ttnext = ephys[2]['%s_time_to_next_%s' % (labels[i], labels[i -1])]
                #pdb.set_trace()
                hist, bin_edges = np.histogram(np.concatenate((-ttprev, ttnext)), binEdges)
                ephysDict['%s_crossCorr_%s' % (labels[i],labels[i -1])] = np.row_stack((binCenters,hist))
        else:
            ephysDict['%s_exist' % labels[i]] = np.array([False])

    return ephysDict

######################################################
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#####################################################
def calculateShuffledStridebasedPSTH(pawPos,spikes,swingStanceD,psth):
    nShuffles = 300
    jitterSTD = 0.5
    psthStanceOnset = {}
    psthSwingOnset = {}
    for i in range(4):
        psthStanceOnset[i] = []
        psthSwingOnset[i] = []
    for n in range(nShuffles):
        spikesJittered = spikes + np.random.normal(0,jitterSTD,len(spikes))
        psthJitter = calculateStridebasedPSTH(pawPos,spikesJittered,swingStanceD)
        for i in range(4):
            psthStanceOnset[i].append(psthJitter[i]['psth_stanceOnsetAligned_allSteps'][1])
            psthSwingOnset[i].append(psthJitter[i]['psth_swingOnsetAligned_allSteps'][1])

    for i in range(4):
        tempStanceO = np.asarray(psthStanceOnset[i])
        tempSwingO = np.asarray(psthSwingOnset[i])
        psth[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles']= np.percentile(tempStanceO,[5,50,95],axis=0)
        psth[i]['psth_swingOnsetAligned_allSteps_median5-95perentiles'] = np.percentile(tempSwingO, [5, 50, 95], axis=0)


        #pdb.set_trace()

    return psth
#####################################################
def calculateStridebasedPSTH(pawPos,spikes,swingStanceD):
    psth = {}
    col = ['C0','C1','C2','C3']
    beforeRange = [-0.1,0] # range in s
    afterRange = [0,0.1] #  range in s
    isiWalkings = np.diff(spikes[(spikes > 10.) & (spikes <= 52)])
    firingRateWalking = 1. / np.mean(isiWalkings)
    ##############
    psth = {}
    for i in range(4):
        psth[i] = {}
        psth[i]['spikeTimes'] = []
        psth[i]['spikeTimesSorted'] = []
        psth[i]['spikeTimesRescaled'] = []
        psth[i]['spikeTimesRescaledSorted'] = []
        psth[i]['spikeTimesCentered'] = []
        psth[i]['spikeTimesCenteredSorted'] = []
        psth[i]['spikeTimesCenteredSwingStart'] = []
        psth[i]['spikeTimesCenteredSwingStartSorted'] = []
        psth[i]['swingStart'] = []
        psth[i]['swingEnd'] = []
        psth[i]['strideEnd'] = []
        psth[i]['strideEnd2'] = []
        psth[i]['strideEndSorted'] = []
        psth[i]['strideEnd2Sorted'] = []
        psth[i]['swingStartSorted'] = []
        psth[i]['swingEndSorted'] = []
        psth[i]['swingStartRescaled'] = []
        psth[i]['swingStartRescaledSorted'] = []
        psth[i]['indecisive'] = []
        psth[i]['indecisiveSorted'] = []
        psth[i]['indecisiveSorted2'] = []
        psth[i]['indecisiveBool'] = []
        psth[i]['indecisiveBoolSorted'] = []
        idxSwings = swingStanceD['swingP'][i][1]
        recTimes = swingStanceD['forFit'][i][2]
        indecisiveSteps = swingStanceD['swingP'][i][3]
        #pdb.set_trace()
        for n in range(len(idxSwings)-1):
            idxStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][0]]))
            idxEnd = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][1]]))
            idxStartNp1 = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n+1][0]]))
            idxEndNp1 = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n+1][1]]))
            mask = (spikes >= pawPos[i][idxEnd, 0]) & (spikes <= pawPos[i][idxEndNp1, 0])
            mask1 = (spikes >= (pawPos[i][idxEnd, 0]-0.4)) & (spikes <= (pawPos[i][idxEnd, 0]+0.6))
            mask2 = (spikes >= (pawPos[i][idxStart, 0] - 0.4)) & (spikes <= (pawPos[i][idxStart, 0] + 0.6))
            psth[i]['spikeTimes'].append(spikes[mask] -  pawPos[i][idxEnd, 0])
            psth[i]['spikeTimesCentered'].append(spikes[mask1] -  pawPos[i][idxEnd, 0])
            psth[i]['spikeTimesCenteredSwingStart'].append(spikes[mask2] - pawPos[i][idxStart, 0])
            psth[i]['spikeTimesRescaled'].append((spikes[mask] - pawPos[i][idxEnd, 0])/(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0]))
            psth[i]['swingStart'].append(pawPos[i][idxStart, 0]-pawPos[i][idxEnd, 0])
            psth[i]['swingEnd'].append(pawPos[i][idxEnd, 0] - pawPos[i][idxStart, 0])
            psth[i]['strideEnd'].append(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0])
            psth[i]['strideEnd2'].append(pawPos[i][idxStartNp1, 0] - pawPos[i][idxStart, 0])
            psth[i]['swingStartRescaled'].append((pawPos[i][idxStartNp1, 0] - pawPos[i][idxEnd, 0])/(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0]))
            psth[i]['indecisive'].append('black' if indecisiveSteps[n][3] else col[i])
            psth[i]['indecisiveBool'].append(indecisiveSteps[n][3])
    # pdb.set_trace()
    # sort the stance-onset aligned psth based on swing-start time
    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingStart'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            psth[i]['spikeTimesSorted'].append(psth[i]['spikeTimes'][idxSorted[n]])
            psth[i]['spikeTimesCenteredSorted'].append(psth[i]['spikeTimesCentered'][idxSorted[n]])
            psth[i]['swingStartSorted'].append([psth[i]['swingStart'][idxSorted[n]]])
            psth[i]['strideEndSorted'].append([psth[i]['strideEnd'][idxSorted[n]]])
            psth[i]['indecisiveSorted'].append(psth[i]['indecisive'][idxSorted[n]])
            psth[i]['indecisiveBoolSorted'].append(psth[i]['indecisiveBool'][idxSorted[n]])
            #print(i,n,psth[i]['stanceStartRescaled'][idxSorted[n]])

    # sort the swing-onset aligned psth based on swing-end time
    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingEnd'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            #psth[i]['spikeTimesSortedSwing'].append(psth[i]['spikeTimes'][idxSorted[n]])
            psth[i]['spikeTimesCenteredSwingStartSorted'].append(psth[i]['spikeTimesCenteredSwingStart'][idxSorted[n]])
            psth[i]['swingEndSorted'].append([psth[i]['swingEnd'][idxSorted[n]]])
            psth[i]['strideEnd2Sorted'].append([psth[i]['strideEnd2'][idxSorted[n]]])

    # sort the rescaled stance-onset aligned data according to the swing-start time
    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingStartRescaled'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            psth[i]['spikeTimesRescaledSorted'].append(psth[i]['spikeTimesRescaled'][idxSorted[n]])
            psth[i]['swingStartRescaledSorted'].append([psth[i]['swingStartRescaled'][idxSorted[n]]])
            #print(i,n,psth[i]['stanceStartRescaled'][idxSorted[n]])

    tbins2 = np.linspace(-0.4, 0.6, 50 + 1, endpoint=True)
    dt2 = tbins2[1]-tbins2[0]
    for i in range(4):
        #ax1 = plt.subplot(gssub1[i])
        indSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted'], psth[i]['spikeTimesCenteredSorted']) if a]
        surSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted'], psth[i]['spikeTimesCenteredSorted']) if not a]
        cnt, edges = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSorted']).ravel(), bins=tbins2)
        cntInd, edges = np.histogram(np.concatenate(indSteps).ravel(), bins=tbins2)
        cntSur, edges = np.histogram(np.concatenate(surSteps).ravel(), bins=tbins2)
        #ax1.hist(np.concatenate(psth[i]['spikeTimesCenteredSorted']).ravel(),bins=tbins2,histtype='step')
        timePoints = (edges[1:] + edges[:-1])/2
        avgFR =  cnt /(len(psth[i]['spikeTimesCenteredSorted'])* dt2)
        psth[i]['psth_stanceOnsetAligned_indSteps'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntInd / (len(indSteps) * dt2)))
        psth[i]['psth_stanceOnsetAligned_surSteps'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntSur / (len(surSteps) * dt2)))
        psth[i]['psth_stanceOnsetAligned_allSteps'] = np.row_stack((timePoints, avgFR ))
        bb = np.average(avgFR[(timePoints>beforeRange[0])&(timePoints<beforeRange[1])])
        aa = np.average(avgFR[(timePoints > afterRange[0]) & (timePoints < afterRange[1])])
        psth[i]['psth_stanceOnsetAligned_allSteps_change'] = np.array([bb,aa,(aa-bb)/firingRateWalking])
        #ax1.step((edges[1:] + edges[:-1]) / 2, cntInd / (len(indSteps) * dt2),color='black',alpha=0.5)
        #ax1.step((edges[1:] + edges[:-1]) / 2, cntSur / (len(surSteps) * dt2),color='0.5',alpha=0.5)
        #ax1.step((edges[1:] + edges[:-1]) / 2, cnt / (len(psth[i]['spikeTimesCenteredSorted']) * dt2), color=col[i])

    for i in range(4):
        cnt2, edges2 = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSwingStartSorted']).ravel(), bins=tbins2)
        # ax1.hist(np.concatenate(psth[i]['spikeTimesCenteredSorted']).ravel(),bins=tbins2,histtype='step')
        timePoints = (edges2[1:] + edges2[:-1]) / 2
        avgFRS =  cnt2 / (len(psth[i]['spikeTimesCenteredSwingStartSorted']) * dt2)
        psth[i]['psth_swingOnsetAligned_allSteps'] = np.row_stack((timePoints, avgFRS))
        bb = np.average(avgFRS[(timePoints>beforeRange[0])&(timePoints<beforeRange[1])])
        aa = np.average(avgFRS[(timePoints > afterRange[0]) & (timePoints < afterRange[1])])
        psth[i]['psth_swingOnsetAligned_allSteps_change'] = np.array([bb,aa,(aa-bb)/firingRateWalking])
        #ax1.step((edges[1:] + edges[:-1]) / 2, cnt2 / (len(psth[i]['spikeTimesCenteredSwingStartSorted']) * dt3), color=col[i])

    bbins = np.linspace(0, 1, 50 + 1, endpoint=True)
    for i in range(4):
        cnt3, edges3 = np.histogram(np.concatenate(psth[i]['spikeTimesRescaled']).ravel(), bins=bbins)
        # ax1.hist(np.concatenate(psth[i]['spikeTimesRescaled']).ravel(),bins=bbins,histtype='step')
        psth[i]['psth_stanceOnsetAlignedNormalized_allSteps'] = np.row_stack(((edges3[1:] + edges3[:-1]) / 2, cnt3))
        #ax1.step((edges[1:] + edges[:-1]) / 2, cnt)
        #self.layoutOfPanel(ax1, xLabel='rescaled time (0-1 is one stride, 0:stance onset n, 1: stance onset (n+1))', yLabel='PSTH', xyInvisible=[False, False])
    return psth

def analyzeHildebrandPlot (recordingsM,refPaw,listOfRecordings):
    nDays = len(recordingsM)
    pawSeqProb_mouse = []
    countPawSeq_mouse = []
    swingOn_mouse = []
    swingOff_mouse=[]
    iqr_mouse = []
    median_mouse= []
    for n in range(nDays):
        pawSeqProb_day = []
        countPawSeq_day = []
        iqr_Day=[]
        median_Day=[]
        swingOn_day = []
        swingOff_day = []
        for j in range(len(recordingsM[n][4])):

            swingOnset = []
            swingOffset = []
            for i in range(4):
                idxSwings = np.asarray(recordingsM[n][4][j][3][i][1])
                recTimes = np.asarray(recordingsM[n][4][j][4][i][2])
                swingOn = []
                swingOff = []
                for k in range(len(idxSwings) - 1):
                    if (recTimes[idxSwings[k, 0]] > 10.) and (recTimes[idxSwings[k, 0]] < 52):  # only look at steps during motorization period
                        swingOn.append(recTimes[idxSwings[k, 0]])
                        swingOff.append(recTimes[idxSwings[k, 1]])
                swingOnset.append(swingOn)
                swingOffset.append(swingOff)

            # define reference paw

            refPawSwingOn = swingOnset[refPaw]
            refPawSwingOff = swingOffset[refPaw]
            dt = 0.02
            counts = np.zeros((8, int(1 / dt) + 1))
            iOnArray = [[], [], [], []]
            iOffArray = [[], [], [], []]


            for x in range(4):
                for p in range(len(refPawSwingOn) - 1):
                    refPawCycle = refPawSwingOn[p + 1] - refPawSwingOn[p]

                    # normalize to refPawCycle
                    for w in range(len(swingOnset[x])):
                        iOn = (swingOnset[x][w] - refPawSwingOn[p]) / (refPawCycle)
                        iOff = (swingOffset[x][w] - refPawSwingOn[p]) / (refPawCycle)

                        # add only swing that falls in ref paw stride cycle
                        if (0 <= iOn < 1) and (iOff < 1):
                            iOnArray[x].append(iOn)
                            iOffArray[x].append(iOff)
                            counts[2 * x, int(iOn / dt):int(iOff / dt)] += 1
                        elif (0 <= iOn < 1) and (iOff > 1):
                            counts[2 * x, int(iOn / dt):] += 1
                            z = 1
                            aborted = False
                            while (iOff>1) :
                                if ((p + z+1)>=len(refPawSwingOn)):
                                    aborted = True
                                    break
                                iOff = (swingOffset[x][w] - refPawSwingOn[p+z]) / (refPawSwingOn[p + z+1] - refPawSwingOn[p + z])
                                z+=1
                            counts[2 * x, :] += (z-1)
                            if not aborted:
                                counts[2 * x,:int(iOff / dt)] += 1
                                iOffArray[x].append(iOff)
                countsProb = counts / np.max(counts)
            periods = [[], [], [], []]
            avg_periods = [[], [], [], []]
            periodSTD= [[], [], [], []]
            relative_phase= [[], [], [], []]
            for i in range(4):

                for s in range(1, len(iOffArray[i])):
                    periods[i].append(iOffArray[i][s] - iOffArray[i][s-1])
                avg_periods[i].append(sum(periods[i]) / len(periods[i]))
                periodSTD[i].append(np.std(periods[i]))
         #    # calculate relative phase for each paw
         #        relative_phase.append((iOnArray[i] - iOffArray[i]) / avg_periods[i])
         # # calculate coordination between paws
         #    coordination = (abs(relative_phase[0] - relative_phase[1]) +abs(relative_phase[0] - relative_phase[2]) +abs(relative_phase[0] - relative_phase[3]) +abs(relative_phase[1] - relative_phase[2]) +abs(relative_phase[1] - relative_phase[3]) +abs(relative_phase[2] - relative_phase[3])) / 6

            simpleCountsProb=[]
            simpleCounts=[]
            iqr=[]
            median = []

            for c in range(4):
                simpleCountsProb.append(np.array(countsProb[c*2]))
                simpleCounts.append(np.array(counts[c*2]))
                # pdb.set_trace()
                # modes, _ = stats.mode(countsProb[c*2])
                #
                # iqr.append(modes[0])
                # iqr.append(stats.iqr(countsProb[c*2],rng=[70,90]))
                iqr.append(np.std(iOffArray[c]))
                # iqr.append(periodSTD[c][0])
                # iqr.append(stats.iqr(iOnArray[c], rng=[25, 75]))
                # median.append(np.median(countsProb[c*2]))
                # median.append(np.percentile(iOffArray[c],75))
                median.append(np.median(iOffArray[c]))
            # pdb.set_trace()
            swingOn_day.append(iOnArray)
            swingOff_day.append(iOffArray)
            iqr_Day.append(iqr)
            median_Day.append(median)

            pawSeqProb_day.append(simpleCountsProb)
            countPawSeq_day.append(simpleCounts) #use count if imshow representation
        # pdb.set_trace()

        iqr_mouse.append(np.array(iqr_Day))
        median_mouse.append(np.array(median_Day))

        pawSeqProb_mouse.append(pawSeqProb_day)
        countPawSeq_mouse.append(countPawSeq_day)
        swingOn_mouse.append(swingOn_day)
        swingOff_mouse.append(swingOff_day)

    # iqr_mouse=regroupTrialsFromSameDay(iqr_mouse,listOfRecordings)[0]
    # median_mouse=regroupTrialsFromSameDay(median_mouse, listOfRecordings)[0]
    # pawSeqProb_mouse=regroupTrialsFromSameDay(pawSeqProb_mouse, listOfRecordings)[0]
    # countPawSeq_mouse=regroupTrialsFromSameDay(countPawSeq_mouse,listOfRecordings)[0]
    # swingOn_mouse = regroupTrialsFromSameDay(swingOn_mouse, listOfRecordings)[0]
    # swingOff_mouse = regroupTrialsFromSameDay(swingOff_mouse, listOfRecordings)[0]
    # duplicate=regroupTrialsFromSameDay(countPawSeq_mouse,listOfRecordings)[1]

    return (countPawSeq_mouse, pawSeqProb_mouse,iqr_mouse,median_mouse,swingOn_mouse,swingOff_mouse)
def regroupTrialsFromSameDay (array, listOfRecordings):
    newArray=[]
    duplicate=[]

    for k in range(len(array)):
        if k > 0 and listOfRecordings[k][1] == listOfRecordings[k - 1][1]:
            # print("same day found for day number %s recording %s! " % (k, listOfRecordings[k][1]))
            array[k]=np.vstack((array[k-1],array[k] ))
            duplicate.append(k-1)
    #     else:
    #         array[k]=array[k]
    #         duplicate='no'
        newArray.append(array[k])
    # if duplicate=='no':
    #     cleanedArray=newArray
    # else:
    #     duplicate=np.array(duplicate)
        cleanedArray=np.delete(newArray, duplicate)

    return cleanedArray,duplicate
def calculateStridebasedPSTH_condition(pawPos,spikes,swingStanceD, pawSpeed):

    psthCond =[]
    col = ['C0','C1','C2','C3']
    beforeRange = [-0.1,0] # range in s
    afterRange = [0,0.1] #  range in s
    isiWalkings = np.diff(spikes[(spikes > 10.) & (spikes <= 52)])
    firingRateWalking = 1. / np.mean(isiWalkings)
    ##############
    psth = {}

    for i in range(4):
        psth[i] = {}
        psth[i]['spikeTimes'] = []
        psth[i]['spikeTimesSorted'] = []
        psth[i]['spikeTimesRescaled'] = []
        psth[i]['spikeTimesRescaledSorted'] = []
        psth[i]['spikeTimesCentered'] = []
        psth[i]['spikeTimesCenteredSorted'] = []
        psth[i]['spikeTimesCenteredSwingStart'] = []
        psth[i]['spikeTimesCenteredSwingStartSorted'] = []
        psth[i]['swingStart'] = []
        psth[i]['swingEnd'] = []
        psth[i]['strideEnd'] = []
        psth[i]['strideEnd2'] = []
        psth[i]['strideEndSorted'] = []
        psth[i]['strideEnd2Sorted'] = []
        psth[i]['swingStartSorted'] = []
        psth[i]['swingEndSorted'] = []
        psth[i]['swingStartRescaled'] = []
        psth[i]['swingStartRescaledSorted'] = []
        psth[i]['indecisive'] = []
        psth[i]['indecisiveSorted'] = []
        psth[i]['indecisiveSorted2'] = []
        psth[i]['indecisiveBool'] = []
        psth[i]['indecisiveBoolSorted'] = []
        psth[i]['indecisiveBoolSorted2']=[]
        psth[i]['swingLength'] = []
        psth[i]['swingLength_sorted'] = []
        psth[i]['swingLength_sorted2'] = []
        psth[i]['swingLength_sorted_high'] = []
        psth[i]['swingDuration_sorted_Low'] = []
        psth[i]['swingDuration'] = []
        psth[i]['swingDuration_sorted'] = []
        psth[i]['swingDuration_sorted2'] = []
        psth[i]['swingDuration_sorted_high'] = []
        psth[i]['swingDuration_sorted_Low'] = []
        psth[i]['swingSpeed'] = []
        psth[i]['swingSpeed_sorted'] = []
        psth[i]['swingSpeed_sorted2'] = []
        psth[i]['swingSpeed_sorted_high'] = []
        psth[i]['swingSpeed_sorted_Low'] = []
        psth[i]['stanceDuration']=[]
        psth[i]['stanceDuration_sorted']=[]
        psth[i]['stanceDuration_sorted2']=[]
        psth[i]['swingTime']=[]
        psth[i]['swingTime_sorted']=[]
        psth[i]['swingTime_sorted2']=[]
        idxSwings = swingStanceD['swingP'][i][1]
        recTimes = swingStanceD['forFit'][i][2]
        indecisiveSteps = swingStanceD['swingP'][i][3]

        for z in range(len(idxSwings) ):
            idxStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[z][0]]))
            idxEnd = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[z][1]]))

        idxSwings=np.array(idxSwings)#[maskList[q]]
        for n in range(len(idxSwings)-1):
            idxStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][0]]))
            idxEnd = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][1]]))
            idxStartNp1 = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n+1][0]]))
            idxEndNp1 = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n+1][1]]))
            mask = (spikes >= pawPos[i][idxEnd, 0]) & (spikes <= pawPos[i][idxEndNp1, 0])
            mask1 = (spikes >= (pawPos[i][idxEnd, 0]-0.4)) & (spikes <= (pawPos[i][idxEnd, 0]+0.6))
            mask2 = (spikes >= (pawPos[i][idxStart, 0] - 0.4)) & (spikes <= (pawPos[i][idxStart, 0] + 0.6))
            psth[i]['spikeTimes'].append(spikes[mask] -  pawPos[i][idxEnd, 0])
            psth[i]['spikeTimesCentered'].append(spikes[mask1] -  pawPos[i][idxEnd, 0])
            psth[i]['spikeTimesCenteredSwingStart'].append(spikes[mask2] - pawPos[i][idxStart, 0])
            psth[i]['spikeTimesRescaled'].append((spikes[mask] - pawPos[i][idxEnd, 0])/(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0]))
            psth[i]['swingStart'].append(pawPos[i][idxStart, 0]-pawPos[i][idxEnd, 0])
            psth[i]['swingEnd'].append(pawPos[i][idxEnd, 0] - pawPos[i][idxStart, 0])
            psth[i]['strideEnd'].append(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0])
            psth[i]['strideEnd2'].append(pawPos[i][idxStartNp1, 0] - pawPos[i][idxStart, 0])
            psth[i]['swingStartRescaled'].append((pawPos[i][idxStartNp1, 0] - pawPos[i][idxEnd, 0])/(pawPos[i][idxEndNp1, 0] - pawPos[i][idxEnd, 0]))
            psth[i]['indecisive'].append('black' if indecisiveSteps[n][3] else col[i])
            psth[i]['indecisiveBool'].append(indecisiveSteps[n][3])
            psth[i]['swingLength'].append(pawPos[i][:, 1][idxEnd] - pawPos[i][:, 1][idxStart])
            psth[i]['swingDuration'].append(pawPos[i][:, 0][idxEnd] - pawPos[i][:, 0][idxStart])
            psth[i]['swingSpeed'].append(np.mean((pawSpeed[i][:, 1][idxStart:idxEnd])))
            psth[i]['swingTime'].append(np.median((pawPos[i][:, 0][idxStart:idxEnd])))
            psth[i]['stanceDuration'].append(pawPos[i][:, 0][idxStartNp1] - pawPos[i][:, 0][idxEnd])

    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingStart'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            psth[i]['spikeTimesSorted'].append(psth[i]['spikeTimes'][idxSorted[n]])
            psth[i]['spikeTimesCenteredSorted'].append(psth[i]['spikeTimesCentered'][idxSorted[n]])
            psth[i]['swingStartSorted'].append([psth[i]['swingStart'][idxSorted[n]]])
            psth[i]['strideEndSorted'].append([psth[i]['strideEnd'][idxSorted[n]]])
            psth[i]['indecisiveSorted'].append(psth[i]['indecisive'][idxSorted[n]])
            psth[i]['indecisiveBoolSorted'].append(psth[i]['indecisiveBool'][idxSorted[n]])
            psth[i]['swingLength_sorted'].append(psth[i]['swingLength'][idxSorted[n]])
            psth[i]['swingDuration_sorted'].append(psth[i]['swingDuration'][idxSorted[n]])
            psth[i]['swingSpeed_sorted'].append(psth[i]['swingSpeed'][idxSorted[n]])
            psth[i]['stanceDuration_sorted'].append(psth[i]['stanceDuration'][idxSorted[n]])
            psth[i]['swingTime_sorted'].append(psth[i]['swingTime'][idxSorted[n]])

    # sort the swing-onset aligned psth based on swing-end time
    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingEnd'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            #psth[i]['spikeTimesSortedSwing'].append(psth[i]['spikeTimes'][idxSorted[n]])
            psth[i]['spikeTimesCenteredSwingStartSorted'].append(psth[i]['spikeTimesCenteredSwingStart'][idxSorted[n]])
            psth[i]['swingEndSorted'].append([psth[i]['swingEnd'][idxSorted[n]]])
            psth[i]['strideEnd2Sorted'].append([psth[i]['strideEnd2'][idxSorted[n]]])
            psth[i]['indecisiveBoolSorted2'].append(psth[i]['indecisiveBool'][idxSorted[n]])
            psth[i]['swingLength_sorted2'].append(psth[i]['swingLength'][idxSorted[n]])
            psth[i]['swingDuration_sorted2'].append(psth[i]['swingDuration'][idxSorted[n]])
            psth[i]['swingSpeed_sorted2'].append(psth[i]['swingSpeed'][idxSorted[n]])
            psth[i]['stanceDuration_sorted2'].append(psth[i]['stanceDuration'][idxSorted[n]])
            psth[i]['swingTime_sorted2'].append(psth[i]['swingTime'][idxSorted[n]])
    # sort the rescaled stance-onset aligned data according to the swing-start time
    for i in range(4):
        idxSorted = np.argsort(psth[i]['swingStartRescaled'])
        #pdb.set_trace()
        for n in range(len(idxSorted)):
            #idx  = np.where(idxSorted==n)[0][0]
            psth[i]['spikeTimesRescaledSorted'].append(psth[i]['spikeTimesRescaled'][idxSorted[n]])
            psth[i]['swingStartRescaledSorted'].append([psth[i]['swingStartRescaled'][idxSorted[n]]])
            #print(i,n,psth[i]['stanceStartRescaled'][idxSorted[n]])

    tbins2 = np.linspace(-0.4, 0.6, 50 + 1, endpoint=True)
    dt2 = tbins2[1]-tbins2[0]
    
    for i in range(4):
        psth[i]['psth_stanceOnsetAligned_Low'] =[]
        psth[i]['psth_stanceOnsetAligned_High'] =[]
        psth[i]['spikeTimesCenteredSorted_Low']=[]
        psth[i]['spikeTimesCenteredSorted_High']=[]
        psth[i]['swingStartSorted_Low']=[]
        psth[i]['strideEndSorted_Low']=[]
        psth[i]['strideEndSorted_High']=[]
        psth[i]['swingStartSorted_High']=[]

        #ax1 = plt.subplot(gssub1[i])
        conditions=['high','low']
        variables=[psth[i]['swingDuration_sorted'], psth[i]['swingLength_sorted'],psth[i]['swingSpeed_sorted'],psth[i]['stanceDuration_sorted'],psth[i]['swingTime_sorted']]

        indSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted'], psth[i]['spikeTimesCenteredSorted']) if a]
        surSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted'], psth[i]['spikeTimesCenteredSorted']) if not a]
        for v in range(len(variables)):
            # pdb.set_trace()
            psth[i]['spikeTimesCenteredSorted_Low'].append(np.array(psth[i]['spikeTimesCenteredSorted'])[variables[v] < np.median(variables[v])])
            psth[i]['spikeTimesCenteredSorted_High'].append(np.array(psth[i]['spikeTimesCenteredSorted'])[variables[v] > np.median(variables[v])])
            psth[i]['swingStartSorted_Low'].append(np.array(psth[i]['swingStartSorted'])[variables[v] < np.median(variables[v])])
            psth[i]['swingStartSorted_High'].append(np.array(psth[i]['swingStartSorted'])[variables[v] > np.median(variables[v])])
            psth[i]['strideEndSorted_Low'].append(np.array(psth[i]['strideEndSorted'])[variables[v] < np.median(variables[v])])
            psth[i]['strideEndSorted_High'].append(np.array(psth[i]['strideEndSorted'])[variables[v] > np.median(variables[v])])
            low=[b for a, b in zip((variables[v]<np.median(variables[v])), psth[i]['spikeTimesCenteredSorted']) if a]
            high = [b for a, b in zip((variables[v]>np.median(variables[v])), psth[i]['spikeTimesCenteredSorted']) if a]
            cntShort, edges = np.histogram(np.concatenate(low).ravel(), bins=tbins2)
            cntHigh, edges = np.histogram(np.concatenate(high).ravel(), bins=tbins2)
            psth[i]['psth_stanceOnsetAligned_Low'].append(np.row_stack(((edges[1:] + edges[:-1]) / 2, cntShort / (len(low) * dt2))))
            psth[i]['psth_stanceOnsetAligned_High'].append(np.row_stack(((edges[1:] + edges[:-1]) / 2, cntHigh / (len(high) * dt2))))

        cnt, edges = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSorted']).ravel(), bins=tbins2)
        cntInd, edges = np.histogram(np.concatenate(indSteps).ravel(), bins=tbins2)
        cntSur, edges = np.histogram(np.concatenate(surSteps).ravel(), bins=tbins2)

        timePoints = (edges[1:] + edges[:-1])/2
        avgFR =  cnt /(len(psth[i]['spikeTimesCenteredSorted'])* dt2)
        psth[i]['psth_stanceOnsetAligned_indSteps'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntInd / (len(indSteps) * dt2)))
        psth[i]['psth_stanceOnsetAligned_surSteps'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntSur / (len(surSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_shortDuration'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntShortDuration / (len(shortDurationSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_highDuration'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntHighDuration / (len(highDurationSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_lowSpeed'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntShortSpeed / (len(shortSpeedSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_highSpeed'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntHighSpeed / (len(highSpeedSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_shortLength'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntShortLength / (len(shortLengthSteps) * dt2)))
        # psth[i]['psth_stanceOnsetAligned_highLength'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntHighLength / (len(highLengthSteps) * dt2)))
        psth[i]['psth_stanceOnsetAligned_allSteps'] = np.row_stack((timePoints, avgFR ))
        bb = np.average(avgFR[(timePoints>beforeRange[0])&(timePoints<beforeRange[1])])
        aa = np.average(avgFR[(timePoints > afterRange[0]) & (timePoints < afterRange[1])])
        psth[i]['psth_stanceOnsetAligned_allSteps_change'] = np.array([bb,aa,(aa-bb)/firingRateWalking])
        #ax1.step((edges[1:] + edges[:-1]) / 2, cntInd / (len(indSteps) * dt2),color='black',alpha=0.5)
        #ax1.step((edges[1:] + edges[:-1]) / 2, cntSur / (len(surSteps) * dt2),color='0.5',alpha=0.5)
        #ax1.step((edges[1:] + edges[:-1]) / 2, cnt / (len(psth[i]['spikeTimesCenteredSorted']) * dt2), color=col[i])

    for i in range(4):            
        psth[i]['psth_swingOnsetAligned_Low'] =[]
        psth[i]['psth_swingOnsetAligned_High'] =[]
        psth[i]['spikeTimesCenteredSwingStartSorted_High'] =[]
        psth[i]['spikeTimesCenteredSwingStartSorted_Low'] =[]
        psth[i]['swingEndSorted_Low']=[]
        psth[i]['swingEndSorted_High']=[]
        psth[i]['strideEnd2Sorted_Low']=[]
        psth[i]['strideEnd2Sorted_High']=[]
        indSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted2'], psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        surSteps = [b for a, b in zip(psth[i]['indecisiveBoolSorted2'], psth[i]['spikeTimesCenteredSwingStartSorted']) if not a]
        variables=[psth[i]['swingDuration_sorted2'], psth[i]['swingLength_sorted2'],psth[i]['swingSpeed_sorted2'],psth[i]['stanceDuration_sorted2'],psth[i]['swingTime_sorted2']]
        # pdb.set_trace()
        for v in range(len(variables)):
            psth[i]['spikeTimesCenteredSwingStartSorted_Low'].append(np.array(psth[i]['spikeTimesCenteredSwingStartSorted'])[(variables[v]<np.median(variables[v]))])
            psth[i]['spikeTimesCenteredSwingStartSorted_High'].append(np.array(psth[i]['spikeTimesCenteredSwingStartSorted'])[(variables[v]>np.median(variables[v]))])
            psth[i]['swingEndSorted_Low'].append(np.array(psth[i]['swingEndSorted'])[variables[v] < np.median(variables[v])])
            psth[i]['strideEnd2Sorted_High'].append(np.array(psth[i]['strideEnd2Sorted'])[variables[v] > np.median(variables[v])])
            psth[i]['swingEndSorted_High'].append(np.array(psth[i]['swingEndSorted'])[variables[v] > np.median(variables[v])])
            psth[i]['strideEnd2Sorted_Low'].append(np.array(psth[i]['strideEnd2Sorted'])[variables[v] < np.median(variables[v])])
            low=[b for a, b in zip((variables[v]<np.median(variables[v])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
            high = [b for a, b in zip((variables[v]>np.median(variables[v])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
            cntShort, edges = np.histogram(np.concatenate(low).ravel(), bins=tbins2)
            cntHigh, edges = np.histogram(np.concatenate(high).ravel(), bins=tbins2)
            psth[i]['psth_swingOnsetAligned_Low'].append(np.row_stack(((edges[1:] + edges[:-1]) / 2, cntShort / (len(low) * dt2))))
            psth[i]['psth_swingOnsetAligned_High'].append(np.row_stack(((edges[1:] + edges[:-1]) / 2, cntHigh / (len(high) * dt2))))
            
        # shortDurationSteps = [b for a, b in zip((psth[i]['swingDuration_sorted2']<np.median(psth[i]['swingDuration_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # highDurationSteps = [b for a, b in zip((psth[i]['swingDuration_sorted2']>np.median(psth[i]['swingDuration_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # shortLengthSteps = [b for a, b in zip((psth[i]['swingLenght_sorted2']<np.median(psth[i]['swingLenght_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # highLengthSteps = [b for a, b in zip((psth[i]['swingLenght_sorted2']>np.median(psth[i]['swingLenght_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # shortSpeedSteps = [b for a, b in zip((psth[i]['swingSpeed_sorted2']<np.median(psth[i]['swingSpeed_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # highSpeedSteps = [b for a, b in zip((psth[i]['swingSpeed_sorted2']>np.median(psth[i]['swingSpeed_sorted2'])), psth[i]['spikeTimesCenteredSwingStartSorted']) if a]
        # 
        cnt2, edges2 = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSwingStartSorted']).ravel(), bins=tbins2)
        cntInd, edges = np.histogram(np.concatenate(indSteps).ravel(), bins=tbins2)
        cntSur, edges = np.histogram(np.concatenate(surSteps).ravel(), bins=tbins2)

        timePoints = (edges2[1:] + edges2[:-1]) / 2
        avgFRS =  cnt2 / (len(psth[i]['spikeTimesCenteredSwingStartSorted']) * dt2)
        psth[i]['psth_swingOnsetAligned_allSteps'] = np.row_stack((timePoints, avgFRS))
        
        psth[i]['psth_swingOnsetAligned_indSteps'] = np.row_stack(((edges[1:] + edges[:-1])/2,cntInd / (len(indSteps) * dt2)))
        psth[i]['psth_swingOnsetAligned_surSteps'] = np.row_stack(((edges[1:] + edges[:-1]) / 2, cntSur / (len(surSteps) * dt2)))

        bb = np.average(avgFRS[(timePoints>beforeRange[0])&(timePoints<beforeRange[1])])
        aa = np.average(avgFRS[(timePoints > afterRange[0]) & (timePoints < afterRange[1])])
        psth[i]['psth_swingOnsetAligned_allSteps_change'] = np.array([bb,aa,(aa-bb)/firingRateWalking])


    bbins = np.linspace(0, 1, 50 + 1, endpoint=True)
    for i in range(4):
        cnt3, edges3 = np.histogram(np.concatenate(psth[i]['spikeTimesRescaled']).ravel(), bins=bbins)
        # ax1.hist(np.concatenate(psth[i]['spikeTimesRescaled']).ravel(),bins=bbins,histtype='step')
        psth[i]['psth_stanceOnsetAlignedNormalized_allSteps'] = np.row_stack(((edges3[1:] + edges3[:-1]) / 2, cnt3))
        #ax1.step((edges[1:] + edges[:-1]) / 2, cnt)
        #self.layoutOfPanel(ax1, xLabel='rescaled time (0-1 is one stride, 0:stance onset n, 1: stance onset (n+1))', yLabel='PSTH', xyInvisible=[False, False])
    # psthCond.append(psth)
    return psth
###############################################
def calculateShuffledStridebasedPSTH_condition(pawPos,spikes,swingStanceD, pawSpeed,psth,nShuff):
    conditions = ['high', 'low']
    variables = ['swingLength', 'swingDuration', 'swingSpeed', 'stanceDuration', 'swingTime']
    nShuffles = nShuff
    jitterSTD = 0.5
    psthStanceOnset = {}
    psthSwingOnset = {}

    psthLowStanceOnset ={}
    psthLowSwingOnset={}
    psthHighStanceOnset ={}
    psthHighSwingOnset={}
    psthSwingOnsetIndStep = {}
    psthSwingOnsetSurStep = {}
    psthStanceOnsetIndStep = {}
    psthStanceOnsetSurStep = {}
    for i in range(4):
        psthStanceOnset[i] = []
        psthSwingOnset[i] = []
        psthLowStanceOnset[i]=[[],[],[],[],[]]
        psthLowSwingOnset[i]=[[],[],[],[],[]]
        psthHighStanceOnset[i]=[[],[],[],[],[]]
        psthHighSwingOnset[i]=[[],[],[],[],[]]
        psthSwingOnsetIndStep[i]=[]
        psthSwingOnsetSurStep[i]=[]
        psthStanceOnsetIndStep[i]=[]
        psthStanceOnsetSurStep[i]=[]
    for n in range(nShuffles):
        spikesJittered = spikes + np.random.normal(0,jitterSTD,len(spikes))
        psthJitter = calculateStridebasedPSTH_condition(pawPos,spikesJittered,swingStanceD,pawSpeed)

        for i in range(4):
            psthStanceOnset[i].append(psthJitter[i]['psth_stanceOnsetAligned_allSteps'][1])
            psthSwingOnset[i].append(psthJitter[i]['psth_swingOnsetAligned_allSteps'][1])
            psthSwingOnsetIndStep[i].append(psthJitter[i]['psth_swingOnsetAligned_indSteps'][1])
            psthSwingOnsetSurStep[i].append(psthJitter[i]['psth_swingOnsetAligned_surSteps'][1])
            psthStanceOnsetIndStep[i].append(psthJitter[i]['psth_stanceOnsetAligned_indSteps'][1])
            psthStanceOnsetSurStep[i].append(psthJitter[i]['psth_stanceOnsetAligned_surSteps'][1])

            for j in range(len(variables)):
                psthLowStanceOnset[i][j].append(psthJitter[i]['psth_stanceOnsetAligned_Low'][j][1])
                psthHighStanceOnset[i][j].append(psthJitter[i]['psth_stanceOnsetAligned_High'][j][1])
                psthLowSwingOnset[i][j].append(psthJitter[i]['psth_swingOnsetAligned_Low'][j][1])
                psthHighSwingOnset[i][j].append(psthJitter[i]['psth_swingOnsetAligned_High'][j][1])

            # psthShortLengthStanceOnset[i].append(psthJitter[i]['psth_swingOnsetAligned_shortLength'][1])
            # psthShortLengthSwingOnset[i].append(psthJitter[i]['psth_stanceOnsetAligned_shortLength'][1])
            # psthHighLengthStanceOnset[i].append(psthJitter[i]['psth_swingOnsetAligned_highLength'][1])
            # psthHighLengthSwingOnset[i].append(psthJitter[i]['psth_stanceOnsetAligned_highLength'][1])

            
    for i in range(4):
        tempStanceO = np.asarray(psthStanceOnset[i])
        tempSwingO = np.asarray(psthSwingOnset[i])
        tempSwingOnsetIndStep=np.array(psthSwingOnsetIndStep[i])
        tempSwingOnsetSurStep=np.array(psthSwingOnsetSurStep[i])
        tempStanceOnsetIndStep=np.array(psthStanceOnsetIndStep[i])
        tempStanceOnsetSurStep=np.array(psthStanceOnsetSurStep[i])
        psth[i]['psth_stanceOnsetAligned_Low_median5-95perentiles'] =[]
        psth[i]['psth_stanceOnsetAligned_High_median5-95perentiles'] =[]
        psth[i]['psth_swingOnsetAligned_Low_median5-95perentiles'] =[]
        psth[i]['psth_swingOnsetAligned_High_median5-95perentiles'] =[]
        
        psth[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles']= np.percentile(tempStanceO,[5,50,95],axis=0)
        psth[i]['psth_swingOnsetAligned_allSteps_median5-95perentiles'] = np.percentile(tempSwingO, [5, 50, 95], axis=0)

        psth[i]['psth_swingOnsetAligned_IndSteps_median5-95perentiles'] = np.percentile(tempSwingOnsetIndStep, [5, 50, 95], axis=0)
        psth[i]['psth_swingOnsetAligned_SurSteps_median5-95perentiles'] = np.percentile(tempSwingOnsetSurStep, [5, 50, 95], axis=0)
        psth[i]['psth_stanceOnsetAligned_IndSteps_median5-95perentiles']= np.percentile(tempStanceOnsetIndStep,[5,50,95],axis=0)
        psth[i]['psth_stanceOnsetAligned_SurSteps_median5-95perentiles']= np.percentile(tempStanceOnsetSurStep,[5,50,95],axis=0)



        for j in range(len(variables)):
            tempStanceOLow = np.asarray(psthLowStanceOnset[i][j])
            tempStanceOHigh= np.asarray(psthHighStanceOnset[i][j])
            tempSwingOLow = np.asarray(psthLowSwingOnset[i][j])
            tempSwingOHigh= np.asarray(psthHighSwingOnset[i][j])
            psth[i]['psth_stanceOnsetAligned_Low_median5-95perentiles'].append(np.percentile(tempStanceOLow,[5,50,95],axis=0))
            psth[i]['psth_stanceOnsetAligned_High_median5-95perentiles'].append(np.percentile(tempStanceOHigh,[5,50,95],axis=0))
            psth[i]['psth_swingOnsetAligned_Low_median5-95perentiles'].append(np.percentile(tempSwingOLow,[5,50,95],axis=0))
            psth[i]['psth_swingOnsetAligned_High_median5-95perentiles'].append(np.percentile(tempSwingOHigh,[5,50,95],axis=0))

        # pdb.set_trace()
####################################
# singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephysPSTHDict])
# ephysData.append([foldersRecordings[f][0],foldersRecordings[f][1],singleCellList])
def PSTHcorrelation(ephysDict):
    nCells = len(ephysDict)
    pearCorrs = [[],[],[],[]]
    #pdb.set_trace()
    for n in range(nCells):
        nRecs = len(ephysDict[n][2])
        combis = list(itertools.combinations(range(nRecs),2))
        pCorrs = np.zeros((4,len(combis)))
        nCombis=0
        for cc in combis:
            print(cc)
            #pdb.set_trace()
            for i in range(4):
                (pCorrs[i,nCombis],_)=scipy.stats.pearsonr(ephysDict[n][2][cc[0]][2][i]['psth_stanceOnsetAligned_allSteps'][1],ephysDict[n][2][cc[1]][2][i]['psth_stanceOnsetAligned_allSteps'][1])
            nCombis+=1
            #pdb.set_trace()
        ephysDict[n].append(pCorrs)
        #for i in range(4):
        #    pearCorrs[i].append(pCorrs[i,:])
        #pdb.set_trace()
    return ephysDict

#########################
# singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephysPSTHDict])
def calculatePSTH_change(singleCellList):
    percentileRange = [20,80]
    tbins2 = np.linspace(-0.4, 0.6, 50 + 1, endpoint=True)
    dt2 = tbins2[1]-tbins2[0]
    nRecs = len(singleCellList)
    if nRecs == 2:
        earlyLateRecs = [[0],[1]]
    elif nRecs == 3:
        earlyLateRecs = [[0], [2]]
    elif nRecs == 4:
        earlyLateRecs = [[0,1], [2,3]]
    elif nRecs == 5:
        earlyLateRecs = [[0,1], [3,4]]
    elif nRecs == 6:
        earlyLateRecs = [[0,1,2], [3,4,5]]
    else:
        print('case note defined for nRecs : ',nRecs)
    # cnt, edges = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSorted']).ravel(), bins=tbins2)
    #pdb.set_trace()
    psthBA = {}
    for i in range(4):
        psthBA[i] = {}
        psthBA[i]['allEarlySpikeTimes'] = []
        psthBA[i]['earlyConvIntervals'] = []
        psthBA[i]['allLateSpikeTimes'] = []
        psthBA[i]['lateConvIntervals'] = []
        psthBA[i]['allEarlySwingDurations'] = []
        psthBA[i]['allLateSwingDurations'] = []
        for el in range(len(earlyLateRecs)):
            for n in earlyLateRecs[el]:
                if el == 0:
                    psthBA[i]['allEarlySpikeTimes'].extend(singleCellList[n][2][i]['spikeTimesCenteredSorted'])
                    psthBA[i]['allEarlySwingDurations'].extend(singleCellList[n][2][i]['swingDuration_sorted'])
                    psthBA[i]['earlyConvIntervals'].append(singleCellList[n][2][i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'])
                elif el == 1:
                    psthBA[i]['allLateSpikeTimes'].extend(singleCellList[n][2][i]['spikeTimesCenteredSorted'])
                    psthBA[i]['allLateSwingDurations'].extend(singleCellList[n][2][i]['swingDuration_sorted'])
                    psthBA[i]['lateConvIntervals'].append(singleCellList[n][2][i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'])
                #allEarlyLateSpikes[el].extend(np.concatenate(singleCellList[n][2][i]['spikeTimesCenteredSorted']).ravel())
                #allEarlyLateSwingDurations[el].extend(np.concatenate(singleCellList[n][2][i]['swingDuration_sorted']).ravel())  #psth[i]['swingDuration_sorted']
        #pdb.set_trace()
        (shortThreshold,longThreshold) = np.percentile(psthBA[i]['allLateSwingDurations'],(percentileRange[0],percentileRange[1]))
        boolLate = (psthBA[i]['allLateSwingDurations']>=shortThreshold)&(psthBA[i]['allLateSwingDurations']<=longThreshold)
        boolEarly = (psthBA[i]['allEarlySwingDurations']>=shortThreshold)&(psthBA[i]['allEarlySwingDurations']<=longThreshold)
        lateSpikeListWithinBoundaries = [i for (i,v) in zip(psthBA[i]['allLateSpikeTimes'],boolLate) if v]
        earlySpikeListWithinBoundaries = [i for (i,v) in zip(psthBA[i]['allEarlySpikeTimes'],boolEarly) if v]
        # calculate PSTH
        cntLate, edges = np.histogram(np.concatenate(lateSpikeListWithinBoundaries).ravel(), bins=tbins2)
        cntEarly, edges = np.histogram(np.concatenate(earlySpikeListWithinBoundaries).ravel(), bins=tbins2)
        if not np.array_equal(edges,edges):  raise Exception("The binning for early and late spikes is different")
        timePoints = (edges[1:] + edges[:-1])/2
        avgFRLate =  cntLate /(len(lateSpikeListWithinBoundaries)* dt2)
        avgFREarly = cntEarly / (len(earlySpikeListWithinBoundaries) * dt2)
        #psth[i]['psth_stanceOnsetAligned_allSteps'] = np.row_stack((timePoints, avgFRLate ))
        #psth[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles']
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][0][0],ls='--')
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][0][1],ls='--')
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][0][2],ls='--')
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][1][0],ls=':')
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][1][1],ls=':')
        plt.plot(timePoints,psthBA[i]['earlyConvIntervals'][1][2],ls=':')
        plt.plot(timePoints,avgFREarly)
        plt.plot(timePoints, avgFRLate)
        plt.show()
        pdb.set_trace()

################
def getSwingStanceDurations(cPawPos,swingStanceDict):
    dur = {}
    for i in range(4):
        dur[i] = {}
        idxSwings = swingStanceDict['swingP'][i][1]
        indecisiveSteps = swingStanceDict['swingP'][i][3]
        recTimes = swingStanceDict['forFit'][i][2]
        # pdb.set_trace()
        stanceDur = []
        swingDur = []
        for n in range(len(idxSwings)):
            #idxStart = np.argmin(np.abs(cPawPos[i][:, 0] - recTimes[idxSwings[n][0]]))
            #idxEnd = np.argmin(np.abs(cPawPos[i][:, 0] - recTimes[idxSwings[n][1]]))
            swingDur.append(recTimes[idxSwings[n][1]]-recTimes[idxSwings[n][0]])
            #print((recTimes[idxSwings[n][1]]-recTimes[idxSwings[n][0]]),(cPawPos[pawIDtoShow][idxEnd, 0]-cPawPos[pawIDtoShow][idxStart, 0]))
            if n>0:
                stanceDur.append(recTimes[idxSwings[n][0]] - recTimes[idxSwings[n - 1][1]])
        dur[i]['stanceDuration'] = stanceDur
        dur[i]['swingDuration'] = swingDur
    return dur
#############################################
def correlateWheelWithPawSpeed(tracks,pawTracks,showFig=False):
    locoPeriod = [10,53] # locomotor period
    dt = 0.005
    timeBasis = np.linspace(locoPeriod[0],locoPeriod[1],int((locoPeriod[1]-locoPeriod[0])/dt+1))
    interpWheel = interp1d(tracks[2], -tracks[1])
    speed = np.zeros((5,len(timeBasis)))
    speed[0] = interpWheel(timeBasis)
    for i in range(4):
        #mask = ((pawTracks[3][i][:,0])>=min(tracks[2])) & ((pawTracks[3][i][:,0])<=max(tracks[2]))
        interpPaw = interp1d(pawTracks[3][i][:,0],pawTracks[3][i][:,2])
        speed[i+1] = interpPaw(timeBasis)
        #newWheelSpeedAtPawTimes = interp(pawTracks[3][i][:,0][mask])
        #plt.plot(pawTracks[3][i][:,0][mask],0.025*pawTracks[3][i][:,2][mask]-newWheelSpeedAtPawTimes)

    #plt.plot(pawTracks[3][i][:,0][mask],0.025*pawTracks[3][0][:,2][mask] +0.025*pawTracks[3][1][:,2][mask] +0.025*pawTracks[3][2][:,2][mask] +0.025*pawTracks[3][2][:,2][mask] -4.*newWheelSpeedAtPawTimes)
    #plt.plot(tracks[2],-tracks[1])
    if showFig:
        plt.axhline(y=0,ls='--',color='gray')
        for i in range(5):
            if i == 0:
                plt.plot(timeBasis,speed[i],c='black')
            else:
                plt.plot(timeBasis,speed[i]*0.025-speed[0])
        plt.show()
    jointSpeed = np.sum((speed[1:]*0.025-speed[0]),axis=0)
    jointAbsSpeed = np.sum(np.abs((speed[1:]*0.025-speed[0])),axis=0)
    #pdb.set_trace()
    pps = []
    pps.extend(scipy.stats.pearsonr(speed[0],jointSpeed))
    pps.extend(scipy.stats.pearsonr(speed[0], jointAbsSpeed))
    #print('joint speed :',scipy.stats.pearsonr(speed[0],jointSpeed))
    #print('joint absolute speed :', scipy.stats.pearsonr(speed[0], jointAbsSpeed))
    #print('individual speeds:')
    for i in range(4):
        pps.extend(scipy.stats.pearsonr(speed[0],(speed[i+1]*0.025-speed[0])))
        #print('  ', scipy.stats.pearsonr(speed[0],speed[i+1]))
    return pps
    #pdb.set_trace()

#####################################################
def determineObstacleTimesAngles(angleTimes,absoluteSignal, obstacle1, obstacle2,angleRange=None):
    def findClosestIdxToAngle(newAngles,recordedAngles):
        indicies = []
        for i in range(len(newAngles)):
            indicies.append(np.argmin(np.abs(recordedAngles-newAngles[i])))
        return np.asarray(indicies)

    def correctAnglesOutsideRecordedRange(newAngles,recordedAngles):
        if len(newAngles)>0:
            if newAngles[-1]>recordedAngles[-1]:
                newAngles[-1]=recordedAngles[-1]
            if newAngles[0]<0.:
                newAngles[0]=0.
        else:
            newAngles=newAngles
        return newAngles

    # determine all time points when wheel is upper position with obstacle up : time points for obstacle 1 and for 2
    # - at which angle and time point is the wheel in upper postion (i.e. obs1 is up) or lower postion (i.e. obs2 up)
    # - extract the times when absoluteSignal goes from 0 to 1 : use np.diff()
    if angleRange is None:
        startAngle = -8
        endAngle = 55
    else:
        startAngle = angleRange[0]
        endAngle = angleRange[1]
    #
    obstacleUPdic = {}
    obstacleUPdic['angleRange'] = [startAngle,endAngle]

    difference = np.diff(absoluteSignal[:, 1])  # calculate difference
    obs1UP_idx = np.arange(len(absoluteSignal))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure

    absDown = np.arange(len(absoluteSignal))[np.concatenate((np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period

    # check for if wheel obstacle 1 is in the up when absUP -> know whether obs1 is active
    obs1_UP_activated_idx = np.intersect1d(obs1UP_idx, np.arange(len(obstacle1))[obstacle1[:, 1] > 1.5])
    # check for if wheel obstacle 1 is in the up when absUP -> know whether obs1 is active
    # obstacle : convert idx of absolute up transitions to angle, add 180 degrees, determine the time point (idx) when the wheel was at this angle
    #absUp_angle=angleTimes[:,1][absUp]
    obs2UP_angle=angleTimes[:,1][obs1UP_idx]+180.
    # pdb.set_trace()
    if (angleTimes[:,1][obs1UP_idx][0]-180.)>0:  # add possible obs 2 crossing before obs 1 crossing in beginning of recording
        # pdb.set_trace()
        obs2UP_angle = np.concatenate((np.array([angleTimes[:,1][obs1UP_idx][0]-180.]),obs2UP_angle))
    if obs2UP_angle[-1]>angleTimes[:,1][-1]:  # remove last obs 2 crossing if it is larger than the maximally achieved angle during recording
        obs2UP_angle = obs2UP_angle[:-1]


    obs2UP_idx = findClosestIdxToAngle(obs2UP_angle,angleTimes[:,1])
    #for i in range(len(obs2UP_angle)):
    #    obs2UP_idx.append(np.argmin(np.abs(angleTimes[:,1]-obs2UP_angle[i])))

    # check if obstacle 2 was up at this idx -> know whether obs2 is active
    obs2_UP_activated_idx = np.intersect1d(obs2UP_idx, np.arange(len(obstacle2))[obstacle2[:, 1] > 1.5])
    #pdb.set_trace()

    obstacleUPdic['obstacle1UPAndActivated'] = [obs1_UP_activated_idx, angleTimes[obs1_UP_activated_idx]]
    obstacleUPdic['obstacle2UPAndActivated'] = [obs2_UP_activated_idx, angleTimes[obs2_UP_activated_idx]]
    # whenever obstacle is up -> determine time window before and after during which the wheel was -/+ 20 degrees
    # pdb.set_trace()
    angleTimeObs1Up= angleTimes[obs1_UP_activated_idx]
    angleTimeObs2Up = angleTimes[obs2_UP_activated_idx]
    #intervals=[20]
    if len(obs1_UP_activated_idx)>0:
        obs1_up_StartAngle= angleTimes[:,1][obs1_UP_activated_idx]+startAngle
        obs1_up_EndAngle= angleTimes[:,1][obs1_UP_activated_idx]+endAngle
        obs1_up_StartAngle = correctAnglesOutsideRecordedRange(obs1_up_StartAngle, angleTimes[:, 1])
        obs1_up_EndAngle = correctAnglesOutsideRecordedRange(obs1_up_EndAngle, angleTimes[:, 1])
        obs1_up_StartIdx = findClosestIdxToAngle(obs1_up_StartAngle, angleTimes[:, 1])
        obs1_up_EndIdx = findClosestIdxToAngle(obs1_up_EndAngle, angleTimes[:, 1])
        # determine time
        obs1_up_StartTime = angleTimes[:, 0][obs1_up_StartIdx]
        obs1_up_EndTime = angleTimes[:, 0][obs1_up_EndIdx]
        obstacleUPdic['obstacle1UPAndActivatedStartEndIdx'] = np.column_stack((obs1_up_StartIdx, obs1_up_EndIdx))
        obstacleUPdic['obstacle1UPAndActivatedStartEndTime'] = np.column_stack((obs1_up_StartTime, obs1_up_EndTime))
    else:
        obstacleUPdic['obstacle1UPAndActivatedStartEndIdx'] = np.zeros((len(obs2_UP_activated_idx),2))
        obstacleUPdic['obstacle1UPAndActivatedStartEndTime'] = np.zeros((len(obs2_UP_activated_idx),2))
        # pdb.set_trace()
        angleTimeObs1Up = np.zeros((len(angleTimes[obs2_UP_activated_idx]),2))

    if len(obs2_UP_activated_idx)>0:
        obs2_up_StartAngle= angleTimes[:, 1][obs2_UP_activated_idx]+startAngle
        obs2_up_EndAngle = angleTimes[:, 1][obs2_UP_activated_idx]+endAngle
        obs2_up_StartAngle = correctAnglesOutsideRecordedRange(obs2_up_StartAngle, angleTimes[:, 1])
        obs2_up_EndAngle = correctAnglesOutsideRecordedRange(obs2_up_EndAngle, angleTimes[:, 1])
        obs2_up_StartIdx = findClosestIdxToAngle(obs2_up_StartAngle, angleTimes[:, 1])
        obs2_up_EndIdx = findClosestIdxToAngle(obs2_up_EndAngle, angleTimes[:, 1])
        # determine time
        obs2_up_StartTime = angleTimes[:, 0][obs2_up_StartIdx]
        obs2_up_EndTime = angleTimes[:, 0][obs2_up_EndIdx]
        obstacleUPdic['obstacle2UPAndActivatedStartEndIdx'] = np.column_stack((obs2_up_StartIdx, obs2_up_EndIdx))
        obstacleUPdic['obstacle2UPAndActivatedStartEndTime'] = np.column_stack((obs2_up_StartTime, obs2_up_EndTime))
    else:
        obstacleUPdic['obstacle2UPAndActivatedStartEndIdx'] = np.zeros((len(obs1_UP_activated_idx),2))
        obstacleUPdic['obstacle2UPAndActivatedStartEndTime'] = np.zeros((len(obs1_UP_activated_idx),2))
        angleTimeObs2Up = np.zeros((len(angleTimes[obs1_UP_activated_idx]), 2))
    obstacleUPdic['angleTimeObs2Up']=angleTimeObs2Up
    obstacleUPdic['angleTimeObs1Up'] = angleTimeObs1Up

    return obstacleUPdic

######################################################
#startEndExposureTimeAni, videoIdxAni, wTimes, angleTimes, absoluteSignal, obstacle1,obstacle2,
def determineObstacleFrames(startEndExposureTime, videoIdx, angleTimes,absoluteSignal, obstacle1, obstacle2, angleRange=None):

    def getObstacleAngleAtTimes(angleTimes,mFrameTimesRange,obsTimeAngleTop):
        allObsAngles = []
        for n in range(len(mFrameTimesRange)):
            idxT = np.argmin(np.abs(angleTimes[:,0]-mFrameTimesRange[n]))
            angle = angleTimes[idxT,1]
            obsAngle = angle-obsTimeAngleTop[1]
            allObsAngles.append(obsAngle)
        return allObsAngles

    obstacleUPdic = determineObstacleTimesAngles(angleTimes,absoluteSignal, obstacle1, obstacle2,angleRange=angleRange)

    #determine index in video
    obsVideoIdx = []
    obsVideoTimes = []
    obsID = []
    obsNumber=[]
    obsAngle = []
    midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.
    # try:
    allObsStartEndTimes = np.row_stack((obstacleUPdic['obstacle1UPAndActivatedStartEndTime'],obstacleUPdic['obstacle2UPAndActivatedStartEndTime']))
    allObstStartEndTimeAngle = np.row_stack((obstacleUPdic['angleTimeObs1Up'],obstacleUPdic['angleTimeObs2Up'])) #((angleTimeObs1Up,angleTimeObs2Up))

    # except ValueError:
    #     pdb.set_trace()
    allObstIdentityObsNumer = np.concatenate((np.ones(len(obstacleUPdic['obstacle1UPAndActivatedStartEndTime'])),2*np.ones(len(obstacleUPdic['obstacle2UPAndActivatedStartEndTime']))))
    #print('allObsStartEndTimes',len(allObsStartEndTimes),'allObstIdentityObsNumer',len(allObstIdentityObsNumer),'allObstStartEndTimeAngle',len(allObstStartEndTimeAngle))
    #pdb.set_trace()
    allObsData = np.column_stack((allObsStartEndTimes,allObstIdentityObsNumer,allObstStartEndTimeAngle))
    allObsData = allObsData[allObsData[:, 0].argsort()] # sort based on first column
    # pdb.set_trace()
    c=0
    for i in range(len(allObsData)):
        if (allObsData[i,0]!=0 and allObsData[i,1]!=0) :
            if (allObsData[i, 0] > 10) and (allObsData[i, 0] < 52):
                # pdb.set_trace()
                mask = (midFrameTimes>=allObsData[i,0])&(midFrameTimes<=allObsData[i,1])
                #print(i, 'th obstacle', np.sum(mask), 'time points')
                mFrameTimesRange = midFrameTimes[mask]
                obsVideoTimes.extend(mFrameTimesRange)
                obsVideoIdx.extend(videoIdx[mask])
                # pdb.set_trace()
                obsID.extend(np.repeat(allObsData[i,2],np.sum(mask)))
                obsAngleAtFrameTimes = getObstacleAngleAtTimes(angleTimes,mFrameTimesRange,allObsData[i,3:])
                obsAngle.extend(obsAngleAtFrameTimes)
                obsNumber.extend(np.repeat((c+1),np.sum(mask)))
                c += 1
        #print(obsVideoTimes, obsVideoIdx)
    #print()
    #pdb.set_trace()
    #obsVideoTimes = obsVideoTimes.sort()
    #obsVideoIdx = obsVideoIdx.sort()
    obsVideoIdxArr = np.asarray(obsVideoIdx)
    obsVideoTimesArr = np.asarray(obsVideoTimes)
    obsAngleArr = np.asarray(obsAngle)
    obsID = np.asarray(obsID)
    obsNumber = np.asarray(obsNumber)
    obstacleUPdic['obstacleVideoTimes'] = obsVideoTimesArr
    obstacleUPdic['obsVideoIdx'] = obsVideoIdxArr
    obstacleUPdic['obsVideoAngle'] = obsAngleArr
    obstacleUPdic['obsID'] = obsID
    obstacleUPdic['obsNumber'] = obsNumber
    #print(len(obsVideoTimes),len(obsVideoIdx))
    #pdb.set_trace()
    # obs1Mask = (obstacleUPdic['obstacle2UPAndActivatedStartEndTime'][:,0]>
    # obsVideoIdx.append(videoIdx[])
    # Idx_obs1_up_vid = np.empty((len(Time_obs1_up_1),2))
    # Idx_obs2_up_vid = np.empty((len(Time_obs2_up_1),2))
    # Time_obstacle_list = [Time_obs1_up_1, Time_obs1_up_2, Time_obs2_up_1, Time_obs2_up_2]
    # for e in range(len(Time_obstacle_list)):
    #     for f in range(len(Time_obstacle_list[e])):
    #         if e==0:
    #             Idx_obs1_up_vid[f,0]=(np.argmin(np.abs(startEndExposurepIdx[:, 0] - Time_obstacle_list[e][f])))
    #         elif e==1:
    #             Idx_obs1_up_vid[f, 1] = (np.argmin(np.abs(startEndExposurepIdx[:, 0] - Time_obstacle_list[e][f])))
    #         if e==2:
    #             Idx_obs2_up_vid[f,0]=(np.argmin(np.abs(startEndExposurepIdx[:, 0] - Time_obstacle_list[e][f])))
    #         elif e==3:
    #             Idx_obs2_up_vid[f, 1] = (np.argmin(np.abs(startEndExposurepIdx[:, 0] - Time_obstacle_list[e][f])))
    #
    #
    # Idx_obs1_up_vid_range=[]
    # Idx_obs2_up_vid_range=[]
    # for n in range(len(Idx_obs1_up_vid)):
    #     Idx_obs1_up_vid_range.append(np.arange(Idx_obs1_up_vid[n][0],Idx_obs1_up_vid[n][1],1))
    # for m in range(len(Idx_obs2_up_vid)):
    #     Idx_obs2_up_vid_range.append(np.arange(Idx_obs2_up_vid[m][0],Idx_obs2_up_vid[m][1],1))
    #
    #
    # Time_obs1_up_vid=np.empty((len(Time_obs1_up_1),2))
    # Time_obs2_up_vid = np.empty((len(Time_obs2_up_1), 2))
    # Time_obs1_up_vid[:,0]=startEndExposurepIdx[:, 0][Idx_obs1_up_vid[:,0].astype(int)]
    # Time_obs1_up_vid[:, 1] = startEndExposurepIdx[:, 0][Idx_obs1_up_vid[:, 1].astype(int)]
    # Time_obs2_up_vid[:, 0] = startEndExposurepIdx[:, 0][Idx_obs2_up_vid[:, 0].astype(int)]
    # Time_obs2_up_vid[:, 1] = startEndExposurepIdx[:, 0][Idx_obs2_up_vid[:, 1].astype(int)]
    #
    # Time_obs1_up_vid_range=[]
    # Time_obs2_up_vid_range=[]
    # for n in range(len(Time_obs1_up_vid)):
    #     Time_obs1_up_vid_range.append(startEndExposurepIdx[:, 0][(Idx_obs1_up_vid[n,0].astype(int)):(Idx_obs1_up_vid[n,1].astype(int))])
    # for m in range(len(Time_obs2_up_vid)):
    #     Time_obs2_up_vid_range.append(startEndExposurepIdx[:, 0][(Idx_obs2_up_vid[m,0].astype(int)):(Idx_obs2_up_vid[m,1].astype(int))])
    #
    #
    # obstacle_up_time_dic['Time_obs1']=Time_obs1_up_vid
    # obstacle_up_time_dic['Time_obs2'] = Time_obs2_up_vid
    # obstacle_up_time_dic['Time_range_obs1']=Time_obs1_up_vid_range
    # obstacle_up_time_dic['Time_range_obs2'] = Time_obs2_up_vid_range
    #
    # obstacle_up_Idx_dic['Idx_obs1'] = Idx_obs1_up_vid
    # obstacle_up_Idx_dic['Idx_obs2'] = Idx_obs2_up_vid
    # obstacle_up_Idx_dic['Idx_range_obs1'] = Idx_obs1_up_vid_range
    # obstacle_up_Idx_dic['Idx_range_obs2'] = Idx_obs2_up_vid_range

    # - we get start and end time during which wheel was [-20,20] degrees with obstacle up
    # - use the time bracket to determine which video frames took place during that time and extract video indicies in a new list containing only those obstacle indicies


    return (obsVideoTimesArr, obsVideoIdxArr, obsNumber, obsID, obstacleUPdic, obsAngleArr)


def findClosestIdxToTime(newTime, oldTime):
    indicies = []
    for i in range(len(newTime)):
        indicies.append(np.argmin(np.abs(oldTime - newTime[i])))
    return np.asarray(indicies)
def obstacleStartEndTimes(date, rec, startEndExposureTimeObstacleWhis, obsWhisNumber, obsWhisID, whiskerTouch, pawPos, obsSpeed,
                          newlinearspeedobs):
    obsTime = pawPos[0][:, 0]  # obstacle is at index 0 of pawPos array

    obsXtrack = pawPos[0][:, 1]
    obsYtrack = pawPos[0][:, 2]
    obsXSpeed = obsSpeed[0][:, 1]  # Index 0 = time, 1 = x speed, 2 = y speed
    obsSpeedTime = obsSpeed[0][:, 0]

    FLytrack = pawPos[2][:, 2]  # [index in PawPos array][x or y axis]
    FLytrackTime = pawPos[2][:, 0]  # [index in PawPos array][x or y axis, or time]
    FLxtrack = pawPos[8][:, 1]
    FLxtrackTime = pawPos[8][:, 0]  # [index in PawPos array][x or y axis, or time]
    FRytrack = pawPos[3][:, 2]
    FRytrackTime = pawPos[3][:, 0]  # [index in PawPos array][x or y axis, or time]
    FRxtrack = pawPos[9][:, 1]
    FRxtrackTime = pawPos[9][:, 0]  # [index in PawPos array][x or y axis, or time]

    current_value = 1
    FRxSpeed = obsSpeed[9][:, 1]  # Index 0 = time, 1 = x speed, 2 = y speed
    FRxSpeedTime = obsSpeed[9][:, 0]
    FRySpeed = obsSpeed[3][:, 1]  # Index 0 = time, 1 = x speed, 2 = y speed
    FRySpeedTime = obsSpeed[3][:, 0]

    FLxSpeed = obsSpeed[8][:, 1]  # Index 0 = time, 1 = x speed, 2 = y speed
    FLxSpeedTime = obsSpeed[8][:, 0]
    FLySpeed = obsSpeed[2][:, 1]  # Index 0 = time, 1 = x speed, 2 = y speed
    FLySpeedTime = obsSpeed[2][:, 0]
    obsNumIndex = 0

    totalNumObs = np.unique(obsWhisNumber)
    # print(f'Expected number of obstacles: {len(totalNumObs)}')
    totalNumObs = totalNumObs[1:-1]

    def findClosestIdxToTime(startEndExposureTimeObstacleWhis, obsTime):
        indicies = []
        for i in range(len(startEndExposureTimeObstacleWhis)):
            indicies.append(np.argmin(np.abs(obsTime - startEndExposureTimeObstacleWhis[i])))
        return np.asarray(indicies)



    obsTimeDict = {}
    for value in totalNumObs:
        obsTimeDict[value] = {}
        wheelDict = {}

    num_changes = 0  # variable for tracking number of 0 to 1 changes (obs detection)
    obsTimes = []

    for value in totalNumObs:

        # add mask time superior to 10 and inferior to 52
        intervalMask = (startEndExposureTimeObstacleWhis > 10) & (startEndExposureTimeObstacleWhis < 52)
        obsTimeMask = (obsTime > 10) & (obsTime < 52)
        FLyobsTimeMask = (FLytrackTime > 10) & (FLytrackTime < 52)
        FLxobsTimeMask = (FLxtrackTime > 10) & (FLxtrackTime < 52)
        FRyobsTimeMask = (FRytrackTime > 10) & (FRytrackTime < 52)
        FRxobsTimeMask = (FRxtrackTime > 10) & (FRxtrackTime < 52)


        FLySpeedTimeMask = (FLySpeedTime > 10) & (FLySpeedTime < 52)
        FLxSpeedTimeMask = (FLxSpeedTime > 10) & (FLxSpeedTime < 52)
        FRySpeedTimeMask = (FRySpeedTime > 10) & (FRySpeedTime < 52)
        FRxSpeedTimeMask = (FRxSpeedTime > 10) & (FRxSpeedTime < 52)
        # define a mask to get subvideos from obswhisnB array
        mask = obsWhisNumber[intervalMask] == value

        subVideoLinearSpeedObs = newlinearspeedobs[intervalMask][mask]

        obsTime = obsTime[obsTimeMask]
        # FLytrackTime = FLytrackTime[obsTimeMask]
        # FLxtrackTime = FLxtrackTime[obsTimeMask]
        # FRytrackTime = FRytrackTime[obsTimeMask]
        # FRxtrackTime = FRxtrackTime[obsTimeMask]
        FLytrackTimeMasked = FLytrackTime[FLyobsTimeMask]
        FLxtrackTimeMasked = FLxtrackTime[FLxobsTimeMask]
        FRytrackTimeMasked = FRytrackTime[FRyobsTimeMask]
        FRxtrackTimeMasked = FRxtrackTime[FRxobsTimeMask]

        FLytrackMasked = FLytrack[FLyobsTimeMask]
        FLxtrackMasked = FLxtrack[FLxobsTimeMask]
        FRytrackMasked = FRytrack[FRyobsTimeMask]
        FRxtrackMasked = FRxtrack[FRxobsTimeMask]


        FLySpeedTimeMasked = FLySpeedTime[FLySpeedTimeMask]
        FLxSpeedTimeMasked = FLxSpeedTime[FLxSpeedTimeMask]
        FRySpeedTimeMasked = FRySpeedTime[FRySpeedTimeMask]
        FRxSpeedTimeMasked = FRxSpeedTime[FRxSpeedTimeMask]

        FLySpeedMasked = FLySpeed[FLySpeedTimeMask]
        FLxSpeedMasked = FLxSpeed[FLxSpeedTimeMask]
        FRySpeedMasked = FRySpeed[FRySpeedTimeMask]
        FRxSpeedMasked = FRxSpeed[FRxSpeedTimeMask]
        

        subVideoTime = startEndExposureTimeObstacleWhis[intervalMask][mask]

        # print('jkjhkhkjhkjh',value,subVideoTime)

        # find index of subvideos in obs trajectory time
        obsIdx = findClosestIdxToTime(subVideoTime, obsTime)
        FLytrackTime_obsIdx = findClosestIdxToTime(subVideoTime, FLytrackTimeMasked)
        FLxtrackTime_obsIdx = findClosestIdxToTime(subVideoTime, FLxtrackTimeMasked)
        FRytrackTime_obsIdx = findClosestIdxToTime(subVideoTime, FRytrackTimeMasked)
        FRxtrackTime_obsIdx = findClosestIdxToTime(subVideoTime, FRxtrackTimeMasked)
        #
        
        FLySpeedTime_obsIdx = findClosestIdxToTime(subVideoTime, FLySpeedTimeMasked)
        FLxSpeedTime_obsIdx = findClosestIdxToTime(subVideoTime, FLxSpeedTimeMasked)
        FRySpeedTime_obsIdx = findClosestIdxToTime(subVideoTime, FRySpeedTimeMasked)
        FRxSpeedTime_obsIdx = findClosestIdxToTime(subVideoTime, FRxSpeedTimeMasked)
        
        normalizedObsTime = obsTime - obsTime[0]
        normalizedSubVideoTime = subVideoTime - subVideoTime[0]



        # normalizedTime_mask = normalizedSubVideoTime > 0.4

        # normalizedSubVideoLinearSpeedObs = subVideoLinearSpeedObs[normalizedTime_mask]

        # maskedSubVideoTime = subVideoTime[normalizedTime_mask]
        # normalizedNormalizedSubVideoTime = normalizedSubVideoTime[normalizedTime_mask] #to remove the first 0.4 sec of video where obstacle trajectory plateaus

        try:
            subWhiskTouch = whiskerTouch['whisker_obstacle_touch'][intervalMask][mask]
            obsWhisIDSubVid = obsWhisID[intervalMask][mask]
            obsWhisIDSubVid = obsWhisIDSubVid[0]
            # date = date.replace('.', '_')
            # subWhiskTouch.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/deg_projects/Whisker_touch_analysis_deepethogram/CSV_obstacle_indexes/whiskerTouch_{value}.csv')

            subVideoObsXtrack = obsXtrack[obsIdx]
            subVideoObsYtrack = obsYtrack[obsIdx]
            obsXSpeedSubVideo = obsXSpeed[obsIdx]

            FLytrackMasked_obs = FLytrackMasked[FLytrackTime_obsIdx]  # extract trajectory when in obstacle range
            FLxtrackMasked_obs = FLxtrackMasked[FLxtrackTime_obsIdx]
            FRytrackMasked_obs = FRytrackMasked[FRytrackTime_obsIdx]
            FRxtrackMasked_obs = FRxtrackMasked[FRxtrackTime_obsIdx]

            FLytrackTimeMasked_obs = FLytrackTimeMasked[FLytrackTime_obsIdx]
            FLxtrackTimeMasked_obs = FLxtrackTimeMasked[FLxtrackTime_obsIdx]
            FRytrackTimeMasked_obs = FRytrackTimeMasked[FRytrackTime_obsIdx]
            FRxtrackTimeMasked_obs = FRxtrackTimeMasked[FRxtrackTime_obsIdx]
            
            FLytrackTimeNorm_obs = FLytrackTimeMasked_obs-FLytrackTimeMasked_obs[0]
            FLxtrackTimeNorm_obs = FLxtrackTimeMasked_obs-FLxtrackTimeMasked_obs[0]
            FRytrackTimeNorm_obs = FRytrackTimeMasked_obs-FRytrackTimeMasked_obs[0]
            FRxtrackTimeNorm_obs = FRxtrackTimeMasked_obs-FRxtrackTimeMasked_obs[0]

            FLySpeedMasked_obs = FLySpeedMasked[FLySpeedTime_obsIdx]  # extract trajectory when in obstacle range
            FLxSpeedMasked_obs = FLxSpeedMasked[FLxSpeedTime_obsIdx]
            FRySpeedMasked_obs = FRySpeedMasked[FRySpeedTime_obsIdx]
            FRxSpeedMasked_obs = FRxSpeedMasked[FRxSpeedTime_obsIdx]

            FLySpeedTimeMasked_obs = FLySpeedTimeMasked[FLySpeedTime_obsIdx]
            FLxSpeedTimeMasked_obs = FLxSpeedTimeMasked[FLxSpeedTime_obsIdx]
            FRySpeedTimeMasked_obs = FRySpeedTimeMasked[FRySpeedTime_obsIdx]
            FRxSpeedTimeMasked_obs = FRxSpeedTimeMasked[FRxSpeedTime_obsIdx]

            FLySpeedTimeNorm_obs = FLySpeedTimeMasked_obs - FLySpeedTimeMasked_obs[0]
            FLxSpeedTimeNorm_obs = FLxSpeedTimeMasked_obs - FLxSpeedTimeMasked_obs[0]
            FRySpeedTimeNorm_obs = FRySpeedTimeMasked_obs - FRySpeedTimeMasked_obs[0]
            FRxSpeedTimeNorm_obs = FRxSpeedTimeMasked_obs - FRxSpeedTimeMasked_obs[0]

        except:
            pdb.set_trace()
            print(f"IndexError at obstacle {value}")

        if sum(subWhiskTouch) != 0:
            touch_index = np.where(subWhiskTouch == 1)[0][0]
            nb = np.where(np.diff(subWhiskTouch) < 0)[0]
            num_changes += len(nb)
            # for i in range(len(subWhiskTouch)):
            #     nb = np.where(np.diff(subWhiskTouch[i]) < 0)
            # num_changes = len(nb)
            # print(num_changes)
            # pdb.set_trace()

            real_touch_time = subVideoTime[
                touch_index]  # Real time at which whisker touches (aka whiskerTouch value = 1)
            normalized_touch_time = normalizedSubVideoTime[
                touch_index]  # Time at which whisker touches (aka whiskerTouch value = 1)

            obsXPosAtTouch = subVideoObsXtrack[touch_index]
            obsYPosAtTouch = subVideoObsYtrack[touch_index]

        else:
            print(f'No whisker touch detected for obstacle {value}')
            touch_index = np.nan
            normalized_touch_time = np.nan
            real_touch_time = np.nan
            obsXPosAtTouch = np.nan
            obsYPosAtTouch = np.nan

        obsTimes.append(subVideoTime)

        # normalized_obsXtrack = subVideoObsXtrack - obsXPosAtTouch

        normalizedToTouch_obsTime = normalizedSubVideoTime - normalized_touch_time

        normalizedToTouch_time_FLy = FLytrackTimeNorm_obs - normalized_touch_time

        normalizedToTouch_time_FLx = FLxtrackTimeNorm_obs - normalized_touch_time

        normalizedToTouch_time_FRy = FRytrackTimeNorm_obs - normalized_touch_time

        normalizedToTouch_time_FRx = FRxtrackTimeNorm_obs - normalized_touch_time

        speedTimeToTouch_time_FLy = FLySpeedTimeNorm_obs - normalized_touch_time

        speedTimeToTouch_time_FLx = FLxSpeedTimeNorm_obs - normalized_touch_time

        speedTimeToTouch_time_FRy = FRySpeedTimeNorm_obs - normalized_touch_time

        speedTimeToTouch_time_FRx = FRxSpeedTimeNorm_obs - normalized_touch_time

        # Obstacle Dictionary
        try:
            obsTimeDict[value]['subVideoTime'] = subVideoTime
            obsTimeDict[value]['normalizedToTouch_obsTime'] = normalizedToTouch_obsTime
            obsTimeDict[value]['normalizedToTouch_time_FLy'] = normalizedToTouch_time_FLy
            obsTimeDict[value]['normalizedToTouch_time_FLx'] = normalizedToTouch_time_FLx
            obsTimeDict[value]['normalizedToTouch_time_FRy'] = normalizedToTouch_time_FRy
            obsTimeDict[value]['normalizedToTouch_time_FRx'] = normalizedToTouch_time_FRx

            obsTimeDict[value]['speedTimeToTouch_time_FLy'] = speedTimeToTouch_time_FLy
            obsTimeDict[value]['speedTimeToTouch_time_FLx'] = speedTimeToTouch_time_FLx
            obsTimeDict[value]['speedTimeToTouch_time_FRy'] = speedTimeToTouch_time_FRy
            obsTimeDict[value]['speedTimeToTouch_time_FRx'] = speedTimeToTouch_time_FRx


            obsTimeDict[value]['FLytrackTimeMasked_obs'] = FLytrackTimeMasked_obs
            obsTimeDict[value]['FLxtrackTimeMasked_obs'] = FLxtrackTimeMasked_obs
            obsTimeDict[value]['FRytrackTimeMasked_obs'] = FRytrackTimeMasked_obs
            obsTimeDict[value]['FRxtrackTimeMasked_obs'] = FRxtrackTimeMasked_obs

            obsTimeDict[value]['obsID'] = int(obsWhisIDSubVid)
            # pdb.set_trace()
            obsTimeDict[value]['FLytrackTimeNorm_obs'] = FLytrackTimeNorm_obs
            obsTimeDict[value]['FLxtrackTimeNorm_obs'] = FLxtrackTimeNorm_obs
            obsTimeDict[value]['FRytrackTimeNorm_obs'] = FRytrackTimeNorm_obs
            obsTimeDict[value]['FRxtrackTimeNorm_obs'] = FRxtrackTimeNorm_obs

            obsTimeDict[value]['FLySpeedTimeNorm_obs'] = FLySpeedTimeNorm_obs
            obsTimeDict[value]['FLxSpeedTimeNorm_obs'] = FLxSpeedTimeNorm_obs
            obsTimeDict[value]['FRySpeedTimeNorm_obs'] = FRySpeedTimeNorm_obs
            obsTimeDict[value]['FRxSpeedTimeNorm_obs'] = FRxSpeedTimeNorm_obs


            obsTimeDict[value]['normalizedObsTime'] = normalizedObsTime

            obsTimeDict[value]['normalizedSubVideoTime'] = normalizedSubVideoTime
            # obsTimeDict[value]['normalized_obsXtrack'] = normalized_obsXtrack
            obsTimeDict[value]['normalized_touch_time'] = normalized_touch_time
            obsTimeDict[value]['real_touch_time'] = real_touch_time
            obsTimeDict[value]['subVideoObsXtrack'] = subVideoObsXtrack
            obsTimeDict[value]['subVideoObsYtrack'] = subVideoObsYtrack
            obsTimeDict[value]['obsXPosAtTouch'] = obsXPosAtTouch
            obsTimeDict[value]['obsYPosAtTouch'] = obsYPosAtTouch
            obsTimeDict[value]['obsXSpeedSubVideo'] = obsXSpeedSubVideo

            obsTimeDict[value]['wheelSpeed'] = subVideoLinearSpeedObs

            obsTimeDict[value]['FLytrackTimeNorm_obs'] = FLytrackTimeNorm_obs
            obsTimeDict[value]['normalizedToTouch_time_FLy'] = normalizedToTouch_time_FLy
            obsTimeDict[value]['normalizedToTouch_time_FLx'] = normalizedToTouch_time_FLx
            obsTimeDict[value]['normalizedToTouch_time_FRy'] = normalizedToTouch_time_FRy
            obsTimeDict[value]['normalizedToTouch_time_FRx'] = normalizedToTouch_time_FRx

            obsTimeDict[value]['FLytrackMasked_obs'] = FLytrackMasked_obs
            obsTimeDict[value]['FLxtrackMasked_obs'] = FLxtrackMasked_obs
            obsTimeDict[value]['FRytrackMasked_obs'] = FRytrackMasked_obs
            obsTimeDict[value]['FRxtrackMasked_obs'] = FRxtrackMasked_obs

            obsTimeDict[value]['FLytrackTimeNorm_obs'] = FLytrackTimeNorm_obs  # top
            obsTimeDict[value]['FLxtrackTimeNorm_obs'] = FLxtrackTimeNorm_obs  # bottom
            obsTimeDict[value]['FRytrackTimeNorm_obs'] = FRytrackTimeNorm_obs  # top
            obsTimeDict[value]['FRxtrackTimeNorm_obs'] = FRxtrackTimeNorm_obs  # bottom
            
            obsTimeDict[value]['FLySpeedMasked_obs'] = FLySpeedMasked_obs
            obsTimeDict[value]['FLxSpeedMasked_obs'] = FLxSpeedMasked_obs
            obsTimeDict[value]['FRySpeedMasked_obs'] = FRySpeedMasked_obs
            obsTimeDict[value]['FRxSpeedMasked_obs'] = FRxSpeedMasked_obs

            obsTimeDict[value]['FLySpeedTimeNorm_obs'] = FLySpeedTimeNorm_obs  # top
            obsTimeDict[value]['FLxSpeedTimeNorm_obs'] = FLxSpeedTimeNorm_obs  # bottom
            obsTimeDict[value]['FRySpeedTimeNorm_obs'] = FRySpeedTimeNorm_obs  # top
            obsTimeDict[value]['FRxSpeedTimeNorm_obs'] = FRxSpeedTimeNorm_obs  # bottom
            obsTimeDict['obsNumber'] = totalNumObs

            # pdb.set_trace()


        except IndexError:
            print("There is an index error!")
            # pdb.set_trace()
    # obsTimeDict['allObsArray']= {}
    # feature_list_key = ['subVideoObsXtrack', 'subVideoObsYtrack']
    #
    # for f in range(len(feature_list_key)):
    #     obsTimeDict['allObsArray'][feature_list_key[f]]=[]
    #
    # for f in range(len(feature_list_key)):
    #
    #     for value in totalNumObs:
    #         time_mask = obsTimeDict[value]['normalizedNormalizedSubVideoTime'] <= 1
    #         print(len(obsTimeDict[value][feature_list_key[f]][time_mask]))
    #         obsTimeDict['allObsArray'][feature_list_key[f]].append(obsTimeDict[value][feature_list_key[f]][time_mask])
    #     # obsTimeDict['avg_subVideoObsXtrack'] = np.mean(obsTimeDict['allObsArray']['subVideoObsXtrack'], axis=0)
    #     # obsTimeDict['avg_subVideoObsYtrack'] = np.mean(obsTimeDict['allObsArray']['subVideoObsYtrack'], axis=0)
    #     obsTimeDict['mean_time'] = normalizedNormalizedSubVideoTime[time_mask]

    return obsTimeDict, obsTimes

def linearizeTracks(tracks, pawTracks,rungMotion) :
    matplotlib.use('TkAgg')
    speedDiffThresh = 10  # cm/s Speed threshold, determine with variance
    minimalLengthOfSwing = 3 # number of frames @ 200 Hz
    minimalLengthOfStance = 4
    #thStance = 10
    #thSwing = 2
    trailingStart = 1
    trailingEnd = 1
    bounds = [1400, 5320]
    wheelCircumsphere = 80.4 # in cm
    # if not obstacle:
    #     pawIdx=[0,1,2,3]
    # else:
    #     #indicies of bottom paw tracks

    # error function : difference betweeen paw and wheel speed ; the inverse of the absolute difference is used to emphasize small values which would be the stance phases
    errfunc = lambda p, x1, y1, x2, y2, x3, y3, x4, y4: np.sum(1./np.abs(x1-p*y1)) + np.sum(1./np.abs(x2-p*y2))+ np.sum(1./np.abs(x3-p*y3)) + np.sum(1./np.abs(x4-p*y4))
    # guess some fit parameters
    p0 = 0.025
    # calculate wheel speed at the frame times : requires interpolation of the wheel speed
    interpAngle = interp1d(tracks[5][:,0],tracks[5][:,1])
    interp = interp1d(tracks[2], -tracks[1])
    forFit = []
    for i in range(len(pawTracks[-1])):
        # print(pawTracks[-1][i])
        mask = ((pawTracks[3][i][:,0])>=min(tracks[2])) & ((pawTracks[3][i][:,0])<=max(tracks[2]))
        newWheelSpeedAtPawTimes = interp(pawTracks[3][i][:,0][mask])
        #
        maskAngle = ((pawTracks[5][i][:,0])>=min(tracks[5][:,0])) & ((pawTracks[5][i][:,0])<=max(tracks[5][:,0]))
        newWheelAngleAtPawTimes = interpAngle(pawTracks[5][i][:,0][maskAngle])
        newX  = (pawTracks[5][i][:,1][maskAngle])*0.025 + (newWheelAngleAtPawTimes*80./360.)
        newY  = (pawTracks[5][i][:,2][maskAngle])*0.025#- (pawTracks[5][i][:,1][maskAngle][0])*0.025                                  #   pawPos[i][:,0][maskAngle]
        forFit.append([newWheelSpeedAtPawTimes, pawTracks[3][i][:,2][mask],pawTracks[3][i][:,0][mask],mask,np.array(pawTracks[3][i][:,4][mask],dtype=int),np.column_stack((pawTracks[5][i][:,0][maskAngle],newX,newY))])
    return forFit



def calculate_mean_sem(arrays):
    newArrayX=[]
    newArrayY=[]
    for n in range(len(arrays)):

        for p in range(len(arrays[n])):
            for a in range(len(arrays[n][p])) :

                if p==0:
                    newArrayX.append(arrays[n][p] )
                else:
                    newArrayY.append(arrays[n][p])

    meanX_array=np.nanmean(newArrayX, axis=0)
    meanY_array = np.nanmean(newArrayY, axis=0)

    semX_array=stats.sem(newArrayX,axis=0,nan_policy='omit')
    semY_array = stats.sem(newArrayY, axis=0, nan_policy='omit')

    mean_array=[meanX_array,meanY_array]
    sem_array = [semX_array,semY_array]

    return mean_array, sem_array
def bin_x_and_get_corresponding_y(x_values, y_values, num_bins):
    bin_edges = np.linspace(0, x_values.max(), num_bins-1)
    #
    bin_indices = np.digitize(x_values, bin_edges)

    bin_sum = np.bincount(bin_indices, weights=y_values)
    bin_count = np.bincount(bin_indices)
    bin_average = bin_sum / bin_count

    return bin_edges, bin_average
def getMeanTrajectoriesObstacle(recordingsM):
    non_obstacleFR = []
    obstacleFR = []
    sessionMeansObs = []
    sessionMeansNonObs = []
    dates=[]
    swingDic={}
    for f in range(len(recordingsM)):
        swingDic[f] = {}
        for r in range(len(recordingsM[f])):
            swingDic[f][r] = {}
    for f in range(len(recordingsM)):
        non_obstacleFR_binned = []
        obstacleFR_binned = []
        dates.append(recordingsM[f][4][0][1][:10])
        swingDic[f]['mean_FR_obsArray'] =[]
        swingDic[f]['mean_FR_non_obsArray'] =[]
        for r in range(len(recordingsM[f])):

            if r < 10:
                print(recordingsM[f][4][r][1], recordingsM[f][4][r][2])
                # forFit.append([newWheelSpeedAtPawTimes, pawTracks[3][i][:,2][mask],pawTracks[3][i][:,0][mask],mask,np.array(pawTracks[3][i][:,4][mask],dtype=int),np.column_stack((pawTracks[5][i][:,0][maskAngle],newX,newY))])
                forFitAll = recordingsM[f][4][r][-1]


                rawPawPos=recordingsM[f][4][r][-2]
                swingP = recordingsM[f][4][r][3]
                # FL_X_swing = []
                # FL_x_swing_Obs = []
                # ['obstacle', 'nose_top', 'front_left_top', 'front_right_top', 'hind_left_top', 'hind_right_top',
                #  'tail_base_top', 'nose_bottom', 'front_left_bottom', 'front_right_bottom', 'hind_left_bottom',
                #  'hind_right_bottom', 'tail_base_bottom']

                # FL_X_time = forFitAll[8][5][:, 0]
                # FL_X_position = forFitAll[8][5][:, 1]
                #
                # FL_Z_time = forFitAll[2][5][:, 0]
                # FL_Z_position = forFitAll[2][5][:, 2]  # [::-1]

                FR_Z_time = forFitAll[3][-1][:, 0]
                FR_X_time = forFitAll[9][-1][:, 0]
                # zTimemask = (FR_Z_time > 10) & (FR_Z_time < 52)
                # FR_Z_time = FR_Z_time  # [zTimemask]
                FR_Z_position = forFitAll[3][-1][:, 2]  # [zTimemask]

                # xTimemask = (FR_X_time > 10) & (FR_X_time < 52)
                FR_X_position = forFitAll[9][-1][:, 1]  # [xTimemask]


                raw_FR_X_pos=rawPawPos[9][:,1]
                raw_FR_X_time = rawPawPos[9][:, 0]
                raw_FR_Z_pos=rawPawPos[3][:,2]

                #
                # xposition = [FL_X_position, FR_X_position]
                # zposition = [FL_Z_position, FR_Z_position]

                recTimes = recordingsM[f][4][r][-3][1][2]
                idxSwings = swingP[1][1]
                rungNumbers = swingP[1][2][1:]
                rungNumbers= np.asarray(rungNumbers)
                obsIdentity = swingP[1][-1][1:]
                obsIdentity= np.asarray(obsIdentity)
                idxSwings = np.asarray(idxSwings)
                try:
                    obsSwing=idxSwings[obsIdentity]
                except:
                    print('missmatch between rung number array and swing number array')
                    swingNb=len(idxSwings)
                    stanceNb=len(rungNumbers)
                    if swingNb>stanceNb:
                        lim=swingNb-stanceNb

                        obsSwing = idxSwings[:-lim][obsIdentity]
                    else:
                        lim=stanceNb-swingNb
                        obsSwing = idxSwings[obsIdentity[:-lim]]

                mask = (recTimes[idxSwings[:, 0]] > 10.) & (recTimes[idxSwings[:, 0]] < 52)
                idxSwings = idxSwings[mask]

                # rungNumbers=rungNumbers[mask]
                # obsIdentity=obsIdentity[mask]
                # pdb.set_trace()
                swingDic[f][r]['obsIdentity'] = obsIdentity
                swingDic[f][r]['idxSwings']=idxSwings
                swingDic[f][r]['recTimes']=recTimes

                swingDic[f][r]['raw_FR_X_pos']=raw_FR_X_pos
                swingDic[f][r]['raw_FR_X_time'] = raw_FR_X_time
                swingDic[f][r]['raw_FR_Z_pos']=raw_FR_Z_pos

                swingDic[f][r]['FR_X_position']=FR_X_position
                swingDic[f][r]['FR_X_time'] = FR_X_time
                swingDic[f][r]['FR_Z_position']=FR_Z_position
                swingDic[f][r]['FR_Z_time'] = FR_Z_time


                swingDic[f][r]['idxStartX']=[]
                swingDic[f][r]['idxEndX'] = []

                swingDic[f][r]['idxStartZ']=[]
                swingDic[f][r]['idxEndZ'] = []

                swingDic[f][r]['normXswing']=[]
                swingDic[f][r]['normZswing'] = []

                swingDic[f][r]['normXswing']=[]
                swingDic[f][r]['normZswing'] = []

                swingDic[f][r]['normXswing_bin']=[]
                swingDic[f][r]['normZswing_bin'] = []
                
                swingDic[f][r]['normXswing_before']=[]
                swingDic[f][r]['normZswing_before'] = []
                
                swingDic[f][r]['normXswing_before_bin']=[]
                swingDic[f][r]['normZswing_before_bin'] = []
                swingDic[f][r]['obs_swingIdx']=[]
                swingDic[f][r]['non_obs_swingIdx'] = []
                swingDic[f][r]['obs_swingTime']=[]
                swingDic[f][r]['non_obs_swingStartTime']=[]
                swingDic[f][r]['non_obs_swingEndTime']=[]
                swingDic[f][r]['obs_swingStartTime']=[]
                swingDic[f][r]['obs_swingEndTime']=[]
                # obsINb=np.count_nonzero(np.diff(obsIdentity))
                # obsNormalNb=len(np.unique(obstacleUPdic['obsNumber']))
                # if obsINb > obsNormalNb:
                #     print('expected obs', obsINb, obsNormalNb)
                #     pdb.set_trace()

                # ax = plt.subplot(gssub[0])
                # ax1=plt.subplot(gssub[1])

                # ax.plot(FR_X_time,FR_X_position, c='k')
                # ax1.plot(FR_Z_time, FR_Z_position, c='k')
                # pdb.set_trace()
                obs=0
                for s in range(len(idxSwings) - 1):
                    # if (forFitAll[9][5][s, 0]>10) and (forFitAll[9][5][s, 0]<52):

                    # idxStart = idxSwings[s, 0]
                    # idxEnd = idxSwings[s, 1]
                    idxStanceEnd = idxSwings[s + 1, 0]
                    # if (pawPos[8][n,0]>10) and (pawPos[8][n,0]<52):
                    idxStartX = np.argmin(np.abs(FR_X_time - recTimes[idxSwings[s][0]]))
                    idxEndX = np.argmin(np.abs(FR_X_time - recTimes[idxSwings[s][1]]))
                    idxStartZ = np.argmin(np.abs(FR_X_time - recTimes[idxSwings[s][0]]))
                    idxEndZ = np.argmin(np.abs(FR_X_time - recTimes[idxSwings[s][1]]))
                    #
                    # idxSwingStart = np.argmin(np.abs(swingDic[f][r]['raw_FR_X_time'] - swingDic[f][r]['recTimes'][
                    #     swingDic[f][r]['idxSwings'][n][0]]))
                    # idxStanceStart = np.argmin(np.abs(swingDic[f][r]['raw_FR_X_time'] - swingDic[f][r]['recTimes'][
                    #     swingDic[f][r]['idxSwings'][n][1]]))
                    # ax.plot(FR_X_time[idxStartX:idxEndX],FR_X_position[idxStartX:idxEndX], c='red')
                    # ax1.plot(FR_Z_time[idxStartZ:idxEndZ], FR_Z_position[idxStartZ:idxEndZ], c='red')
                    rangeStartX=idxStartX-5
                    rangeEndX=idxEndX

                    rangeStartZ=idxStartZ-5
                    rangeEndZ=idxEndZ

                    xSwing = FR_X_position[rangeStartX :rangeEndX]
                    zSwing = FR_Z_position[rangeStartZ :rangeEndZ]
                    xSwingStart = FR_X_position[rangeStartX]
                    zSwingStart = FR_Z_position[rangeStartZ]

                    swingStartTime = FR_X_time[rangeStartX]
                    swingEndTime = FR_X_time[rangeEndX]



                    normXswing = xSwing - xSwingStart
                    normZswing = zSwing - zSwingStart

                    xSwing_before = FR_X_position[idxStartX - 2:idxEndX-1]
                    zSwing_before = FR_Z_position[idxStartZ - 2:idxEndZ-1]
                    xSwingStart_before = FR_X_position[idxStartX - 2]
                    zSwingStart_before = FR_Z_position[idxStartZ - 2]

                    normXswing_before = xSwing_before - xSwingStart_before
                    normZswing_before = zSwing_before - zSwingStart_before

                    swingDic[f][r]['idxStartX'].append(idxStartX)
                    swingDic[f][r]['idxEndX'].append(idxEndX)

                    swingDic[f][r]['idxStartZ'].append(idxStartZ)
                    swingDic[f][r]['idxEndZ'].append(idxEndZ)

                    swingDic[f][r]['normXswing'].append(normXswing)
                    swingDic[f][r]['normZswing'].append(normZswing)

                    # swingDic[f][r]['normXswing'].append(normXswing)
                    # swingDic[f][r]['normZswing'].append(idxEndX)


                    swingDic[f][r]['normXswing_before'].append(normXswing_before)
                    swingDic[f][r]['normZswing_before'].append(normZswing_before)



                    if len(normXswing) == len(normZswing):
                        binnedXswing, binnedZswing = bin_x_and_get_corresponding_y(normXswing, normZswing,
                                                                                                10)
                        normXswing_before_bin,normZswing_before_bin=bin_x_and_get_corresponding_y(normXswing_before,normZswing_before,10)

                        swingDic[f][r]['normXswing_before_bin'].append(normXswing_before_bin)
                        swingDic[f][r]['normZswing_before_bin'].append(normZswing_before_bin)
                        swingDic[f][r]['normXswing_bin'].append(binnedXswing)
                        swingDic[f][r]['normZswing_bin'].append(binnedZswing)
                        for o in range(len(obsSwing)):
                            if (idxSwings[s][0] == obsSwing[o][0]):
                                obstacleFR.append([normXswing, normZswing])
                                obstacleFR_binned.append([binnedXswing, binnedZswing])
                                swingDic[f][r]['obs_swingIdx'].append(s)
                                swingDic[f][r]['obs_swingStartTime'].append(swingStartTime)
                                swingDic[f][r]['obs_swingEndTime'].append(swingEndTime)
                                obs+=1
                                print('obstacle swing detected at session',f, 'trial', r, 'obstacle nb=', obs)

                            else:

                                non_obstacleFR_binned.append([binnedXswing, binnedZswing])

                                non_obstacleFR.append([normXswing, normZswing])
                                swingDic[f][r]['non_obs_swingIdx'].append(s)
                                swingDic[f][r]['non_obs_swingStartTime'].append(swingStartTime)
                                swingDic[f][r]['non_obs_swingEndTime'].append(swingEndTime)

        mean_FR_obsArray, sem_FR_obsArray = calculate_mean_sem(obstacleFR_binned)
        mean_FR_non_obsArray, sem_FR_non_obsArray = calculate_mean_sem(non_obstacleFR_binned)
        swingDic[f]['mean_FR_obsArray'].append(mean_FR_obsArray)
        swingDic[f]['mean_FR_non_obsArray'].append(mean_FR_non_obsArray)
        sessionMeansObs.append(mean_FR_obsArray)
        sessionMeansNonObs.append(mean_FR_non_obsArray)
    obstacleTrajDic={}
    obstacleTrajDic['sessionMeanObs']=sessionMeansObs
    obstacleTrajDic['sessionMeanNonObs']=sessionMeansNonObs
    obstacleTrajDic['date']=dates
    return obstacleTrajDic, swingDic

#######################################################
def coordinateTransformRotationToLinear(points):
    R = 1536.405/2
    #point = np.array([518,600-81])

    if np.all(points[1]<300): # points are in the coordinate system with the origin in the upper left for the y-axis
        points[1] = 600. - points[1]

    origin = np.array([370.677,-360.308])

    leftPoint = np.array([0,600-287.8])

    py = np.sqrt(np.sum((points-origin[:,np.newaxis])**2,axis=0)) - R

    vectorA = leftPoint - origin
    vectorB = points - origin[:,np.newaxis]

    px = R*(np.arccos(np.dot(vectorA,vectorB)/(np.sqrt(np.sum(vectorA**2))*np.sqrt(np.sum(vectorB**2,axis=0)))))

    #print('the new coordinates are :', px,py)

    return (px,py)

import h5py

def readEMG(path):
    try:
        # Initialize an empty dictionary to store the EMG data
        EMG = {}

        # Open the HDF5 file
        with h5py.File(path, 'r') as file:
            # Assuming your data is stored in a dataset named 'data' within the HDF5 file
            data = file['data']

            # Calculate the number of channels
            EMG['nChan'] = data.shape[0]

            if EMG['nChan'] > 0:
                for n in range(EMG['nChan']):
                    currentKey = f'current_chan{n}'
                    EMG[currentKey] = data[n]
            else:
                EMG['current_chan0'] = data[0]

            # Extract other relevant data, e.g., 'time' dataset
            time_dataset = file['info/1/values']
            EMG['time'] = time_dataset[()]

        return EMG

    except Exception as e:
        print(f"Error: {str(e)}")
        return None



