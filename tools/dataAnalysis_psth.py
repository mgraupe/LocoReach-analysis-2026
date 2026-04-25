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
#from numpy import trapz
import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import tools.groupAnalysis as groupAnalysis

#######################
def getLaserActivationStartAndEnd(time,data):
    laserDict = {}
    difference = np.diff(data)  # calculate difference
    laserDict['laserStartIdx'] = np.arange(len(data))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
    laserDict['laserEndIdx'] = np.arange(len(data))[np.concatenate((np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
    laserDict['laserStartTime'] = time[laserDict['laserStartIdx']]
    laserDict['laserEndTime'] = time[laserDict['laserEndIdx']]
    return laserDict

########################
def determineLaserActivationDistribution(successRate, idxSwings, recTimes, pawPos, pawID, laserDict):
    # successRate[case]['normStepCycleBins'] = np.zeros(201)
    # successRate[case]['normTimeStepCycle'] =  np.linspace(0,2,201)
    normTimeStepCycle = successRate['normTimeStepCycle']
    unmatchedSwingCases = 0
    unmatchedStanceCases = 0
    for j in range(len(idxSwings) - 2):
        idxStart = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][0]]))
        idxEnd = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][1]]))
        idxStartNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][0]]))
        idxEndNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][1]]))
        idxStartNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][0]]))
        idxEndNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][1]]))
        swingNStart = pawPos[pawID][idxStart, 0]
        swingNEnd = pawPos[pawID][idxEnd, 0]
        swingNPOStart = pawPos[pawID][idxStartNext, 0]
        swingNPOEnd = pawPos[pawID][idxEndNext, 0]
        swingNPTStart = pawPos[pawID][idxStartNNext, 0]
        swingNPTEnd = pawPos[pawID][idxEndNNext, 0]
        laserActivationTimeSwing = np.argwhere((laserDict['laserStartTime'] >= swingNStart) & (laserDict['laserStartTime'] < swingNEnd))
        laserActivationTimeStance = np.argwhere((laserDict['laserStartTime'] >= swingNEnd) & (laserDict['laserStartTime'] < swingNPOStart))
        if len(laserActivationTimeSwing) > 0:
            successRate['swingWithLaserActivation'] += 1
            successRate['swingWithLaserIxd'].append(j)
        else:
            successRate['swingNoLaserActivation'] += 1
        # if len(laserActivationTimeSwing)>1 :
        #     print(laserActivationTimeSwing)
        #     print('more than one activations during the swing')
        #     pdb.set_trace()
        if len(laserActivationTimeStance) > 0: successRate['stanceWithLaserActivation'] += 1
        # if len(laserActivationTimeStance)>1:
        #     print(laserActivationTimeStance)
        #     print('more than one activations during the stance')
        #     pdb.set_trace()
        swingDuration = swingNEnd - swingNStart
        stanceDuration = swingNPOStart - swingNEnd
        swingNextDuration = swingNPOEnd - swingNPOStart
        stanceNextDuration = swingNPTStart - swingNPOEnd
        for k in laserActivationTimeSwing:
            addBins = True
            normStart = (laserDict['laserStartTime'][k] - swingNStart) / swingDuration
            start_bin = np.digitize(normStart, normTimeStepCycle[:101]) - 1
            if laserDict['laserEndTime'][k] < swingNEnd:
                normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / swingDuration
                assert normDuration < 1, print('The normalized duration in case one should be smaller than one!')
                normEnd = normStart + normDuration
                end_bin = np.digitize(normEnd, normTimeStepCycle) - 1  # unique_bins = np.arange(start_bin, end_bin + 1)  # normStepCycleBins[np.unique(unique_bins)] += 1
                unique_bins = np.arange(start_bin, end_bin + 1)
            elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] < swingNPOStart):
                # normDuration = (laserDict['laserEndTime'][k]-laserDict['laserStartTime'][k])/swingDuration
                normDurationStance = (laserDict['laserEndTime'][k] - swingNEnd) / stanceDuration
                assert normDurationStance < 1, print('The normalized duration in case two should be smaller than one!')
                # if normDurationStance < 1 : print('It\'s happening again in case two!')
                # start_bin = np.digitize(0, normTimeStepCycle[100:]) - 1 + 100
                end_bin = np.digitize(1. + normDurationStance, normTimeStepCycle) - 1
                unique_bins = np.arange(start_bin, end_bin + 1)
            elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                normDurationStance = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                assert normDurationStance < 1, print('The normalized duration %s in case twoB should be smaller than one!' % normDurationSwing)
                # if  normDurationStance < 1 : print('It\'s happening again in case twoB !')
                end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                unique_bins = np.concatenate((np.arange(start_bin, 101), np.arange(0, end_bin)))
                successRate['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
            elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] >= swingNPOEnd) and (laserDict['laserEndTime'][k] < swingNPTStart):
                normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                assert normDurationStance < 1, print('The normalized duration %s in case twoC should be smaller than one!' % normDurationStance)
                end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                unique_bins = np.arange(start_bin, end_bin + 1)
                successRate['normStepCycleBins'] += 1  # in this condition, laser is active during a whole stance and swing  # successRate[case]['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
            else:
                unmatchedSwingCases +=1
                addBins = False
            if addBins :
                successRate['normStepCycleBins'][np.unique(unique_bins)] += 1
        for k in laserActivationTimeStance:
            addBins = True
            normStart = (laserDict['laserStartTime'][k] - swingNEnd) / stanceDuration
            start_bin = np.digitize(normStart + 1., normTimeStepCycle) - 1
            if laserDict['laserEndTime'][k] < swingNPOStart:
                normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / stanceDuration
                assert normDuration < 1, print('The normalized duration in case three should be smaller than one!')
                normEnd = 1 + normStart + normDuration
                end_bin = np.digitize(normEnd, normTimeStepCycle) - 1
                unique_bins = np.arange(start_bin, end_bin + 1)
            elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                normDurationSwing = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                assert normDurationSwing < 1, print('The normalized duration %s in case four should be smaller than one!' % normDurationSwing)
                end_bin = np.digitize(normDurationSwing, normTimeStepCycle) - 1
                unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(0, end_bin + 1)))  # np.arange(start_bin, end_bin + 1)
            elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd) and (laserDict['laserEndTime'][k] < swingNPTStart):
                normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                assert normDurationStance < 1, print('The normalized duration %s in case five should be smaller than one!' % normDurationStance)
                end_bin = np.digitize(normDurationStance + 1, normTimeStepCycle) - 1
                unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(100, end_bin + 1)))
                successRate['normStepCycleBins'][np.arange(0, 101)] += 1  # in this condition, laser is active during a whole swing
            else:
                unmatchedStanceCases +=1
                addBins = False
            #elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd) and (laserDict['laserEndTime'][k] >= swingNPTStart):
            #    normDurationStance = (laserDict['laserEndTime'][k] - swingNPTStart) / swingNextDuration
            if addBins:
                successRate['normStepCycleBins'][np.unique(unique_bins)] += 1

    print('unmatched casesd during swing  : ', unmatchedSwingCases)
    print('unmatched casesd during stance : ', unmatchedStanceCases)

#######################
def calculateStrideProperties(swingStanceD,pawPos,pawSpeed):
    strideProp = {}
    for i in range(4):
        strideProp[i] = {}

        stepParameters=calculateStepPar(pawPos,swingStanceD)
        strideProp[i]['stepPos'] =stepParameters['non_linear'][i]['stepPos']
        strideProp[i]['stepTime'] =  stepParameters['non_linear'][i]['stepTime']
        strideProp[i]['stepSpeed'] = stepParameters['non_linear'][i]['stepSpeed']
        strideProp[i]['stepLength']=stepParameters['non_linear'][i]['stepLength']
        strideProp[i]['stepDuration']=stepParameters['non_linear'][i]['stepDuration']
        strideProp[i]['stepMeanSpeed']=stepParameters['non_linear'][i]['stepMeanSpeed']


        strideProp[i]['indicies'] = []
        strideProp[i]['indecisiveBool'] = []
        strideProp[i]['swingLength'] = []
        strideProp[i]['swingDuration'] = []
        strideProp[i]['swingSpeed'] = []
        strideProp[i]['swingTime'] = []
        strideProp[i]['stanceDuration'] = []
        strideProp[i]['swingLengthLinear']=[]
        strideProp[i]['rungCrossed']=[]

        newVariablesList = [ "acceleration",
            "max_acceleration",
            "mean_acceleration",
            "acceleration_phases",
            "max_deceleration",
            "mean_deceleration",
            "deceleration_phases",
            "acc_duration",
            "dec_duration"]
        for var in newVariablesList:
                strideProp[i][var]=[]
        idxSwings =np.array(swingStanceD['swingP'][i][1])
        recTimes = np.array(swingStanceD['forFit'][i][2])
        linearPawPos=np.array(swingStanceD['forFit'][i][5])

        wSpeed = swingStanceD['forFit'][i][0]
        xSpeed = swingStanceD['forFit'][i][1]
        indecisiveSteps = swingStanceD['swingP'][i][3]
        rungNumbers = np.array(swingStanceD['swingP'][i][2])

        rungCrossed = np.diff(rungNumbers)

        for n in range(len(idxSwings) - 1):
            idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][0]]))
            idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][1]]))
            idxSwingStartNext = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n + 1][0]]))
            idxStanceStartNext = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n + 1][1]]))
            swingMask = (linearPawPos[:, 0] >= recTimes[idxSwings[n, 0]]) & (linearPawPos[:, 0] <= recTimes[idxSwings[n, 1]])
            # fill dictionary
            strideProp[i]['indicies'].append([idxSwingStart,idxStanceStart,idxSwingStartNext,idxStanceStartNext])
            strideProp[i]['indecisiveBool'].append(indecisiveSteps[n][3])
            strideProp[i]['swingLength'].append(pawPos[i][:, 1][idxStanceStart] - pawPos[i][:, 1][idxSwingStart])
            strideProp[i]['swingLengthLinear'].append(indecisiveSteps[n][7])
            strideProp[i]['rungCrossed'].append(rungCrossed[n])

            swingSpeed = abs(xSpeed[idxSwings[n, 0]:idxSwings[n, 1]] * 0.025 - wSpeed[idxSwings[n, 0]:idxSwings[n, 1]])
            swingSpeedTime = recTimes[idxSwings[n, 0]:idxSwings[n, 1]]
            accelerationDic = groupAnalysis.calc_acceleration(
                swingSpeed, swingSpeedTime)
            for key, value in accelerationDic.items():
                if key not in ["acceleration"]:
                    strideProp[i][key].extend([value])
            strideProp[i]['swingDuration'].append(pawPos[i][:, 0][idxStanceStart] - pawPos[i][:, 0][idxSwingStart])
            strideProp[i]['swingSpeed'].append(np.mean(swingSpeed))
            strideProp[i]['swingTime'].append(np.median((pawPos[i][:, 0][idxSwingStart:idxStanceStart])))
            strideProp[i]['stanceDuration'].append(pawPos[i][:, 0][idxSwingStartNext] - pawPos[i][:, 0][idxStanceStart])
            strideProp[i]['acceleration'].append(np.mean(accelerationDic['acceleration']))
    
    return strideProp

###########################################################
def calculateStridePropertiesFlatSurface(swingStanceD, pawPos, pawSpeed):
    strideProp = {}
    for i in range(4):
        strideProp[i] = {}

        # stepParameters = calculateStepPar(pawPos, swingStanceD)
        # strideProp[i]['stepPos'] = stepParameters['non_linear'][i]['stepPos']
        # strideProp[i]['stepTime'] = stepParameters['non_linear'][i]['stepTime']
        # strideProp[i]['stepSpeed'] = stepParameters['non_linear'][i]['stepSpeed']
        # strideProp[i]['stepLength'] = stepParameters['non_linear'][i]['stepLength']
        # strideProp[i]['stepDuration'] = stepParameters['non_linear'][i]['stepDuration']
        # strideProp[i]['stepMeanSpeed'] = stepParameters['non_linear'][i]['stepMeanSpeed']

        strideProp[i]['indicies'] = []
        strideProp[i]['indecisiveBool'] = []
        strideProp[i]['swingLength'] = []
        strideProp[i]['swingDuration'] = []
        strideProp[i]['swingSpeed'] = []
        strideProp[i]['swingTime'] = []
        strideProp[i]['stanceDuration'] = []
        strideProp[i]['swingLengthLinear'] = []
        strideProp[i]['rungCrossed'] = []

        newVariablesList = ["acceleration",
                            "max_acceleration",
                            "mean_acceleration",
                            "acceleration_phases",
                            "max_deceleration",
                            "mean_deceleration",
                            "deceleration_phases",
                            "acc_duration",
                            "dec_duration"]
        for var in newVariablesList:
            strideProp[i][var] = []
        idxSwings = np.array(swingStanceD['swingP'][i][1])
        recTimes = np.array(swingStanceD['forFit'][i][2])
        linearPawPos = np.array(swingStanceD['forFit'][i][5])

        wSpeed = swingStanceD['forFit'][i][0]
        xSpeed = swingStanceD['forFit'][i][1]





        for n in range(len(idxSwings) - 1):
            idxSwingStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][0]]))
            idxStanceStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n][1]]))
            idxSwingStartNext = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n + 1][0]]))
            idxStanceStartNext = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[n + 1][1]]))
            swingMask = (linearPawPos[:, 0] >= recTimes[idxSwings[n, 0]]) & (
                        linearPawPos[:, 0] <= recTimes[idxSwings[n, 1]])
            # fill dictionary
            strideProp[i]['indicies'].append([idxSwingStart, idxStanceStart, idxSwingStartNext, idxStanceStartNext])
            strideProp[i]['swingLength'].append(pawPos[i][:, 1][idxStanceStart] - pawPos[i][:, 1][idxSwingStart])


            swingSpeed = abs(xSpeed[idxSwings[n, 0]:idxSwings[n, 1]] * 0.025 - wSpeed[idxSwings[n, 0]:idxSwings[n, 1]])
            swingSpeedTime = recTimes[idxSwings[n, 0]:idxSwings[n, 1]]
            accelerationDic = groupAnalysis.calc_acceleration(
                swingSpeed, swingSpeedTime)
            for key, value in accelerationDic.items():
                if key not in ["acceleration"]:
                    strideProp[i][key].extend([value])
            strideProp[i]['swingDuration'].append(pawPos[i][:, 0][idxStanceStart] - pawPos[i][:, 0][idxSwingStart])
            strideProp[i]['swingSpeed'].append(np.mean(swingSpeed))
            strideProp[i]['swingTime'].append(np.median((pawPos[i][:, 0][idxSwingStart:idxStanceStart])))
            strideProp[i]['stanceDuration'].append(pawPos[i][:, 0][idxSwingStartNext] - pawPos[i][:, 0][idxStanceStart])
            strideProp[i]['acceleration'].append(np.mean(accelerationDic['acceleration']))

    return strideProp
##########################
def getConditionName(cond,percent=None):
    conditionName = ''.join(str(e)+'_' for e in cond)
    if percent is not None and conditionName!='rungCrossed_':
        conditionName+=str(percent[0])+'_'+str(percent[1])
    elif percent is not None and conditionName=='rungCrossed_':
        conditionName+=str(percent)


    else:
        conditionName = conditionName[:-1]
        # print(conditionName)
        # # pdb.set_trace()
    return conditionName

###########################
def defineCondition(condition, singleCellList):
    nRecs = len(singleCellList)
    intervals = 5
    spacing = int(100/intervals)
    percentiles = [[i*spacing,(i+1)*spacing] for i in range(intervals)]

    if condition[0] == 'allSteps':
        return [condition[0],[condition[0],None],None]

    elif  (condition[0]=='rungCrossed'):
        rungCrossed=[0,1,2]
        cc = []
        for j in range(len(rungCrossed)):
            thresholds = []
            for i in range(4):
                thresholds.append(rungCrossed[j])
                conditionName = getConditionName(condition, percent=rungCrossed[j])
            cc.append([conditionName, condition, thresholds])


        return cc

    elif condition[0] in ['swingDuration','swingLength','swingLengthLinear', 'swingSpeed', 'stepLength', 'stepDuration','stepMeanSpeed']:
        if condition[1] == 'allRecs': # all recordings
            var = [[],[],[],[]]
            for i in range(4):
                for r in range(nRecs):
                    var[i].extend(singleCellList[r][6][i][condition[0]])
            cc = []
            for j in range(intervals):
                thresholds = []
                for i in range(4):
                    thresholds.append(np.percentile(var[i],(percentiles[j][0],percentiles[j][1])))
                conditionName = getConditionName(condition,percent=(percentiles[j][0],percentiles[j][1]))
                #pdb.set_trace()
                cc.append([conditionName,condition,thresholds])
            return cc

        elif condition[1] == 'lastRec': # only one recording
            strideProps = singleCellList[nRecs-1][6]
            thresholds = []
            for i in range(4):
                thresholds.append(np.percentile(strideProps[i][condition[0]],(condition[2],condition[3])))
            conditionName = getConditionName(condition)
            return [conditionName, condition, thresholds]

    elif condition[0] == 'indecisiveSteps':  # only one recording
        thresholds = []
        for i in range(4):
            thresholds.append(True)
        conditionName = getConditionName(condition)
        return [conditionName, [condition[0],None], thresholds]

    elif condition[0] == 'certainSteps':  # only one recording
        thresholds = []
        for i in range(4):
            thresholds.append(False)
        conditionName = getConditionName(condition)
        return [conditionName, [condition[0],None], thresholds]


    # pdb.set_trace()
    # if 1:
    #     pass
    # elif condition == 'swingLengthLastRec20-80':
    #     strideProps = singleCellList[nRecs - 1][6]
    #     thresholds = []
    #     for i in range(4):
    #         thresholds.append(np.percentile(strideProps[i]['swingLength'], (20, 80)))
    #     return [condition,thresholds]
    # elif condition == 'swingSpeedLastRec20-80':
    #     strideProps = singleCellList[nRecs - 1][6]
    #     thresholds = []
    #     for i in range(4):
    #         thresholds.append(np.percentile(strideProps[i]['swingSpeed'], (20, 80)))
    #     return [condition,thresholds]
    # elif condition == 'stanceDurationLastRec20-80':
    #     strideProps = singleCellList[nRecs - 1][6]
    #     thresholds = []
    #     for i in range(4):
    #         thresholds.append(np.percentile(strideProps[i]['stanceDuration'], (20, 80)))
    #     return [condition,thresholds]
    # elif condition == 'indecisiveSteps':
    #     strideProps = singleCellList[nRecs - 1][6]
    #     thresholds = []
    #     for i in range(4):
    #         thresholds.append([True])
    #     return [condition,thresholds]
    # elif condition == 'not_indecisiveSteps':
    #     strideProps = singleCellList[nRecs - 1][6]
    #     thresholds = []
    #     for i in range(4):
    #         thresholds.append([False])
    #     return [condition,thresholds]
    # if control_var!=None and (control_var!='rungCrossed'):
    #     conditionIntervals=[[0,20],[20,40],[40,60],[60,80],[80,100]]
    #     for c in range(len(conditionIntervals)):
    #         if condition == f'{control_var}AllRecs{conditionIntervals[c][0]}-{conditionIntervals[c][1]}':
    #             thresholds = []
    #             for i in range(4):
    #                 swingLenArray = np.array([])
    #                 for r in range(nRecs):
    #                     strideProps = singleCellList[r][6]
    #                     swingLenArray=np.concatenate((swingLenArray,strideProps[i][f'{control_var}']))
    #                 thresholds.append(np.percentile(swingLenArray, (conditionIntervals[c][0], conditionIntervals[c][1])))
    #             return [condition,thresholds]
    # elif  (control_var=='rungCrossed'):
    #     conditionIntervals=[0,1,2,3]
    #     for c in range(len(conditionIntervals)):
    #         if condition == f'{control_var}AllRecs{conditionIntervals[c]}':
    #             thresholds = []
    #             for i in range(4):
    #                 thresholds.append(conditionIntervals[c])
    #             #     rungCrossedArray = np.array([])
    #             #     for r in range(nRecs):
    #             #         strideProps = singleCellList[r][6]
    #             #         swingLenArray=np.concatenate((rungCrossedArray,strideProps[i][f'{control_var}']))
    #             #     thresholds.append(np.percentile(rungCrossedArray, (conditionIntervals[c][0], conditionIntervals[c][1])))
    #
    #             return [condition,thresholds]



###########################
def conditionMatched(n,i,condition,strideProp):
    if condition[1][0] == 'allSteps':
        return True
    elif condition[1][0] in ['swingDuration','swingLength','swingLengthLinear','swingSpeed', 'stepLength', 'stepDuration','stepMeanSpeed']:
        if ((strideProp[i][condition[1][0]][n]>=condition[2][i][0])&(strideProp[i][condition[1][0]][n]<condition[2][i][1])):
            return True
        else:
            return False
    elif condition[1][0] == 'rungCrossed':
        if condition[2][i]==2:
            if (strideProp[i][condition[1][0]][n] >= condition[2][i]):
                return True
            else:
                return False
        else :
            if (strideProp[i][condition[1][0]][n] == condition[2][i]):
                return True
            else:
                return False
    elif condition[1][0] == 'indecisiveSteps':
        if (strideProp[i]['indecisiveBool'][n] == True):
            return True
        else:
            return False
    elif condition[1][0] == 'certainSteps':
        if (strideProp[i]['indecisiveBool'][n] == False):
            return True
        else:
            return False

    # pdb.set_trace()
    # if 1:
    #     pass
    # elif condition[0] == 'swingDurationLastRec20-80':
    #     if ((strideProp[i]['swingDuration'][n]>condition[1][i][0])&(strideProp[i]['swingDuration'][n]<condition[1][i][1])):
    #         return True
    #     else:
    #         return False
    # elif condition[0] == 'swingLengthLastRec20-80':
    #     if ((strideProp[i]['swingLength'][n]>condition[1][i][0])&(strideProp[i]['swingLength'][n]<condition[1][i][1])):
    #         return True
    #     else:
    #         return False
    # elif condition[0] == 'swingSpeedLastRec20-80':
    #     if ((strideProp[i]['swingSpeed'][n]>condition[1][i][0])&(strideProp[i]['swingSpeed'][n]<condition[1][i][1])):
    #         return True
    #     else:
    #         return False
    # elif condition[0] == 'stanceDurationLastRec20-80':
    #     if ((strideProp[i]['stanceDuration'][n]>condition[1][i][0])&(strideProp[i]['stanceDuration'][n]<condition[1][i][1])):
    #         return True
    #     else:
    #         return False
    # elif condition[0] == 'indecisiveSteps':
    #     if ((strideProp[i]['indecisiveBool'][n]==True)):
    #         return True
    #     else:
    #         return False
    # elif condition[0] == 'not_indecisiveSteps':
    #     if ((strideProp[i]['indecisiveBool'][n]==False)):
    #         return True
    #     else:
    #         return False
    # if control_var!=None and (control_var!='rungCrossed'):
    #     conditionIntervals=[[0,20],[20,40],[40,60],[60,80],[80,100]]
    #     for c in range(len(conditionIntervals)):
    #         if condition[0] == f'{control_var}AllRecs{conditionIntervals[c][0]}-{conditionIntervals[c][1]}':
    #             if ((strideProp[i][f'{control_var}'][n] > condition[1][i][0]) & (
    #                     strideProp[i][f'{control_var}'][n] < condition[1][i][1])):
    #                 return True
    #             else:
    #                 return False
    # elif control_var=='rungCrossed':
    #     conditionIntervals=[0,1,2,3]
    #     for c in range(len(conditionIntervals)):
    #         if c<3:
    #             if condition[0] == f'{control_var}AllRecs{conditionIntervals[c]}':
    #
    #                 if (strideProp[i][f'{control_var}'][n] == condition[1][i]):
    #                     return True
    #                 else:
    #                     return False
    #         elif c==3:
    #             if condition[0] == f'{control_var}AllRecs{conditionIntervals[c]}':
    #                 if (strideProp[i][f'{control_var}'][n] >=condition[1][i]):
    #
    #                     return True
    #                 else:
    #                     return False

############################ spikes, pawPos, pawSpeed, swingStanceDict, strideProps, cond
def calculateStridebasedPSTH(spikes,pawPos,pawSpeed, swingStanceD,strideProp,cond): #, control_var):
    col = ['C0','C1','C2','C3']
    spikeBeforeAfter = [0.4,0.6] # in sec
    speedBeforeAfter = [0.5,0.7]
    dt = 0.02
    dtRes = 0.05
    dtSpeed = 0.01
    timePSTH = np.linspace(-spikeBeforeAfter[0],spikeBeforeAfter[1],int((spikeBeforeAfter[1]+spikeBeforeAfter[0])/dtSpeed)+1,endpoint=True)
    rescaledTimePSTH = np.linspace(0,1,int(1/dtRes)+1,endpoint=True)
    #beforeRange = [-0.1,0] # range in s
    #afterRange = [0,0.1] #  range in s
    # isiWalkings = np.diff(spikes[(spikes > 10.) & (spikes <= 53)])
    #firingRateWalking = 1. / np.mean(isiWalkings)
    ##############

    possibleConditions = ['allSteps','swingDuration','swingLength','swingLengthLinear', 'rungCrossed','swingSpeed', 'stepLength', 'stepDuration','stepMeanSpeed','indecisiveSteps','certainSteps']

    #pdb.set_trace()
    if not (cond[1][0] in possibleConditions): raise Exception('PSTH condition %s not implemented (yet)!' % cond[1][0])
    psth= {}
    for i in range(4):
        psth[i] = {}
        if cond[1][0] in  ['stepLength', 'stepDuration', 'stepMeanSpeed']:
            psth[i]['stepLength']=[]
            psth[i]['stepDuration'] = []
            psth[i]['stepMeanSpeed'] = []
    for i in range(4):

        #psth[condition][i]['spikeTimes'] = []
        #psth[condition][i]['spikeTimesSorted'] = []
        #psth[condition][i]['spikeTimesRescaled'] = []
        #psth[condition][i]['spikeTimesRescaledSorted'] = []
        psth[i]['spikeTimesCenteredStanceStart'] = []
        psth[i]['spikeTimesCenteredStanceStartSorted'] = []
        psth[i]['spikeTimesCenteredSwingStart'] = []
        psth[i]['spikeTimesCenteredSwingStartSorted'] = []
        psth[i]['spikeTimesCenteredExpectedStanceStart'] = []
        psth[i]['speedCenteredStanceStart'] = []
        psth[i]['speedCenteredSwingStart'] = []
        psth[i]['speedCenteredExpectedStanceStart'] = []
        psth[i]['speedCenteredStanceStartRescaled'] = []
        psth[i]['swingStart'] = []
        psth[i]['stanceStart'] = []
        psth[i]['swingOnset'] = []
        psth[i]['stanceOnset'] = []
        psth[i]['expectedStanceOnset'] = []
        psth[i]['expectedStanceOnsetIdx'] = []
        psth[i]['strideEndSwingCentered'] = []
        psth[i]['strideEndSwingCenteredSorted'] = []
        psth[i]['strideEndStanceCentered'] = []
        psth[i]['strideEndStanceCenteredSorted'] = []
        psth[i]['swingLength'] = []
        psth[i]['swingLengthLinear']=[]
        psth[i]['swingDuration'] = []
        psth[i]['swingSpeed'] = []
        #psth[i]['strideEnd2Sorted'] = []
        psth[i]['swingStartSorted'] = []
        psth[i]['stanceStartSorted'] = []
        psth[i]['swingIndicies']=[]
        psth[i]['stanceDuration']=[]
        # new for rescaled PSTH
        #psth[i]['spikeTimesCenteredSwingStartRescaled'] = []
        #psth[i]['spikeTimesCenteredSwingStartRescaledSorted'] = []
        psth[i]['spikeTimesCenteredStanceStartRescaled'] = []
        psth[i]['spikeTimesCenteredStanceStartRescaledSorted'] = []
        # rescaled PSTH
        psth[i]['indecisive'] = []
        psth[i]['rungCrossed']=[]
        psth[i]['indecisiveSorted'] = []
        psth[i]['indecisiveSorted2'] = []
        #psth[i]['indecisiveSorted2'] = []
        #psth[i]['indecisiveBool'] = []
        #psth[i]['indecisiveBoolSorted'] = []
        idxSwings = np.array(swingStanceD['swingP'][i][1])
        wSpeed = np.array(swingStanceD['forFit'][i][0])
        xSpeed = np.array(swingStanceD['forFit'][i][1])
        recTimes= np.array(swingStanceD['forFit'][i][2])
        stepCharater = swingStanceD['swingP'][i][3]
        # rungNumbers = np.array(swingStanceD['swingP'][i][2])
        #
        # rungCrossed = np.diff(rungNumbers)
        newVariablesList = [ "acceleration",
            "max_acceleration",
            "mean_acceleration",
            "acceleration_phases",
            "max_deceleration",
            "mean_deceleration",
            "deceleration_phases",
            "acc_duration",
            "dec_duration"]
        for var in newVariablesList:
                psth[i][var]=[]
        #recTimes = swingStanceD['forFit'][i][2]
        if not (len(idxSwings)==(len(strideProp[i]['indicies'])+1)): raise Exception('Not the same lenght! : Swing indicies and swing properties')
        #pdb.set_trace()
        nStride=0
        nStrideNotMatched=0
        matchingStrideId=[]
        if cond[1][0] in ['stepLength', 'stepDuration', 'stepMeanSpeed']:
            if len(strideProp[i]['stepLength'])<len(idxSwings):
                swingNumber=len(strideProp[i]['stepLength'])
            else:
                swingNumber=len(idxSwings)
        else:
            swingNumber=len(idxSwings)
        for n in range(swingNumber-1):
            if conditionMatched(n,i,cond,strideProp):
                nStride+=1
                idxSwingStart = strideProp[i]['indicies'][n][0]
                idxStanceStart = strideProp[i]['indicies'][n][1]
                idxSwingStartNext = strideProp[i]['indicies'][n][2]
                idxStanceStartNext = strideProp[i]['indicies'][n][3]
                #pdb.set_trace()
                psth[i]['indecisive'].append(stepCharater[n][3])
                expectedStanceIdx = stepCharater[n][10]
                psth[i]['expectedStanceOnsetIdx'].append(expectedStanceIdx)
                # this is now computed directly in the swing-stance script `extractSwingStancePhases.py`
                # if stepCharater[n][3]:
                #     closeIndices = stepCharater[n][4]
                #     firstExpectedStancePeriod = np.where((closeIndices[:,1]-closeIndices[:,0])>=2)[0]
                #     if len(firstExpectedStancePeriod)>0:
                #         expectedStanceIdx = closeIndices[firstExpectedStancePeriod[0],0]
                #     else:
                #         expectedStanceIdx = closeIndices[0,0]
                #     psth[i]['expectedStanceOnsetIdx'].append(expectedStanceIdx)
                # else:
                #     expectedStanceIdx = idxStanceStart - idxSwingStart
                #     psth[i]['expectedStanceOnsetIdx'].append(expectedStanceIdx)

                maskSwingStart = (spikes >= (pawPos[i][idxSwingStart, 0] - spikeBeforeAfter[0])) & (spikes <= (pawPos[i][idxSwingStart, 0] + spikeBeforeAfter[1]))
                maskStanceStart = (spikes >= (pawPos[i][idxStanceStart, 0]- spikeBeforeAfter[0])) & (spikes <= (pawPos[i][idxStanceStart, 0]+spikeBeforeAfter[1]))
                maskExpectedStanceStart = (spikes >= (pawPos[i][(idxSwingStart+expectedStanceIdx), 0]- spikeBeforeAfter[0])) & (spikes <= (pawPos[i][(idxSwingStart+expectedStanceIdx), 0]+spikeBeforeAfter[1]))
                maskSpeedSwingStart = (pawSpeed[i][:,0] > (pawPos[i][idxSwingStart, 0] - speedBeforeAfter[0]))&(pawSpeed[i][:,0] <= (pawPos[i][idxSwingStart, 0] + speedBeforeAfter[1])) #pawSpeed[i][:,0] # is time  -  pawSpeed[i][:,2] is paw x-speed
                maskSpeedStanceStart = (pawSpeed[i][:, 0] > (pawPos[i][idxStanceStart, 0] - speedBeforeAfter[0])) & (pawSpeed[i][:, 0] <= (pawPos[i][idxStanceStart, 0] + speedBeforeAfter[1]))  # pawSpeed[i][:,0] # is time  -  pawSpeed[i][:,2] is paw x-speed
                maskSpeedExpectedStanceStart = (pawSpeed[i][:, 0] > (pawPos[i][(idxSwingStart+expectedStanceIdx), 0] - speedBeforeAfter[0])) & (pawSpeed[i][:, 0] <= (pawPos[i][(idxSwingStart+expectedStanceIdx), 0] + speedBeforeAfter[1]))
                #psth[i]['spikeTimes'].append(spikes[mask] -  pawPos[i][idxEnd, 0])
                psth[i]['spikeTimesCenteredSwingStart'].append(spikes[maskSwingStart] - pawPos[i][idxSwingStart, 0])
                psth[i]['spikeTimesCenteredStanceStart'].append(spikes[maskStanceStart] -  pawPos[i][idxStanceStart, 0])
                psth[i]['spikeTimesCenteredExpectedStanceStart'].append(spikes[maskExpectedStanceStart] - pawPos[i][(idxSwingStart+expectedStanceIdx), 0])
                # psth[i]['rungCrossed'].append(rungCrossed[n])
                maskSwingSpikes = (spikes >= pawPos[i][idxSwingStart, 0]) & (spikes <= pawPos[i][idxStanceStart, 0]) #
                maskStanceSpikes =(spikes >= pawPos[i][idxStanceStart, 0]) & (spikes <= pawPos[i][idxSwingStartNext, 0])
                swingSpikesRes = (spikes[maskSwingSpikes] - pawPos[i][idxStanceStart, 0])/(pawPos[i][idxStanceStart, 0] - pawPos[i][idxSwingStart, 0]) + 1.
                stanceSpikesRes = (spikes[maskStanceSpikes] - pawPos[i][idxStanceStart, 0])/(pawPos[i][idxSwingStartNext, 0] - pawPos[i][idxStanceStart, 0]) + 1.
                #pdb.set_trace()
                interpSpeedSwingStart = interp1d(pawSpeed[i][:,0][maskSpeedSwingStart]-pawPos[i][idxSwingStart, 0], pawSpeed[i][:,2][maskSpeedSwingStart], fill_value='extrapolate')  # ,kind='cubic')
                interpSpeedStanceStart = interp1d(pawSpeed[i][:, 0][maskSpeedStanceStart]-pawPos[i][idxStanceStart, 0], pawSpeed[i][:, 2][maskSpeedStanceStart], fill_value='extrapolate')
                interpSpeedExpectedStanceStart = interp1d(pawSpeed[i][:, 0][maskSpeedExpectedStanceStart]-pawPos[i][(idxSwingStart+expectedStanceIdx), 0], pawSpeed[i][:, 2][maskSpeedExpectedStanceStart], fill_value='extrapolate')
                psth[i]['speedCenteredSwingStart'].append(interpSpeedSwingStart(timePSTH))
                psth[i]['speedCenteredStanceStart'].append(interpSpeedStanceStart(timePSTH))
                psth[i]['speedCenteredExpectedStanceStart'].append(interpSpeedExpectedStanceStart(timePSTH))
                maskSwingSpeed = (pawSpeed[i][:,0] >= (pawPos[i][idxSwingStart, 0]-0.2)) & (pawSpeed[i][:,0] <= (pawPos[i][idxStanceStart, 0]+0.2))
                maskStanceSpeed = (pawSpeed[i][:,0] >= (pawPos[i][idxStanceStart, 0]-0.2))& (pawSpeed[i][:,0] <= (pawPos[i][idxSwingStartNext, 0]+0.2))
                swingTimeRes = (pawSpeed[i][:,0][maskSwingSpeed] - pawPos[i][idxStanceStart, 0])/(pawPos[i][idxStanceStart, 0] - pawPos[i][idxSwingStart, 0]) + 1.
                stanceTimeRes = (pawSpeed[i][:,0][maskStanceSpeed] - pawPos[i][idxStanceStart, 0])/(pawPos[i][idxSwingStartNext, 0] - pawPos[i][idxStanceStart, 0]) + 1.

                speedSwingRes = np.interp(rescaledTimePSTH, swingTimeRes, pawSpeed[i][:,2][maskSwingSpeed])
                speedStanceRes = np.interp(rescaledTimePSTH+1., stanceTimeRes, pawSpeed[i][:, 2][maskStanceSpeed])
                #pdb.set_trace()
                psth[i]['speedCenteredStanceStartRescaled'].append(np.concatenate((speedSwingRes,speedStanceRes)))
                #
                psth[i]['spikeTimesCenteredStanceStartRescaled'].append(np.concatenate((swingSpikesRes,stanceSpikesRes)))
                psth[i]['swingStart'].append(pawPos[i][idxSwingStart, 0]-pawPos[i][idxStanceStart, 0])
                psth[i]['stanceStart'].append(pawPos[i][idxStanceStart, 0] - pawPos[i][idxSwingStart, 0])
                psth[i]['swingOnset'].append(pawPos[i][idxSwingStart, 0])
                psth[i]['stanceOnset'].append(pawPos[i][idxStanceStart, 0])
                psth[i]['expectedStanceOnset'].append(pawPos[i][(idxSwingStart+expectedStanceIdx), 0])
                psth[i]['strideEndSwingCentered'].append(pawPos[i][idxSwingStartNext, 0] - pawPos[i][idxSwingStart, 0])
                psth[i]['strideEndStanceCentered'].append(pawPos[i][idxSwingStartNext, 0] - pawPos[i][idxStanceStart, 0])

                #pdb.set_trace()
                #psth[i]['swingLength'].append(stepCharater[n][4]) #6
                #psth[i]['swingLengthLinear'].append(stepCharater[n][5]) #7
                #psth[i]['swingDuration'].append(stepCharater[n][3]) #5 if not flat surface
                psth[i]['stanceDuration'].append(pawPos[i][idxSwingStartNext,0] - pawPos[i][idxStanceStart,0])
                psth[i]['swingIndicies'].append(n)
                swingSpeed =abs(xSpeed[idxSwings[n, 0]:idxSwings[n, 1]] * 0.025 - wSpeed[idxSwings[n, 0]:idxSwings[n, 1]])
                swingSpeedTime = recTimes[idxSwings[n, 0]:idxSwings[n, 1]]
                accelerationDic= groupAnalysis.calc_acceleration(swingSpeed, swingSpeedTime)
                for key, value in accelerationDic.items():
                    if key not in ["acceleration"]:
                        psth[i][key].extend([value])
                psth[i]['swingSpeed'].append(np.mean(swingSpeed))
                psth[i]['acceleration'].append(np.mean(accelerationDic['acceleration']))
                if cond[1][0] in ['stepLength', 'stepDuration', 'stepMeanSpeed']:
                    psth[i]['stepLength'].append(strideProp[i]['stepLength'][n])
                    psth[i]['stepDuration'].append(strideProp[i]['stepDuration'][n])
                    psth[i]['stepMeanSpeed'].append(strideProp[i]['stepMeanSpeed'][n])
            else:
                nStrideNotMatched+=1
        # if (nStride+nStrideNotMatched)!=(len(swingNumber)-1) : raise Exception('Problem in condition implementation when extracting spikes')
        # print(nStride, 'matched condition !!!')
        if nStride==0:
            print('No stride match condition !!!!!!!!!!!!!!!!!!!!!!!!!!')
            psth[i]={}
                #psth[i]['indecisiveBool'].append(indecisiveSteps[n][3])

    # pdb.set_trace()

    # sort the stance-onset aligned psth based on swing-start time
    for i in range(4):
        if len(psth[i]) >0:
            idxSorted_swing = np.argsort(psth[i]['swingStart'])
            # pdb.set_trace()
            for n in range(len(idxSorted_swing)):
                #idx  = np.where(idxSorted==n)[0][0]
                #psth[i]['spikeTimesSorted'].append(psth[i]['spikeTimes'][idxSorted[n]])
                psth[i]['spikeTimesCenteredStanceStartSorted'].append(psth[i]['spikeTimesCenteredStanceStart'][idxSorted_swing[n]])
                psth[i]['swingStartSorted'].append([psth[i]['swingStart'][idxSorted_swing[n]]])
                psth[i]['strideEndStanceCenteredSorted'].append([psth[i]['strideEndStanceCentered'][idxSorted_swing[n]]])
                psth[i]['spikeTimesCenteredStanceStartRescaledSorted'].append(psth[i]['spikeTimesCenteredStanceStartRescaled'][idxSorted_swing[n]])
                #psth[i]['indecisiveSorted'].append(psth[i]['indecisive'][idxSorted[n]])
                #psth[i]['indecisiveBoolSorted'].append(psth[i]['indecisiveBool'][idxSorted[n]])
                #print(i,n,psth[i]['stanceStartRescaled'][idxSorted[n]])

    # sort the swing-onset aligned psth based on swing-end time
    for i in range(4):
        if len(psth[i]) > 0:
            idxSorted_stance = np.argsort(psth[i]['stanceStart'])
            #pdb.set_trace()
            for n in range(len(idxSorted_stance)):
                #idx  = np.where(idxSorted==n)[0][0]
                #psth[i]['spikeTimesSortedSwing'].append(psth[i]['spikeTimes'][idxSorted[n]])
                psth[i]['spikeTimesCenteredSwingStartSorted'].append(psth[i]['spikeTimesCenteredSwingStart'][idxSorted_stance[n]])
                psth[i]['stanceStartSorted'].append([psth[i]['stanceStart'][idxSorted_stance[n]]])
                psth[i]['strideEndSwingCenteredSorted'].append([psth[i]['strideEndSwingCentered'][idxSorted_stance[n]]])


        # bin spike counts and calcualte histogram
    tbins = np.linspace(-spikeBeforeAfter[0], spikeBeforeAfter[1], int((spikeBeforeAfter[0]+spikeBeforeAfter[1])/dt) + 1, endpoint=True)
    tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True) # rescaled time goes from 0 to 2, 1 is swing-onset
    for i in range(4):
        if len(psth[i]) > 0:
            # stance-onset aligned
            cnt, edges = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredStanceStartSorted']).ravel(), bins=tbins)
            timePoints = (edges[1:] + edges[:-1])/2
            avgFRStance =  cnt /(len(psth[i]['spikeTimesCenteredStanceStartSorted'])* dt)
            psth[i]['psth_stanceOnsetAligned'] = np.row_stack((timePoints, avgFRStance))

            # stance-onset alinged rescaled
            cnt1, edges1 = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredStanceStartRescaledSorted']).ravel(), bins=tbinsRes)
            timePoints1 = (edges1[1:] + edges1[:-1]) / 2
            avgFRStance1 = cnt1 / (len(psth[i]['spikeTimesCenteredStanceStartRescaledSorted']) * dtRes)
            psth[i]['psth_stanceOnsetAlignedRescaled'] = np.row_stack((timePoints1, avgFRStance1))

            # expected stance-onset aligned
            successful_Combined = np.concatenate([arr for arr, flag in zip(psth[i]['spikeTimesCenteredStanceStart'], psth[i]['indecisive']) if not flag]).ravel()
            #successful_Combined = sum([lst for lst, flag in zip(psth[i]['indecisive'], psth[i]['spikeTimesCenteredStanceStart']) if not flag], [])
            #missstep_Combined = sum([lst for lst, flag in zip(psth[i]['indecisive'], psth[i]['spikeTimesCenteredExpectedStanceStart']) if flag], [])
            missstep_Combined = np.concatenate([arr for arr, flag in zip(psth[i]['spikeTimesCenteredExpectedStanceStart'], psth[i]['indecisive']) if flag]).ravel()
            missstep_Combined2 = np.concatenate([arr for arr, flag in zip(psth[i]['spikeTimesCenteredStanceStart'], psth[i]['indecisive']) if flag]).ravel()
            cnt3, edges3 = np.histogram(np.array(successful_Combined).ravel(), bins=tbins)
            cnt4, edges4 = np.histogram(np.array(missstep_Combined).ravel(), bins=tbins)
            cnt5, edges5 = np.histogram(np.array(missstep_Combined2).ravel(), bins=tbins)
            timePoints3 = (edges3[1:] + edges3[:-1])/2
            timePoints4 = (edges4[1:] + edges4[:-1]) / 2
            timePoints5 = (edges5[1:] + edges5[:-1]) / 2
            avgFRStance3 =  cnt3/(np.sum(np.invert(psth[i]['indecisive']))* dt)   # normalized by the number of successful steps
            avgFRStance4 =  cnt4/(np.sum(psth[i]['indecisive']) * dt)  # normalized by the number of miss steps
            avgFRStance5 = cnt5 /(np.sum(psth[i]['indecisive']) * dt)  # normalized by the number of miss steps
            #pdb.set_trace()
            psth[i]['psth_stanceOnsetAlignedSuccessful'] = np.row_stack((timePoints3, avgFRStance3))
            psth[i]['psth_expectedStanceOnsetAlignedMissstep'] = np.row_stack((timePoints4, avgFRStance4))
            psth[i]['psth_stanceOnsetAlignedMissstep'] = np.row_stack((timePoints5, avgFRStance5))

            # swing-onset aligned
            cnt2, edges2 = np.histogram(np.concatenate(psth[i]['spikeTimesCenteredSwingStartSorted']).ravel(), bins=tbins)
            timePoints2 = (edges2[1:] + edges2[:-1]) / 2
            avgFRSwing =  cnt2 / (len(psth[i]['spikeTimesCenteredSwingStartSorted']) * dt)
            psth[i]['psth_swingOnsetAligned'] = np.row_stack((timePoints2, avgFRSwing))
            # swing-onset segreated in successful and mis-steps
            successful_SwingCombined = np.concatenate([arr for arr, flag in zip(psth[i]['spikeTimesCenteredSwingStartSorted'], psth[i]['indecisive']) if not flag]).ravel()
            missstep_SwingCombined = np.concatenate([arr for arr, flag in zip(psth[i]['spikeTimesCenteredSwingStartSorted'], psth[i]['indecisive']) if flag]).ravel()
            cnt3, edges3 = np.histogram(np.array(successful_SwingCombined).ravel(), bins=tbins)
            cnt4, edges4 = np.histogram(np.array(missstep_SwingCombined).ravel(), bins=tbins)
            timePoints3 = (edges3[1:] + edges3[:-1])/2
            timePoints4 = (edges4[1:] + edges4[:-1]) / 2
            avgFRStance3 =  cnt3/(np.sum(np.invert(psth[i]['indecisive']))* dt)   # normalized by the number of successful steps
            avgFRStance4 =  cnt4/(np.sum(psth[i]['indecisive']) * dt)  # normalized by the number of miss steps
            psth[i]['psth_swingOnsetAlignedSuccessful'] = np.row_stack((timePoints3, avgFRStance3))
            psth[i]['psth_swingOnsetAlignedMissstep'] = np.row_stack((timePoints4, avgFRStance4))

            # speed psth
            avgSpeedStance = np.average(np.asarray(psth[i]['speedCenteredStanceStart']),axis=0)
            avgSpeedSwing = np.average(np.asarray(psth[i]['speedCenteredSwingStart']), axis=0)
            avgSpeedStanceRescaled = np.average(np.asarray(psth[i]['speedCenteredStanceStartRescaled']),axis=0)
            #pdb.set_trace()
            psth[i]['psth_speedStanceOnsetAligned'] = np.row_stack((timePSTH, avgSpeedStance))
            psth[i]['psth_speedSwingOnsetAligned'] = np.row_stack((timePSTH, avgSpeedSwing))
            psth[i]['psth_speedStanceOnsetAlignedRescaled'] = np.row_stack((np.concatenate((rescaledTimePSTH,rescaledTimePSTH+1)), avgSpeedStanceRescaled))

            avgSpeedStanceSuccessful = np.average(np.asarray(psth[i]['speedCenteredStanceStart'])[np.invert(psth[i]['indecisive'])], axis=0)
            avgSpeedStanceMissstep = np.average(np.asarray(psth[i]['speedCenteredStanceStart'])[psth[i]['indecisive']], axis=0)
            avgSpeedStanceExpectedStanceOnsetMissstep = np.average(np.asarray(psth[i]['speedCenteredExpectedStanceStart'])[psth[i]['indecisive']], axis=0)
            psth[i]['psth_speedStanceOnsetAlignedSuccessful'] = np.row_stack((timePSTH, avgSpeedStanceSuccessful))
            psth[i]['psth_speedExpectedStanceOnsetAlignedMissstep'] = np.row_stack((timePSTH, avgSpeedStanceExpectedStanceOnsetMissstep))
            psth[i]['psth_speedStanceOnsetAlignedMissstep'] = np.row_stack((timePSTH, avgSpeedStanceMissstep))

            avgSpeedSwingSuccessful = np.average(np.asarray(psth[i]['speedCenteredSwingStart'])[np.invert(psth[i]['indecisive'])], axis=0)
            avgSpeedSwingMissstep = np.average(np.asarray(psth[i]['speedCenteredSwingStart'])[psth[i]['indecisive']], axis=0)
            psth[i]['psth_speedSwingOnsetAlignedSuccessful'] = np.row_stack((timePSTH, avgSpeedSwingSuccessful))
            psth[i]['psth_speedSwingOnsetAlignedMissstep'] = np.row_stack((timePSTH, avgSpeedSwingMissstep))
    return psth

###############
def calculatePSTHForShuffles(psth, spikes, pawPos, pawSpeed, swingStanceDict, strideProps, condition, nShuffles): #, control_var):
    jitterSTD = 0.5
    psthStanceOnset = {}
    psthStanceOnsetRes = {}
    psthSwingOnset = {}
    psthSwingOnsetSuccess = {}
    psthSwingOnsetMiss = {}
    psthStanceOnsetSuccess = {}
    psthStanceOnsetMiss = {}
    psthExpectedStanceOnsetMiss ={}
    before = [-0.1, 0.]
    after = [0., 0.1]
    #extended range for count
    before1 = [-0.2, 0.]
    after1 = [0., 0.2]
    #during=[-0.05,0.05]
    for i in range(4):
        psthStanceOnset[i] = []
        psthStanceOnsetRes[i] = []
        psthStanceOnsetSuccess[i] = []
        psthStanceOnsetMiss[i] = []
        psthExpectedStanceOnsetMiss[i] = []
        psthSwingOnset[i] = []
        psthSwingOnsetSuccess[i] = []
        psthSwingOnsetMiss[i] = []
    for n in range(nShuffles):
        #randomState=np.random.RandomState(100)
        spikesJittered = spikes + np.random.normal(0,jitterSTD,len(spikes))
        # spikesJittered = spikes + randomState.normal(0, jitterSTD, len(spikes))
        psthJitter = calculateStridebasedPSTH(spikesJittered,pawPos,pawSpeed,swingStanceDict,strideProps,condition) #,control_var)

        for i in range(4):
            if len(psthJitter[i]) > 0:
                psthStanceOnset[i].append(psthJitter[i]['psth_stanceOnsetAligned'][1])
                psthSwingOnset[i].append(psthJitter[i]['psth_swingOnsetAligned'][1])
                psthSwingOnsetSuccess[i].append(psthJitter[i]['psth_swingOnsetAlignedSuccessful'][1])
                psthSwingOnsetMiss[i].append(psthJitter[i]['psth_swingOnsetAlignedMissstep'][1])
                psthStanceOnsetRes[i].append(psthJitter[i]['psth_stanceOnsetAlignedRescaled'][1])
                psthStanceOnsetSuccess[i].append(psthJitter[i]['psth_stanceOnsetAlignedSuccessful'][1])
                psthStanceOnsetMiss[i].append(psthJitter[i]['psth_stanceOnsetAlignedMissstep'][1])
                psthExpectedStanceOnsetMiss[i].append(psthJitter[i]['psth_expectedStanceOnsetAlignedMissstep'][1])
                dt = np.diff(psthJitter[i]['psth_stanceOnsetAligned'][0])[0]
                dtRes = np.diff(psthJitter[i]['psth_stanceOnsetAlignedRescaled'][0])[0]
            else:
                pass

    #print('dt is :',dt)
    for i in range(4):
        if len(psthJitter[i]) > 0:
            tempStance = np.asarray(psthStanceOnset[i])
            tempStanceRes = np.asarray(psthStanceOnsetRes[i])
            tempSwing = np.asarray(psthSwingOnset[i])
            tempSwingSuc = np.asarray(psthSwingOnsetSuccess[i])
            tempSwingMis = np.asarray(psthSwingOnsetMiss[i])
            tempStanceSuc = np.asarray(psthStanceOnsetSuccess[i])
            tempStanceMis = np.asarray(psthStanceOnsetMiss[i])
            tempExpStanceMis = np.asarray(psthExpectedStanceOnsetMiss[i])
            psth[i]['psth_stanceOnsetAligned_5-50-95perentiles']= np.percentile(tempStance,[5,50,95],axis=0)
            psth[i]['psth_stanceOnsetAlignedRescaled_5-50-95perentiles'] = np.percentile(tempStanceRes, [5, 50, 95], axis=0)
            psth[i]['psth_swingOnsetAligned_5-50-95perentiles'] = np.percentile(tempSwing,[5, 50, 95],axis=0)
            psth[i]['psth_swingOnsetAlignedSuccessful_5-50-95perentiles'] = np.percentile(tempSwingSuc, [5, 50, 95], axis=0)
            psth[i]['psth_swingOnsetAlignedMissstep_5-50-95perentiles'] = np.percentile(tempSwingMis, [5, 50, 95], axis=0)
            psth[i]['psth_stanceOnsetAlignedSuccessful_5-50-95perentiles'] = np.percentile(tempStanceSuc, [5, 50, 95], axis=0)
            psth[i]['psth_stanceOnsetAlignedMissstep_5-50-95perentiles'] = np.percentile(tempStanceMis, [5, 50, 95], axis=0)
            psth[i]['psth_expectedStanceOnsetAlignedMissstep_5-50-95perentiles'] = np.percentile(tempExpStanceMis, [5, 50, 95], axis=0)
            #
            psth[i]['psth_stanceOnsetAligned_mean-std']=np.row_stack([np.mean(tempStance,axis=0),np.std(tempStance,axis=0)])
            psth[i]['psth_stanceOnsetAlignedRescaled_mean-std'] = np.row_stack([np.mean(tempStanceRes, axis=0), np.std(tempStanceRes, axis=0)])
            psth[i]['psth_swingOnsetAligned_mean-std']=np.row_stack([np.mean(tempSwing,axis=0),np.std(tempSwing,axis=0)])
            psth[i]['psth_swingOnsetAlignedSuccessful_mean-std'] = np.row_stack([np.mean(tempSwingSuc, axis=0), np.std(tempSwingSuc, axis=0)])
            psth[i]['psth_swingOnsetAlignedMissstep_mean-std'] = np.row_stack([np.mean(tempSwingMis, axis=0), np.std(tempSwingMis, axis=0)])
            psth[i]['psth_stanceOnsetAlignedSuccessful_mean-std'] = np.row_stack([np.mean(tempStanceSuc, axis=0), np.std(tempStanceSuc, axis=0)])
            psth[i]['psth_stanceOnsetAlignedMissstep_mean-std'] = np.row_stack([np.mean(tempStanceMis, axis=0), np.std(tempStanceMis, axis=0)])
            psth[i]['psth_expectedStanceOnsetAlignedMissstep_mean-std'] = np.row_stack([np.mean(tempExpStanceMis, axis=0), np.std(tempExpStanceMis, axis=0)])
            #
            psth[i]['psth_stanceOnsetAligned_z-scored']=(psth[i]['psth_stanceOnsetAligned']-np.mean(tempStance,axis=0))/np.std(tempStance,axis=0)
            psth[i]['psth_stanceOnsetAlignedRescaled_z-scored']=(psth[i]['psth_stanceOnsetAlignedRescaled'] - np.mean(tempStanceRes, axis=0)) / np.std(tempStanceRes, axis=0)
            psth[i]['psth_swingOnsetAligned_z-scored']=(psth[i]['psth_swingOnsetAligned']-np.mean(tempSwing,axis=0))/np.std(tempSwing,axis=0)
            psth[i]['psth_swingOnsetAlignedSuccessful_z-scored'] = (psth[i]['psth_swingOnsetAlignedSuccessful'] - np.mean(tempSwingSuc, axis=0)) / np.std(tempSwingSuc, axis=0)
            psth[i]['psth_swingOnsetAlignedMissstep_z-scored'] = (psth[i]['psth_swingOnsetAlignedMissstep'] - np.mean(tempSwingMis, axis=0)) / np.std(tempSwingMis, axis=0)
            psth[i]['psth_stanceOnsetAlignedSuccessful_z-scored'] = (psth[i]['psth_stanceOnsetAlignedSuccessful'] - np.mean(tempStanceSuc, axis=0)) / np.std(tempStanceSuc, axis=0)
            psth[i]['psth_stanceOnsetAlignedMissstep_z-scored'] = (psth[i]['psth_stanceOnsetAlignedMissstep'] - np.mean(tempStanceMis, axis=0)) / np.std(tempStanceMis, axis=0)
            psth[i]['psth_expectedStanceOnsetAlignedMissstep_z-scored'] = (psth[i]['psth_expectedStanceOnsetAlignedMissstep'] - np.mean(tempExpStanceMis, axis=0)) / np.std(tempExpStanceMis, axis=0)
            timeInterval=[0.1,0.15,0.2,0.25,0.3]

            for t in range(len(timeInterval)):
                zScore_AUC_Key_stance=['before_stanceOnset_z-score_AUC_%s'%timeInterval[t],'after_stanceOnset_z-score_AUC_%s'%timeInterval[t]]
                zScore_AUC_Key_swing=['before_swingOnset_z-score_AUC_%s'%timeInterval[t],'after_swingOnset_z-score_AUC_%s'%timeInterval[t]]
                zScore_peak_Key_stance=['before_stanceOnset_z-score_peak_%s'%timeInterval[t],'after_stanceOnset_z-score_peak_%s'%timeInterval[t]]
                zScore_peak_Key_swing=['before_swingOnset_z-score_peak_%s'%timeInterval[t],'after_swingOnset_z-score_peak_%s'%timeInterval[t]]
                #calculate z-score AUC
                beforeStanceOnsetMask=(psth[i]['psth_stanceOnsetAligned'][0]>-timeInterval[t]) & (psth[i]['psth_stanceOnsetAligned'][0]<0)
                afterStanceOnsetMask=(psth[i]['psth_stanceOnsetAligned'][0]>0) & (psth[i]['psth_stanceOnsetAligned'][0]<timeInterval[t])
                psth[i][zScore_AUC_Key_stance[0]]=np.trapz(psth[i]['psth_stanceOnsetAligned_z-scored'][1][beforeStanceOnsetMask], dx=dt)
                psth[i][zScore_AUC_Key_stance[1]] = np.trapz(psth[i]['psth_stanceOnsetAligned_z-scored'][1][afterStanceOnsetMask],dx=dt)
                psth[i][zScore_peak_Key_stance[0]]=np.max(abs(psth[i]['psth_stanceOnsetAligned_z-scored'][1][beforeStanceOnsetMask]))
                psth[i][zScore_peak_Key_stance[1]] = np.max(abs(psth[i]['psth_stanceOnsetAligned_z-scored'][1][afterStanceOnsetMask]))

                beforeSwingOnsetMask=(psth[i]['psth_swingOnsetAligned'][0]>-timeInterval[t]) & (psth[i]['psth_swingOnsetAligned'][0]<0)
                afterSwingOnsetMask=(psth[i]['psth_swingOnsetAligned'][0]>0) & (psth[i]['psth_swingOnsetAligned'][0]<timeInterval[t])
                psth[i][zScore_AUC_Key_swing[0]]= np.trapz(psth[i]['psth_swingOnsetAligned_z-scored'][1][beforeSwingOnsetMask],dx=dt)
                psth[i][zScore_AUC_Key_swing[1]] = np.trapz(psth[i]['psth_swingOnsetAligned_z-scored'][1][afterSwingOnsetMask],dx=dt)
                psth[i][zScore_peak_Key_swing[0]]=np.max(abs(psth[i]['psth_swingOnsetAligned_z-scored'][1][beforeSwingOnsetMask]))
                psth[i][zScore_peak_Key_swing[1]] = np.max(abs(psth[i]['psth_swingOnsetAligned_z-scored'][1][afterSwingOnsetMask]))
        else:
            pass
    #determine category of modulation
    labels = '↑↑', '↑-', '↑↓', '-↓', '↓↓', '↓-', '↓↑', '-↑', '--'
    for i in range(4):
        if len(psth[i])>0:
            pieCounts = np.zeros((2,9))
            modsBefore = []
            modsAfter = []
            modRecB = []
            modRecA = []
            countBefore=[]
            peakBefore=[]
            countAfter = []
            peakAfter=[]
            modulation = []
            kkeys = ['psth_swingOnsetAligned','psth_stanceOnsetAligned']

            for k in range(2): # loop over swing and stance onset aligned psth's
                ttime = psth[i][kkeys[k]][0]
                beforeMask = (ttime>before[0])&(ttime<before[1])
                afterMask = (ttime>after[0])&(ttime<after[1])
                #duringMask=(ttime>during[0])&(ttime<during[1]) #to implement later
                beforeMask1 = (ttime>before1[0])&(ttime<before1[1])
                afterMask1 = (ttime>after1[0])&(ttime<after1[1])
                pqq = psth[i][kkeys[k] + '_5-50-95perentiles']
                psth_base = psth[i][kkeys[k]][1]

                beforeEffect = True
                afterEffect = True

                #pdb.set_trace()
                conditionHighB = psth_base[beforeMask]>pqq[2,beforeMask]
                conditionLowB = psth_base[beforeMask]<pqq[0,beforeMask]
                #pdb.set_trace()
                consExcursions = 1 # means that at least (n+1) neighboring psth_base values have to cross the confidence interval

                if checkThresholdCrossingsForConsecutiveValues(conditionHighB,consExcursions):
                    bb = np.mean(psth_base[beforeMask][psth_base[beforeMask]>pqq[2,beforeMask]]/pqq[1,beforeMask][psth_base[beforeMask]>pqq[2,beforeMask]])-1.
                    before_peak=np.max(psth_base[beforeMask][psth_base[beforeMask]>pqq[2,beforeMask]]/pqq[1,beforeMask][psth_base[beforeMask]>pqq[2,beforeMask]])
                elif checkThresholdCrossingsForConsecutiveValues(conditionLowB, consExcursions):
                    bb = np.mean(psth_base[beforeMask][psth_base[beforeMask] < pqq[0,beforeMask]] / pqq[1,beforeMask][psth_base[beforeMask] < pqq[0,beforeMask]])-1.
                    before_peak = np.max(psth_base[beforeMask][psth_base[beforeMask] < pqq[0,beforeMask]] / pqq[1,beforeMask][psth_base[beforeMask] < pqq[0,beforeMask]])
                else:
                    beforeEffect = False

                conditionHighA = psth_base[afterMask]>pqq[2,afterMask]
                conditionLowA = psth_base[afterMask]<pqq[0,afterMask]
                if checkThresholdCrossingsForConsecutiveValues(conditionHighA, consExcursions):
                    aa =  np.mean(psth_base[afterMask][psth_base[afterMask]>pqq[2,afterMask]]/pqq[1,afterMask][psth_base[afterMask]>pqq[2,afterMask]])-1.
                    after_peak=np.max(psth_base[afterMask][psth_base[afterMask]>pqq[2,afterMask]]/pqq[1,afterMask][psth_base[afterMask]>pqq[2,afterMask]])
                elif checkThresholdCrossingsForConsecutiveValues(conditionLowA, consExcursions):
                    aa = np.mean(psth_base[afterMask][psth_base[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth_base[afterMask] < pqq[0,afterMask]])-1.
                    after_peak = np.max(psth_base[afterMask][psth_base[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth_base[afterMask] < pqq[0,afterMask]])
                else:
                    afterEffect = False
                #count the number of bins beyond the 100ms limit, here the limit is extended to 200ms
                conditionHighB1 = psth_base[beforeMask1]>pqq[2,beforeMask1]
                conditionLowB1 = psth_base[beforeMask1]<pqq[0,beforeMask1]
                if (beforeEffect) and (sum(conditionHighB1)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighB1 ) if key])>consExcursions):
                    before_count=len(psth_base[beforeMask1][psth_base[beforeMask1]>pqq[2,beforeMask1]])
                    #print('after-count :', before_count)
                elif (beforeEffect) and (sum(conditionLowB1)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowB1 ) if key])>consExcursions):
                    before_count = len(psth_base[beforeMask1][psth_base[beforeMask1] < pqq[0, beforeMask1]])
                    #print('after-count :', before_count)

                conditionHighA1 = psth_base[afterMask1]>pqq[2,afterMask1]
                conditionLowA1 = psth_base[afterMask1]<pqq[0,afterMask1]
                if (afterEffect) and (sum(conditionHighA1)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighA1 ) if key])>consExcursions):
                    after_count=len(psth_base[afterMask][psth_base[afterMask]>pqq[2,afterMask]])
                    #print('after-count :', after_count)
                elif (afterEffect) and (sum(conditionLowA1)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowA1 ) if key])>consExcursions):
                    after_count = len(psth_base[afterMask][psth_base[afterMask] < pqq[0, afterMask]])
                    #print('after-count :', after_count)


                #pointPlot = [False,-1,-1]
                if beforeEffect and afterEffect:
                    #ax3.plot(bb,aa,symList[(idxSym % len(symList))],color=cols[n],alpha=0.3+0.7*j/nRecs)
                    #pointPlot = [True,bb,aa]
                    if bb > 0 and aa > 0: pieCounts[k,0]+=1
                    elif bb>0 and aa<0: pieCounts[k,2]+=1
                    elif bb<0 and aa<0: pieCounts[k,4] += 1
                    elif bb<0 and aa>0: pieCounts[k,6] += 1
                elif beforeEffect and not afterEffect:
                    #pointPlot = [True,bb,0]
                    if bb>0: pieCounts[k,1] += 1
                    elif bb<0: pieCounts[k,5] += 1
                elif not beforeEffect and afterEffect:
                    #pointPlot = [True,0,aa]
                    if aa>0: pieCounts[k,7] += 1
                    elif aa<0: pieCounts[k,3] += 1
                elif not beforeEffect and not afterEffect:
                    pieCounts[k, 8] += 1
                if beforeEffect:
                    modRecB.append(bb)
                    countBefore.append(before_count)
                    peakBefore.append(before_peak)
                else:
                    modRecB.append(0)
                    countBefore.append(0)
                    peakBefore.append(0)
                if afterEffect:
                    modRecA.append(aa)
                    countAfter.append(after_count)
                    peakAfter.append(after_peak)
                else:
                    modRecA.append(0)
                    countAfter.append(0)
                    peakAfter.append(0)
                if (beforeEffect==True):
                    modulation.append([True, 'before'])
                elif afterEffect==True:
                    modulation.append([True, 'after'])
                else:
                    modulation.append([False, 'none'])
                event = kkeys[k][5:-7] # swing or stance onset
                psth[i]['modulation_%s' % event ]=modulation[k]
                psth[i]['modulation_before_%s' % event ] = modRecB[k]
                psth[i]['modulation_after_%s' % event ] = modRecA[k]

                psth[i]['modulation_count_before_%s' % event] = countBefore[k]
                psth[i]['modulation_count_after_%s' % event] = countAfter[k]

                psth[i]['modulation_peak_before_%s' % event] = peakBefore[k]
                psth[i]['modulation_peak_after_%s' % event] = peakAfter[k]

                psth[i]['modulation_category_%s' % event] = [pieCounts[k], labels[np.argmax(pieCounts[k]==1)]]
        else:
            pass
###############
def checkThresholdCrossingsForConsecutiveValues(cond,consExcursions):
    return (sum(cond)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( cond ) if key])>consExcursions)

###############
def calculatePSTH(pawPos,pawSpeed, spikes,swingStanceDict,strideProps,cond): #List,control_var):
    # create dictionary
    #psth = {}
    nShuffles = 400
    showFig = False
    # first calculate properties of all steps
    #strideProps = calculateStrideProperties(swingStanceDict, pawPos, pawSpeed)
    # calculate PSTH for specific condition
    #pdb.set_trace()
    psth = calculateStridebasedPSTH(spikes, pawPos, pawSpeed, swingStanceDict, strideProps, cond) #,control_var)
    # check if entries were added to the dictionary
    # create shuffles of the psth
    calculatePSTHForShuffles(psth, spikes, pawPos, pawSpeed,swingStanceDict, strideProps, cond, nShuffles) # cList ,nShuffles, control_var)

    if showFig:
        for i in range(4):
            plt.plot(psth[i]['psth_stanceOnsetAligned'][0],psth[i]['psth_stanceOnsetAligned'][1])
            plt.plot(psth[i]['psth_stanceOnsetAligned'][0],psth[i]['psth_stanceOnsetAligned_5-50-95perentiles'][0],c='0.6',ls=':')
            plt.plot(psth[i]['psth_stanceOnsetAligned'][0],psth[i]['psth_stanceOnsetAligned_5-50-95perentiles'][1],c='0.1',ls=':')
            plt.plot(psth[i]['psth_stanceOnsetAligned'][0],psth[i]['psth_stanceOnsetAligned_5-50-95perentiles'][2],c='0.6',ls=':')
            plt.show()
    return psth
###########################################################
def calculatePSTHopto(spkTimesImec,alignments):
    psthStartEnd = [-1,1]
    dt = 0.01
    pp = {}
    tbins = np.linspace(psthStartEnd[0], psthStartEnd[1], int((psthStartEnd[0] + psthStartEnd[1]) / dt) + 1, endpoint=True)
    for k in alignments:
        dtts = (spkTimesImec[:,None] -  alignments[k]).flatten() # compute all time difference between spikes and the event times

        cnt, edges = np.histogram(dtts, bins=tbins)
        timePoints = (edges[1:] + edges[:-1]) / 2
        avgFR = cnt / (len(dtts) * dt)
        pp[k] = np.row_stack((timePoints, avgFR))
    return pp




###########################################################
def fit_line(trial_num, data, ax):
    # Use polyfit to fit a straight line to the trial data
    slope, intercept = np.polyfit(trial_num, data, 1)
    # Generate points on the fitted line
    fit = np.polyfit(trial_num, data, 1)
    fit_fn = np.poly1d(fit)
    x_fit = np.linspace(min(trial_num), max(trial_num))
    y_fit = fit_fn(x_fit)

    ax.plot(x_fit, y_fit, '--k', alpha=0.1)
    # setting coordinates for slope value plotting
    y_last = data[-1]
    y_offset = 0.25 * (max(data) - min(data))
    y_text = y_last - y_offset
    x_text = max(trial_num)-max(trial_num)*0.05

    # Plot the slope value text
    ax.text(x_text, y_text, "Slope: {:.2f}".format(slope), transform=ax.transData, fontsize=8, color='grey')


def time_to_peak(signal, time, start_time, end_time):
    peak_indices, _ = scipy.signal.find_peaks(signal)
    peak_times = time[peak_indices]
    peak_times = peak_times[(peak_times >= start_time) & (peak_times <= end_time)]
    highest_peak_index = np.argmax(signal[peak_indices])
    timeTopeak=peak_times[highest_peak_index] - start_time
    return timeTopeak

def calculateStepParameters (pawPos, swingStanceD):
    linear=[False, True]
    type=['non_linear', 'linear']
    stepParameters= {}
    for l in range(2):
        stepParameters[type[l]]={}
        for i in range(4):
            stepParameters[type[l]][i]={}
            stepParameters[type[l]][i]['stepPos'] = []
            stepParameters[type[l]][i]['stepTime'] = []
            stepParameters[type[l]][i]['stepSpeed'] = []
            stepParameters[type[l]][i]['stepLength']=[]
            stepParameters[type[l]][i]['stepDuration']=[]
            stepParameters[type[l]][i]['stepMeanSpeed']=[]
        if linear[l]==True:
            pawPos=[]
            for i in range(4):
                pawPos.append(np.array(swingStanceD['forFit'][i][5]))
        else:
            pawPos=pawPos
        idxSwings_FL = swingStanceD['swingP'][0][1]
        idxSwings_FR = swingStanceD['swingP'][1][1]
        recTimes_FL = swingStanceD['forFit'][0][2]
        recTimes_FR = swingStanceD['forFit'][1][2]
    
        time_FL = pawPos[0][:, 0]
        time_FR = pawPos[1][:, 0]
    
        for j in range(len(idxSwings_FL) - 1):
            # get start and end indices of the swing
            if linear[l]==True:
                idxStart_FL = idxSwings_FL[j][0]
                idxEnd_FL = idxSwings_FL[j][1]
                idxStart1_FL = idxSwings_FL[j+1][0]
            else:
                idxStart_FL = np.argmin(np.abs(time_FL - recTimes_FL[idxSwings_FL[j][0]]))
                idxEnd_FL = np.argmin(np.abs(time_FL - recTimes_FL[idxSwings_FL[j][1]]))
                idxStart1_FL = np.argmin(np.abs(time_FL - recTimes_FL[idxSwings_FL[j + 1][0]]))
    
            stancePeriod_FL = pawPos[0][idxEnd_FL:idxStart1_FL, 0]
            stanceLength_FL = pawPos[0][idxEnd_FL:idxStart1_FL, 1]
            swingPeriod_FL = pawPos[0][idxStart_FL:idxEnd_FL, 0]
            swingPos_FL = pawPos[0][idxStart_FL:idxEnd_FL, 1]
            swingLength_FL=pawPos[0][idxEnd_FL,1]-pawPos[0][idxStart_FL,1]
            # if swingLength_FL < 0:
            #     print('negative step length spotted', swingPos_FL)
            #     print('paw position', pawPos[0][idxStart_FL:idxEnd_FL, 1])
            #     plt.plot(pawPos[0][:, 0], pawPos[0][:, 1])
            #     plt.plot(pawPos[1][:, 0], pawPos[1][:, 1])
            #     plt.plot(pawPos[0][idxStart_FL:idxEnd_FL, 0], pawPos[0][idxStart_FL:idxEnd_FL, 1], c='red')
            #     plt.xlim(pawPos[0][idxStart_FL, 0] - 2, (pawPos[0][idxStart_FL, 0] - 2) + 6)
            #     # plt.plot(FL_stepPos, FL_stepTime, c='red')
            #     plt.show()
            #     pdb.set_trace()
    
            for k in range(len(idxSwings_FR) - 1):
                if linear[l]==True:
                    idxStart_FR = idxSwings_FR[k][0]
                    idxEnd_FR= idxSwings_FR[k][1]
                    idxStart1_FR = idxSwings_FR[k + 1][0]
                else:
                    idxStart_FR = np.argmin(np.abs(time_FR - recTimes_FR[idxSwings_FR[k][0]]))
                    idxEnd_FR = np.argmin(np.abs(time_FR - recTimes_FR[idxSwings_FR[k][1]]))
                    idxStart1_FR = np.argmin(np.abs(time_FR - recTimes_FR[idxSwings_FR[k + 1][0]]))
    
                swingPeriod_FR = pawPos[1][idxStart_FR:idxEnd_FR, 0]
                swingLength_FR = pawPos[1][idxStart_FR:idxEnd_FR, 1]
    
                stancePeriod_FR = pawPos[1][idxEnd_FR:idxStart1_FR, 0]
                stanceLength_FR = pawPos[1][idxEnd_FR:idxStart1_FR, 1]
    
                maskFLStance = (pawPos[1][idxStart_FR:idxEnd_FR, 0] <= np.max(stancePeriod_FL)) & (
                        pawPos[1][idxStart_FR:idxEnd_FR, 0] >= np.min(
                    stancePeriod_FL))
    
                maskFL_position = (pawPos[1][idxStart_FR:idxEnd_FR, 1][maskFLStance] >= pawPos[0][idxStart_FR:idxEnd_FR, 1][
                    maskFLStance])
    
                #for FL paw, swing has to be in the stance period of FR
                maskFRStance = (pawPos[0][idxStart_FL:idxEnd_FL, 0] < np.max(stancePeriod_FR)) & (
                        pawPos[0][idxStart_FL:idxEnd_FL, 0] > np.min(stancePeriod_FR))  # & (pawPos[0][:,0]!=stancePeriod_FL)
    
                maskFR_position = (pawPos[0][idxStart_FL:idxEnd_FL, 1][maskFRStance] >= pawPos[1][idxStart_FL:idxEnd_FL, 1][
                    maskFRStance])
    
                FL_stepPos = ((pawPos[0][idxStart_FL:idxEnd_FL, 1][maskFRStance])[maskFR_position])
                # FL_stepPos = ((pawPos[0][idxStart_FL:idxEnd_FL, 1]))
                FL_stepTime = ((pawPos[0][idxStart_FL:idxEnd_FL, 0][maskFRStance])[maskFR_position])
                # FL_stepTime = ((pawPos[0][idxStart_FL:idxEnd_FL, 0]))
                FL_stepSpeed = np.diff(FL_stepPos) / np.diff(FL_stepTime)
    
    
    
    
    
                FR_stepPos = (pawPos[1][idxStart_FR:idxEnd_FR, 1][maskFLStance][maskFL_position])
                FR_stepTime = (pawPos[1][idxStart_FR:idxEnd_FR, 0][maskFLStance][maskFL_position])
                FR_stepSpeed = np.diff(FR_stepPos) / np.diff(FR_stepTime)
                if len(FL_stepPos) > 1:
                    FL_stepLength = FL_stepPos[-1] - FL_stepPos[0]
                    # if FL_stepLength<0:
                    #     print('negative step length spotted', FL_stepPos)
                    #     mask=(pawPos[0][:,0]>FL_stepTime[0]) & (pawPos[0][:,0]<FL_stepTime[-1])
                    #     plt.plot(pawPos[0][:,0], pawPos[0][:,1])
                    #     plt.plot(pawPos[1][:, 0], pawPos[1][:, 1])
                    #     plt.plot(pawPos[0][:,0][mask], pawPos[0][:,1][mask], c='red')
                    #     plt.xlim(np.min(pawPos[0][:,0][mask])-2, (np.min(pawPos[0][:,0][mask])-2)+6)
                    #     # plt.plot(FL_stepPos, FL_stepTime, c='red')
                    #     plt.show()
                    #     pdb.set_trace()
    
                    FL_stepDuration = FL_stepTime[-1] - FL_stepTime[0]
                    FL_stepMeanSpeed = np.mean(FL_stepSpeed)
                    stepParameters[type[l]][0]['stepLength'].append(FL_stepLength)
                    stepParameters[type[l]][0]['stepDuration'].append(FL_stepDuration)
                    stepParameters[type[l]][0]['stepMeanSpeed'].append(FL_stepMeanSpeed)
                    stepParameters[type[l]][0]['stepPos'].append(FL_stepPos)
                    stepParameters[type[l]][0]['stepTime'].append(FL_stepTime)
                    stepParameters[type[l]][0]['stepSpeed'].append(FL_stepSpeed)
                if len(FR_stepPos) > 1:
                    FR_stepLength = FR_stepPos[-1] - FR_stepPos[0]
                    FR_stepDuration = FR_stepTime[-1] - FR_stepTime[0]
                    FR_stepMeanSpeed = np.mean(FR_stepSpeed)
                    stepParameters[type[l]][1]['stepLength'].append(FR_stepLength)
                    stepParameters[type[l]][1]['stepDuration'].append(FR_stepDuration)
                    stepParameters[type[l]][1]['stepMeanSpeed'].append(FR_stepMeanSpeed)
                    stepParameters[type[l]][1]['stepPos'].append(FR_stepPos)
                    stepParameters[type[l]][1]['stepTime'].append(FR_stepTime)
                    stepParameters[type[l]][1]['stepSpeed'].append(FR_stepSpeed)
    
        idxSwings_HL = swingStanceD['swingP'][2][1]
        idxSwings_HR = swingStanceD['swingP'][3][1]
        recTimes_HL = swingStanceD['forFit'][2][2]
        recTimes_HR = swingStanceD['forFit'][3][2]
        time_HL = pawPos[2][:, 0]
        time_HR = pawPos[3][:, 0]
    
        for j in range(len(idxSwings_HL) - 1):
            # get start and end indices of the swing
            if linear[l]==True:
                idxStart_HL = idxSwings_HL[j][0]
                idxEnd_HL = idxSwings_HL[j][1]
                idxStart1_HL = idxSwings_HL[j + 1][0]
            else:
                idxStart_HL = np.argmin(np.abs(time_HL - recTimes_HL[idxSwings_HL[j][0]]))
                idxEnd_HL = np.argmin(np.abs(time_HL - recTimes_HL[idxSwings_HL[j][1]]))
                idxStart1_HL = np.argmin(np.abs(time_HL - recTimes_HL[idxSwings_HL[j + 1][0]]))
    
            stancePeriod_HL = pawPos[2][idxEnd_HL:idxStart1_HL, 0]
            stanceLength_HL = pawPos[2][idxEnd_HL:idxStart1_HL, 1]
            swingPeriod_HL = pawPos[2][idxStart_HL:idxEnd_HL, 0]
            swingLength_HL = pawPos[2][idxStart_HL:idxEnd_HL, 1]
    
            for k in range(len(idxSwings_HR) - 1):
                if linear[l]==True:
                    idxStart_HR = idxSwings_HR[k][0]
                    idxEnd_HR = idxSwings_HR[k][1]
                    idxStart1_HR = idxSwings_HR[k + 1][0]
                else:
                    idxStart_HR = np.argmin(np.abs(time_HR - recTimes_HR[idxSwings_HR[k][0]]))
                    idxEnd_HR = np.argmin(np.abs(time_HR - recTimes_HR[idxSwings_HR[k][1]]))
                    idxStart1_HR = np.argmin(np.abs(time_HR - recTimes_HR[idxSwings_HR[k + 1][0]]))
    
                swingPeriod_HR = pawPos[3][idxStart_HR:idxEnd_HR, 0]
                swingLength_HR = pawPos[3][idxStart_HR:idxEnd_HR, 1]
    
                stancePeriod_HR = pawPos[3][idxEnd_HR:idxStart1_HR, 0]
                stanceLength_HR = pawPos[3][idxEnd_HR:idxStart1_HR, 1]
    
                maskHLStance = (pawPos[3][idxStart_HR:idxEnd_HR, 0] <= np.max(stancePeriod_HL)) & (
                        pawPos[2][idxStart_HR:idxEnd_HR, 0] >= np.min(
                    stancePeriod_HL))  # & np.invert((pawPos[1][:,0]<np.max(stancePeriod_HR)) & (pawPos[1][:,0]>np.min(stancePeriod_HR)))
    
                maskHL_position = (pawPos[3][idxStart_HR:idxEnd_HR, 1][maskHLStance] >= pawPos[2][idxStart_HR:idxEnd_HR, 1][
                    maskHLStance])
    
                maskHRStance = (pawPos[3][idxStart_HL:idxEnd_HL, 0] < np.max(stancePeriod_HR)) & (
                        pawPos[2][idxStart_HL:idxEnd_HL, 0] > np.min(
                    stancePeriod_HR))  # & (pawPos[2][:,0]!=stancePeriod_HL)
    
                maskHR_position = (pawPos[3][idxStart_HL:idxEnd_HL, 1][maskHRStance] >= pawPos[2][idxStart_HL:idxEnd_HL, 1][
                    maskHRStance])
    
                HL_stepPos = (pawPos[0][idxStart_HL:idxEnd_HL, 1][maskHRStance][maskHR_position])
                HL_stepTime = (pawPos[0][idxStart_HL:idxEnd_HL, 0][maskHRStance][maskHR_position])
                HL_stepSpeed = np.diff(HL_stepPos) / np.diff(HL_stepTime)
    
                HR_stepPos = (pawPos[1][idxStart_HR:idxEnd_HR, 1][maskHLStance][maskHL_position])
                HR_stepTime = (pawPos[1][idxStart_HR:idxEnd_HR, 0][maskHLStance][maskHL_position])
                HR_stepSpeed = np.diff(HR_stepPos) / np.diff(HR_stepTime)
                if len(HL_stepPos) > 1:
                    HL_stepLength = HL_stepPos[-1] - HL_stepPos[0]
                    HL_stepDuration = HL_stepTime[-1] - HL_stepTime[0]
                    HL_stepMeanSpeed = np.mean(HL_stepSpeed)
                    stepParameters[type[l]][2]['stepLength'].append(HL_stepLength)
                    stepParameters[type[l]][2]['stepDuration'].append(HL_stepDuration)
                    stepParameters[type[l]][2]['stepMeanSpeed'].append(HL_stepMeanSpeed)
                    stepParameters[type[l]][2]['stepPos'].append(HL_stepPos)
                    stepParameters[type[l]][2]['stepTime'].append(HL_stepTime)
                    stepParameters[type[l]][2]['stepSpeed'].append(HL_stepSpeed)
                if len(HR_stepPos) > 1:
                    HR_stepLength = HR_stepPos[-1] - HR_stepPos[0]
                    HR_stepDuration = HR_stepTime[-1] - HR_stepTime[0]
                    HR_stepMeanSpeed = np.mean(HR_stepSpeed)
                    stepParameters[type[l]][3]['stepLength'].append(HR_stepLength)
                    stepParameters[type[l]][3]['stepDuration'].append(HR_stepDuration)
                    stepParameters[type[l]][3]['stepMeanSpeed'].append(HR_stepMeanSpeed)
                    stepParameters[type[l]][3]['stepPos'].append(HR_stepPos)
                    stepParameters[type[l]][3]['stepTime'].append(HR_stepTime)
                    stepParameters[type[l]][3]['stepSpeed'].append(HR_stepSpeed)

    return stepParameters


def calculateStepPar(pawPos, swingStanceD):
    linear = [False, True]
    type = ['non_linear', 'linear']
    stepParameters = {}
    
    for l in range(2):
        stepParameters[type[l]] = {}
        if linear[l] == True:
            pawPos = []
            for i in range(4):
                pawPos.append(np.array(swingStanceD['forFit'][i][5]))
        else:
            pawPos = pawPos
        for i in range(4):
            stepParameters[type[l]][i] = {}
            stepParameters[type[l]][i]['stepPos'] = []
            stepParameters[type[l]][i]['stepTime'] = []
            stepParameters[type[l]][i]['stepSpeed'] = []
            stepParameters[type[l]][i]['stepLength'] = []
            stepParameters[type[l]][i]['stepDuration'] = []
            stepParameters[type[l]][i]['stepMeanSpeed'] = []

            idxSwings= swingStanceD['swingP'][i][1]
            recTimes = swingStanceD['forFit'][i][2]
            pawTimes= pawPos[i][:, 0]
            for j in range(len(idxSwings) - 1):
                # get start and end indices of the swing
                if linear[l] == True:
                    idxStart= idxSwings[j][0]
                    idxEnd= idxSwings[j][1]
                    idxStart1= idxSwings[j + 1][0]
                else:
                    idxStart = np.argmin(np.abs(pawTimes - recTimes[idxSwings[j][0]]))
                    idxEnd = np.argmin(np.abs(pawTimes - recTimes[idxSwings[j][1]]))
                    idxStart1 = np.argmin(np.abs(pawTimes - recTimes[idxSwings[j + 1][0]]))
                if i==0:
                    refPawPositionTime=pawPos[1][:, 0]
                    refPawPosition = pawPos[1][:, 1]
                elif i==1:
                    refPawPositionTime=pawPos[0][:, 0]
                    refPawPosition = pawPos[0][:, 1]
                elif i==2:
                    refPawPositionTime=pawPos[3][:, 0]
                    refPawPosition = pawPos[3][:, 1]
                elif i==3:
                    refPawPositionTime=pawPos[2][:, 0]
                    refPawPosition = pawPos[2][:, 1]
                swingPeriod  = pawPos[i][idxStart :idxEnd , 0]
                swingPos  = pawPos[i][idxStart :idxEnd , 1]
                swingLength  = pawPos[i][idxEnd , 1] - pawPos[i][idxStart , 1]
                mask_swing= (refPawPositionTime>=pawPos[i][idxStart , 0]) & (refPawPositionTime<pawPos[i][idxEnd , 0])
                if len(swingPos) >len(refPawPosition[mask_swing]):
                    lengthLimit=len(refPawPosition[mask_swing])
                else:
                    lengthLimit = len(swingPos)
                mask_Foward=(swingPos[:lengthLimit]>refPawPosition[mask_swing][:lengthLimit])
                stepPos = pawPos[i][idxStart :idxEnd , 1][:lengthLimit][mask_Foward]
                stepTime=pawPos[i][idxStart :idxEnd , 0][:lengthLimit][mask_Foward]
                if len(stepPos)>2:
                    stepLength = stepPos[-1] - stepPos[0]
                    stepSpeed = np.diff(stepPos) / np.diff(stepTime)
                    stepDuration = stepTime[-1] - stepTime[0]
                    stepMeanSpeed = np.mean(stepSpeed)
                    # if stepLength<0:
                    # print('negative step length spotted', stepLength)
                    #mask=(pawPos[0][:,0]>stepTime[0]) & (pawPos[0][:,0]<stepTime[-1])
                    #plt.plot(pawPos[0][:,0], pawPos[0][:,1])
                    #"plt.plot(pawPos[1][:, 0], pawPos[1][:, 1])
                    #plt.plot(pawPos[0][idxStart :idxEnd , 0], pawPos[0][idxStart :idxEnd , 1], c='green')
                    #plt.xlim(np.min(pawPos[0][:,0][mask])-2, (np.min(pawPos[0][:,0][mask])-2)+6)
                    #plt.plot(stepTime,stepPos, c='red', alpha=0.5)
                    #plt.show()
                    stepParameters[type[l]][i]['stepLength'].append(stepLength)
                    stepParameters[type[l]][i]['stepDuration'].append(stepDuration)
                    stepParameters[type[l]][i]['stepMeanSpeed'].append(stepMeanSpeed)
                    stepParameters[type[l]][i]['stepPos'].append(stepPos)
                    stepParameters[type[l]][i]['stepTime'].append(stepTime)
                    stepParameters[type[l]][i]['stepSpeed'].append(stepSpeed)
    return stepParameters