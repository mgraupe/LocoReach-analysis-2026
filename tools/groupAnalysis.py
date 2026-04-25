import time

import numpy
import numpy as np
import statistics
import sys
import os
import scipy, scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import pdb
import math
from statsmodels.formula.api import ols
import scipy.ndimage
import itertools
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
# import cv2
from scipy import signal
from scipy.signal import find_peaks
from scipy import stats
import pickle
import pandas as pd
import seaborn as sns
from scipy import ndimage
from matplotlib import rcParams
import matplotlib.pyplot as plt
#import pymer4.models as models
from mpl_toolkits.axes_grid1 import AxesGrid
#import rpy2.robjects as rob
#from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.packages import STAP
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import inf
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaResults
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pingouin as pg
from scipy.ndimage import gaussian_filter1d
import tools.dataAnalysis_psth as dataAnalysis_psth
from sklearn.preprocessing import normalize
from tools.pyqtgraph.Qt import QtGui, QtCore
import tools.pyqtgraph as pyg
# matplotlib.use('WxAgg')
numpy.set_printoptions(threshold=sys.maxsize)

groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/'
###########################################################
def getArrayTolerantAverage(Array):
    avg = []

    for n in range(10):
        oneSession = np.zeros(1)
        nAnimal = 0
        for i in range(len(Array)):
            if len(Array[i][1]) > n:
                oneSession += np.array(Array[i][1][n])
                nAnimal += 1
        avg.append([n, nAnimal, oneSession / nAnimal])
    avgValue = [avg[r][2] for r in range(len(avg))]
    return (avg,avgValue)

###########################################################
def get2DArrayTolerantAverage(Array):
    Avg = []
    AvgValue=[]

    for n in range(14):
        shape = np.shape(Array[0][1])
        oneSession = np.zeros(shape[1])
        nAnimal = 0
        for i in range(len(Array)):
            #oneSession = np.zeros(len(Array[i][1][n]))
            if len(Array[i][1]) > n:
                #oneSession = np.zeros(len(Array[i][1][n]))
                oneSession += np.array(Array[i][1][n])
                nAnimal += 1
        Avg.append([n, nAnimal, oneSession / nAnimal])
    AvgValue = [Avg[r][2] for r in range(len(Avg))]
    return (Avg,AvgValue)
###########################################################
def tolerant_mean(arrs):
    arrs1= [arrs[r][1] for r in range(len(arrs))]
    lens = [len(i) for i in arrs1]
    arr = np.ma.empty((np.max(lens),len(arrs1)))
    arr.mask = True
    for idx, l in enumerate(arrs1):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), np.std(arr,axis=-1)

def getMeanStdNan (array):
    array1= [array[r][1] for r in range(len(array))]
    #array1=np.asarray(array1)
    a=np.empty((len(array1),10))
    a[:]=np.nan
    for i in range(len(a)):
        nRecs=len(array1[i])
        a[i, :nRecs]=array1[i]

    arrayStd=np.nanstd((a),axis=0)
    arrayMean=np.nanmean((a),axis=0)
    arraySem=stats.sem((a),axis=0,nan_policy='omit')

    return (arrayMean,arrayStd, arraySem)

def getMeanStdNan2DArray (array):
    array1= [array[r][1] for r in range(len(array))]
    shape=np.shape(array1)
    a=np.empty((len(array1),14,shape[1]))
    a[:]=np.nan
    for i in range(len(a)):
        nRecs = len(array1[i])
        for j in range(shape[1]):
           a[i, :nRecs][j]=array1[i][j]
    arrayStd=np.nanstd((a),axis=0)
    arrayMean=np.nanmean((a),axis=0)
    arraySem=stats.sem((a),axis=0,nan_policy='omit')
    return (arrayMean,arrayStd, arraySem)
###########################################################
def convertListToPandasDF(data, trialValues, treatments, fixedLength=None):

    if ( trialValues==True and treatments==False) :
        maxDim = 4
    elif ( trialValues==True and treatments==True):
        maxDim=5
    elif ( trialValues==False and treatments==True):
        maxDim = 4
    else:
        maxDim = 3
    my_array = np.empty((1, maxDim),dtype=object)
    # pdb.set_trace()
    for i in range(len(data)):
        values = np.asanyarray(data[i][1])
        if fixedLength is None:

            nRecDays = len(values)
        else:
            nRecDays = fixedLength
        if  trialValues:

             trials = np.shape(values)[1]
            #print( trials)
        else:
             trials=1
        tempArray = np.empty((nRecDays* trials, maxDim),dtype=object)
        tempArray[:, 0] = np.repeat(data[i][0], nRecDays* trials)
        tempArray[:, 1] = np.repeat(np.arange(nRecDays)+1, trials)
        #pdb.set_trace()
        tempArray[:, 2] = values[:nRecDays].flatten()

        if  trialValues==True and treatments==False:
            tempArray[:, 3] = np.tile(np.arange( trials),nRecDays)
        elif  trialValues==True and treatments==True:
            tempArray[:, 3] = np.tile(np.arange( trials), nRecDays)
            tempArray[:,4]=np.repeat(data[i][2], nRecDays* trials)
        elif  trialValues == False and treatments == True:
            tempArray[:, 3] = np.repeat(data[i][2], nRecDays*1)
        my_array = np.concatenate((my_array, tempArray))

    my_array = my_array[1:] # remove first line which was created before
    # print('my array',my_array)
    if  trialValues==True and treatments==False:
        df = pd.DataFrame(my_array, columns=['mouse','recordingDay','measuredValue','trialNumber'])
        convert_dict = {'mouse': str,'recordingDay': int,'measuredValue':float,'trialNumber':int}
    elif  trialValues==True and treatments==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue', 'trialNumber','treatments'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'trialNumber': int, 'treatments': str}

    elif  trialValues==False and treatments==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue','treatments'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'treatments': str}

    else:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue'])
        convert_dict = {'mouse': str, 'recordingDay': int,'measuredValue': float}

    df = df.astype(convert_dict)
    # print(df.describe())

    return df


###########################################################
def performRepeatedMeasuresANOVA(allMiceData,sessionValues=False, treatments=False):
    # determine animals with the shortest number of recording days
    nShotestRecDays = 100
    for i in range(len(allMiceData)):

        if len(allMiceData[i][1])<nShotestRecDays:
            nShotestRecDays = len(allMiceData[i][1])
    print('shortest recording length (days) : ',nShotestRecDays)

    df = convertListToPandasDF(allMiceData,sessionValues,treatments,fixedLength=nShotestRecDays)
    if sessionValues==True :
        AnovaRMRes=AnovaRM(data=df, depvar='measuredValue', subject='mouse', within=['recordingDay'],aggregate_func='sum').fit()

        AnovaRMSesRes=AnovaRM(data=df, depvar='measuredValue', subject='mouse', within=['sessionNumber'], aggregate_func='sum').fit()

        if AnovaRMSesRes.anova_table.iloc[0,3]<0.05:
            print(sp.posthoc_tukey(df, val_col='measuredValue', group_col='sessionNumber'))

            posthocS=sp.posthoc_tukey(df, val_col='measuredValue', group_col='sessionNumber')
            posthocR=sp.posthoc_tukey(df, val_col='measuredValue', group_col='recordingDay')
            anovaResults=[AnovaRMRes, AnovaRMSesRes,posthocS,posthocR]
        return (anovaResults)
    else:
        AnovaRMRes=AnovaRM(data=df, depvar='measuredValue', subject='mouse', within=['recordingDay'], missing='drop').fit()
        if AnovaRMRes.anova_table.iloc[0, 3] < 0.05:
            print(sp.posthoc_tukey(df, val_col='measuredValue', group_col='recordingDay'))
            posthocS=sp.posthoc_tukey(df, val_col='measuredValue', group_col='recordingDay')
            anovaResults = [AnovaRMRes, posthocS]
        return (anovaResults)

def performMixedTwoWayANOVA(allMiceData,sessionValues=False, treatments=False):
    print('session values', sessionValues, 'treatment', treatments)
    if sessionValues == True and treatments==False:
        df = convertListToPandasDF(allMiceData,sessionValues, treatments)
        anovaTW = pg.mixed_anova(data=df, dv='measuredValue', within=['recordingDay','sessionNumber'], subject='mouse',effsize="np2").round(3)
    elif sessionValues == True and treatments==True:
        df = convertListToPandasDF(allMiceData, sessionValues, treatments)
        anovaTW = pg.mixed_anova(data=df, dv='measuredValue', subject='mouse', within='recordingDay',
                                 between='treatments', effsize='np2').round(3)
        posthoc = pg.pairwise_ttests(data=df, dv='measuredValue', subject='mouse',within='recordingDay', interaction=True, within_first=True, marginal=True, between='treatments')
        posthoc = pairwise_tukeyhsd(endog=df['measuredValue'], groups=df['treatments'], alpha=0.05)
        posthocR=posthoc[['A','B','T','dof','p-unc']].copy()
        df.to_csv('/home/andry/Documents/stats/Step_Number_Muscimol.csv')
    elif sessionValues == False and treatments==True:
        df = convertListToPandasDF(allMiceData, sessionValues, treatments)
        anovaTW = pg.mixed_anova(data=df, dv='measuredValue', subject='mouse',within='recordingDay', between='treatments',effsize='np2').round(3)
        # posthoc=MultiComparison(df['measuredValue'],groups=df['recordingDay'])
        # posthocR=posthoc.allpairtest(0.05)
        # posthocR.summary()
        #posthocR = sp.posthoc_nemenyi_friedman(df, y_col='measuredValue', block_col='treatments', group_col='recordingDay', melted=True)
        posthoc = pg.pairwise_ttests(data=df, dv='measuredValue', subject='mouse',within='recordingDay', interaction=True, within_first=True, marginal=True, between='treatments')
        posthocR=posthoc[['A','B','T','dof','p-unc']].copy()
        # posthoc=pairwise_tukeyhsd(endog=df['measuredValue'], groups=df['treatments'], alpha=0.05)
        # posthocR = pd.DataFrame(data=posthoc._results_table.data[1:], columns=posthoc._results_table.data[0])


    else:
        # for i in range(len(allMiceData)):
        #     #newList.append([allMiceData[i][0],allMiceData[i][1]])
        df = convertListToPandasDF(allMiceData, sessionValues, treatments)
        anovaTW = pg.mixed_anova(data=df, dv='measuredValue', subject='mouse',within=['recordingDay'],
                              effsize="np2").round(3)

    df.to_csv(groupAnalysisDir+'anova.csv', sep=',')
    return (anovaTW,posthocR)

###########################################################
def performMixedLinearModelRegression(allMiceData, treatments, trialValues):
    print('trial values', trialValues, 'treatment', treatments)
    # pdb.set_trace()
    if trialValues == True and treatments==False:

        df = convertListToPandasDF(allMiceData,trialValues, treatments)
        md = smf.mixedlm("measuredValue ~ recordingDay*trialNumber", df, groups=df["mouse"],missing='drop')
        mdf = md.fit()
        # print(mdf.summary())
        conf_int = pd.DataFrame(mdf.conf_int())
        pvalues = {"recording":mdf.pvalues['recordingDay'], "trial":mdf.pvalues['trialNumber']}
        stars = {"recording": starMultiplier(pvalues['recording']), "trial": starMultiplier(pvalues['trial'])}
    elif trialValues == True and treatments==True:
        df = convertListToPandasDF(allMiceData, trialValues, treatments)
        md = smf.mixedlm("measuredValue ~ recordingDay*treatments*trialNumber", df, groups=df["mouse"], missing='drop')
        mdf = md.fit()
        # print(mdf.summary())
        conf_int = pd.DataFrame(mdf.conf_int())
        pvalues = {"recording": mdf.pvalues['recordingDay'], "trial": mdf.pvalues['trialNumber'], "treatment":mdf.pvalues['treatments[T.saline]']}
        stars = {"recording": starMultiplier(pvalues['recording']), "trial": starMultiplier(pvalues['trial']), "treatments": starMultiplier(pvalues['treatment'])}
    elif trialValues == False and treatments==True:
        df = convertListToPandasDF(allMiceData, trialValues, treatments)
        md = smf.mixedlm("measuredValue ~ treatments*recordingDay", df, groups=df["mouse"], missing='drop')
        mdf = md.fit()
        print(mdf.summary())
        conf_int = pd.DataFrame(mdf.conf_int())
        pvalues = {"recording": mdf.pvalues['recordingDay'], "treatment":mdf.pvalues['treatments[T.saline]']}
        stars = {"recording": starMultiplier(pvalues['recording']), "treatments": starMultiplier(pvalues['treatment'])}
    else:
        # for i in range(len(allMiceData)):
        #     #newList.append([allMiceData[i][0],allMiceData[i][1]])
        df = convertListToPandasDF(allMiceData, trialValues, treatments)
        # pdb.set_trace()
        md = smf.mixedlm("measuredValue ~ recordingDay ", df, groups=df["mouse"], missing='drop')
        mdf = md.fit()
        # print(mdf.summary())
        conf_int = pd.DataFrame(mdf.conf_int())
        pvalues = {"recording": mdf.pvalues['recordingDay']}
        stars = {"recording": starMultiplier(pvalues['recording'])}
    return (df,mdf,conf_int,pvalues,stars)


def extractStepNumber(mouseDict, experiment, treatments=True):
    sumStepsAllMice = []
    averageStepsAllMice = []
    AvgAllPawsStepNumber = []
    trialNbArray=[]
    dayNbArray=[]


    mice_swingNumber_paws_trials_days=[]

    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            listOfRecordings=mouseDict[a]['listOfRecordings']
            maxTrials = 10
            maxDays = 10
        else:
            listOfRecordings = mouseDict[a]['foldersRecordings']
            maxTrials = 5
            maxDays = 10
        pawsStepNumber_mouse=[]
        nDays = len(recs)
        for n in range(nDays):
            allPawstrialStepNumber=np.empty(4)
            print('number of recordings : ', len(recs[n][4]))
            nTrial=len(recs[n][4])

            for j in range(nTrial):
                pawStepNumber = []
                wSpeedArray = []
                for i in range(4):
                    # print(len(recs[n][4][j][3][i][1]))
                    idxSwings = recs[n][4][j][3][i][1]
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    wSpeed = recs[n][4][j][4][i][0]
                    wSpeed = np.asarray(wSpeed)
                    mask = (recTimes[idxSwings[:, 0]] > 10.) & (recTimes[idxSwings[:, 0]] < 52)
                    #get step number for 5 sessions of recording day
                    pawStepNumber=np.append(pawStepNumber, len(idxSwings[mask])) #/ len(recs[n][4])
                allPawstrialStepNumber=np.vstack((allPawstrialStepNumber,pawStepNumber))
            pawsStepNumber_mouse.append(allPawstrialStepNumber[1:])


        if experiment=='ephy':
            pawsStepNumber_mouse=equalizeTrialNumberWithNan(pawsStepNumber_mouse,listOfRecordings)
        else:
            pawsStepNumber_mouse=pawsStepNumber_mouse
        pawsStepNumber_mouse_nan=np.empty((maxDays,maxTrials,4))
        pawsStepNumber_mouse_nan.fill(np.nan)
        for w in range(len(pawsStepNumber_mouse)):
            for x in range(len(pawsStepNumber_mouse[w])):
                for c in range(len(pawsStepNumber_mouse[w][x])):
                    pawsStepNumber_mouse_nan[w][x][c]=pawsStepNumber_mouse[w][x][c]

        AvgPawsStepNumber=np.nanmean(pawsStepNumber_mouse_nan, axis=1)
        dayStepNumber=np.nanmean(pawsStepNumber_mouse_nan, axis=2)
        dayAvgStepNumber=np.nanmean(dayStepNumber,axis=1)
        # pdb.set_trace()
        #equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(maxTrials)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(dayAvgStepNumber)<maxDays:
            dayAvgStepNumber=np.insert(dayAvgStepNumber,len(dayAvgStepNumber),np.nan)
            dayStepNumber=np.vstack((dayStepNumber,nanArrayTrials))
            AvgPawsStepNumber=np.vstack((AvgPawsStepNumber,nanArrayPaws))


        if treatments==False :
            averageStepsAllMice.append([mouseDict[a]['mouseName'],dayAvgStepNumber])
            sumStepsAllMice.append([mouseDict[a]['mouseName'],np.array(dayStepNumber)])
            AvgAllPawsStepNumber.append([mouseDict[a]['mouseName'], AvgPawsStepNumber])
            mice_swingNumber_paws_trials_days.append([mouseDict[a]['mouseName'],pawsStepNumber_mouse_nan])
        else:
            averageStepsAllMice.append([mouseDict[a]['mouseName'],dayAvgStepNumber,mouseDict[a]['treatment']])
            sumStepsAllMice.append([mouseDict[a]['mouseName'],np.array(dayStepNumber), mouseDict[a]['treatment']])
            AvgAllPawsStepNumber.append([mouseDict[a]['mouseName'],AvgPawsStepNumber, mouseDict[a]['treatment']])
            mice_swingNumber_paws_trials_days.append([mouseDict[a]['mouseName'], pawsStepNumber_mouse_nan,mouseDict[a]['treatment']])
    return (averageStepsAllMice, sumStepsAllMice,AvgAllPawsStepNumber,listOfRecordings,mice_swingNumber_paws_trials_days)


###########################################################
def extractStepDuration(mouseDict, experiment, treatments):
    maxDays=10
    maxTrials=5
    mice_swingDuration_Mouse_Day_Avg = []
    mice_swingDuration_Mouse_Day_Trials_Avg = []
    mice_swingDuration_Mouse_Paw_Avg = []
    mice_stanceDuration_Mouse_Day_Avg = []
    mice_stanceDuration_Mouse_Day_Trials_Avg = []
    mice_stanceDuration_Mouse_Paw_Avg = []
    mice_swingDuration_Mouse_Day_Trials_Paw=[]
    mice_stanceDuration_Mouse_Day_Trials_Paw=[]
    mice_f_Mouse_Day_Trials_Paw=[]
    mice_swingDurStd_Mouse_Day_Trials_Paw = []
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            maxTrials = 10
            listOfRecordings=mouseDict[a]['listOfRecordings']
        else:
            maxTrials = 5
            listOfRecordings = mouseDict[a]['foldersRecordings']

        lpaw = ['FR', 'FL', 'HL', 'HR']

        nDays = len(recs)
        swingDuration_Mouse = []
        stanceDuration_Mouse = []
        f_Mouse = []
        swingDurStd_Mouse = []
        for n in range(nDays):
            swingDuration_Day=[]
            stanceDuration_Day=[]
            f_Day=[]
            swingDurStd_Day= []
            for j in range(len(recs[n][4])):
                swingDuration_Paw=[]
                stanceDuration_Paw=[]
                f_Paw=[]
                swingDurStd_Paw=[]
                for i in range(4):
                    swingD=[]
                    stanceD=[]
                    f=[]

                    idxSwings = recs[n][4][j][3][i][1]
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    pawPos = np.array(recs[n][2][j][5][i])
                    linearPawPos = recs[n][4][j][4][i][5]
                    # pdb.set_trace()
                    # only look at steps during motorization period
                    pawPos=np.asarray(pawPos)
                    recTimes = np.asarray(recTimes)
                    idxSwings = np.asarray(idxSwings)
                    for k in range(len(idxSwings)-1):
                        idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                        idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))
                        idxSwingStartNext = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k + 1][0]]))
                        idxStanceStartNext = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k + 1][1]]))
                        if (recTimes[idxSwings[k, 0]] > 10.) and (recTimes[idxSwings[k, 0]]<52):  # only look at steps during motorization period
                            swingDuration=pawPos[:, 0][idxStanceStart] - pawPos[:, 0][idxSwingStart]
                            stanceDuration=pawPos[:, 0][idxSwingStartNext] - pawPos[:, 0][idxStanceStart]
                            swingD.append(swingDuration)
                            stanceD.append(stanceDuration)
                            f.append(1/swingDuration+stanceDuration)


                    #using median duration of swings for each paws
                    swingDurMean=np.mean(swingD)
                    swingDurStd=np.std(swingD)
                    stanceDurMean=np.mean(stanceD)
                    fMean=np.mean(f)
                    #regrouping paw values
                    swingDuration_Paw.append(swingDurMean)
                    stanceDuration_Paw.append(stanceDurMean)
                    f_Paw.append(fMean)
                    swingDurStd_Paw.append(swingDurStd)
                # pdb.set_trace()
                #regroup individual paw values
                swingDuration_Day.append(swingDuration_Paw)
                stanceDuration_Day.append(stanceDuration_Paw)
                f_Day.append(f_Paw)
                swingDurStd_Day.append(swingDurStd_Paw)
                 #regrouping trial values
            swingDuration_Mouse.append(swingDuration_Day)
            stanceDuration_Mouse.append(stanceDuration_Day)
            f_Mouse.append(f_Day)
            swingDurStd_Mouse.append(swingDurStd_Day)
        if experiment=='ephy':
            swingDuration_Mouse = equalizeTrialNumberWithNan(swingDuration_Mouse, listOfRecordings)
            stanceDuration_Mouse = equalizeTrialNumberWithNan(stanceDuration_Mouse, listOfRecordings)
            f_Mouse = equalizeTrialNumberWithNan(f_Mouse, listOfRecordings)
            swingDurStd_Mouse=equalizeTrialNumberWithNan(swingDurStd_Mouse, listOfRecordings)
        else:
            swingDuration_Mouse=swingDuration_Mouse
            stanceDuration_Mouse=stanceDuration_Mouse
            f_Mouse =f_Mouse
            swingDurStd_Mouse=swingDurStd_Mouse
        swingDuration_mouse_nan=np.empty((maxDays,maxTrials,4))
        swingDuration_mouse_nan.fill(np.nan)
        stanceDuration_mouse_nan=np.empty((maxDays,maxTrials,4))
        stanceDuration_mouse_nan.fill(np.nan)
        f_mouse_nan=np.empty((maxDays,maxTrials,4))
        f_mouse_nan.fill(np.nan)
        swingDurStd_Mouse_nan=np.empty((maxDays,maxTrials,4))
        swingDurStd_Mouse_nan.fill(np.nan)
        for w in range(len(swingDuration_Mouse)):
            for x in range(len(swingDuration_Mouse[w])):
                for c in range(len(swingDuration_Mouse[w][x])):
                    swingDuration_mouse_nan[w][x][c]=swingDuration_Mouse[w][x][c]
                    stanceDuration_mouse_nan[w][x][c] = stanceDuration_Mouse[w][x][c]
                    f_mouse_nan[w][x][c] = f_Mouse[w][x][c]
                    swingDurStd_Mouse_nan[w][x][c] = swingDurStd_Mouse[w][x][c]
        swingDuration_Mouse_Day_Trials_Avg = np.nanmean(swingDuration_mouse_nan, axis=2)
        swingDuration_Mouse_Paw_Avg = np.nanmean(swingDuration_mouse_nan, axis=1)
        swingDuration_Mouse_Day_Avg = np.nanmean(swingDuration_Mouse_Paw_Avg, axis=1)
        
        stanceDuration_Mouse_Day_Trials_Avg = np.nanmean(stanceDuration_mouse_nan, axis=2)
        stanceDuration_Mouse_Paw_Avg = np.nanmean(stanceDuration_mouse_nan, axis=1)
        stanceDuration_Mouse_Day_Avg = np.nanmean(stanceDuration_Mouse_Paw_Avg, axis=1)

        # pdb.set_trace()
        # equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(maxTrials)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(swingDuration_Mouse_Day_Avg) < maxDays:
            swingDuration_Mouse_Day_Avg = np.insert(swingDuration_Mouse_Day_Avg,len(swingDuration_Mouse_Day_Avg), np.nan)
            swingDuration_Mouse_Day_Trials_Avg = np.vstack((swingDuration_Mouse_Day_Trials_Avg, nanArrayTrials))
            swingDuration_Mouse_Paw_Avg = np.vstack((swingDuration_Mouse_Paw_Avg, nanArrayPaws))
        while len(stanceDuration_Mouse_Day_Avg) < maxDays:
            stanceDuration_Mouse_Day_Avg = np.insert(stanceDuration_Mouse_Day_Avg,len(stanceDuration_Mouse_Day_Avg), np.nan)
            stanceDuration_Mouse_Day_Trials_Avg = np.vstack((stanceDuration_Mouse_Day_Trials_Avg, nanArrayTrials))
            stanceDuration_Mouse_Paw_Avg = np.vstack((stanceDuration_Mouse_Paw_Avg, nanArrayPaws))

        if treatments==False :
            mice_swingDuration_Mouse_Day_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Day_Avg])
            mice_swingDuration_Mouse_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Day_Trials_Avg])
            mice_swingDuration_Mouse_Paw_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Paw_Avg])
            mice_swingDuration_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingDuration_mouse_nan])

            mice_stanceDuration_Mouse_Day_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Day_Avg])
            mice_stanceDuration_Mouse_Day_Trials_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Day_Trials_Avg])
            mice_stanceDuration_Mouse_Paw_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Paw_Avg])
            mice_stanceDuration_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], stanceDuration_mouse_nan])

            mice_f_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], f_mouse_nan])
            mice_swingDurStd_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingDurStd_Mouse_nan])
        else:
            mice_swingDuration_Mouse_Day_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Day_Avg,mouseDict[a]['treatment']])
            mice_swingDuration_Mouse_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_swingDuration_Mouse_Paw_Avg.append([mouseDict[a]['mouseName'], swingDuration_Mouse_Paw_Avg,mouseDict[a]['treatment']])
            mice_swingDuration_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingDuration_mouse_nan,mouseDict[a]['treatment']])

            mice_stanceDuration_Mouse_Day_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Day_Avg,mouseDict[a]['treatment']])
            mice_stanceDuration_Mouse_Day_Trials_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_stanceDuration_Mouse_Paw_Avg.append([mouseDict[a]['mouseName'], stanceDuration_Mouse_Paw_Avg,mouseDict[a]['treatment']])
            mice_stanceDuration_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], stanceDuration_mouse_nan,mouseDict[a]['treatment']])

            mice_f_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], f_mouse_nan,mouseDict[a]['treatment']])
            mice_swingDurStd_Mouse_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingDurStd_Mouse_nan, mouseDict[a]['treatment']])
    # pdb.set_trace()
    return (mice_swingDuration_Mouse_Day_Avg,mice_swingDuration_Mouse_Day_Trials_Avg,mice_stanceDuration_Mouse_Day_Avg,mice_stanceDuration_Mouse_Day_Trials_Avg,mice_swingDuration_Mouse_Paw_Avg,mice_stanceDuration_Mouse_Paw_Avg,mice_swingDuration_Mouse_Day_Trials_Paw,mice_stanceDuration_Mouse_Day_Trials_Paw,mice_f_Mouse_Day_Trials_Paw,mice_swingDurStd_Mouse_Day_Trials_Paw)






###########################################################
def getStepSpeed(mouseDict, experiment, treatments):
    mice_swingSpeed_Day_Trials_Avg = []
    mice_swingSpeed_Paw_Avg =[]
    mice_swingSpeed_Day_Avg =[]
    mice_swingSpeed_Day_Trials_paw = []
    mice_fastSwingsFraction_Day_Trials_paw=[]
    mice_highAcceleration_Day_Trials_paw=[]
    mice_accelerationNb_Day_Trials_paw = []
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy' or experiment=='test':
            maxTrials = 10
            listOfRecordings=mouseDict[a]['listOfRecordings']
        else:
            listOfRecordings = mouseDict[a]['foldersRecordings']
            maxTrials = 5
        nDays = len(recs)
        swingSpeed_Mouse = []
        pawSpeedDay=[]
        pawSpeedDayAvg=[]
        fastSwingsFraction_Mouse=[]
        highAcceleration_Mouse=[]
        accelerationNb_Mouse = []
        for n in range(nDays):
           swingSpeed_Day = []
           fastSwingsFraction_day=[]
           highAcceleration_day=[]
           accelerationNb_day = []
           for j in range(len(recs[n][4])):
                swingSpeed_Paw = []
                fastSwingsFraction_paw=[]
                highAcceleration_paw=[]
                accelerationNb_paw=[]
                pawList = ['FL', 'FR', 'HL', 'HR']
                for i in range(4):
                    swingSpeed_Array=[]
                    meanFastSwingFraction=[]
                    acceleration_Array = []
                    meanAccelerationNb=[]
                    meanSwingSpeed=[]
                    # pawPos   = recs[n][2][j][5][i]
                    # linearPawPos = recs[n][4][j][4][i][5]
                    pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    recTimes = recs[n][4][j][4][i][2]
                    xSpeed = recs[n][4][j][4][i][1]
                    wSpeed = recs[n][4][j][4][i][0]
                    idxSwings = np.asarray(idxSwings)
                    xSpeed = np.asarray(xSpeed)
                    wSpeed = np.asarray(wSpeed)
                    pawPos = recs[n][2][j][5][i]
                    linearPawPos = recs[n][4][j][4][i][5]
                    stepCharacter = recs[n][4][j][3][i][3]
                    rungNumbers = recs[n][4][j][3][i][2]
                    newXtime=linearPawPos[:,0]
                    speedTimes=recTimes
                    # if (recTimes[:]>10) & (recTimes[:]<52):
                    totalDistance=np.sum((wSpeed*np.gradient(recTimes)))
                    fig = plt.figure(figsize=(20, 7))

                    ax1 = fig.add_subplot(1, 2, 1)
                    ax2 = fig.add_subplot(1, 2, 2)

                    ax1.plot(speedTimes,xSpeed * 0.025 - wSpeed)
                    ax1.plot(speedTimes, wSpeed)
                    ax2.plot(pawPos[:,0],pawPos[:,1])

                    ax1.set_xlim(40, 45)
                    ax1.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                    ax1.set_xlabel('time (s)')
                    ax1.set_ylabel('speed (cm)')
                    ax2.set_xlim(40, 45)
                    # ax2.set_ylim(150, 250)
                    ax2.set_title(f'{pawList[i]} paw', color=f'C{i}', fontsize=16)
                    ax2.set_xlabel('time (s)')
                    ax2.set_ylabel('position (cm)')

                    for k in range(len(idxSwings)):

                        if (recTimes[idxSwings[k, 0]] > 10.) and (recTimes[idxSwings[k, 0]] < 52) :  # only look at steps during motorization period
                            idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                            idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))

                            pawPosSwing=pawPos[:,1][idxSwingStart:idxStanceStart]
                            pawPosSwing_time=pawPos[:,0][idxSwingStart:idxStanceStart]

                            swingSpeed=abs(xSpeed[idxSwings[k, 0]:idxSwings[k, 1]] * 0.025 - wSpeed[idxSwings[k, 0]:idxSwings[k, 1]])
                            pawSpeedTime=recTimes[idxSwings[k, 0]:idxSwings[k, 1]]
                            # acceleration=np.gradient(swingSpeed,pawSpeedTime)
                            ax1.plot(pawSpeedTime, swingSpeed, color='C1', alpha=0.6)
                            ax2.plot(pawPosSwing_time,pawPosSwing, color='C1', alpha=0.6)

                            accelerationDic = calc_acceleration(swingSpeed, pawSpeedTime)
                            meanSwingSpeed.append(np.mean(swingSpeed))
                            acceleration=np.array(accelerationDic['acceleration'])
                            pdb.set_trace()


                            #
                            # swingSpeed_Array=np.append(swingSpeed_Array,swingSpeed)
                            # acceleration_Array=np.append(acceleration_Array, acceleration)
                            #
                            # meanAccelerationNb.append(accelerationDic['acceleration_phases'])

                    # swingSpeed_Array_Scaled=NormalizeData(swingSpeed_Array)
                    # acceleration_Array_Scaled = NormalizeData(acceleration_Array)
                    swingSpeed_Paw.append(np.mean(meanSwingSpeed))
                    # fastSwingsFraction_paw.append(len(swingSpeed_Array_Scaled[swingSpeed_Array_Scaled>np.percentile(swingSpeed_Array_Scaled,75)])/len(swingSpeed_Array_Scaled))
                    highAcceleration_paw.append(np.mean(acceleration_Array))
                    accelerationNb_paw.append(np.mean(meanAccelerationNb))
                # print(swingSpeed_Array)
                # print(swingSpeed_Paw)

                swingSpeed_Day.append(swingSpeed_Paw)
                highAcceleration_day.append(highAcceleration_paw)
                fastSwingsFraction_day.append(fastSwingsFraction_paw)
                accelerationNb_day.append(accelerationNb_paw)
                #
                # print('paw',i, 'rec',j, 'day', n, swingSpeed_Day)
                # pdb.set_trace()
           swingSpeed_Mouse.append(swingSpeed_Day)
           highAcceleration_Mouse.append(highAcceleration_day)
           fastSwingsFraction_Mouse.append(fastSwingsFraction_day)
           accelerationNb_Mouse.append(accelerationNb_day)
        if experiment=='ephy':
            swingSpeed_Mouse = equalizeTrialNumberWithNan(swingSpeed_Mouse, listOfRecordings)
            highAcceleration_Mouse = equalizeTrialNumberWithNan(highAcceleration_Mouse, listOfRecordings)
            # fastSwingsFraction_Mouse = equalizeTrialNumberWithNan(fastSwingsFraction_Mouse, listOfRecordings)
            accelerationNb_Mouse = equalizeTrialNumberWithNan(accelerationNb_Mouse, listOfRecordings)
        else:
            swingSpeed_Mouse=swingSpeed_Mouse
            highAcceleration_Mouse = highAcceleration_Mouse
            # fastSwingsFraction_Mouse=fastSwingsFraction_Mouse
            accelerationNb_Mouse = accelerationNb_Mouse

        swingSpeed_Mouse_nan=np.empty((maxDays,maxTrials,4))
        highAcceleration_Mouse_nan=np.empty((maxDays,maxTrials,4))
        fastSwingsFraction_Mouse_nan=np.empty((maxDays,maxTrials,4))
        accelerationNb_Mouse_nan=np.empty((maxDays,maxTrials,4))

        highAcceleration_Mouse_nan.fill(np.nan)
        fastSwingsFraction_Mouse_nan.fill(np.nan)
        accelerationNb_Mouse_nan.fill(np.nan)
        swingSpeed_Mouse_nan.fill(np.nan)
        for w in range(len(swingSpeed_Mouse)):
            for x in range(len(swingSpeed_Mouse[w])):
                for c in range(len(swingSpeed_Mouse[w][x])):
                    if swingSpeed_Mouse[w][x][c]==inf:
                        swingSpeed_Mouse_nan[w][x][c] == np.nan
                    else:
                        swingSpeed_Mouse_nan[w][x][c]=swingSpeed_Mouse[w][x][c]
                        # fastSwingsFraction_Mouse_nan[w][x][c] = fastSwingsFraction_Mouse[w][x][c]
                        accelerationNb_Mouse_nan[w][x][c] = accelerationNb_Mouse[w][x][c]
                        highAcceleration_Mouse_nan[w][x][c] = highAcceleration_Mouse[w][x][c]


        swingSpeed_Day_Trials_Avg=np.nanmean(swingSpeed_Mouse_nan, axis=2)
        swingSpeed_Paw_Avg=np.nanmean(swingSpeed_Mouse_nan,axis=1)
        swingSpeed_Day_Avg=np.nanmean(swingSpeed_Paw_Avg, axis=1)
        # pdb.set_trace()
        # swingSpeed_Day_Avg[swingSpeed_Day_Avg== -inf ]=np.nan
        # swingSpeed_Mouse_nan[swingSpeed_Mouse_nan == inf] = np.nan
        # swingSpeed_Day_Avg[swingSpeed_Day_Avg== -inf ]=np.nan
        # swingSpeed_Day_Avg[swingSpeed_Day_Avg == inf] = np.nan
        #equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(maxTrials)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(swingSpeed_Day_Avg)<maxDays:
            swingSpeed_Day_Avg=np.insert(swingSpeed_Day_Avg,len(swingSpeed_Day_Avg),np.nan)
            swingSpeed_Day_Trials_Avg=np.vstack((swingSpeed_Day_Trials_Avg,nanArrayTrials))
            swingSpeed_Paw_Avg=np.vstack((swingSpeed_Paw_Avg,nanArrayPaws))

                    #print(np.percentile(pawSpeed[i], 90, overwrite_input=False, interpolation='linear', keepdims=True))

        if treatments==False :
            mice_swingSpeed_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingSpeed_Day_Trials_Avg])
            mice_swingSpeed_Day_Avg.append([mouseDict[a]['mouseName'],swingSpeed_Day_Avg])
            mice_swingSpeed_Paw_Avg.append([mouseDict[a]['mouseName'],swingSpeed_Paw_Avg])
            mice_swingSpeed_Day_Trials_paw.append([mouseDict[a]['mouseName'],swingSpeed_Mouse_nan])
            mice_fastSwingsFraction_Day_Trials_paw .append([mouseDict[a]['mouseName'],fastSwingsFraction_Mouse_nan])
            mice_highAcceleration_Day_Trials_paw .append([mouseDict[a]['mouseName'],highAcceleration_Mouse_nan])
            mice_accelerationNb_Day_Trials_paw .append([mouseDict[a]['mouseName'],accelerationNb_Mouse_nan])
        else:
            mice_swingSpeed_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingSpeed_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_swingSpeed_Day_Avg.append([mouseDict[a]['mouseName'], swingSpeed_Day_Avg,mouseDict[a]['treatment']])
            mice_swingSpeed_Paw_Avg.append([mouseDict[a]['mouseName'],swingSpeed_Paw_Avg,mouseDict[a]['treatment']])
            mice_swingSpeed_Day_Trials_paw.append([mouseDict[a]['mouseName'],swingSpeed_Mouse_nan,mouseDict[a]['treatment']])
            mice_fastSwingsFraction_Day_Trials_paw .append([mouseDict[a]['mouseName'],fastSwingsFraction_Mouse_nan,mouseDict[a]['treatment']])
            mice_highAcceleration_Day_Trials_paw .append([mouseDict[a]['mouseName'],highAcceleration_Mouse_nan,mouseDict[a]['treatment']])
            mice_accelerationNb_Day_Trials_paw .append([mouseDict[a]['mouseName'],accelerationNb_Mouse_nan,mouseDict[a]['treatment']])
    # pdb.set_trace()
    return (mice_swingSpeed_Day_Trials_Avg,mice_swingSpeed_Day_Avg,mice_swingSpeed_Paw_Avg,mice_swingSpeed_Day_Trials_paw,mice_fastSwingsFraction_Day_Trials_paw,mice_highAcceleration_Day_Trials_paw,mice_accelerationNb_Day_Trials_paw)

###########################################################
def getFractionRungCrossed(mouseDict,experiment, treatments):
    mice_twoRungCrossFraction_Day_Trials_paw=[]
    mice_twoRungCrossFraction_Day_Avg=[]
    mice_twoRungCrossFraction_Day_Trials_Avg=[]
    mice_twoRungCrossFraction_Paw_Avg=[]
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            maxTrials = 10
            listOfRecordings=mouseDict[a]['listOfRecordings']
        else:
            maxTrials = 5
            listOfRecordings = mouseDict[a]['foldersRecordings']
        recs=mouseDict[a]['pawData']
        nDays = len(recs)
        twoRungCrossFraction_Mouse=[]
        for n in range(nDays):

            nTrials=len(recs[n][4])

            twoRungCrossFraction_Day=[]
            for j in range(len(recs[n][4])):
                twoRungCrossFraction=[[],[],[],[]]
                for i in range(4):
                    idxSwings = recs[n][4][j][3][i][1]
                    rungNumbers = recs[n][4][j][3][i][2]
                    # for k in range(len(rungNumbers)):
                    rungCrossed=np.diff(rungNumbers)
                    twoRungCrossFraction[i]=np.count_nonzero(np.greater_equal(rungCrossed,[2]))/len(rungNumbers)
                twoRungCrossFraction_Day.append(twoRungCrossFraction)
            twoRungCrossFraction_Mouse.append(twoRungCrossFraction_Day)
        if experiment=='ephy':
            twoRungCrossFraction_Mouse = equalizeTrialNumberWithNan(twoRungCrossFraction_Mouse, listOfRecordings)
        else:
            twoRungCrossFraction_Mouse=twoRungCrossFraction_Mouse

        twoRungCrossFraction_Mouse_nan=np.empty((maxDays,maxTrials,4))
        twoRungCrossFraction_Mouse_nan.fill(np.nan)
        for w in range(len(twoRungCrossFraction_Mouse)):
            for x in range(len(twoRungCrossFraction_Mouse[w])):
                for c in range(len(twoRungCrossFraction_Mouse[w][x])):
                    twoRungCrossFraction_Mouse_nan[w][x][c]=twoRungCrossFraction_Mouse[w][x][c]

        twoRungCrossFraction_Day_Trials_Avg = np.nanmean(twoRungCrossFraction_Mouse_nan, axis=2)
        twoRungCrossFraction_Paw_Avg = np.nanmean(twoRungCrossFraction_Mouse_nan, axis=1)
        twoRungCrossFraction_Day_Avg = np.nanmean(twoRungCrossFraction_Paw_Avg, axis=1)
        # equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(maxTrials)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(twoRungCrossFraction_Day_Avg) < maxDays:
            twoRungCrossFraction_Day_Avg = np.insert(twoRungCrossFraction_Day_Avg, len(twoRungCrossFraction_Day_Avg), np.nan)
            twoRungCrossFraction_Day_Trials_Avg = np.vstack((twoRungCrossFraction_Day_Trials_Avg, nanArrayTrials))
            twoRungCrossFraction_Paw_Avg = np.vstack((twoRungCrossFraction_Paw_Avg, nanArrayPaws))
        # pdb.set_trace()
        if treatments==False :
            mice_twoRungCrossFraction_Day_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Day_Avg])
            mice_twoRungCrossFraction_Day_Trials_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Day_Trials_Avg])
            mice_twoRungCrossFraction_Paw_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Paw_Avg])
            mice_twoRungCrossFraction_Day_Trials_paw.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Mouse_nan])
        else:
            mice_twoRungCrossFraction_Day_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Day_Avg,mouseDict[a]['treatment']])
            mice_twoRungCrossFraction_Day_Trials_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_twoRungCrossFraction_Paw_Avg.append([mouseDict[a]['mouseName'], twoRungCrossFraction_Paw_Avg,mouseDict[a]['treatment']])
            mice_twoRungCrossFraction_Day_Trials_paw.append(
                [mouseDict[a]['mouseName'], twoRungCrossFraction_Mouse_nan, mouseDict[a]['treatment']])
    return (mice_twoRungCrossFraction_Day_Trials_Avg,mice_twoRungCrossFraction_Day_Avg,mice_twoRungCrossFraction_Paw_Avg,mice_twoRungCrossFraction_Day_Trials_paw)

###########################################################
def getStrideLength(mouseDict, experiment, treatments):
    mice_swingLen_Day_Trials_Avg =[]
    mice_swingLen_Day_Paw_Avg =[]
    mice_swingLen_Day_Avg = []
    mice_swingLen_Day_Trials_Paw =[]
    mice_swingLenStd_Day_Trials_Paw =[]
    mice_wheelDistance=[]
    mice_wheelSpeed=[]
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            maxTrials = 10
            listOfRecordings=mouseDict[a]['listOfRecordings']
        else:
            maxTrials = 5
            listOfRecordings = mouseDict[a]['foldersRecordings']

        recs=mouseDict[a]['pawData']
        nDays = len(recs)
        swingLenStd_Mouse=[]
        swingLen_Mouse=[]
        wheelDistance_Mouse=[]
        wheelSpeed_Mouse = []
        for n in range(nDays):
            wheelDistance_Day=[]
            swingLen_Day=[]
            swingLenStd_Day=[]
            wheelSpeed_Day=[]
            for j in range(len(recs[n][4])):
                pawPosRecordingSwing = np.empty((4,250, 1000))
                pawPosRecordingSwing.fill(np.nan)
                pawTimeRecordingSwing = np.empty((4,250, 1000))
                pawTimeRecordingSwing.fill(np.nan)
                swingLen_Paws=[]
                swingLenStd_Paws = []
                wheelLinearSpeed = recs[n][1][j][1]
                wheelTime = recs[n][1][j][2]
                wheelTimeMask=(wheelTime>10)&(wheelTime<52)
                # pdb.set_trace()
                wheelDistance = np.sum(wheelLinearSpeed[wheelTimeMask] * np.gradient(wheelTime[wheelTimeMask]))
                avgLinWheelSpeed=np.mean(wheelLinearSpeed[wheelTimeMask])
                wheelSpeed_Day.append([avgLinWheelSpeed,avgLinWheelSpeed,avgLinWheelSpeed,avgLinWheelSpeed])
                wheelDistance_Day.append([wheelDistance,wheelDistance,wheelDistance,wheelDistance])
                for i in range(4):

                    recTimes = recs[n][4][j][4][i][2]
                    pawPos = recs[n][2][j][5][i]
                    linearPawPos = recs[n][4][j][4][i][5]
                    pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    #stepCharacter = recs[n][4][j][3][i][3]
                    #stepC[i, n, 1] += len(stepCharacter)
                    wheelAngularSpeed=recs[n][1][j][0]

                    wheelsTimestamp = recs[n][1][j][3]
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    swingLen = []
                    swingTimes = []
                    pawPosSwing=[]
                    # pdb.set_trace()
                    for k in range(len(idxSwings)):
                        idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                        idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))
                        mask = (linearPawPos[:, 0] >= recTimes[idxSwings[k, 0]]) & (linearPawPos[:, 0] <= recTimes[idxSwings[k, 1]])
                        linSwingPawPos = linearPawPos[:, 1][idxSwingStart:idxStanceStart]
                        swingTime = linearPawPos[:, 0][idxSwingStart:idxStanceStart]
                        pawPosSwing.append(linSwingPawPos)
                        swingLen.append(linearPawPos[:, 1][idxStanceStart] - linearPawPos[:, 1][idxSwingStart])

                        swingTimes.append(swingTime)

                    # plt.plot(linearPawPos[:, 0][idxSwings[k, 0]:idxSwings[k, 1]],linearPawPos[:, 1][idxSwings[k, 0]:idxSwings[k, 1]])
                    # plt.show()
                    swingLenArray=[]
                    for s in range (len(pawPosSwing)):
                        swingLenArray.append(len(pawPosSwing[s]))
                    maxSwingLen=np.max(swingLenArray)

                    swingLen_Paws.append(np.mean(swingLen))
                    swingLenStd_Paws.append(np.std(swingLen))
                # pdb.set_trace()
                swingLen_Day.append(swingLen_Paws)
                swingLenStd_Day.append(swingLenStd_Paws)
            swingLen_Mouse.append(swingLen_Day)
            swingLenStd_Mouse.append(swingLenStd_Day)
            wheelDistance_Mouse.append(wheelDistance_Day)
            wheelSpeed_Mouse.append(wheelSpeed_Day)
        # pdb.set_trace()
        if experiment=='ephy':
            swingLen_Mouse=equalizeTrialNumberWithNan(swingLen_Mouse,listOfRecordings)
            swingLenStd_Mouse=equalizeTrialNumberWithNan(swingLenStd_Mouse,listOfRecordings)
            wheelDistance_Mouse=equalizeTrialNumberWithNan(wheelDistance_Mouse,listOfRecordings)
            wheelSpeed_Mouse=equalizeTrialNumberWithNan(wheelSpeed_Mouse,listOfRecordings)
        else:
            swingLen_Mouse=swingLen_Mouse
            swingLenStd_Mouse=swingLenStd_Mouse
            wheelDistance_Mouse=wheelDistance_Mouse
            wheelSpeed_Mouse=wheelSpeed_Mouse
        wheelSpeed_Mouse_nan=np.empty((maxDays,maxTrials,4))
        wheelSpeed_Mouse_nan.fill(np.nan)
        wheelDistance_Mouse_nan=np.empty((maxDays,maxTrials,4))
        wheelDistance_Mouse_nan.fill(np.nan)
        swingLen_Mouse_nan=np.empty((maxDays,maxTrials,4))
        swingLen_Mouse_nan.fill(np.nan)
        swingLenStd_Mouse_nan=np.empty((maxDays,maxTrials,4))
        swingLenStd_Mouse_nan.fill(np.nan)
        for w in range(len(swingLen_Mouse)):
            for x in range(len(swingLen_Mouse[w])):
                # pdb.set_trace()

                for c in range(len(swingLen_Mouse[w][x])):
                    swingLen_Mouse_nan[w][x][c]=swingLen_Mouse[w][x][c]
                    swingLenStd_Mouse_nan[w][x][c]=swingLenStd_Mouse[w][x][c]
                    wheelDistance_Mouse_nan[w][x][c] = wheelDistance_Mouse[w][x][c]
                    wheelSpeed_Mouse_nan[w][x][c] = wheelSpeed_Mouse[w][x][c]
        wheelDistance_Mouse_nan_mean=np.nanmean(wheelDistance_Mouse_nan,axis=1)
        wheelSpeed_Mouse_nan_mean=np.nanmean(wheelSpeed_Mouse_nan,axis=1)
        
        swingLen_Day_Trials_Avg=np.nanmean(swingLen_Mouse_nan, axis=2)
        swingLen_Day_Paw_Avg=np.nanmean(swingLen_Mouse_nan,axis=1)
        swingLen_Day_Avg=np.nanmean(swingLen_Day_Paw_Avg, axis=1)
        #equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(maxTrials)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(swingLen_Day_Avg)<maxDays:
            wheelDistance_Mouse=np.insert(wheelDistance_Mouse,len(wheelDistance_Mouse),np.nan)
            swingLen_Day_Avg=np.insert(swingLen_Day_Avg,len(swingLen_Day_Avg),np.nan)
            swingLen_Day_Trials_Avg=np.vstack((swingLen_Day_Trials_Avg,nanArrayTrials))
            swingLen_Day_Paw_Avg=np.vstack((swingLen_Day_Paw_Avg,nanArrayPaws))

        # pdb.set_trace()
        if treatments==False :
            #trials values
            mice_swingLen_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Trials_Avg])
            #Paw values
            mice_swingLen_Day_Paw_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Paw_Avg])
            #average accross trials and paws
            mice_swingLen_Day_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Avg])
            mice_swingLen_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingLen_Mouse_nan])
            mice_swingLenStd_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingLenStd_Mouse_nan])
            mice_wheelDistance.append([mouseDict[a]['mouseName'], wheelDistance_Mouse_nan])
            mice_wheelSpeed.append([mouseDict[a]['mouseName'], wheelSpeed_Mouse_nan])
        else:
            mice_swingLen_Day_Trials_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_swingLen_Day_Paw_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Paw_Avg,mouseDict[a]['treatment']])
            mice_swingLen_Day_Avg.append([mouseDict[a]['mouseName'], swingLen_Day_Avg,mouseDict[a]['treatment']])
            mice_swingLen_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingLen_Mouse_nan,mouseDict[a]['treatment']])
            mice_swingLenStd_Day_Trials_Paw.append([mouseDict[a]['mouseName'], swingLenStd_Mouse_nan,mouseDict[a]['treatment']])
            mice_wheelDistance.append([mouseDict[a]['mouseName'], wheelDistance_Mouse_nan,mouseDict[a]['treatment']])
            mice_wheelSpeed.append([mouseDict[a]['mouseName'], wheelSpeed_Mouse_nan,mouseDict[a]['treatment']])
    return (mice_swingLen_Day_Avg,mice_swingLen_Day_Trials_Avg,mice_swingLen_Day_Paw_Avg,mice_swingLen_Day_Trials_Paw,mice_swingLenStd_Day_Trials_Paw,mice_wheelDistance,mice_wheelSpeed)

def getStepQuality(mouseDict, experiment, treatments):
    mice_indecisiveSwingFrac_Day_Trials_Avg =[]
    mice_indecisiveSwingFrac_Paw_Avg = []
    mice_indecisiveSwingFrac_Day_Avg = []
    mice_indecisiveSwingFrac_Day_Trials_Paw =[]
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            listOfRecordings=mouseDict[a]['listOfRecordings']
            maxTrials = 10
        else:
            listOfRecordings = mouseDict[a]['foldersRecordings']
            maxTrials = 5
        indeciveStepNb=[]
        avgIndeciveStep=[]
        recs=mouseDict[a]['pawData']
        nDays = len(recs)
        indecisiveSwingFrac_Mouse=[]
        stepC = np.zeros((4, nDays, 2))
        for n in range(nDays):

            indecisiveSwingFrac_Day=[]
            for j in range(len(recs[n][4])):
                indecisiveSwingFrac_Paw=[]
                for i in range(4):

                    indecisiveSwing=[]
                    pawPos = recs[n][2][j][5][i]
                    linearPawPos = recs[n][4][j][4][i][5]
                    # pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    stepCharacter = recs[n][4][j][3][i][3]
                    stepC[i, n, 1] += len(stepCharacter)
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    for k in range(len(idxSwings)):
                        # pdb.set_trace()
                        # only look at steps during motorization period
                            if (recTimes[idxSwings[k, 0]] > 10.) and (recTimes[idxSwings[k, 0]] < 52):
                                indecisiveSwing.append(stepCharacter[k][3])
                    indecisiveSwingFrac_Paw.append(np.count_nonzero(indecisiveSwing)/len(indecisiveSwing))
                indecisiveSwingFrac_Day.append(indecisiveSwingFrac_Paw)
            indecisiveSwingFrac_Mouse.append(indecisiveSwingFrac_Day)
        if experiment=='ephy':
            indecisiveSwingFrac_Mouse = equalizeTrialNumberWithNan(indecisiveSwingFrac_Mouse, listOfRecordings)
        else:
            indecisiveSwingFrac_Mouse=indecisiveSwingFrac_Mouse

        indecisiveSwingFrac_Mouse_nan=np.empty((maxDays,maxTrials,4))
        indecisiveSwingFrac_Mouse_nan.fill(np.nan)
        for w in range(len(indecisiveSwingFrac_Mouse)):
            for x in range(len(indecisiveSwingFrac_Mouse[w])):
                for c in range(len(indecisiveSwingFrac_Mouse[w][x])):
                    indecisiveSwingFrac_Mouse_nan[w][x][c]=indecisiveSwingFrac_Mouse[w][x][c]

        indecisiveSwingFrac_Day_Trials_Avg = np.nanmean(indecisiveSwingFrac_Mouse_nan, axis=2)
        indecisiveSwingFrac_Paw_Avg = np.nanmean(indecisiveSwingFrac_Mouse_nan, axis=1)
        indecisiveSwingFrac_Day_Avg = np.nanmean(indecisiveSwingFrac_Paw_Avg, axis=1)
        # equalize all the arrays to the max number of days
        nanArrayTrials = np.empty(10)
        nanArrayTrials.fill(np.nan)
        nanArrayPaws = np.empty(4)
        nanArrayPaws.fill(np.nan)
        while len(indecisiveSwingFrac_Paw_Avg) < 10:
            indecisiveSwingFrac_Day_Avg = np.insert(indecisiveSwingFrac_Day_Avg, len(indecisiveSwingFrac_Day_Avg), np.nan)
            indecisiveSwingFrac_Day_Trials_Avg = np.vstack((indecisiveSwingFrac_Day_Trials_Avg, nanArrayTrials))
            indecisiveSwingFrac_Paw_Avg = np.vstack((indecisiveSwingFrac_Paw_Avg, nanArrayPaws))


        if treatments==False :
            mice_indecisiveSwingFrac_Day_Trials_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Day_Trials_Avg])
            mice_indecisiveSwingFrac_Day_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Day_Avg])
            mice_indecisiveSwingFrac_Paw_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Paw_Avg])
            mice_indecisiveSwingFrac_Day_Trials_Paw.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Mouse_nan])
        else:
            mice_indecisiveSwingFrac_Day_Trials_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Day_Trials_Avg,mouseDict[a]['treatment']])
            mice_indecisiveSwingFrac_Day_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Day_Avg,mouseDict[a]['treatment']])
            mice_indecisiveSwingFrac_Paw_Avg.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Paw_Avg,mouseDict[a]['treatment']] )
            mice_indecisiveSwingFrac_Day_Trials_Paw.append([mouseDict[a]['mouseName'], indecisiveSwingFrac_Mouse_nan,mouseDict[a]['treatment']] )
    return (mice_indecisiveSwingFrac_Day_Trials_Avg,mice_indecisiveSwingFrac_Day_Avg,mice_indecisiveSwingFrac_Paw_Avg,mice_indecisiveSwingFrac_Day_Trials_Paw)
def getTreatmentAverages1D (ArrayWithTreatment):
    #this function split an array containing treatment name to 2 arrays for each treatment and return Average for each treatment
    #takes in account the fact that different animals have different recording days number
    saline=[]
    muscimol=[]

    for s in range(len(ArrayWithTreatment)):
        if ArrayWithTreatment[s][2]=='saline':
            if type(ArrayWithTreatment[s][1]) is numpy.ndarray:
                saline.append(ArrayWithTreatment[s][1].tolist())
            else:
                saline.append(ArrayWithTreatment[s][1])
        if ArrayWithTreatment[s][2]=='muscimol':
            if type(ArrayWithTreatment[s][1]) is numpy.ndarray:
                muscimol.append(ArrayWithTreatment[s][1].tolist())
            else:
                muscimol.append(ArrayWithTreatment[s][1])
    # pdb.set_trace()
    for row in saline:
        while len(row) < 10:
            row.append(np.nan)
    for row in muscimol:
        while len(row) < 10:
            row.append(np.nan)

    nSaline=len(saline)
    stdSaline = np.nanstd((saline), axis=0)
    avgValueSaline = np.nanmean((saline), axis=0)
    semSaline = stats.sem((saline), axis=0, nan_policy='omit')
    salineAvg =[avgValueSaline, stdSaline, semSaline]

    nMuscimol = len(muscimol)
    stdMuscimol = np.nanstd((muscimol), axis=0)
    avgValueMuscimol = np.nanmean((muscimol), axis=0)
    semMuscimol = stats.sem((muscimol), axis=0, nan_policy='omit')
    MuscimolAvg =[avgValueMuscimol, stdMuscimol, semMuscimol]
    return (salineAvg,MuscimolAvg, saline, muscimol,nSaline,nMuscimol)

def splitGroups (ArrayWithTreatment):
    #this function split an array containing treatment name to 2 arrays for each treatment and return separated arrays
    #takes in account the fact that different animals have different recording days number
    salineArray=[]
    muscimolArray=[]

    for s in range(len(ArrayWithTreatment)):
        if ArrayWithTreatment[s][2]=='saline':
            if type(ArrayWithTreatment[s][1]) is numpy.ndarray:
                salineArray.append(ArrayWithTreatment[s][1].tolist())
            else:
                salineArray.append(ArrayWithTreatment[s][1])
        else:
            if type(ArrayWithTreatment[s][1]) is numpy.ndarray:
                muscimolArray.append(ArrayWithTreatment[s][1].tolist())
            else:
                muscimolArray.append(ArrayWithTreatment[s][1])
    for row in salineArray:
        while len(row) < 10:
            row.append([np.nan,np.nan,np.nan,np.nan])
    for row in muscimolArray:
        while len(row) < 10:
            row.append([np.nan,np.nan,np.nan,np.nan])

    salineArray=np.asanyarray(salineArray)
    muscimolArray=np.asanyarray(muscimolArray)
    #pdb.set_trace()
    return (salineArray, muscimolArray)

def getAverageSingleGroup (Array):
    #this function  return the average of an arrays with the structure: [animalName, [pawValues,pawValues,pawValues,pawValues]]
    #takes in account the fact that different animals have different recording days number
    OutArray=[]
    for s in range(len(Array)):
            if type(Array[s][1]) is numpy.ndarray:
                OutArray.append(Array[s][1].tolist())
            else:
                OutArray.append(Array[s][1])

    for row in OutArray:
        while len(row) < 10:
            row.append([np.nan,np.nan,np.nan,np.nan])

    Array=np.asanyarray(OutArray)
    # pdb.set_trace()
    # (anovaResults) = performRepeatedMeasuresANOVA(OutArray,sessionValues=False,treatments=False)

    arrayStd=np.nanstd((OutArray),axis=0)
    arrayMean=np.nanmean((OutArray),axis=0)
    arraySem=stats.sem((OutArray),axis=0,nan_policy='omit')

    return (Array,arrayMean,arraySem)

def PawListToPandasDFAndMxLM(data,trialValues, treatments, fixedLength=None):
    #convert to panda and perform mixedML analysis on array with paw values, multiples days and animals, return pvalues for each paw

    if (trialValues==True and treatments==False) :
        maxDim = 4
    elif (trialValues==True and treatments==True):
        maxDim=5


    my_array = np.empty((1, maxDim),dtype=object)
    print(maxDim)
    for i in range(len(data)):
        values = np.asanyarray(data[i][1])
        if fixedLength is None:

            nRecDays = len(values)
        else:
            nRecDays = fixedLength
        if trialValues:

            sessions = 4
            #print(sessions)
        else:
            sessions=1
        tempArray = np.empty((nRecDays*sessions, maxDim),dtype=object)
        tempArray[:, 0] = np.repeat(data[i][0], nRecDays*sessions)
        tempArray[:, 1] = np.repeat(np.arange(nRecDays),sessions)
        #pdb.set_trace()
        tempArray[:, 2] = values[:nRecDays].flatten()

        if trialValues==True and treatments==False:
            tempArray[:, 3] = np.tile(np.arange(sessions),nRecDays)
        elif trialValues==True and treatments==True:
            tempArray[:, 3] = np.tile(np.arange(sessions), nRecDays)
            tempArray[:,4]=np.repeat(data[i][2], nRecDays*sessions)

        my_array = np.concatenate((my_array, tempArray))

    my_array = my_array[1:] # remove first line which was created before
    # print('my array',my_array)
    if trialValues==True and treatments==False:
        df = pd.DataFrame(my_array, columns=['mouse','recordingDay','measuredValue','paw'])
        convert_dict = {'mouse': str,'recordingDay': int,'measuredValue':float,'paw':int}
    elif trialValues==True and treatments==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue', 'paw','treatments'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'sessionNumber': int, 'treatments': str}

    df = df.astype(convert_dict)
    # print(df)
    # df=df.dropna()

    # FRpd= FRpd.dropna()
    FLpd = df.loc[df['paw'] == 0]
    FRpd=df.loc[df['paw']==1]
    HLpd = df.loc[df['paw'] == 2]
    HRpd = df.loc[df['paw'] == 3]

    # sns.lineplot(x='recordingDay', y='measuredValue', data=FRpd)
    # plt.show()
    # pdb.set_trace()
    if trialValues == True and treatments==False:
        mdFL=smf.mixedlm("measuredValue ~ recordingDay", FLpd, groups=FLpd["mouse"], missing='drop').fit()
        mdFR=smf.mixedlm("measuredValue ~ recordingDay", FRpd, groups=FRpd["mouse"], missing='drop').fit()
        mdHL=smf.mixedlm("measuredValue ~ recordingDay", HLpd, groups=HLpd["mouse"], missing='drop').fit()
        mdHR=smf.mixedlm("measuredValue ~ recordingDay", HRpd, groups=HRpd["mouse"], missing='drop').fit()

        print('front left',mdFL.summary())
        print('front right',mdFR.summary())
        print('hind left',mdHL.summary())
        print('hind right',mdHR.summary())
        # pawPvalues = [mdFL.pvalues['paw == 0[T.True]'],mdFR.pvalues['paw == 1[T.True]'],mdHL.pvalues['paw == 2[T.True]'],mdHR.pvalues['paw == 3[T.True]']]

        pawPvalues = [mdFL.pvalues['recordingDay'], mdFR.pvalues['recordingDay'],mdHL.pvalues['recordingDay'], mdHR.pvalues['recordingDay']]
        #pawPvalues = [mdFL.pvalues['paw == 0[T.True]:recordingDay'], mdFR.pvalues['paw == 1[T.True]:recordingDay'],mdHL.pvalues['paw == 2[T.True]:recordingDay'], mdHR.pvalues['paw == 3[T.True]:recordingDay']]
        print(pawPvalues)
        # print(df)

    elif trialValues == True and treatments == True:

        mdFL=smf.mixedlm("measuredValue ~ recordingDay+treatment", FLpd, groups=FLpd["mouse"], missing='drop').fit()
        mdFR=smf.mixedlm("measuredValue ~ recordingDay+treatment", FRpd, groups=FRpd["mouse"], missing='drop').fit()
        mdHL=smf.mixedlm("measuredValue ~ recordingDay+treatment", HLpd, groups=HLpd["mouse"], missing='drop').fit()
        mdHR=smf.mixedlm("measuredValue ~ recordingDay+treatment", HRpd, groups=HRpd["mouse"], missing='drop').fit()
        # conf_int = pd.DataFrame(mdf.conf_int())
    # posthoc=sp.posthoc_tukey(df, val_col='measuredValue', group_col='recordingDay')
    # plt.imshow(posthoc)
    # plt.show()
    # print(df)

    return (df, pawPvalues)
def pandaDataFrameAndMixedMLCompleteData(data, treatments, varName, fixedLength=None):
    #convert to panda and perform mixedML analysis on array with paw values, multiples days and animals, return pvalues for each paw

    if treatments==True :
        maxDim = 7
    else:
        maxDim=6
    for m in range(len(data)):
        nDays=len(data[m][1])
        for t in range(nDays):
            nTrials=len(data[m][1][t])

    my_array = np.empty((1, maxDim),dtype=object)

    for i in range(len(data)):
        values = np.asanyarray(data[i][1])
        if fixedLength is None:

            nRecDays = len(values)
        else:
            nRecDays = fixedLength

        sessions = nTrials
        paws=4
        if 'f' in data[i][0]:
            sex='female'
        else:
            sex='male'
        tempArray = np.empty((nRecDays*sessions*paws, maxDim),dtype=object)
        tempArray[:, 0] = np.repeat(data[i][0], nRecDays*sessions*paws)
        tempArray[:, 1] = np.repeat(np.arange(nRecDays)+1,sessions*paws)

        tempArray[:, 2] = values.flatten()#values[:nRecDays].flatten()

        tempArray[:, 3] = np.tile(np.arange(paws),sessions*nRecDays)

        tempArray[:,4]=np.tile(np.repeat(np.arange(sessions), paws),nRecDays)
        tempArray[:,5]=np.repeat(sex, nRecDays*sessions*paws)
        if treatments==True:
            tempArray[:,6]=np.repeat(data[i][2], nRecDays*paws*sessions)

        my_array = np.concatenate((my_array, tempArray))

    my_array = my_array[1:] # remove first line which was created before
    # print('my array',my_array)


    if treatments==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue', 'paw', 'trial','sex','treatments'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'trial': int, 'paw':int, 'sex':str, 'treatments': str}
    else:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue', 'paw', 'trial','sex'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'paw': int, 'trial': int, 'sex':str}

    df = df.astype(convert_dict)
    # print(df)
    # df=df.dropna()

    # FRpd= FRpd.dropna()
    FLpd = df.loc[df['paw'] == 0]
    FRpd=df.loc[df['paw']==1]
    HLpd = df.loc[df['paw'] == 2]
    HRpd = df.loc[df['paw'] == 3]

    # sns.lineplot(x='recordingDay', y='measuredValue', data=FRpd)
    # plt.show()

    if treatments == False:
        md_paw = [[], [], [], []]
        pawPvalues = []
        df_anova=df.dropna()
        mdf=smf.mixedlm("measuredValue ~ paw*recordingDay+paw*trial+recordingDay*trial", df, groups='mouse', missing='drop').fit()


        pvalues_all = {"day": mdf.pvalues['recordingDay'], "trial": mdf.pvalues['trial'], "paw":mdf.pvalues['paw']}
        stars_all = {"day": starMultiplier(pvalues_all['day']), "trial": starMultiplier(pvalues_all['trial']), "paw":starMultiplier(pvalues_all['paw'])}
        df_tukey=df.dropna()
        df_tukey_paw=df_tukey.groupby(['recordingDay', 'mouse','paw'])['measuredValue'].agg(['mean']).reset_index()


        tukey_paw = pairwise_tukeyhsd(df_tukey_paw['mean'], df_tukey_paw['paw'])
        tukey_day = pairwise_tukeyhsd(df_tukey_paw['mean'], df_tukey_paw['recordingDay'])
        # print(tukey_paw.summary())

        # print(varName)
        # print(mdf.summary())
        # pdb.set_trace()

        # print(tukey_day.summary())
        # print(tukey.summary())
        # pdb.set_trace()
    else:
        md_paw = [[[], [], [], []], [[], [], [], []]]
        pawPvalues= [[],[]]
        muscimol = df.loc[df['treatments'] == 'muscimol']
        saline = df.loc[df['treatments'] == 'saline']

        mdf = smf.mixedlm("measuredValue ~ recordingDay*treatments+trial*treatments+paw*treatments+paw*recordingDay+paw*trial+recordingDay*trial", df, groups='mouse', missing='drop').fit()
        # mdf = smf.mixedlm("measuredValue ~recordingDay+trial+treatments+paw+ recordingDay*treatments*trial*paw", df, groups='mouse', missing='drop').fit()
        # mdf = smf.mixedlm("measuredValue ~ recordingDay+trial+paw+treatments+recordingDay:trial:paw:treatments", df, groups='mouse',
        #                   missing='drop').fit()
        conf_int_all = pd.DataFrame(mdf.conf_int())

        pvalues_all = {"day": mdf.pvalues['recordingDay'], "trial": mdf.pvalues['trial'],"treatment": mdf.pvalues['treatments[T.saline]'], "paw":mdf.pvalues['paw']}
        stars_all = {"day": starMultiplier(pvalues_all['day']), "trial": starMultiplier(pvalues_all['trial']),"treatments": starMultiplier(pvalues_all['treatment']), "paw":starMultiplier(pvalues_all['paw'])}
        # print('big stats %s'%varName, mdf.summary())
        mdf_muscimol=smf.mixedlm("measuredValue ~ recordingDay+trial+paw+sex", muscimol, groups='mouse', missing='drop').fit()
        pvalues_muscimol = {"day": mdf_muscimol.pvalues['recordingDay'], "trial": mdf_muscimol.pvalues['trial'], "paw":mdf_muscimol.pvalues['paw']}
        stars_muscimol = {"day": starMultiplier(pvalues_muscimol['day']), "trial": starMultiplier(pvalues_muscimol['trial']),"paw":starMultiplier(pvalues_muscimol['paw'])}
        # print('muscimol stats', mdf_muscimol.summary())
        mdf_saline=smf.mixedlm("measuredValue ~ recordingDay+trial+paw+sex", saline, groups='mouse', missing='drop').fit()
        pvalues_saline = {"day": mdf_saline.pvalues['recordingDay'], "trial": mdf_saline.pvalues['trial'], "paw":mdf_saline.pvalues['paw']}
        stars_saline = {"day": starMultiplier(pvalues_saline['day']), "trial": starMultiplier(pvalues_saline['trial']), "paw":starMultiplier(pvalues_saline['paw'])}
        # print('saline stats', mdf_saline.summary())
        # print(varName)
        # print(mdf.summary())
        # pdb.set_trace()
        # mouseDf=pd.DataFrame(mdf.random_effects).transpose()
        # plt.plot(mouseDf)
        # plt.show()
    treatmentsList=['saline', 'muscimol']

    for p in range(4):
        if  treatments==False:

            md_paw[p]=smf.mixedlm("measuredValue ~ recordingDay*trial", df.loc[df['paw'] == p], groups=df.loc[df['paw'] == p]["mouse"], missing='drop').fit()
            # mdFR=smf.mixedlm("measuredValue ~ recordingDay", FRpd, groups=FRpd["mouse"], missing='drop').fit()
            # mdHL=smf.mixedlm("measuredValue ~ recordingDay", HLpd, groups=HLpd["mouse"], missing='drop').fit()
            # mdHR=smf.mixedlm("measuredValue ~ recordingDay", HRpd, groups=HRpd["mouse"], missing='drop').fit()

            # print('paw stats',md_paw[p].summary())
        # print('front right',mdFR.summary())
        # print('hind left',mdHL.summary())
        # print('hind right',mdHR.summary())

        #pawPvalues = [mdFL.pvalues['recordingDay'], mdFR.pvalues['recordingDay'],mdHL.pvalues['recordingDay'], mdHR.pvalues['recordingDay']]
        #pawPvalues = [mdFL.pvalues['paw == 0[T.True]:recordingDay'], mdFR.pvalues['paw == 1[T.True]:recordingDay'],mdHL.pvalues['paw == 2[T.True]:recordingDay'], mdHR.pvalues['paw == 3[T.True]:recordingDay']]
            pawPvalues.append([md_paw[p].pvalues['recordingDay'],md_paw[p].pvalues['trial']])
        # print(df)

        if  treatments == True:
            for t in range(len(treatmentsList)):
                md_paw[t][p] = smf.mixedlm("measuredValue ~ recordingDay+trial+treatments", df.loc[(df['paw'] == p) & (df['treatments']==treatmentsList[t])],groups="mouse", missing='drop').fit()
        # mdFL=smf.mixedlm("measuredValue ~ recordingDay+treatment", FLpd, groups=FLpd["mouse"], missing='drop').fit()
        # mdFR=smf.mixedlm("measuredValue ~ recordingDay+treatment", FRpd, groups=FRpd["mouse"], missing='drop').fit()
        # mdHL=smf.mixedlm("measuredValue ~ recordingDay+treatment", HLpd, groups=HLpd["mouse"], missing='drop').fit()
        # mdHR=smf.mixedlm("measuredValue ~ recordingDay+treatment", HRpd, groups=HRpd["mouse"], missing='drop').fit()
        # conf_int = pd.DataFrame(mdf.conf_int())
                pawPvalues[t].append([md_paw[t][p].pvalues['recordingDay'],md_paw[t][p].pvalues['trial']])
    # posthoc=sp.posthoc_nemenyi_friedman(df, y_col='measuredValue', block_col='recordingDay',group_col='trial', melted=True)
    # posthoc=sp.posthoc_mannwhitney(df, val_col='measuredValue', group_col='recordingDay')
    # print(posthoc)
    # heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
    #                 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    # sp.sign_plot(posthoc, **heatmap_args)
    # # pdb.set_trace()
    # # cmap = cm.get_cmap('tab20')
    # # significant=np.ma.masked_where(posthoc <0.05, posthoc)
    # # cmap.set_bad(color='white')
    # # im=plt.imshow(significant)
    # # cbar=plt.colorbar(im)
    # # cbar.set_label("colorbar")
    # plt.show()
    df.to_csv(groupAnalysisDir+'data.csv')
    if treatments==False:
        return (df, mdf, md_paw, pawPvalues,stars_all,tukey_paw)
    else:
        return (df, mdf, md_paw, pawPvalues,stars_all, mdf_saline,mdf_muscimol,stars_saline,stars_muscimol)
def starMultiplier(pvalue):
    multiplier = 0

    if pvalue < 0.001:
        multiplier = 3
        stars='*'*multiplier
    elif pvalue < 0.01:
        multiplier= 2
        stars='*'*multiplier
    elif pvalue < 0.05:
        multiplier = 1
        stars='*'*multiplier
    else:
        stars=''
    return (stars)

def analyzeStepTiming(mouseDict, refPaw, experiment, treatments):
    pawSeqProb_mice=[]
    countPawSeq_mice=[]
    iqr_mice=[]
    stanceOnMedian_mice = []
    iOnArray_mice = []
    iOffArray_mice = []
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy' or experiment=='test':
            maxTrials=10
            listOfRecordings=mouseDict[a]['listOfRecordings']
        else:
            maxTrials=5
            listOfRecordings = mouseDict[a]['foldersRecordings']
        recs = mouseDict[a]['pawData']
        nDays = len(recs)
        pawSeqProb_mouse=[]
        countPawSeq_mouse=[]
        iqr_mouse=[]
        stanceOnMedian_mouse = []
        iOnArray_mouse = []
        iOffArray_mouse = []
        for n in range(nDays):
            pawSeqProb_day=[]
            countPawSeq_day=[]
            iqr_day=[]
            stanceOnMedian_day=[]
            iOnArray_day=[]
            iOffArray_day=[]
            for j in range(len(recs[n][4])):
                swingOnset=[]
                swingOffset = []
                for i in range(4):
                    swingOn =[]
                    swingOff=[]

                    idxSwings = np.asarray(recs[n][4][j][3][i][1])
                    recTimes = np.asarray(recs[n][4][j][4][i][2])
                    pawPos = recs[n][2][j][5][i]
                    for k in range(len(idxSwings)-1):
                        idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                        idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))
                        if (recTimes[idxSwings[k, 0]] > 10) and (recTimes[idxSwings[k, 0]] <52):  # only look at steps during motorization period
                            swingOn.append(pawPos[idxSwingStart, 0])
                            swingOff.append(pawPos[idxStanceStart, 0])
                    swingOnset.append(swingOn)
                    swingOffset.append(np.array(swingOff))

                # define reference paw

                refPawSwingOn = swingOnset[refPaw]
                refPawSwingOff = swingOffset[refPaw]
                dt = 0.02
                # counts = np.zeros((8, int(1 / dt) + 1))
                counts=np.zeros((4,int(1/dt)+1))
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
                                counts[ x, int(iOn / dt):int(iOff / dt)] += 1
                            elif (0 <= iOn < 1) and (iOff > 1):
                                counts[ x, int(iOn / dt):] += 1
                                z = 1
                                aborted = False
                                while (iOff > 1):
                                    if ((p + z + 1) >= len(refPawSwingOn)):
                                        aborted = True
                                        break
                                    iOff = (swingOffset[x][w] - refPawSwingOn[p + z]) / (
                                                refPawSwingOn[p + z + 1] - refPawSwingOn[p + z])
                                    z += 1
                                counts[x, :] += (z - 1)
                                if not aborted:
                                    counts[ x, :int(iOff / dt)] += 1
                                    iOffArray[x].append(iOff)

                countsProb=counts/np.max(counts)
                iqr = []
                stanceOnMedian=[]
                for c in range(4):
                    # iqr.append(stats.iqr(countsProb[c], rng=[70, 90]))
                    iqr.append(np.std(iOffArray[c]))
                    # stanceOnMedian.append(np.median(iOffArray[c]))
                    stanceOnMedian.append(np.percentile(iOffArray[c],50))
                iqr_day.append(iqr)
                pawSeqProb_day.append(countsProb)
                countPawSeq_day.append(counts)
                stanceOnMedian_day.append(stanceOnMedian)
                iOnArray_day.append(iOnArray)
                iOffArray_day .append(iOffArray)

            pawSeqProb_mouse.append(pawSeqProb_day)
            countPawSeq_mouse.append(countPawSeq_day)
            iqr_mouse.append(iqr_day)
            stanceOnMedian_mouse.append(stanceOnMedian_day)
            iOnArray_mouse.append(iOnArray_day)
            iOffArray_mouse.append(iOffArray_day)
        if experiment=='ephy':
            iqr_mouse=equalizeTrialNumberWithNan(iqr_mouse,listOfRecordings)
            # pawSeqProb_mouse=regroupTrialsFromSameDayGroupSimple(pawSeqProb_mouse,listOfRecordings)
            # countPawSeq_mouse=regroupTrialsFromSameDayGroupSimple(countPawSeq_mouse,listOfRecordings)
            stanceOnMedian_mouse = equalizeTrialNumberWithNan(stanceOnMedian_mouse, listOfRecordings)
            # iOnArray_mouse = regroupTrialsFromSameDayGroupSimple(iOnArray_mouse, listOfRecordings)
            # iOffArray_mouse = regroupTrialsFromSameDayGroupSimple(iOffArray_mouse, listOfRecordings)
        else:
            iqr_mouse=iqr_mouse
            pawSeqProb_mouse=pawSeqProb_mouse
            countPawSeq_mouse=countPawSeq_mouse
            stanceOnMedian_mouse = stanceOnMedian_mouse
            iOnArray_mouse =iOnArray_mouse
            iOffArray_mouse = iOffArray_mouse

        pawProb_nan=np.empty((maxDays,maxTrials,4,51))
        pawProb_nan.fill(np.nan)
        swingCount_nan=np.empty((maxDays,maxTrials,4,51))
        swingCount_nan.fill(np.nan)
        iqr_mouse_nan=np.empty((maxDays,maxTrials,4))
        iqr_mouse_nan.fill(np.nan)
        stanceOnMedian_nan=np.empty((maxDays,maxTrials,4))
        stanceOnMedian_nan.fill(np.nan)
        iOnArray_mouse_nan=np.empty((maxDays,maxTrials,4,250))
        iOnArray_mouse_nan.fill(np.nan)
        iOffArray_mouse_nan=np.empty((maxDays,maxTrials,4,250))
        iOffArray_mouse_nan.fill(np.nan)
        for d in range(len(pawSeqProb_mouse)):
            for f in range(len(pawSeqProb_mouse[d])):
                iqr_mouse_nan[d][f]=iqr_mouse[d][f]
                stanceOnMedian_nan[d][f] = stanceOnMedian_mouse[d][f]
                for g in range(len(pawSeqProb_mouse[d][f])):
                    pawProb_nan[d][f][g]=pawSeqProb_mouse[d][f][g]
                    swingCount_nan[d][f][g]=countPawSeq_mouse[d][f][g]
                for h in range(len(iOnArray_mouse[d][f])):
                    for j in range(len(iOnArray_mouse[d][f][h])):
                        iOnArray_mouse_nan[d][f][h][j]=iOnArray_mouse[d][f][h][j]
                    for k in range(len(iOnArray_mouse[d][f][h])):
                        iOffArray_mouse_nan[d][f][h][k]=iOffArray_mouse[d][f][h][k]

        # pdb.set_trace()
        if treatments==False:
            pawSeqProb_mice.append([mouseDict[a]['mouseName'],pawProb_nan])
            countPawSeq_mice.append([mouseDict[a]['mouseName'],swingCount_nan])
            iqr_mice.append([mouseDict[a]['mouseName'],iqr_mouse_nan])
            stanceOnMedian_mice.append([mouseDict[a]['mouseName'],stanceOnMedian_nan])
            iOnArray_mice.append([mouseDict[a]['mouseName'],iOnArray_mouse_nan])
            iOffArray_mice.append([mouseDict[a]['mouseName'],iOffArray_mouse_nan])
        else:
            pawSeqProb_mice.append([mouseDict[a]['mouseName'],pawProb_nan,mouseDict[a]['treatment']])
            countPawSeq_mice.append([mouseDict[a]['mouseName'],swingCount_nan,mouseDict[a]['treatment']])
            iqr_mice.append([mouseDict[a]['mouseName'],iqr_mouse_nan,mouseDict[a]['treatment']])
            stanceOnMedian_mice.append([mouseDict[a]['mouseName'],stanceOnMedian_nan,mouseDict[a]['treatment']])
            iOnArray_mice.append([mouseDict[a]['mouseName'],iOnArray_mouse_nan,mouseDict[a]['treatment']])
            iOffArray_mice.append([mouseDict[a]['mouseName'],iOffArray_mouse_nan,mouseDict[a]['treatment']])

    return (pawSeqProb_mice,countPawSeq_mice,iqr_mice,stanceOnMedian_mice,iOnArray_mice,iOffArray_mice)
def getPawTrajectories(mouseDict, experiment, treatments):
    pawTrajectory_Mice = []
    timePawTrajectory_Mice = []
    rungCrossed_trials_Mice = []
    meanPawTrajectories_Mice = []
    meanPawTrajectories_Time_Mice = []
    stdPawTrajectories_Mice = []
    varRungCrossed_Mice = []

    excludeMissSteps=True
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy':
            maxTrials = 10
        else:
            maxTrials = 5
        recs=mouseDict[a]['pawData']
        nDays = len(recs)

        pawTrajectory_Mouse = []
        timePawTrajectory_Mouse = []
        rungCrossed_trials_Mouse = []
        for n in range(nDays):
            pawTrajectory_Mouse.append([[], [], [], []])
            timePawTrajectory_Mouse.append([[], [], [], []])
            rungCrossed_trials_Mouse.append([[], [], [], []])
            for j in range(len(recs[n][4])):

                for i in range(4):
                    swingLen = []
                    swingTimes = []
                    pawPosSwing=[]

                    pawPos = recs[n][2][j][5][i]
                    linearPawPos = recs[n][4][j][4][i][5]
                    pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    #stepCharacter = recs[n][4][j][3][i][3]
                    #stepC[i, n, 1] += len(stepCharacter)
                    wheelAngularSpeed=recs[n][1][j][0]
                    xSpeed = recs[n][4][j][4][i][1]
                    wSpeed = recs[n][4][j][4][i][0]

                    wheelsTimestamp = recs[n][1][j][3]
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    mask = (recTimes[idxSwings[:, 0]] > 10.) & (recTimes[idxSwings[:, 0]] < 52)

                    # pdb.set_trace()
                    # rungNumbers=np.array(rungNumbers)[mask]
                    stepCharacter = recs[n][4][j][3][i][3]
                    rungNumbers = recs[n][4][j][3][i][2]
                    rungCrossed = np.insert(np.diff(rungNumbers), 0, 0)
                    # rungCrossed= np.insert(rungCrossed, -1, 0)
                    missStepsIdx=[]
                    for l in range(len(idxSwings)):
                        missStepsIdx.append(np.invert(stepCharacter[l][3]))
                    if excludeMissSteps:
                        # rungCrossed=rungCrossed[missStepsIdx]
                        idxSwings=idxSwings[missStepsIdx]

                    # rungCrossed=rungCrossed[mask]
                    # sortedRungCrossed = np.argsort(rungCrossed)

                    rungCrossed_trials_Mouse[-1][i].extend(rungCrossed)
                    for k in range(len(idxSwings)):
                        if (recTimes[idxSwings[k,0]]>10) and  (recTimes[idxSwings[k,0]]<52): # only look at steps during motorization period

                            idxStart = np.argmin(np.abs(linearPawPos[:, 0] - recTimes[idxSwings[k][0]]))
                            idxEnd = np.argmin(np.abs(linearPawPos[:, 0] - recTimes[idxSwings[k][1]]))

                            pawSwingPos = linearPawPos[:, 1][idxStart:idxEnd] - linearPawPos[:, 1][idxStart]
                            pawSwingPos_Norm = linearPawPos[:, 1][idxStart:idxEnd] - linearPawPos[:, 1][idxStart]
                            timePawSwingTrajectory_Norm = linearPawPos[:, 0][idxStart:idxEnd] - linearPawPos[:, 0][idxStart]
                            if len(pawSwingPos_Norm)<100:
                                while len(pawSwingPos_Norm)<100:
                                    pawSwingPos_Norm = np.insert(pawSwingPos_Norm,len(pawSwingPos_Norm), np.nan)
                                    timePawSwingTrajectory_Norm=np.insert(timePawSwingTrajectory_Norm,len(timePawSwingTrajectory_Norm), np.nan)
                            pawTrajectory_Mouse[-1][i].extend([pawSwingPos_Norm[:100]])
                            timePawTrajectory_Mouse[-1][i].extend([timePawSwingTrajectory_Norm[:100]])

        pawTrajectory_Mouse=np.array(pawTrajectory_Mouse)
        timePawTrajectory_Mouse=np.array(timePawTrajectory_Mouse)
        meanPawTrajectories_Mouse=[[],[],[],[]]
        meanPawTrajectories_Time_Mouse=[[],[],[],[]]
        stdPawTrajectories_Mouse=[[],[],[],[]]
        varRungCrossed=[[],[],[],[]]
        varRungCrossed_sem=[[],[],[],[]]
        # pdb.set_trace()
        for p in range(4):
            for n in range(len(pawTrajectory_Mouse[:,p])):
                meanPawTrajectories_Mouse[p].append(np.nanmean(pawTrajectory_Mouse[:,p][n], axis=0))
                meanPawTrajectories_Time_Mouse[p].append(np.nanmean(timePawTrajectory_Mouse[:,p][n], axis=0))
                dayPawTrajectories=np.array(pawTrajectory_Mouse[:,p][n])
                stdPawTrajectories_Mouse[p].append(stats.variation(dayPawTrajectories, ddof=1,axis=0, nan_policy='omit'))

                varRungCrossed[p].append(stats.variation(rungCrossed_trials_Mouse[n][p], ddof=1,axis=0, nan_policy='omit'))
                varRungCrossed_sem[p].append(stats.sem(rungCrossed_trials_Mouse[n][p], axis=0, nan_policy='omit'))

       # pdb.set_trace()
        if treatments==False :

            pawTrajectory_Mice.append([mouseDict[a]['mouseName'], pawTrajectory_Mouse])
            timePawTrajectory_Mice.append([mouseDict[a]['mouseName'], timePawTrajectory_Mouse])
            #average accross trials and paws
            rungCrossed_trials_Mice.append([mouseDict[a]['mouseName'], rungCrossed_trials_Mouse])
            meanPawTrajectories_Mice.append([mouseDict[a]['mouseName'], meanPawTrajectories_Mouse])
            meanPawTrajectories_Time_Mice.append([mouseDict[a]['mouseName'], meanPawTrajectories_Time_Mouse])
            stdPawTrajectories_Mice.append([mouseDict[a]['mouseName'], stdPawTrajectories_Mouse])
            varRungCrossed_Mice.append([mouseDict[a]['mouseName'], varRungCrossed])
        else:
            pawTrajectory_Mice.append([mouseDict[a]['mouseName'], pawTrajectory_Mouse,mouseDict[a]['treatment']])
            timePawTrajectory_Mice.append([mouseDict[a]['mouseName'], timePawTrajectory_Mouse,mouseDict[a]['treatment']])
            #average accross trials and paws
            rungCrossed_trials_Mice.append([mouseDict[a]['mouseName'], rungCrossed_trials_Mouse,mouseDict[a]['treatment']])
            meanPawTrajectories_Mice.append([mouseDict[a]['mouseName'], meanPawTrajectories_Mouse,mouseDict[a]['treatment']])
            meanPawTrajectories_Time_Mice.append([mouseDict[a]['mouseName'], meanPawTrajectories_Time_Mouse,mouseDict[a]['treatment']])
            stdPawTrajectories_Mice.append([mouseDict[a]['mouseName'], stdPawTrajectories_Mouse,mouseDict[a]['treatment']])
            varRungCrossed_Mice.append([mouseDict[a]['mouseName'], varRungCrossed,mouseDict[a]['treatment']])
    return (pawTrajectory_Mice,timePawTrajectory_Mice,rungCrossed_trials_Mice,meanPawTrajectories_Mice,meanPawTrajectories_Time_Mice,stdPawTrajectories_Mice,varRungCrossed_Mice)

def splitMaleFemale(Array):
    femaleArray=[]
    maleArray=[]
    #pdb.set_trace()
    for s in range(len(Array)):
        if 'f' in Array[s][0]:
            if type(Array[s][1]) is numpy.ndarray:
                femaleArray.append(Array[s][1].tolist())
            else:
                femaleArray.append(Array[s][1])
        else:
            if type(Array[s][1]) is numpy.ndarray:
                maleArray.append(Array[s][1].tolist())
            else:
                maleArray.append(Array[s][1])
    for row in femaleArray:
        while len(row) < 14:
            row.append(np.nan)
    for row in maleArray:
        
        while len(row) < 14:
            row.append(np.nan)

    femaleArray=np.asanyarray(femaleArray)
    maleArray=np.asanyarray(maleArray)
    nFemale=len(femaleArray)
    nMale = len(maleArray)
    femaleArrayMean=np.nanmean((femaleArray),axis=0)
    femaleArraySem=stats.sem((femaleArray),axis=0,nan_policy='omit')
    maleArrayMean=np.nanmean((maleArray),axis=0)
    maleArraySem=stats.sem((maleArray),axis=0,nan_policy='omit')
    female=[femaleArray,femaleArrayMean,femaleArraySem]
    male=[maleArray,maleArrayMean,maleArraySem]

    return (female, male)


def splitTreatmentAvg(Array):
    #split arrays with average values into saline and muscimol arrays
    salineArray = []
    muscimolArray = []
    # pdb.set_trace()
    for s in range(len(Array)):
        if 'muscimol' in Array[s][2]:
            if type(Array[s][1]) is numpy.ndarray:
                muscimolArray.append(Array[s][1].tolist())
            else:
                muscimolArray.append(Array[s][1])
        else:
            if type(Array[s][1]) is numpy.ndarray:
                salineArray.append(Array[s][1].tolist())
            else:
                salineArray.append(Array[s][1])
    for row in muscimolArray:
        while len(row) < 10:
            row.append(np.nan)
    for row in salineArray:

        while len(row) < 10:
            row.append(np.nan)

    muscimolArray = np.asanyarray(muscimolArray)
    salineArray = np.asanyarray(salineArray)
    nmuscimol = len(muscimolArray)
    nsaline = len(salineArray)
    muscimolArrayMean = np.nanmean((muscimolArray), axis=0)
    muscimolArraySem = stats.sem((muscimolArray), axis=0, nan_policy='omit')
    salineArrayMean = np.nanmean((salineArray), axis=0)
    salineArraySem = stats.sem((salineArray), axis=0, nan_policy='omit')
    muscimol = [muscimolArray, muscimolArrayMean, muscimolArraySem]
    saline = [salineArray, salineArrayMean, salineArraySem]

    return (muscimol, saline)
def ListToPandasDFAndStatsSex(data,sessionValues, sex, fixedLength=None):
    #convert to panda and perform mixedML analysis on array with paw values, multiples days and animals, return pvalues for each paw
    #if you analyze sessions instead of paw, set session number= 5
    if (sessionValues==True and sex==False) :
        maxDim = 4
    elif (sessionValues==False and sex==True):
        maxDim=4
    elif (sessionValues==True and sex==True):
        maxDim=5
    for s in range(len(data)):
        if 'f' in data[s][0]:
            data[s].append('female')
        else:
            data[s].append('male')

    my_array = np.empty((1, maxDim),dtype=object)
    print(maxDim)
    for i in range(len(data)):
        values = np.asanyarray(data[i][1][:10])
        if fixedLength is None:

            nRecDays = len(values)
        else:
            nRecDays = fixedLength
        if sessionValues:

            sessions = 4
            #print(sessions)
        else:
            sessions=1
        tempArray = np.empty((nRecDays*sessions, maxDim),dtype=object)
        tempArray[:, 0] = np.repeat(data[i][0], nRecDays*sessions)
        tempArray[:, 1] = np.repeat(np.arange(nRecDays),sessions)
        #pdb.set_trace()
        tempArray[:, 2] = values[:nRecDays].flatten()
        tempArray[:, 3] = np.repeat(data[i][2], nRecDays)
        if sessionValues==True and sex==False:
            tempArray[:, 3] = np.tile(np.arange(sessions),nRecDays)
        elif sessionValues==True and sex==True:
            tempArray[:, 3] = np.tile(np.arange(sessions), nRecDays)
            tempArray[:,4]=np.repeat(data[i][2], nRecDays*sessions)

        my_array = np.concatenate((my_array, tempArray))

    my_array = my_array[1:] # remove first line which was created before
    # print('my array',my_array)
    #print(my_array)
    if sessionValues==True and sex==False:
        df = pd.DataFrame(my_array, columns=['mouse','recordingDay','measuredValue','paw'])
        convert_dict = {'mouse': str,'recordingDay': int,'measuredValue':float,'paw':int}
    elif sessionValues==True and sex==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue', 'paw','sex'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'sessionNumber': int, 'treatments': str}
    elif sessionValues==False and sex==True:
        df = pd.DataFrame(my_array, columns=['mouse', 'recordingDay', 'measuredValue','sex'])
        convert_dict = {'mouse': str, 'recordingDay': int, 'measuredValue': float, 'sex': str}

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199

    df = df.astype(convert_dict)
    df=df.dropna()
    df.to_csv(groupAnalysisDir+'stepsNumber_sex.csv')
    # path="/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stepsNumber_sex.csv"
    # r=rob.r
    # r['source']("/home/andry/Analysis/LocoRungs/tools/TwoWayAnova.r")
    # twoWayANOVA_R=rob.globalenv['twoWayANOVA']
    # df_r=pandas2ri.ri2py(df)
    #
    # statsList=twoWayANOVA_R(df_r)
    # pdb.set_trace()

    if sessionValues == False and sex == True:
        # AnovaRMRes=AnovaRM(data=df, depvar='measuredValue', subject='mouse', within=['recordingDay'],aggregate_func='sum').fit()
        # anova = pg.rm_anova(dv='measuredValue', within=['recordingDay', 'sex'], subject='mouse', data=df)
        # print(anova)
        md = smf.mixedlm("measuredValue ~ recordingDay+sex", df, groups=df["mouse"], missing='drop').fit()
        print(md.summary())
        sexPvalue=md.pvalues['recordingDay']

    # FRpd= FRpd.dropna()

    return (df, sexPvalue)

def analyzeStride(mouseDict, experiment):
    pawId=['FL','FR','HL','HR']
    variablesList=[]
    coordinates_time_list=[]
    for m in range(len(mouseDict)):
        recs=mouseDict[m]['pawData']
        nDays = len(recs)
        for n in range(nDays):
            nRecs = len(recs[n][4])
            for j in range(nRecs):
                wheelLinearSpeed = recs[n][1][j][1]
                wheelTime = recs[n][1][j][2]

                swingOnset = [[],[],[],[]]
                stanceOnset = [[],[],[],[]]
                swingOnsetArray = [swingOnset[0],swingOnset[1],swingOnset[2],swingOnset[3]]
                stanceOffsetArray = [stanceOnset[0],stanceOnset[1],stanceOnset[2],stanceOnset[3]]
                indecisiveFraction =[]
                twoRungCrossFraction= []
                rungCrossedStd=[]
                for i in range(4):
                    strideParameters={}
                    idxSwings = np.asarray(recs[n][4][j][3][i][1])
                    recTimes = np.asarray(recs[n][4][j][4][i][2])
                    xSpeed = np.array(recs[n][4][j][4][i][1])
                    wSpeed = np.asarray(recs[n][4][j][4][i][0])
                    pawPos = np.asarray(recs[n][2][j][5][i])
                    linearPawPos = recs[n][4][j][4][i][5]
                    rungNumbers = recs[n][4][j][3][i][2]
                    stepCharacter = recs[n][4][j][3][i][3]

                    rungCrossed = np.diff(rungNumbers)

                    wheelTime = recs[n][1][j][2]
                    wheelTimeMask = (wheelTime > 10) & (wheelTime < 52)
                    indecisive=[]
                    # mask = (recTimes[idxSwings[:, 0]] > 10) & (recTimes[idxSwings[:, 0]] < 52)
                    # twoRungCrossFraction = np.count_nonzero(np.greater_equal(rungCrossed, [2])) / len(rungNumbers)


                    
                    for k in range(len(idxSwings) - 1):
                        if (recTimes[idxSwings[k, 0]] > 10) and (recTimes[idxSwings[k, 0]] < 52):
                            idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                            idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))
                            idxSwingStartNext = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k + 1][0]]))
                            stanceOnset[i].append(pawPos[idxStanceStart, 0])
                            swingOnset[i].append(pawPos[idxSwingStart, 0])
                            indecisive.append(stepCharacter[k][3])
                    indecisiveFraction.append(np.count_nonzero(indecisive)/len(indecisive))
                    twoRungCrossFraction.append(np.count_nonzero(np.greater_equal(rungCrossed, [2])) / len(rungNumbers))
                    rungCrossedStd.append(np.std(rungCrossed))

                # coordination=analyse_paw_coordination(swingOnsetArray,stanceOffsetArray, ref_paw=0, dt=0.02)
                swingCoordination, swingTiming=analyseCoordination(swingOnsetArray,stanceOffsetArray)

                swingStanceD={}
                swingStanceD['swingP']=recs[n][4][j][3]
                swingStanceD['forFit']=recs[n][4][j][4]
                pawPosRaw=recs[n][2][j][5]
                # pdb.set_trace()
                # stepParameters = dataAnalysis_psth.calculateStepPar(pawPosRaw, swingStanceD)
                for i in range(4):

                    print('current position:','mouse:', m+1, 'day:',n+1,'trial:',j+1,'paw:',pawId[i])
                    # try:
                    idxSwings = np.asarray(recs[n][4][j][3][i][1])
                    # except IndexError:
                    #     pdb.set_trace()
                    recTimes = np.asarray(recs[n][4][j][4][i][2])
                    xSpeed = np.array(recs[n][4][j][4][i][1])
                    wSpeed = np.asarray(recs[n][4][j][4][i][0])
                    pawPos = np.asarray(recs[n][2][j][5][i])
                    pawSpeed = np.asarray(recs[n][2][j][3][i])
                    linearPawPos = recs[n][4][j][4][i][5]
                    rungNumbers = recs[n][4][j][3][i][2]
                    stepCharacter = recs[n][4][j][3][i][3]
                    newXtime=linearPawPos[:,0]
                    speedTimes=recTimes
                    rungCrossed = np.diff(rungNumbers)

                    wheelTime = recs[n][1][j][2]
                    wheelTimeMask = (wheelTime > 10) & (wheelTime < 52)

                    mask = (recTimes[idxSwings[:, 0]] > 10) & (recTimes[idxSwings[:, 0]] < 52)



                    for k in range(len(idxSwings)-1):
                        if (recTimes[idxSwings[k, 0]] > 10) and (recTimes[idxSwings[k, 0]] < 52):
                            idxSwingStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][0]]))
                            idxStanceStart = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k][1]]))
                            idxSwingStartNext = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k + 1][0]]))
                            idxStanceStartNext = np.argmin(np.abs(pawPos[:, 0] - recTimes[idxSwings[k + 1][1]]))
                            iStart=idxSwings[k][0]
                            iEnd= idxSwings[k][1]
                            iStartNext=idxSwings[k+1][0]

                        strideParameters = {}
                        strideTrajectories = {}
                        if (recTimes[idxSwings[k, 0]] > 10) and (recTimes[idxSwings[k, 0]] <52):  # only look at steps during motorization period
                            speedDiff = np.abs(xSpeed[iStart:iEnd] * 0.025 - wSpeed[iStart:iEnd])
                            pawSpeedTime =  (speedTimes[iStart:iEnd])

                            strideParameters['mouseId'] = mouseDict[m]['mouseName']
                            if 'f' in mouseDict[m]['mouseName']:
                                sex = 'female'
                            else:
                                sex = 'male'
                            
                            strideParameters['sex'] = sex
                            strideParameters['paw'] = pawId[i]
                            strideParameters['day'] = n + 1
                            strideParameters['trial'] = j + 1
                            strideParameters['trialNumber'] = nRecs
                            strideParameters['dayNumber'] = nDays
                            # strideParameters['dayId'] = mouseDict[m]['foldersRecordings'][n][0]
                            strideParameters['swingId'] = k + 1
                            strideParameters['rungId'] = rungNumbers[k]
                            if experiment=='muscimol':
                                strideParameters['treatment'] = mouseDict[m]['treatment']
                                strideTrajectories['treatment'] = mouseDict[m]['treatment']
                            elif experiment=='opto':
                                strideParameters['treatment'] = mouseDict[m]['treatment']
                                strideTrajectories['treatment'] = mouseDict[m]['treatment']
                            if nRecs > 1:
                                if j == 0:
                                    strideParameters['trial_category'] = 'first'
                                elif j == nRecs - 1:
                                    strideParameters['trial_category'] = 'last'
                                if j < nRecs / 2:
                                    strideParameters['trial_type'] = 'early'
                                elif j >= nRecs / 2:
                                    strideParameters['trial_type'] = 'late'
                            if n < nDays / 2:
                                strideParameters['day_category'] = 'early'
                            elif n >= nDays / 2:
                                strideParameters['day_category'] = 'late'
                            strideParameters['sex'] = sex
                            strideParameters['paw'] = pawId[i]
                            strideParameters['day'] = n + 1
                            strideParameters['trial'] = j + 1
                            strideParameters['trialNumber'] = nRecs
                            strideParameters['dayNumber'] = nDays
                            # strideParameters['dayId'] = mouseDict[m]['foldersRecordings'][n][0]
                            strideParameters['swingId'] = k + 1
                            strideParameters['rungId'] = rungNumbers[k]
                            if nRecs > 1:
                                if j == 0:
                                    strideParameters['trial_category'] = 'first'
                                elif j == nRecs - 1:
                                    strideParameters['trial_category'] = 'last'
                                if j < nRecs / 2:
                                    strideParameters['trial_type'] = 'early'
                                elif j >= nRecs / 2:
                                    strideParameters['trial_type'] = 'late'
                            if n < nDays / 2:
                                strideParameters['day_category'] = 'early'
                            elif n >= nDays / 2:
                                strideParameters['day_category'] = 'late'
                            #coordination
                            # strideParameters['stance_on_std']=coordination[i]['stance_on_std']
                            # strideParameters['swing_on_std']=coordination[i]['swing_on_std']
                            # strideParameters['stance_on_median']=coordination[i]['stance_on_median']
                            # strideParameters['swing_on_median']=coordination[i]['swing_on_median']
                            # strideParameters['relative_phase_mean']=coordination[i]['relative_phase_mean']
                            # strideParameters['relative_phase_std']=coordination[i]['relative_phase_std']
                            # strideParameters['phase_difference_std']=coordination[i]['phase_difference_std']
                            # strideParameters['phase_difference_mean']=coordination[i]['phase_difference_mean']
                            # swingCoordination[f'ref_{pawId[pawNb]}'][c][f'stanceOnMedian_ref_{pawId[pawNb]}']
                            for key, value in swingCoordination['ref_FL'][i].items():
                                strideParameters[key] = value
                            key_list = [key for key in swingCoordination['ref_FL'][i]]
                            strideParameters['keys']=key_list

                            strideParameters['twoRungsFraction'] = twoRungCrossFraction[i]
                            strideParameters['rungsCrossed_std']=rungCrossedStd[i]
                            strideParameters['indecisiveFraction']=indecisiveFraction[i]
                            strideParameters['indecisiveStrides']=stepCharacter[k][3]
                            strideParameters['wheelDistance'] = np.sum(wheelLinearSpeed[wheelTimeMask] * np.gradient(wheelTime[wheelTimeMask]))
                            strideParameters['LinWheelSpeed_avg'] = np.mean(wheelLinearSpeed[wheelTimeMask])
                            strideParameters['stanceOnset']=pawPos[idxStanceStart, 0]
                            strideParameters['swingOnset']=pawPos[idxSwingStart, 0]
                            strideParameters['swingNumber'] = len(idxSwings)
                            strideParameters['swingSpeed']=np.mean(speedDiff)
                            strideParameters['swingSpeed_Max'] = np.max(speedDiff)
                            strideParameters['swingLength'] = pawPos[:, 1][idxStanceStart] - pawPos[:, 1][
                                idxSwingStart]
                            if linearPawPos[idxSwings[k + 1, 0]][1] - linearPawPos[idxSwings[k, 0]][1]>-2:
                                strideParameters['swingLengthLinear']=linearPawPos[idxSwings[k+1, 0]][1] - linearPawPos[idxSwings[k, 0]][1]
                            else:
                                strideParameters['swingLengthLinear'] = abs( linearPawPos[idxSwings[k + 1, 0]][1] - \
                                                                  linearPawPos[idxSwings[k, 0]][1])/2
                            if pawPos[:, 0][idxStanceStart] - pawPos[:, 0][idxSwingStart]<2:
                                strideParameters['swingDuration']= pawPos[:, 0][idxStanceStart] - pawPos[:, 0][idxSwingStart]
                            else:
                                strideParameters['swingDuration'] = (pawPos[:, 0][idxStanceStart] - pawPos[:, 0][
                                    idxSwingStart])/10
                            strideParameters['stanceDuration']= pawPos[:, 0][idxSwingStartNext] - pawPos[:, 0][idxStanceStart]
                            strideParameters['strideDuration']=pawPos[:, 0][idxSwingStartNext] - pawPos[:, 0][idxSwingStart]
                            strideParameters['frequency']=1/strideParameters['strideDuration']
                            strideParameters['dutyFactor']=strideParameters['stanceDuration']/strideParameters['strideDuration']
                            accelerationDic= calc_acceleration(speedDiff, pawSpeedTime)

                            strideParameters.update({key: value for key, value in accelerationDic.items() if key != "acceleration"})
                            strideParameters.update({key: value for key, value in swingCoordination['ref_FL'][i].items()})
                            strideParameters.update({key: value for key, value in swingCoordination['ref_FR'][i].items()})
                            strideParameters['acceleration'] =np.mean(accelerationDic['acceleration'])
                            strideParameters['rungCrossed'] = rungCrossed[k]

                            strideTrajectories['mouseId'] = mouseDict[m]['mouseName']
                            strideTrajectories['sex'] = sex
                            strideTrajectories['paw'] = pawId[i]
                            strideTrajectories['day'] = n + 1
                            strideTrajectories['trial'] = j + 1
                            strideTrajectories['trialNumber'] = nRecs
                            strideTrajectories['dayNumber'] = nDays
                            if nRecs > 1:
                                if j == 0:
                                    strideTrajectories['trial_category'] = 'first'
                                elif j == nRecs - 1:
                                    strideTrajectories['trial_category'] = 'last'
                                if j < nRecs / 2:
                                    strideTrajectories['trial_type'] = 'early'
                                elif j >= nRecs / 2:
                                    strideTrajectories['trial_type'] = 'late'
                            if n < nDays / 2:
                                strideTrajectories['day_category'] = 'early'
                            elif n >= nDays / 2:
                                strideTrajectories['day_category'] = 'late'
                            # strideTrajectories['dayId'] = mouseDict[m]['foldersRecordings'][n][0]
                            strideTrajectories['swingId'] = k + 1
                            strideTrajectories['rungId'] = rungNumbers[k]
                            strideTrajectories['rungCrossed'] = rungCrossed[k]
                            strideTrajectories['x_pos_time'] = pawPos[:, 0][idxSwingStart:idxStanceStart]
                            strideTrajectories['x_pos'] = pawPos[:, 1][idxSwingStart:idxStanceStart]
                            strideTrajectories['x_pos_linear_time'] = linearPawPos[:,0][iStart:iEnd]
                            strideTrajectories['x_pos_linear'] = linearPawPos[:,1][iStart:iEnd]
                            strideTrajectories['x_speed'] = speedDiff
                            strideTrajectories['x_speed_time'] = pawSpeedTime
                            strideTrajectories['swingOnset'] = speedDiff
                            # strideTrajectories['FL_HL_time'] = swingTiming['FL_HL_time']
                            # strideTrajectories['FR_HR_time'] = swingTiming['FR_HR_time']
                            # strideTrajectories['FL_FR_time'] = swingTiming['FL_FR_time']
                            # strideTrajectories['FL_HR_time'] = swingTiming['FL_HR_time']
                            # strideTrajectories['FR_HL_time'] = swingTiming['FR_HL_time']
                            strideTrajectories.update({key: value for key, value in swingTiming['ref_FL'][i].items()})
                            strideTrajectories.update({key: value for key, value in swingTiming['ref_FR'][i].items()})
                            #step
                            # for s in range(len(stepParameters['non_linear'][i]['stepLength'])):
                            #     strideParameters['stepLength']=stepParameters['non_linear'][i]['stepLength'][s]
                            #     strideParameters['stepDuration'] = stepParameters['non_linear'][i]['stepDuration'][s]
                            #     strideParameters['stepSpeed'] = stepParameters['non_linear'][i]['stepMeanSpeed'][s]

                            variablesList.append(strideParameters)
                            coordinates_time_list.append(strideTrajectories)
    df = pd.DataFrame(variablesList)
    df_coord = pd.DataFrame(coordinates_time_list)
    df_coord_keys = list(df_coord.columns.values[12:])
    for key in df_coord_keys:
        try:
            df_coord[key]=df_coord[key].apply(lambda r: tuple(r)).apply(np.asarray)
        except TypeError:
            pass
    pickleFileName=groupAnalysisDir + f'/behavior_data_{experiment}'
    pickleFileName1 = groupAnalysisDir + f'/behavior_data_trajectories_{experiment}'
    pickle.dump(df, open(pickleFileName, 'wb'))
    pickle.dump(df_coord, open(pickleFileName1, 'wb'))
    # df.to_csv(groupAnalysisDir + 'behavior_data_%s.csv'%experiment)
    # df_coord.to_csv(groupAnalysisDir + 'behavior_data_trajectories_%s.csv' % experiment)
    return df,df_coord




def equalizeDataLengthWithNan(data, maxDay):
    nanDay=np.empty((4,10))
    nanDay=nanDay.fill(np.nan)
    for m in range(len(data)):
        nDays = len(data[m][0])
        pdb.set_trace()
        while len(data[m][0])<maxDay:
            data[m][0]=np.concatenate((data[m][0],nanDay),axis=0)
        for n in range(nDays):
            for trial in data[m][0][n]:
                for index in range(len(trial)):
                    if trial[index]==0:
                        trial[index]=np.nan
            for i in range(len(data[m][0][n])):
                nTrial = len(data[m][0][n][i])

    return data

def equalizeAndAverage(data, maxDay):
    #for data with structure (mouse,paw) pawArray=[day1_value, day2_value]
    for m in range(len(data)):
        for i in range(4):
            nDays = len(data[m][i])
            # pdb.set_trace()
            while len(data[m][i]) < maxDay:
                data[m][i].append(np.nan)
    data=np.array(data)

    average=np.nanmean(data, axis=0)
    sem=stats.sem(data,axis=0,nan_policy='omit')
    # pdb.set_trace()
    return average,sem

def equalizeTrialNumberWithNan (array, listOfRecordings):
    cleanedArray=array
    newCleanedArray=[]
    try:
        for row in cleanedArray:
            while len(row) < 10:
                row=np.vstack((row,[[np.nan,np.nan,np.nan,np.nan]]))
            newCleanedArray.append(row)
    except:
        newCleanedArray=cleanedArray
        for n in range(len(newCleanedArray)):
            newCleanedArray[n]=np.array(newCleanedArray[n])
            while len(newCleanedArray[n]) < 10:
                newCleanedArray[n] = np.insert(newCleanedArray[n], len(newCleanedArray[n]), np.nan)
    return newCleanedArray
def regroupTrialsFromSameDayGroupSimple (array, listOfRecordings):
    newArray=[]
    duplicate=[]
    # pdb.set_trace()
    for k in range(len(array)):
        if k > 0 and listOfRecordings[k][1] == listOfRecordings[k - 1][1]:
            # print("same day found for day number %s recording %s! " % (k, listOfRecordings[k][1]))
            #regroup same day trials
            # array[k]=np.insert(array[k-1],len(array[k-1]),array[k]) #when single values
            array[k]=np.concatenate((array[k-1],array[k]))
            duplicate.append(k-1)

        newArray.append(array[k])
    cleanedArray=np.delete(newArray, duplicate)
    cleanedArray=np.array(cleanedArray)

    return cleanedArray
def pawAhead(mouseDict, experiment, treatments):
    fractionFLAhead_trials_mice=[]
    fractionFRAhead_trials_mice = []
    maxDays=10
    maxTrials=5
    for a in range(len(mouseDict)):
        recs=mouseDict[a]['pawData']
        if experiment=='ephy' or experiment=='test':
            listOfRecordings=mouseDict[a]['listOfRecordings']
            maxTrials = 10
        else:
            maxTrials = 5
            listOfRecordings = mouseDict[a]['foldersRecordings']
        recs=mouseDict[a]['pawData']
        nDays = len(recs)
        fractionFLAhead_mouse = []
        fractionFRAhead_mouse = []
        for n in range(nDays):
            fractionFLAhead_day = []
            fractionFRAhead_day=[]
            for j in range(len(recs[n][4])):
                xFL = recs[n][4][j][4][0][5]
                xFR = recs[n][4][j][4][1][5]
                recTimes = recs[n][4][j][4][0][2]
                timeFLAhead=[]
                timeFRAhead=[]
                for x in range(len(xFL)-1):
                    if x <len(xFR):
                        if xFL[x,1]>xFR[x,1]:
                            timeFLAhead.append(xFL[x,0])
                        elif xFR[x, 1] > xFL[x, 1]:
                            timeFRAhead.append(xFR[x, 0])

                fractionFLAhead=len(timeFLAhead)/len(xFL)
                fractionFRAhead=len(timeFRAhead)/len(xFR)
                fractionFLAhead_day.append(fractionFLAhead)
                fractionFRAhead_day.append(fractionFRAhead)
            fractionFLAhead_mouse.append(fractionFLAhead_day)
            fractionFRAhead_mouse.append(fractionFRAhead_day)
        # if experiment=='ephy':
        #     fractionFLAhead_mouse=regroupTrialsFromSameDayGroupSimple(fractionFLAhead_mouse,listOfRecordings)
        #     fractionFRAhead_mouse = regroupTrialsFromSameDayGroupSimple(fractionFRAhead_mouse, listOfRecordings)
        # else:
        #     fractionFLAhead_mouse=fractionFLAhead_mouse
        #     fractionFRAhead_mouse=fractionFRAhead_mouse
        # pdb.set_trace()
        fractionFLAhead_mouse_nan=np.empty((maxDays,maxTrials))
        fractionFRAhead_mouse_nan = np.empty((maxDays, maxTrials))
        fractionFLAhead_mouse_nan.fill(np.nan)
        fractionFRAhead_mouse_nan.fill(np.nan)

        for d in range(len(fractionFLAhead_mouse)):
            for f in range(len(fractionFLAhead_mouse[d])):
                fractionFLAhead_mouse_nan[d][f]=fractionFLAhead_mouse[d][f]
                fractionFRAhead_mouse_nan[d][f] = fractionFRAhead_mouse[d][f]
        # fractionFLAhead_avg_mice=np.nanmean(fractionFLAhead_mouse_nan, axis=1)
        if treatments==False:
            fractionFLAhead_trials_mice.append([mouseDict[a]['mouseName'], fractionFLAhead_mouse_nan])
            fractionFRAhead_trials_mice.append([mouseDict[a]['mouseName'], fractionFRAhead_mouse_nan])
        else:
            fractionFLAhead_trials_mice.append([mouseDict[a]['mouseName'], fractionFLAhead_mouse_nan,mouseDict[a]['treatment']])
            fractionFRAhead_trials_mice.append([mouseDict[a]['mouseName'], fractionFRAhead_mouse_nan,mouseDict[a]['treatment']])

    return(fractionFLAhead_trials_mice,fractionFRAhead_trials_mice)

def calculate_correlation_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    rvalues=dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            rvalues[r][c] = round(stats.pearsonr(df[r], df[c])[0], 4)
    mask = np.triu(np.ones_like(pvalues, dtype=np.bool))
    pvalues_cut_off=[0.05, 0.01,0.001]
    rvalues_cut_off=0.05

    # mask |= pvalues > cut_off[0]
    # mask=(pvalues>pvalues_cut_off[0]) | ((abs(rvalues)>rvalues_cut_off))
    mask=(pvalues>pvalues_cut_off[0]) | ((abs(rvalues)<abs(rvalues_cut_off)))
    # mask = (abs(rvalues)<rvalues_cut_off)
    # pdb.set_trace()
    # pvalues=pvalues[~mask]
    annot = [[('' if abs(val) > pvalues_cut_off[0] else '★')  # add one star if abs(val) >= extreme_1
              + ('' if abs(val) > pvalues_cut_off[1] else '★')  # add an extra star if abs(val) >= extreme_2
              + ('' if abs(val) > pvalues_cut_off[2] else '★')  # add yet an extra star if abs(val) >= extreme_3
              for val in row] for row in pvalues.to_numpy()]
    # pdb.set_trace()
    return pvalues, rvalues, annot, mask
def NormalizeData(sample_mat):
    interval_min = -2
    interval_max = 4
    scaled_mat = (sample_mat - np.min(sample_mat) / (np.max(sample_mat) - np.min(sample_mat)) * (
                interval_max - interval_min) + interval_min)
    return scaled_mat


import numpy as np

def calc_acceleration(SwingSpeed, pawSpeedTime):
    # Calculate acceleration
    if len(SwingSpeed>=2):
        delta_v = np.diff(SwingSpeed)
        delta_t = np.diff(pawSpeedTime)
        acceleration = delta_v / delta_t
        acceleration_smoothed=gaussian_filter1d(acceleration, 1)
        acceleration_phases = np.sign(acceleration_smoothed)
    else:
        acceleration=0
        acceleration_phases=0

    acc_phases = np.where(acceleration_phases == 1)[0]
    dec_phases = np.where(acceleration_phases == -1)[0]
    # max mean acceleration
    if len(acc_phases) > 0:
        max_acceleration = np.max(acceleration_smoothed[acc_phases])
        mean_acceleration = np.mean(acceleration_smoothed[acc_phases])
    else:
        max_acceleration = 0
        mean_acceleration = 0
    if len(dec_phases) > 0:
        max_deceleration = np.min(acceleration_smoothed[dec_phases])
        mean_deceleration = np.mean(acceleration_smoothed[dec_phases])
    else:
        max_deceleration = 0
        mean_deceleration=0

    # duration of acceleration phase
    if len(acc_phases) > 0:
        acc_start_time = pawSpeedTime[acc_phases[0] + 1]
        acc_end_time = pawSpeedTime[acc_phases[-1] + 1]
        acc_duration = acc_end_time - acc_start_time
    else:
        acc_duration = 0
    # duration of deceleration phase
    if len(dec_phases) > 0:
        dec_start_time = pawSpeedTime[dec_phases[0] + 1]
        dec_end_time = pawSpeedTime[dec_phases[-1] + 1]
        dec_duration = dec_end_time - dec_start_time
    else:
        dec_duration = 0

    accelerationDic = {
        "acceleration": acceleration_smoothed,
        "max_acceleration": max_acceleration,
        "mean_acceleration": mean_acceleration,
        "acceleration_phases": len(acc_phases),
        "max_deceleration": max_deceleration,
        "mean_deceleration": mean_deceleration,
        "deceleration_phases": len(dec_phases),
        "acc_duration": acc_duration,
        "dec_duration": dec_duration
    }
    return accelerationDic


import numpy as np
from scipy import stats


def analyseCoordination(swingOnset, swingOffset):
    # define reference paws and their labels
    refPaw = [0, 1, 2, 3]
    pawId = ['FL', 'FR', 'HL', 'HR']

    swingCoordination = {}
    swingTiming = {}
    dt = 0.02  # time resolution for counts

    for pawNb in range(len(refPaw)):
        swingCoordination[f'ref_{pawId[pawNb]}'] = {}
        swingTiming[f'ref_{pawId[pawNb]}'] = {}

        refPawSwingOn = swingOnset[pawNb]
        refPawSwingOff = swingOffset[pawNb]
        counts = np.zeros((4, int(1 / dt) + 1))
        iOnArray = [[], [], [], []]
        iOffArray = [[], [], [], []]

        for x in range(4):
            swingCoordination[f'ref_{pawId[pawNb]}'][x] = {}
            swingTiming[f'ref_{pawId[pawNb]}'][x] = {}

            for p in range(len(refPawSwingOn) - 1):
                refPawCycle = refPawSwingOn[p + 1] - refPawSwingOn[p]

                # normalize timings to refPawCycle
                for w in range(len(swingOnset[x])):
                    iOn = (swingOnset[x][w] - refPawSwingOn[p]) / refPawCycle
                    iOff = (swingOffset[x][w] - refPawSwingOn[p]) / refPawCycle

                    # include only swings that occur within the ref paw stride cycle
                    if (0 <= iOn < 1) and (iOff < 1):
                        iOnArray[x].append(iOn)
                        iOffArray[x].append(iOff)
                        counts[x, int(iOn / dt):int(iOff / dt)] += 1
                    elif (0 <= iOn < 1) and (iOff > 1):
                        counts[x, int(iOn / dt):] += 1
                        z = 1
                        aborted = False
                        while iOff > 1:
                            if (p + z + 1) >= len(refPawSwingOn):
                                aborted = True
                                break
                            iOff = (swingOffset[x][w] - refPawSwingOn[p + z]) / (
                                    refPawSwingOn[p + z + 1] - refPawSwingOn[p + z])
                            z += 1
                        counts[x, :] += (z - 1)
                        if not aborted:
                            counts[x, :int(iOff / dt)] += 1
                            iOffArray[x].append(iOff)

        countsProb = counts / np.max(counts) if np.max(counts) > 0 else counts

        for c in range(4):
            swingTiming[f'ref_{pawId[pawNb]}'][c][f'swingOnset_ref_{pawId[pawNb]}'] = iOnArray[c]
            swingTiming[f'ref_{pawId[pawNb]}'][c][f'swingOffset_ref_{pawId[pawNb]}'] = iOffArray[c]
            swingTiming[f'ref_{pawId[pawNb]}'][c][f'swingProb_ref_{pawId[pawNb]}'] = countsProb[c]
            swingTiming[f'ref_{pawId[pawNb]}'][c][f'swingCount_ref_{pawId[pawNb]}'] = counts[c]

            swingCoordination[f'ref_{pawId[pawNb]}'][c][f'iqr_25_75_ref_{pawId[pawNb]}'] = stats.iqr(countsProb[c],
                                                                                                     rng=[25, 75])
            swingCoordination[f'ref_{pawId[pawNb]}'][c][f'iqr_70_90_ref_{pawId[pawNb]}'] = stats.iqr(countsProb[c],
                                                                                                     rng=[70, 90])
            swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOffStd_ref_{pawId[pawNb]}'] = np.std(iOffArray[c])

            if c != pawNb:
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOnMedian_ref_{pawId[pawNb]}'] = np.percentile(
                    iOnArray[c], 50)

            try:
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOffMedian_ref_{pawId[pawNb]}'] = np.percentile(
                    iOffArray[c], 50)
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOff_iqr_25_75_ref_{pawId[pawNb]}'] = stats.iqr(
                    iOffArray[c], rng=[25, 75])
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOff_iqr_25_ref_{pawId[pawNb]}'] = np.percentile(
                    iOffArray[c], 25)
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOff_iqr_75_ref_{pawId[pawNb]}'] = np.percentile(
                    iOffArray[c], 75)
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOff_iqr_70_90_ref_{pawId[pawNb]}'] = stats.iqr(
                    iOffArray[c], rng=[70, 90])
            except IndexError:
                swingCoordination[f'ref_{pawId[pawNb]}'][c][f'swingOffMedian_ref_{pawId[pawNb]}'] = np.nan

    # Safe aggregation of combined swing timing variables
    # Using median across relevant groups, with key presence checks
    def safe_median_swing_time(ref_key, index):
        if ref_key in swingTiming:
            if index in swingTiming[ref_key]:
                key = f'swingOffset_ref_{ref_key[-2:]}'  # e.g., ref_HL -> 'swingOffset_ref_HL'
                if key in swingTiming[ref_key][index]:
                    data = swingTiming[ref_key][index][key]
                    if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                        return np.median(data)
        return np.nan

    combinedSwingTiming = {}
    combinedSwingTiming['FL_HL_time'] = safe_median_swing_time('ref_HL', 0)
    combinedSwingTiming['FR_HR_time'] = safe_median_swing_time('ref_HR', 1)
    combinedSwingTiming['FL_FR_time'] = safe_median_swing_time('ref_FL', 1)
    combinedSwingTiming['FL_HR_time'] = safe_median_swing_time('ref_HR', 0)
    combinedSwingTiming['FR_HL_time'] = safe_median_swing_time('ref_HL', 1)

    # Add the combined timings to swingTiming dict for convenience
    swingTiming.update(combinedSwingTiming)

    return swingCoordination, swingTiming


def analyse_paw_coordination(swing_onset, stance_onset, ref_paw=0, dt=0.02): #using stance probability
    ref_paw_swing_on = swing_onset[ref_paw]
    ref_paw_stance_on = stance_onset[ref_paw]
    counts_stance = np.zeros((4, int(1 / dt) + 1))
    counts_swing = np.zeros((4, int(1 / dt) + 1))
    relative_stance_onset_array = [[], [], [], []]
    relative_stance_offset_array = [[], [], [], []]
    phase_difference = [[] for _ in range(4)]
    relative_phase = [[] for _ in range(4)]

    for paw_num in range(4):
        for cycle_num in range(len(ref_paw_stance_on) - 1):
            ref_paw_cycle = ref_paw_stance_on[cycle_num + 1] - ref_paw_stance_on[cycle_num]
            for stance_num in range(len(stance_onset[paw_num])-1):
                relative_stance_onset = (stance_onset[paw_num][stance_num] - ref_paw_stance_on[cycle_num]) / ref_paw_cycle
                relative_stance_offset = (swing_onset[paw_num][stance_num + 1] - ref_paw_stance_on[cycle_num]) / ref_paw_cycle
                if (0 <= relative_stance_onset < 1) and (0 <relative_stance_offset < 1):
                    relative_stance_onset_array[paw_num].append(relative_stance_onset)
                    relative_stance_offset_array[paw_num].append(relative_stance_offset)
                    counts_stance[paw_num, int(relative_stance_onset / dt):int(relative_stance_offset / dt)] += 1
                    counts_swing[paw_num, int(relative_stance_offset / dt):int(1/dt)] += 1
                elif (0 <= relative_stance_onset < 1) and (relative_stance_offset > 1):
                    counts_stance[paw_num, int(relative_stance_onset / dt):] += 1
                    z = 1
                    while (relative_stance_offset > 1) :
                        if ((cycle_num + z + 1) >= len(ref_paw_stance_on)):
                            break
                        relative_stance_offset = (stance_onset[paw_num][stance_num] - ref_paw_stance_on[cycle_num + z]) / (
                                ref_paw_stance_on[cycle_num + z + 1] - ref_paw_stance_on[cycle_num + z])
                        counts_swing[paw_num, :int(1 / dt)] += 1
                        z += 1
                    counts_stance[paw_num, :] += (z - 1)
                    if not (cycle_num + z + 1) >= len(ref_paw_stance_on):
                        if relative_stance_offset >0:
                            counts_stance[paw_num, :int(relative_stance_offset / dt)] += 1
                            counts_swing[paw_num, int(relative_stance_offset / dt):int(1 / dt)] += 1
                            relative_stance_offset_array[paw_num].append(relative_stance_offset)
                if stance_num < len(relative_stance_onset_array[ref_paw]):
                    phase_difference[paw_num].append(relative_stance_onset - relative_stance_onset_array[ref_paw][stance_num])
                    relative_phase[paw_num].append(
                        relative_stance_onset - relative_stance_onset_array[ref_paw][stance_num] + (relative_stance_onset > relative_stance_onset_array[ref_paw][stance_num]))

    counts_stance_prob = counts_stance / np.max(counts_stance)
    counts_swing_prob = counts_swing / np.max(counts_swing)
    stance_on_std = []
    stance_on_median = []
    swing_on_median = []
    swing_on_std = []
    meanPhaseDifference=[]
    meanRelativePhase=[]
    stdPhaseDifference=[]
    stdRelativePhase=[]
    coordination={}
    for paw_num in range(4):
        coordination[paw_num]= {}
        stance_on_std.append(np.std(relative_stance_offset_array[paw_num]))
        swing_on_std.append(np.std(relative_stance_onset_array[paw_num]))
        stance_on_median.append(np.percentile(relative_stance_offset_array[paw_num], 50))
        swing_on_median.append(np.percentile(relative_stance_onset_array[paw_num], 50))
        meanPhaseDifference.append(np.mean(phase_difference[paw_num]))
        meanRelativePhase.append(np.mean(relative_phase[paw_num]))
        stdPhaseDifference.append(np.std(phase_difference[paw_num]))
        stdRelativePhase.append(np.std(relative_phase[paw_num]))

        coordination [paw_num]= {
            'counts_stance': counts_stance[paw_num],
            'counts_stance_prob': counts_stance_prob[paw_num],
            'counts_swing': counts_swing[paw_num],
            'counts_swing_prob': counts_swing_prob[paw_num],
            'stance_on_std': stance_on_std[paw_num],
            'stance_on_median': stance_on_median[paw_num],
            'swing_on_median': swing_on_median[paw_num],
            'swing_on_std': swing_on_std[paw_num],
            'phase_difference': np.mean(phase_difference[paw_num]),
            'relative_phase': np.mean(relative_phase[paw_num]),
            'relative_stance_onset':relative_stance_onset_array[paw_num],
            'relative_stance_offset': relative_stance_offset_array[paw_num],
            'relative_phase_mean':np.mean(meanRelativePhase[paw_num]),
            'relative_phase_std': np.mean(stdRelativePhase[paw_num]),
            'phase_difference_mean': np.mean(meanPhaseDifference[paw_num]),
            'phase_difference_std': np.mean(stdPhaseDifference[paw_num])
        }

    return coordination

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def perform_mixedlm_obstacle(df, dependent_var, treatment=False):
    # for ind in range(len(independent_var)):
    if treatment:
        independent_vars = ['session', 'trial', 'whikers[T.half_trimmed]']
        formula = f'{dependent_var} ~ session+trial+whikers'
    else:
        independent_vars = ['session', 'trial']
        formula = f'{dependent_var} ~ session+trial'
    results = smf.mixedlm(formula, data=df, groups='mouse', missing='drop').fit()
    print(results.summary())

    pValues = {}
    stars = {}
    for ind in independent_vars:
        pValues[ind] = results.pvalues[ind]
        stars[ind] = starMultiplier(pValues[ind])
    summary = {}
    summary['table'] = results
    summary['pvalues'] = pValues
    summary['stars'] = stars

    return summary

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# If you already have this elsewhere, keep yours and remove this.
def starMultiplier(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    return '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else ''

def _model_to_rows(model, *, exp, dependent_var, condition, paw, extra_notes=None):
    """
    Convert a fitted MixedLMResults to tidy rows.
    Columns: exp, dependent_var, condition, paw, term, coef, se, z, p, stars, ci_low, ci_high, n, notes
    """
    rows = []
    if model is None or isinstance(model, float) and np.isnan(model):
        return rows
    try:
        params = model.params
        bse = model.bse
        pvals = model.pvalues
        # conf_int can fail for some models; guard it.
        try:
            ci = model.conf_int()
            # handle named index if present
            ci.columns = ['ci_low', 'ci_high']
        except Exception:
            ci = pd.DataFrame({'ci_low': pd.Series(index=params.index, dtype=float),
                               'ci_high': pd.Series(index=params.index, dtype=float)})
        # statsmodels MixedLM stores z-values in tvalues (for z-tests); name is 'z' here for clarity
        zvals = getattr(model, 'tvalues', pd.Series(index=params.index, dtype=float))

        nobs = getattr(model, 'nobs', np.nan)

        for term in params.index:
            if term.startswith('Group Var') or 'Intercept RE' in term:
                # skip variance components in the fixed-effects summary CSV
                continue
            coef = params.get(term, np.nan)
            se = bse.get(term, np.nan)
            z = zvals.get(term, np.nan)
            p = pvals.get(term, np.nan)
            ci_low = ci.loc[term, 'ci_low'] if term in ci.index else np.nan
            ci_high = ci.loc[term, 'ci_high'] if term in ci.index else np.nan
            rows.append({
                'exp': exp,
                'dependent_var': dependent_var,
                'condition': condition,
                'paw': paw,
                'term': term,
                'coef': coef,
                'se': se,
                'z': z,
                'p': p,
                'stars': starMultiplier(p),
                'ci_low': ci_low,
                'ci_high': ci_high,

                'notes': extra_notes or ''
            })
    except Exception as e:
        rows.append({
            'exp': exp,
            'dependent_var': dependent_var,
            'condition': condition,
            'paw': paw,
            'term': 'ERROR',
            'coef': np.nan, 'se': np.nan, 'z': np.nan, 'p': np.nan, 'stars': '',
            'ci_low': np.nan, 'ci_high': np.nan, 'n': np.nan,
            'notes': f'Failed to extract model results: {e}'
        })
    return rows

def perform_mixedlm(df, dependent_var, treatments=False, exp='behavior', version='v1',csv_folder=None):
    """
    Runs your MixedLM(s) as before, PLUS writes a tidy CSV summary with:
      - effect of day, trial, paw (and interactions you include in the formula)
      - coefficients, SE, z, p-values, stars, 95% CIs
      - per-paw models and overall models
    CSV is saved as: f"{exp}_{dependent_var}_{version}.csv"
    Returns your original summary dict (unchanged) and the CSV path.
    """
    if exp == 'muscimol':
        control = 'saline'
        treat = 'muscimol'
    else:
        control = 'tdTomato'
        treat = 'opsin'

    pawId = ['FL', 'FR', 'HL', 'HR']
    independent_vars = ['day', 'trial']

    tidy_rows = []  # will collect rows for the CSV

    if not treatments:
        # === Overall model (no treatments) ===
        # you had two formulas; the second overwrote the first — preserving your effective formula:
        formula = f'{dependent_var} ~ day+trial+paw'
        results = smf.mixedlm(formula, data=df, groups='mouseId', missing='drop').fit()

        # Original p-values + stars dicts (kept)
        pValues = {}
        stars = {}
        for ind in independent_vars:
            pValues[ind] = results.pvalues.get(ind, np.nan)
            stars[ind] = starMultiplier(pValues[ind])

        # Add overall model rows
        tidy_rows.extend(_model_to_rows(results, exp=exp, dependent_var=dependent_var,
                                        condition='all', paw='ALL'))

        # === Per-paw models ===
        paw_results = {}
        pawPvalues = {}
        pawStars = {}
        for i in range(4):
            paw = pawId[i]
            pawPvalues[paw] = {}
            pawStars[paw] = {}
            formula_paw = f'{dependent_var} ~ day+trial'
            try:
                mdl = smf.mixedlm(formula_paw, df.loc[df['paw'] == paw],
                                  groups="mouseId", missing='drop').fit()
                paw_results[paw] = mdl
                for ind in independent_vars:
                    p = mdl.pvalues.get(ind, np.nan)
                    pawPvalues[paw][ind] = p
                    pawStars[paw][ind] = starMultiplier(p)
                # Add per-paw rows
                tidy_rows.extend(_model_to_rows(mdl, exp=exp, dependent_var=dependent_var,
                                                condition='all', paw=paw))
            except Exception:
                # keep your behavior
                for ind in independent_vars:
                    pawPvalues[paw][ind] = np.nan
                    pawStars[paw][ind] = np.nan
                paw_results[paw] = np.nan

        summary = {
            'table': results,
            'pvalues': pValues,
            'stars': stars,
            'paw_table': paw_results,
            'paw_pvalues': pawPvalues,
            'paw_stars': pawStars
        }

    else:
        # === With treatments ===
        independent_vars = ['day', 'trial', f'treatment[T.{control}]',
                            f'trial:treatment[T.{control}]', f'day:treatment[T.{control}]']
        independent_vars_treatments = ['day', 'trial']

        # Your effective formula (last one wins)
        formula = f'{dependent_var} ~ day*treatment+trial*treatment+paw*treatment + day*paw + trial*paw + trial*day'
        formula_treatments = f'{dependent_var} ~ day+trial+paw'

        muscimol = df.loc[df['treatment'] == treat]
        saline = df.loc[df['treatment'] == control]

        results = {}
        results['all'] = smf.mixedlm(formula, data=df, groups='mouseId', missing='drop').fit()

        residuals = results['all'].resid
        norm, pvalueNorm = stats.normaltest(residuals.values)
        # Add an entry with residual normality (as a note on Intercept if available)
        tidy_rows.extend(_model_to_rows(results['all'], exp=exp, dependent_var=dependent_var,
                                        condition='all', paw='ALL',
                                        extra_notes=f'residual_normality_p={pvalueNorm:.4g}'))

        # Per-treatment subsets
        results[treat] = smf.mixedlm(formula_treatments, data=muscimol, groups='mouseId', missing='drop').fit()
        results[control] = smf.mixedlm(formula_treatments, data=saline, groups='mouseId', missing='drop').fit()

        # Tidy rows for subset models
        tidy_rows.extend(_model_to_rows(results[treat], exp=exp, dependent_var=dependent_var,
                                        condition=treat, paw='ALL'))
        tidy_rows.extend(_model_to_rows(results[control], exp=exp, dependent_var=dependent_var,
                                        condition=control, paw='ALL'))

        conditions = ['all', control, treat]

        # Original p-values + stars dicts (kept)
        pValues = {cond: {} for cond in conditions}
        stars = {cond: {} for cond in conditions}
        for ind in independent_vars:
            p = results['all'].pvalues.get(ind, np.nan)
            pValues['all'][ind] = p
            stars['all'][ind] = starMultiplier(p)
        for ind_treat in independent_vars_treatments:
            for t in conditions[1:]:
                p = results[t].pvalues.get(ind_treat, np.nan)
                pValues[t][ind_treat] = p
                stars[t][ind_treat] = starMultiplier(p)

        # === Per-paw models with/without treatment ===
        paw_results = {}
        pawPvalues = {}
        pawStars = {}
        for paw in ['FL', 'FR', 'HL', 'HR']:
            paw_results[paw] = {}
            pawPvalues[paw] = {}
            pawStars[paw] = {}

            formula_paw_all = f'{dependent_var} ~ day*treatment+treatment*trial+trial*day'
            formula_paw_treatment = f'{dependent_var} ~ day+trial+day*trial'

            # all
            mdl_all = smf.mixedlm(formula_paw_all, df.loc[df['paw'] == paw],
                                  groups='mouseId', missing='drop').fit()
            paw_results[paw]['all'] = mdl_all
            tidy_rows.extend(_model_to_rows(mdl_all, exp=exp, dependent_var=dependent_var,
                                            condition='all', paw=paw))

            # control
            mdl_ctrl = smf.mixedlm(formula_paw_treatment,
                                   df.loc[(df['paw'] == paw) & (df['treatment'] == control)],
                                   groups='mouseId', missing='drop').fit()
            paw_results[paw][control] = mdl_ctrl
            tidy_rows.extend(_model_to_rows(mdl_ctrl, exp=exp, dependent_var=dependent_var,
                                            condition=control, paw=paw))

            # treat
            mdl_treat = smf.mixedlm(formula_paw_treatment,
                                    df.loc[(df['paw'] == paw) & (df['treatment'] == treat)],
                                    groups='mouseId', missing='drop').fit()
            paw_results[paw][treat] = mdl_treat
            tidy_rows.extend(_model_to_rows(mdl_treat, exp=exp, dependent_var=dependent_var,
                                            condition=treat, paw=paw))

            # fill p-values/stars dicts (kept)
            pawPvalues[paw]['all'] = {}
            pawStars[paw]['all'] = {}
            for ind in independent_vars:
                try:
                    p = mdl_all.pvalues.get(ind, np.nan)
                    pawPvalues[paw]['all'][ind] = p
                    pawStars[paw]['all'][ind] = starMultiplier(p)
                except Exception:
                    pass

            for cond_mdl, cond_name in [(mdl_ctrl, control), (mdl_treat, treat)]:
                pawPvalues[paw][cond_name] = {}
                pawStars[paw][cond_name] = {}
                for ind_treat in independent_vars_treatments:
                    p = cond_mdl.pvalues.get(ind_treat, np.nan)
                    pawPvalues[paw][cond_name][ind_treat] = p
                    pawStars[paw][cond_name][ind_treat] = starMultiplier(p)

        summary = {
            'table': results,
            'pvalues': pValues,
            'stars': stars,
            'paw_table': paw_results,
            'paw_pvalues': pawPvalues,
            'paw_stars': pawStars
        }

    # === Write tidy CSV ===
    tidy_df = pd.DataFrame(tidy_rows)
    # nice ordering of columns
    cols = ['exp', 'dependent_var', 'condition', 'paw', 'term',
            'coef', 'se', 'z', 'p', 'stars', 'ci_low', 'ci_high', 'notes']
    tidy_df = tidy_df.reindex(columns=cols)
    if csv_folder is not None:
        csv_path = f"{csv_folder}/{exp}_{dependent_var}_{version}.csv"
    else:
        csv_path = f"{exp}_{dependent_var}_{version}.csv"
    tidy_df.to_csv(csv_path, index=False)

    # Return your original structure AND the CSV path (handy for the caller)
    return summary
def perform_mixedlm_treatment_single_paw(df, dependent_var):
    independent_vars = ['day', 'trial', 'treatment[T.tdTomato]','trial:treatment[T.tdTomato]', 'day:treatment[T.tdTomato]']
    # independent_vars = ['day', 'trial', 'treatment[T.saline]']

    formula= f'{dependent_var} ~ day*treatment+trial*treatment+day*trial'
    # formula= f'{dependent_var} ~ day+treatment+trial'
    formula_treatments = f'{dependent_var} ~ day+trial'
    muscimol = df.loc[df['treatment'] == 'opsin']
    saline = df.loc[df['treatment'] == 'tdTomato']
    results={}
    results['all'] = smf.mixedlm(formula, data=df, groups='mouseId', missing='drop').fit()
    results['tdTomato'] = smf.mixedlm(formula_treatments, data=saline, groups='mouseId', missing='drop').fit()
    results['opsin'] = smf.mixedlm(formula_treatments, data=muscimol, groups='mouseId', missing='drop').fit()
    # print(results.summary())
    # print(    results['all'].summary() )
    # print(    results['saline'].summary() )
    # pdb.set_trace()
    pValues = {}
    stars = {}
    cond=['all', 'tdTomato', 'opsin']
    for c in range(3):
        stars[cond[c]]={}
        pValues[cond[c]]={}
    # for c in range(3):
        if c==0:
            for ind in independent_vars:
                pValues[cond[c]][ind] = results[cond[c]].pvalues[ind]
                stars[cond[c]][ind] = starMultiplier(pValues[cond[c]][ind])
        else:
            for ind_ in independent_vars[:2]:
                pValues[cond[c]][ind_] = results[cond[c]].pvalues[ind_]
                stars[cond[c]][ind_] = starMultiplier(pValues[cond[c]][ind_])
    summary = {}
    summary['table'] = results
    summary['pvalues'] = pValues
    summary['stars'] = stars
    return summary
def performLMMANOVA(df, dependent_var,random_var, treatments=False):
    pawId=['FL', 'FR', 'HL', 'HR']
    independent_vars=['day','trial','paw']
    # for n in independent_vars:
    #
    #     indLevels = df[independent_vars[n]].unique()
    #     indList= {}
    #     indList[n]=indLevels.astype(str).tolist()
    trials = df['trial'].unique()
    days = df['day'].unique()
    trialList = trials.astype(str).tolist()
    daysList = days.astype(str).tolist()
    if not treatments :
        formula = f'{dependent_var} ~ day+trial+sex+day*sex+sex*trial+day*trial+(1|{random_var})'
        formula = f'{dependent_var} ~ day+trial+paw+(1|{random_var})'#day:paw+ trial:paw' #+ day:trial'
        formula = f'{dependent_var} ~ day*trial+trial*paw+day*paw+(1|{random_var})'
        model = models.Lmer(formula, data=df)
        model.fit(factors={"paw":["FL","FR","HL","HR"], "trial":trialList, "day":daysList},ordered=True)
        # model.anova()
        results=model.anova()
        # marginal_estimates_paw_day, comparisons_paw_day = model.post_hoc(marginal_vars="paw", grouping_vars=["day"])
        # marginal_estimates_paw_paw, comparisons_paw_paw = model.post_hoc(marginal_vars="paw", grouping_vars=["paw"])
        # marginal_estimates_day_day, comparisons_day_day = model.post_hoc(marginal_vars="day", grouping_vars=["day"])
        # marginal_estimates_trial_trial, comparisons_trial_trial = model.post_hoc(marginal_vars="trial", grouping_vars=["trial"])

        pValues={}
        stars={}
        
        for ind in independent_vars:
            pValues[ind]=results['P-val'][ind]
            stars[ind]=starMultiplier(pValues[ind])
        paw_model={}
        paw_results = {}
        pawPvalues = {}
        pawStars={}
        for i in range(4):
            pawPvalues[pawId[i]]={}
            pawStars[pawId[i]] = {}
            # paw_results[pawId[i]]={}
            formula_paw = f'{dependent_var} ~ day+trial+(1|{random_var})'

            paw_model[pawId[i]]  = models.Lmer(formula_paw, data=df)
            paw_model[pawId[i]] .fit(factors={"trial": trialList, "day": daysList}, ordered=True)
            # paw_model.anova()
            paw_results[pawId[i]]  = paw_model[pawId[i]].anova()

            # if dependent_var=='swingLength':
            #     print(f'{pawId[i]}!!!!!!!!!!!!!!',paw_model[pawId[i]].anova())
            #     print(results)
            #     if i==3:
            #         pdb.set_trace()
            #     print(results.summary())
            #     print(paw_results[pawId[i]].summary())
            #     pdb.set_trace()
            independent_vars_paw=['day','trial']
            for ind_paw in independent_vars_paw:
                pawPvalues[pawId[i]][ind_paw]= paw_results[pawId[i]]['P-val'][ind_paw]
                pawStars[pawId[i]][ind_paw]= starMultiplier(pawPvalues[pawId[i]][ind_paw])
        summary={}
        summary['table']=results
        summary['pvalues'] = pValues
        summary['stars'] = stars
        summary['paw_table'] = paw_results
        summary['paw_pvalues'] = pawPvalues
        summary['paw_stars'] = pawStars
        return summary
    elif treatments:
        results={}
        independent_vars = [ 'day', 'trial','paw','treatment']
        independent_vars_treatments = ['day', 'trial', 'paw']
        formula = f'{dependent_var} ~ day*treatment+trial*treatment+paw+day*trial+(1|{random_var})'#day:paw+ trial:paw' #+ day:trial'
        formula_treatments = f'{dependent_var} ~ day+trial+paw+(1|{random_var})'
        
        model_all = models.Lmer(formula, data=df)
        model_all.fit(factors={"treatment":["saline","muscimol"], "paw":["FL","FR","HL","HR"], "trial":trialList, "day":daysList},ordered=True)
        # model_all.anova()
        results['all']=model_all.anova()
        muscimol = df.loc[df['treatment'] == 'muscimol']
        saline = df.loc[df['treatment'] == 'saline']


        model_muscimol = models.Lmer(formula_treatments, data=muscimol)
        model_muscimol.fit(factors={"paw":["FL","FR","HL","HR"], "trial":trialList, "day":daysList},ordered=True)
        # model_muscimol.anova()
        results['muscimol']= model_muscimol.anova()
        model_saline = models.Lmer(formula_treatments, data=saline)
        model_saline.fit(factors={"paw":["FL","FR","HL","HR"], "trial":trialList, "day":daysList},ordered=True)
        # model_saline.anova()
        results['saline']= model_saline.anova()


        print(results['all'])
        print(results['muscimol'])
        print(results['saline'])
        # pdb.set_trace()
        pValues={}
        stars={}
        conditions=['all', 'saline', 'muscimol']
        for cond in conditions:
            pValues[cond]={}
            stars[cond]={}
        for ind in independent_vars:
            pValues['all'][ind] = results['all']['P-val'][ind]
            stars['all'][ind] = starMultiplier(pValues['all'][ind])
        for ind_treat in independent_vars_treatments:
            for t in conditions[1:]:
                pValues[t][ind_treat] = results[t]['P-val'][ind_treat]
                stars[t][ind_treat] = starMultiplier(pValues[t][ind_treat])
        paw_model= {}
        paw_results = {}
        pawPvalues = {}
        pawStars={}
        for i in range(4):
            independent_vars_paw = ['day', 'trial', 'treatment']
            pawPvalues[pawId[i]]={}
            pawStars[pawId[i]] = {}
            # paw_results[pawId[i]]={}
            formula_paw = f'{dependent_var} ~ day*treatment+trial*treatment+(1|{random_var})'
            paw_model[pawId[i]] = models.Lmer(formula_paw, data=df.loc[df['paw'] == pawId[i]])
            paw_model[pawId[i]].fit(factors={"treatment":["saline","muscimol"], "trial":trialList, "day":daysList},ordered=True)
            paw_results[pawId[i]]=paw_model[pawId[i]].anova()
            for ind_paw in independent_vars_paw:
                pawPvalues[pawId[i]][ind_paw]= paw_results[pawId[i]]['P-val'][ind_paw]
                pawStars[pawId[i]][ind_paw]= starMultiplier(pawPvalues[pawId[i]][ind_paw])
        summary={}
        summary['table']=results
        summary['pvalues'] = pValues
        summary['stars'] = stars
        summary['paw_table'] = paw_results
        summary['paw_pvalues'] = pawPvalues
        summary['paw_stars'] = pawStars
        return summary
def expfunc(x, a,b, tau):
    # return a * np.exp(-x*tau)
    return b + a * np.exp(-x / tau)

    # tukey = pairwise_tukeyhsd(df[dependent_var], df[independent_var])
