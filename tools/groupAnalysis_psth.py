import pdb

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import tools.groupAnalysis as groupAnalysis
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_psth as dataAnalysis_psth
import seaborn as sns
import statsmodels.genmod.generalized_linear_model as glm
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import kruskal
from statsmodels.stats import multitest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/'

def collectModulatedCells (mice, PSTHAll, condition, recordings):
    pawList = ['FL', 'FR', 'HL', 'HR']
    # ephysDataPerCell.append([foldersRecordings[f][0], foldersRecordings[f][1], singleCellList, ephysPSTHDict])
    # psth = psthAllRec[r]['allSteps']
    # PSTHSummaryAllAnimals.append([n, mice[n], PSTHData])
    # cPawPos = singleCellList[r][2]
    # spikes = singleCellList[r][3]
    # swingStanceDict = singleCellList[r][4]
    # pawSpeed = singleCellList[r][5]
    # strideProps = singleCellList[r][6]
    modCell=0
    modCellSwing=0
    cellNb=[]
    miceCells_allTrialsPar=[]
    mice_allTrialsPar = {}
    micePSTHParameterDict={}
    nMice=len(mice)
    variablesList=[]
    psth_cellList=[]
    cell_glob_Id=0
    recordingGlobalId=0
    for m in range(nMice):
        #c is a cell [c][1] is the recording day it is from
        #count cell number per animal
        nCell=len(PSTHAll[m][2])
        cellNb.append(nCell)
        mouseId=mice[m]

        mouseCells_allTrialsPar=[]
        mice_allTrialsPar[mouseId]=[]
        dayNb=0
        for c in range(nCell): #look at a single cell
            date=PSTHAll[m][2][c][1]
            nTrial=len(PSTHAll[m][2][c][3])
            parameters = ['modulation_count_before_swingOnset','modulation_count_after_swingOnset','modulation_peak_before_swingOnset', 'modulation_peak_after_swingOnset','modulation_after_swingOnset','modulation_before_swingOnset',
                      'modulation_count_before_stanceOnset', 'modulation_count_after_stanceOnset','modulation_peak_before_stanceOnset','modulation_peak_after_stanceOnset','modulation_before_stanceOnset','modulation_after_stanceOnset']
            if (c>=1) & (PSTHAll[m][2][c-1][1] !=PSTHAll[m][2][c][1]):
                dayNb+=1
            else:
                dayNb=dayNb

            for r in range(nTrial):
                #psth of a single cell
                recordingGlobalId+=1
                psth=PSTHAll[m][2][c][3][r][condition]
                spikes=PSTHAll[m][2][c][2][r][3]
                if r==0:
                    cell_glob_Id+=1
                if (len(psth[0]) != 0) & (len(psth[1]) != 0) & (len(psth[2]) != 0) & (len(psth[3]) != 0):
                    swingOnsetArray=[psth[0]['swingOnset'],psth[1]['swingOnset'],psth[2]['swingOnset'],psth[3]['swingOnset']]
                    swingOffsetArray = [psth[0]['stanceOnset'], psth[1]['stanceOnset'], psth[2]['stanceOnset'],psth[3]['stanceOnset']]
                    (countsProb, counts,stanceOnStd,stanceOnMedian)=analyseCoordination(swingOnsetArray,swingOffsetArray)
                for i in range(4):
                    strideProp = PSTHAll[m][2][c][2][r][6][i]
                    psth_cell={}
                    psth_cell['paw']=pawList[i]
                    psth_cell['trial'] = r+1
                    psth_cell['cell']=c
                    psth_cell['cell_global_Id'] = cell_glob_Id
                    psth_cell['rec_global_Id'] = recordingGlobalId
                    psth_cell['cell_trial_Nb'] = nTrial
                    psth_cell['date'] = date
                    psth_cell['mouse']=mouseId
                    psth_cell['dayNb']=dayNb
                    if nTrial>1:
                        if  r==0:
                            psth_cell['trial_category']='first'
                        elif r==nTrial-1:
                            psth_cell['trial_category'] = 'last'
                        if r < nTrial / 2:
                            psth_cell['trial_type'] = 'early'
                            psth_cell['trial_type_bin'] = 0
                        elif r >= nTrial / 2:
                            psth_cell['trial_type'] = 'late'
                            psth_cell['trial_type_bin'] = 1
                    if c < nCell/2:
                        psth_cell['day_category']='early'
                        psth_cell['day_category_bin'] = 0
                    elif c >=nCell/2:
                        psth_cell['day_category'] = 'late'
                        psth_cell['day_category_bin'] = 1
                    if len(psth[i])!=0:
                        psth_cell['psth_stanceOnsetAligned']=psth[i]['psth_stanceOnsetAligned'][1]
                        psth_cell['psth_swingOnsetAligned'] = psth[i]['psth_swingOnsetAligned'][1]
                        psth_cell['psth_stanceOnsetAligned_time']=psth[i]['psth_stanceOnsetAligned'][0]

                        psth_cell['psth_swingOnsetAligned_time'] = psth[i]['psth_swingOnsetAligned'][0]
                        psth_cell['psth_stanceOnsetAligned_zscore']=  psth[i]['psth_stanceOnsetAligned_z-scored']
                        psth_cell['psth_swingOnsetAligned_zscore'] = psth[i]['psth_swingOnsetAligned_z-scored']
                    variables={}
                    variables['paw']=pawList[i]
                    variables['cell']=c
                    variables['cell_global_Id'] = cell_glob_Id
                    variables['rec_global_Id'] = recordingGlobalId
                    variables['cell_trial_Nb'] = nTrial
                    variables['date'] = date
                    variables['mouse']=mouseId
                    variables['dayNb']=dayNb
                    variables['trial'] = r+1
                    if nTrial>1:
                        if  r==0:
                            variables['trial_category']='first'
                        elif r==nTrial-1:
                            variables['trial_category'] = 'last'
                        if r < nTrial / 2:
                            variables['trial_type'] = 'early'
                            variables['trial_type_bin'] = 0
                        elif r >= nTrial / 2:
                            variables['trial_type'] = 'late'
                            variables['trial_type_bin'] = 1
                    if c < nCell/2:
                        variables['day_category']='early'
                        variables['day_category_bin'] = 0
                    elif c >=nCell/2:
                        variables['day_category'] = 'late'
                        variables['day_category_bin'] = 1


                    if len(psth[i])!=0:
                        variables['modulation_swingOnset']=psth[i]['modulation_swingOnset'][1]
                        variables['modulation_stanceOnset']=psth[i]['modulation_stanceOnset'][1]
                        variables['modulation_category_swingOnset']=psth[i]['modulation_category_swingOnset'][1]
                        variables['modulation_category_stanceOnset']=psth[i]['modulation_category_stanceOnset'][1]
                        variables['modulation_category_before_swingOnset']=psth[i]['modulation_category_swingOnset'][1][0]
                        variables['modulation_category_before_stanceOnset']=psth[i]['modulation_category_stanceOnset'][1][0]
                        variables['modulation_category_after_swingOnset']=psth[i]['modulation_category_swingOnset'][1][1]
                        variables['modulation_category_after_stanceOnset']=psth[i]['modulation_category_stanceOnset'][1][1]

                    # newVariablesList = ["acceleration",
                    #                     "max_acceleration",
                    #                     "mean_acceleration",
                    #                     "acceleration_phases",
                    #                     "max_deceleration",
                    #                     "mean_deceleration",
                    #                     "deceleration_phases",
                    #                     "acc_duration",
                    #                     "dec_duration"]
                    # for var in newVariablesList:
                    #     variables[var]=np.mean(psth[i][var])
                    variables['swingDuration']=np.mean(strideProp['swingDuration'])
                    variables['stanceDuration']=np.mean(strideProp['stanceDuration'])
                    variables['swingLength'] = np.mean(strideProp['swingLength'])
                    variables['stepLength'] = np.mean(strideProp['stepLength'])
                    variables['rungCrossed'] = np.mean(strideProp['rungCrossed'])
                    variables['stepDuration'] = np.mean(strideProp['stepDuration'])
                    variables['stepMeanSpeed'] = np.mean(strideProp['stepMeanSpeed'])
                    variables['swingLengthLinear'] = np.mean(strideProp['swingLengthLinear'])
                    variables['swingSpeed'] = np.mean(strideProp['swingSpeed'])
                    variables['stanceOnsetStd'] = stanceOnStd[i]
                    variables['stanceOnsetMedian'] = stanceOnMedian[i]
                    if len(psth[i])!=0:
                        for p in range(len(parameters)):
                            if psth[i][parameters[p]] !=0:
                                variables[parameters[p]] = psth[i][parameters[p]]
                            else:
                                variables[parameters[p]] = np.nan
                        timeIntervals = [0.1, 0.15, 0.2, 0.25, 0.3]
                        # timeIntervals = [0.1, 0.2, 0.3]
                        for t in range(len(timeIntervals)):
                            parameters_intervals=['before_swingOnset_z-score_AUC_%s'%timeIntervals[t],'after_swingOnset_z-score_AUC_%s'%timeIntervals[t],'before_stanceOnset_z-score_AUC_%s'%timeIntervals[t],'after_stanceOnset_z-score_AUC_%s'%timeIntervals[t], 'before_swingOnset_z-score_peak_%s'%timeIntervals[t],'after_swingOnset_z-score_peak_%s'%timeIntervals[t],'before_stanceOnset_z-score_peak_%s'%timeIntervals[t],'after_stanceOnset_z-score_peak_%s'%timeIntervals[t]]
                            for y in range(len(parameters_intervals)):
                                 variables[parameters_intervals[y]] = psth[i][parameters_intervals[y]]


                    variablesList.append(variables)
                    psth_cellList.append(psth_cell)
        #     mice_allTrialsPar[mice[m]].append(allTrialsPar)
        #     mouseCells_allTrialsPar.append([c,allTrialsPar])
        # miceCells_allTrialsPar.append([mice[m],mouseCells_allTrialsPar])
    indexList = ['paw', 'trial', 'cell', 'date', 'mouse']
    columns = ['paw', 'trial', 'cell', 'date', 'mouse','before_swingOnset_z-score_AUC', 'after_swingOnset_z-score_AUC', 'before_stanceOnset_z-score_AUC',
                  'after_stanceOnset_z-score_AUC', 'modulation_count_before_swingOnset',
                  'modulation_count_before_stanceOnset', 'modulation_count_after_swingOnset',
                  'modulation_count_after_stanceOnset',
                  'modulation_peak_before_swingOnset', 'modulation_peak_before_stanceOnset',
                  'modulation_peak_after_swingOnset', 'modulation_peak_after_stanceOnset']
    psth_keys=['psth_stanceOnsetAligned','psth_swingOnsetAligned','psth_stanceOnsetAligned_time','psth_swingOnsetAligned_time','psth_stanceOnsetAligned_zscore','psth_swingOnsetAligned_zscore']

    df = pd.DataFrame(variablesList)
    df_psth=pd.DataFrame(psth_cellList)

    for key in psth_keys:
        try:
            df_psth[key]=df_psth[key].apply(lambda r: tuple(r)).apply(np.asarray)
        except TypeError:
            pass

    # df_psth.to_csv(groupAnalysisDir + 'cells_psth_zscore_%s_%s.csv' % (condition, recordings))
    # df.to_csv(groupAnalysisDir+'psth_multi_%s_%s.csv'%(condition, recordings))
    # # pdb.set_trace()

    return df, df_psth     # mice_allTrialsPar.append
def collectComplexSpikes (mice, PSTHAll, condition, recordings):
    pawList = ['FL', 'FR', 'HL', 'HR']
    # ephysDataPerCell.append([foldersRecordings[f][0], foldersRecordings[f][1], singleCellList, ephysPSTHDict])
    # psth = psthAllRec[r]['allSteps']
    # PSTHSummaryAllAnimals.append([n, mice[n], PSTHData])
    # cPawPos = singleCellList[r][2]
    # spikes = singleCellList[r][3]
    # swingStanceDict = singleCellList[r][4]
    # pawSpeed = singleCellList[r][5]
    # strideProps = singleCellList[r][6]
    modCell=0
    modCellSwing=0
    cellNb=[]
    miceCells_allTrialsPar=[]
    mice_allTrialsPar = {}
    micePSTHParameterDict={}
    nMice=len(mice)
    variablesList=[]
    psth_cellList=[]
    cell_glob_Id=0
    recordingGlobalId=0
    for m in range(nMice):
        #c is a cell [c][1] is the recording day it is from
        #count cell number per animal
        nCell=len(PSTHAll[m][2])
        cellNb.append(nCell)
        mouseId=mice[m]

        mouseCells_allTrialsPar=[]
        mice_allTrialsPar[mouseId]=[]
        dayNb=0
        for c in range(nCell): #look at a single cell
            date=PSTHAll[m][2][c][1]
            nTrial=len(PSTHAll[m][2][c][3])

            if (c>=1) & (PSTHAll[m][2][c-1][1] !=PSTHAll[m][2][c][1]):
                dayNb+=1
            else:
                dayNb=dayNb

            for r in range(nTrial):
                #psth of a single cell
                recordingGlobalId+=1
                psth=PSTHAll[m][2][c][3][r][condition]
                spikes=PSTHAll[m][2][c][2][r][3]
                pawPos=PSTHAll[m][2][c][2][r][2]
                pawSpeed = PSTHAll[m][2][c][2][r][5]
                if r==0:
                    cell_glob_Id+=1
                if (len(psth[0]) != 0) & (len(psth[1]) != 0) & (len(psth[2]) != 0) & (len(psth[3]) != 0):
                    swingOnsetArray=[psth[0]['swingOnset'],psth[1]['swingOnset'],psth[2]['swingOnset'],psth[3]['swingOnset']]
                    swingOffsetArray = [psth[0]['stanceOnset'], psth[1]['stanceOnset'], psth[2]['stanceOnset'],psth[3]['stanceOnset']]
                    (countsProb, counts,stanceOnStd,stanceOnMedian)=analyseCoordination(swingOnsetArray,swingOffsetArray)
                for i in range(4):
                    strideProp = PSTHAll[m][2][c][2][r][6][i]
                    psth_cell={}
                    psth_cell['paw']=pawList[i]
                    psth_cell['trial'] = r+1
                    psth_cell['cell']=c
                    psth_cell['cell_global_Id'] = cell_glob_Id
                    psth_cell['rec_global_Id'] = recordingGlobalId
                    psth_cell['cell_trial_Nb'] = nTrial
                    psth_cell['date'] = date
                    psth_cell['mouse']=mouseId
                    psth_cell['dayNb']=dayNb
                    psth_cell['CS_time']=spikes
                    psth_cell['pawPos'] = pawPos[i]
                    psth_cell['pawSpeed'] = pawSpeed[i]
                    psth_cell['swingIndicies']=psth[i]['swingIndicies']
                    psth_cell['swingStart'] = psth[i]['swingStart']
                    psth_cell['stanceStart']=psth[i]['stanceStart']
                    psth_cell['stanceOnset'] = psth[i]['stanceOnset']
                    psth_cell['swingOnset'] = psth[i]['swingOnset']
                    psth_cell['indecisive'] = psth[i]['indecisive']
                    psth_cell['stanceDuration'] =psth[i]['stanceDuration']
                    psth_cell['swingSpeed'] = psth[i]['swingSpeed']
                    psth_cell['swingDuration'] = psth[i]['swingDuration']
                    psth_cell['swingLengthLinear'] =psth[i]['swingLengthLinear']
                    psth_cell['swingLength'] = psth[i]['swingLength']
                    if nTrial>1:
                        if  r==0:
                            psth_cell['trial_category']='first'
                        elif r==nTrial-1:
                            psth_cell['trial_category'] = 'last'
                        if r < nTrial / 2:
                            psth_cell['trial_type'] = 'early'
                            psth_cell['trial_type_bin'] = 0
                        elif r >= nTrial / 2:
                            psth_cell['trial_type'] = 'late'
                            psth_cell['trial_type_bin'] = 1
                    if c < nCell/2:
                        psth_cell['day_category']='early'
                        psth_cell['day_category_bin'] = 0
                    elif c >=nCell/2:
                        psth_cell['day_category'] = 'late'
                        psth_cell['day_category_bin'] = 1



                    psth_cellList.append(psth_cell)

    psth_keys=['CS_time','pawPos','pawSpeed','swingIndicies', 'swingStart', 'stanceStart','stanceOnset', 'swingOnset', 'indecisive']

    df_CS=pd.DataFrame(psth_cellList)

    for key in psth_keys:
        try:
            df_CS[key]=df_CS[key].apply(lambda r: tuple(r)).apply(np.asarray)
        except TypeError:
            pass
    cellCSDicList = []
    for i in [0,1]:
        df_CS_paw = df_CS[(df_CS['paw'] == pawList[i])]
        df_CS_paw = df_CS_paw.drop(
            df_CS_paw[df_CS_paw['cell_trial_Nb'] == 1].index).reset_index()
        cells = np.unique(df_CS_paw['cell_global_Id'])
        df_CS_paw_d = df_CS_paw.groupby(['cell_global_Id', 'day_category']).mean().reset_index()
        dayCatArray = np.array(df_CS_paw_d['day_category'])
        k = 0
        for c in cells:
            k += 1
            # trialType=np.unique(df_CS_paw[(df_CS_paw['cell_global_Id']==cell)]['trial_category'])
            trials = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['cell_trial_Nb'].values[0]
            count = 0
            for t in range(trials):

                cellCS = {}
                if t < 5:
                    trialCat = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['trial_category'].values[t]
                    dayCat = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['day_category'].values[0]
                    spikes = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['CS_time'].values[t]
                    swingStart=df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['swingOnset'].values[t]
                    stanceStart = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['stanceOnset'].values[t]
                    swingIdx=df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['swingIndicies'].values[t]
                    pawPos=df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['pawPos'].values[t]
                    misses = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['indecisive'].values[t]
                    swingLength = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['swingLengthLinear'].values[t]
                    stanceDuration = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['stanceDuration'].values[t]
                    swingSpeed = df_CS_paw[(df_CS_paw['cell_global_Id'] == c)]['swingSpeed'].values[t]

                    xPawPos=pawPos[:,1]
                    tPawPos=pawPos[:,0]
                    # except:
                    #     count=+1
                    #     spikes=np.zeros(1)
                    if len(spikes) > 0:
                        motorTime = (spikes > 10) & (spikes < 50)
                        rest = np.invert(motorTime)
                        # rest=spikes<10
                        locoCS = np.array(spikes[motorTime])
                        restCS = spikes[rest]
                        swingPeriodCs=[]
                        stancePeriodCs = []
                        swingPeriodNoCs=[]
                        stancePeriodNoCs=[]
                        swingIndicies=[]
                        stanceIndicies=[]
                        missSwing=[]
                        missStance = []
                        noCSswing=[]
                        noCSstance = []
                        noCSmissSw=[]
                        noCSmissSt = []
                        swingLengthSwCS=[]
                        swingLengthStCS = []
                        swingLengthPostCSSwing=[]
                        stanceDurationCS=[]
                        swingSpeedStCS = []
                        swingSpeedStNoCS = []
                        swingLengthStNoCS = []
                        stanceDurationNoCS=[]
                        for s in range(len(swingIdx)-1):
                            swingMask=(locoCS>swingStart[s]) & (locoCS<stanceStart[s])
                            stanceMask=(locoCS>stanceStart[s]) & (locoCS<swingStart[s+1])

                            if sum(swingMask)>0:
                                swingPeriodCs.append(locoCS[swingMask])
                                swingIndicies.append(s)
                                missSwing.append(misses[s])
                                swingLengthSwCS.append(swingLength[s])
                            else:
                                noCSswing.append(s)
                                noCSmissSw.append(misses[s])

                            if sum(stanceMask) > 0:
                                stancePeriodCs.append(locoCS[stanceMask])
                                stanceIndicies.append(s)
                                missStance.append(misses[s])
                                swingLengthStCS.append(swingLength[s])
                                swingLengthPostCSSwing.append(swingLength[s+1])
                                stanceDurationCS.append(stanceDuration[s])
                                swingSpeedStCS.append(swingSpeed[s])
                            else:
                                noCSstance.append(s)
                                noCSmissSt.append(misses[s])
                                swingLengthStNoCS.append(swingLength[s])

                                stanceDurationNoCS.append(stanceDuration[s])
                                swingSpeedStNoCS.append(swingSpeed[s])
                        print('swing',len(swingPeriodCs),'stance',len(stancePeriodCs))
                        try:
                            swingPeriodCsArray = np.concatenate(np.array(swingPeriodCs))
                            stancePeriodCsArray =np.concatenate(np.array(stancePeriodCs))
                            stanceIndiciesArray = np.array(stanceIndicies)
                            swingIndiciesArray = np.array(swingIndicies)
                            stanceMissArray = np.array(missStance)
                            swingMissArray = np.array(missSwing)
                        except:
                            pdb.set_trace()


                    if len(spikes) > 0:
                        locoCSFreq = len(locoCS) / 40
                        restCSFreq = len(restCS) / 20
                        print('rest', restCSFreq)
                        print('loco', locoCSFreq)
                        swingCSNb=len(swingPeriodCs)
                        stanceCSNb = len(stancePeriodCs)
                        swingCSProb=swingCSNb/(swingCSNb+stanceCSNb)
                        stanceCSProb = stanceCSNb / (swingCSNb + stanceCSNb)
                        cellCS['swingCSprob'] = swingCSProb
                        cellCS['stanceCSProb'] = stanceCSProb

                        cellCS['swingCSNb'] = swingCSNb
                        cellCS['stanceCSNb'] = stanceCSNb

                        cellCS['swingCSTime'] = swingPeriodCs
                        cellCS['stanceCSTime'] = stancePeriodCs
                        timeIntervalPos = 0.6
                        CSstTimes=[]
                        CSswTimes = []
                        CSswPos=[]
                        CSstPos = []
                        for w in range(len(swingPeriodCsArray)):
                            CSpawPosSwTimeMask=(tPawPos>swingPeriodCsArray[w]-timeIntervalPos) & (tPawPos<swingPeriodCsArray[w]+timeIntervalPos)
                            CSpawPosSwTime=tPawPos[CSpawPosSwTimeMask]-swingPeriodCsArray[w]
                            CSpawPosSw=xPawPos[CSpawPosSwTimeMask]
                            CSswTimes.append(CSpawPosSwTime)
                            CSswPos.append(CSpawPosSw)
                        for st in range(len(stancePeriodCsArray)):
                            # index=np.argmin(stancePeriodCsArray[st])
                            CSpawPosStTimeMask=(tPawPos>(stancePeriodCsArray[st]-timeIntervalPos)) & (tPawPos<(stancePeriodCsArray[st]+timeIntervalPos))
                            CSpawPosStTime=tPawPos[CSpawPosStTimeMask]-stancePeriodCsArray[st]

                            CSpawPosSt = xPawPos[CSpawPosStTimeMask]
                            CSstTimes.append(CSpawPosStTime)
                            CSstPos.append(CSpawPosSt)
                        cellCS['CSstanceIdx'] = stanceIndiciesArray
                        cellCS['CSswingIdx'] = swingIndiciesArray

                        cellCS['noCSstanceIdx'] = np.array(noCSstance)
                        cellCS['noCSswingIdx'] = np.array(noCSswing)

                        cellCS['noCSmissStance'] = np.array(noCSmissSt)
                        cellCS['noCSmissSwing'] = np.array(noCSmissSw)
                        cellCS['swingLengthPostCSSwing'] = np.array(swingLengthPostCSSwing)

                        cellCS['CSstanceMiss'] = stanceMissArray
                        cellCS['CSswingMiss'] = swingMissArray

                        ###################################
                        cellCS['swingLengthSwingCS'] = np.array(swingLengthSwCS)
                        cellCS['swingLengthStanceCS'] = np.array(swingLengthStCS)

                        cellCS['stanceDurationCS'] = np.array(stanceDurationCS)
                        cellCS['swingLengthStanceNoCS'] = np.array(swingLengthStNoCS)

                        cellCS['stanceDurationNoCS'] = np.array(stanceDurationNoCS)

                        cellCS['swingSpeedStNoCS'] = np.array(swingSpeedStNoCS)

                        cellCS['swingSpeedStCS'] = np.array(swingSpeedStCS)


                        cellCS['CSstanceTimes'] = CSstTimes
                        cellCS['CSstancePos'] = CSstPos
                        cellCS['CSswingTimes'] = CSswTimes
                        cellCS['CSswingPos'] = CSswPos
                        # pdb.set_trace()
                        cellCS['restCSFreq'] = restCSFreq

                        cellCS['locoCSFreq'] = locoCSFreq
                        cellCS['restCSFreq'] = restCSFreq
                        try:
                            cellCS['CS_ratio'] = locoCSFreq / restCSFreq
                            # print(cellCS['CS_ratio'])
                        except:
                            cellCS['CS_ratio'] = 1
                        cellCS['spikes'] = spikes
                        cellCS['locoCS'] = locoCS
                        cellCS['restCS'] = restCS
                        cellCS['trial'] = t
                        cellCS['trialCat'] = trialCat
                        cellCS['paw'] = pawList[i]
                        cellCS['Id'] = c
                        cellCS['paw'] = i
                        cellCS['day'] = dayCat
                        cellCSDicList.append(cellCS)

                # pdb.set_trace()
    compCS_df = pd.DataFrame(cellCSDicList)
    # pdb.set_trace()
    # df_psth.to_csv(groupAnalysisDir + 'cells_psth_zscore_%s_%s.csv' % (condition, recordings))
    # df_CS.to_csv(groupAnalysisDir+'psth_complexSpikes_%s_%s.csv'%(condition, recordings))
    # # pdb.set_trace()

    return df_CS, compCS_df     # mice_allTrialsPar.append
def getModulatedcell_Id_count(paw_df, catList, time, event, condition):
    varKey = 'modulation_category_'

    if condition!=None:
        key = (f'{varKey}{time}{event}Onset_{condition}')
    else:
        key = (f'{varKey}{time}{event}Onset')

    modCells_Id = {}
    modCells_count = {}
    overlapCount=0
    # get the modulated cell ids and check for overlaps
    for m, cat in enumerate(catList):
        # print(m)
        # print(key)
        # pdb.set_trace()

        # counts = paw_df[key].value_counts()
        # paw_df = paw_df.dropna(subset=[key])
        modCelldf = paw_df[paw_df[key] == cat]  # .groupby("cell_global_Id").mean()
        modCells_Id[cat] = np.unique(modCelldf['cell_global_Id'])
    modCells_Id['-'] = np.setdiff1d(modCells_Id['-'], modCells_Id['↓'])
    modCells_Id['-'] = np.setdiff1d(modCells_Id['-'], modCells_Id['↑'])
    # check for ovelaps:

    if len(np.intersect1d(modCells_Id['↓'], modCells_Id['↑'])) != 0:
        badCell_Id = np.intersect1d(modCells_Id['↓'], modCells_Id['↑'])
        badCell_mouse_id=np.unique(paw_df[paw_df['cell_global_Id'] == badCell_Id[0]]['mouse'])[0]
        badCell_day = np.unique(paw_df[paw_df['cell_global_Id'] == badCell_Id[0]]['date'])[0]
        badCell_paw = np.unique(paw_df[paw_df['cell_global_Id'] == badCell_Id[0]]['paw'])[0]
        overlapCount+=1
        for b in range(len(badCell_Id)):
            badCelldf = paw_df[paw_df['cell_global_Id'] == badCell_Id[b]]
            badCell_ModulationCount = (badCelldf[key].value_counts())
            if '-' in badCell_ModulationCount.index:
                badCell_ModulationCount = badCell_ModulationCount.drop('-')
            # print(badCell_ModulationCount)
            winingModulation = badCell_ModulationCount.index[np.argmax(badCell_ModulationCount.values)]
            # pdb.set_trace()
            print('there is overlap !!!!','for cell n°', badCell_Id[0], badCell_mouse_id, badCell_day, badCell_paw, time, event, 'most frequent is ', winingModulation)
            if winingModulation == '↑':
                badCell_index = np.argwhere(modCells_Id['↓'] == badCell_Id[b])
                modCells_Id['↓'] = np.delete(modCells_Id['↓'], badCell_index)
            elif winingModulation == '↓':
                badCell_index = np.argwhere(modCells_Id['↑'] == badCell_Id[b])
                modCells_Id['↑'] = np.delete(modCells_Id['↑'], badCell_index)
            # print('overlap fixed!!!!', 'for cell n°', np.intersect1d(modCells_Id['↓'], modCells_Id['↑']), 'most frequent is ', winingModulation)
    print('cells with overlaping modulation ',overlapCount)
    for m in modCells_Id:
        modCells_count[m] = len(modCells_Id[m])
    counts = pd.Series(modCells_count)
    counts = counts.sort_index(ascending=False)
    # counts = (paw_df[key1].value_counts())

    return modCells_Id, modCells_count, counts


def getModCellsDic(df, event, pawId):
    times = ['before_', 'after_']
    # iterate through time (before/after)
    modCells = {}
    modCells[pawId] = {}
    modCells[pawId]['all'] = np.empty(0)
    for t in range(2):
        # define modulation catgories
        catList = ['↓', '↑', '-']
        # time to look at
        time = times[t]
        # get Id and counts of modulated cells
        modCells_Id, modCells_count, counts =getModulatedcell_Id_count(df,
                                                                                           catList,
                                                                                           time, event,
                                                                                           condition=None)
        if t == 0:
            modCells[pawId]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
            modCells[pawId]['not_before'] = modCells_Id['-']
        else:
            modCells[pawId]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
            modCells[pawId]['not_after'] = modCells_Id['-']
    # pdb.set_trace()
    modCells[pawId]['all_mod'] = np.unique(np.concatenate((modCells[pawId]['before'], modCells[pawId]['after'])))
    all_cells = np.arange(64) + 1
    modCells[pawId]['all_non'] = np.setdiff1d(all_cells, modCells[pawId]['all_mod'])


    return modCells

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import  KMeans
import time
def find_optimal_clusters(cellsId, zscoreArrayList, m, z_scoreKey, z_scoreTimeKey, interval, IDs,max_clusters=5, showFig=True):
    zscoreArray = []
    maskedZscoreArray = []

    for c in range(len(cellsId)):
        zscoreSingle = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreKey].values
        zscoreSingleTime = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreTimeKey].values
        timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -0.10)
        # timeMask = (zscoreSingleTime[0] < 0.15) & (zscoreSingleTime[0] > -0.150)
        # timeMask = (zscoreSingleTime[0] >0) & (zscoreSingleTime[0] <0.10)
        zscore = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[c])][z_scoreKey].values
        zscoreArray.append(zscore[0][1])
        maskedZscoreArray.append(zscore[0][1][timeMask])
    perplexity=len(cellsId)-1
    maskedZscoreArray = np.array(maskedZscoreArray)
    zscoreArray = np.array(zscoreArray)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    zscoreArray_tsne = tsne.fit_transform(maskedZscoreArray)

    silhouette_scores = []
    for n_clusters in range(2, min(max_clusters + 1, len(zscoreArray_tsne) - 1)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(zscoreArray_tsne)
        silhouette_scores.append(silhouette_score(zscoreArray_tsne, labels))
    if len(silhouette_scores)>1:
        best_cluster_num = np.argmax(silhouette_scores) +2  # Add 2 to account for range starting from 2
        best_labels = KMeans(n_clusters=best_cluster_num, random_state=42).fit_predict(zscoreArray_tsne)

        cluster_ids = [[] for _ in range(best_cluster_num)]
        cluster_zscores = [[] for _ in range(best_cluster_num)]

        for i, label in enumerate(best_labels):
            cluster_ids[label].append(cellsId[i])
            cluster_zscores[label].append(np.array(zscoreArray[i]))
        cluster_zscores=np.asarray(cluster_zscores)
        cluster_ids = np.asarray(cluster_ids)
        if showFig:
            plt.ioff()
            plt.plot(range(2, min(max_clusters + 1, len(zscoreArray_tsne) - 1)), silhouette_scores, marker='o',
                     linestyle='-')
            plt.axvline(x=best_cluster_num, color='r', linestyle='--', label=f'Best Cluster Number for {IDs}')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score for Different Numbers of Clusters')

            plt.legend()
            plt.show()
            plt.pause(1)  # Pause for 2 seconds
            plt.close()  # Close the plot
            plt.ion()
    else:
        cluster_ids=cellsId
        cluster_zscores=zscoreArray
        best_cluster_num=1

    return best_cluster_num,cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.cluster import _hierarchical_fast
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from scipy.cluster import hierarchy
def find_cell_clusters(cellsId, zscoreArrayList, m, z_scoreKey, z_scoreTimeKey, interval, IDs , method='Kmeans',max_clusters=3, showFig=True):
    zscoreArray = []
    maskedZscoreArray = []

    for c in range(len(cellsId)):
        zscoreSingle = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreKey].values
        zscoreSingleTime = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreTimeKey].values
        timeMask = (zscoreSingleTime[0] < 0.0) & (zscoreSingleTime[0] > -0.1)
        zscore = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[c])][z_scoreKey].values
        zscoreArray.append(zscore[0][1])
        maskedZscoreArray.append(zscore[0][1][timeMask])

    maskedZscoreArray = np.array(maskedZscoreArray)
    zscoreArray = np.array(zscoreArray)

    if method=='Kmeans':
        silhouette_scores = []
        for n_clusters in range(2, min(max_clusters + 1, len(maskedZscoreArray) - 1)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(maskedZscoreArray)
            silhouette_scores.append(silhouette_score(maskedZscoreArray, labels))

        if len(silhouette_scores) > 1:
            best_cluster_num = np.argmax(silhouette_scores) + 2  # Add 2 to account for range starting from 2
            best_labels = KMeans(n_clusters=best_cluster_num, random_state=42).fit_predict(maskedZscoreArray)

            cluster_ids = [[] for _ in range(best_cluster_num)]
            cluster_zscores = [[] for _ in range(best_cluster_num)]

            for i, label in enumerate(best_labels):
                cluster_ids[label].append(cellsId[i])
                cluster_zscores[label].append(np.array(zscoreArray[i]))

            cluster_zscores = np.asarray(cluster_zscores)
            cluster_ids = np.asarray(cluster_ids)

            if showFig:
                plt.plot(range(2, min(max_clusters + 1, len(maskedZscoreArray) - 1)), silhouette_scores, marker='o',
                         linestyle='-')
                plt.axvline(x=best_cluster_num, color='r', linestyle='--', label=f'Best Cluster Number for {IDs}')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score for Different Numbers of Clusters')
                plt.legend()
                plt.show()
                plt.pause(2)  # Pause for 2 seconds
                plt.close()  # Close the plot
        else:
            cluster_ids = cellsId
            cluster_zscores = zscoreArray
            best_cluster_num = 1
        return best_cluster_num,cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime
    elif method == 'DTW':
        distance_matrix = np.zeros((len(zscoreArray), len(zscoreArray)))

        # Calculate pairwise DTW distances
        for i in range(len(zscoreArray)):
            for j in range(i+1, len(zscoreArray)):
                distance, _ = fastdtw(maskedZscoreArray[i], maskedZscoreArray[j], dist=euclidean)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        # Perform hierarchical clustering based on DTW distances
        linkage_matrix = hierarchy.linkage(distance_matrix, method='average')

        # Cut the dendrogram to get the desired number of clusters
        clusters = hierarchy.cut_tree(linkage_matrix, n_clusters=7)

        # Assign cell IDs to their respective clusters
        cluster_ids = [[] for _ in range(max_clusters)]
        cluster_zscores = [[] for _ in range(max_clusters)]
        cluster_numbers = []
        for i, label in enumerate(clusters.flatten()):
            cluster_ids[label].append(cellsId[i])
            cluster_zscores[label].append(zscoreArray[i])
            cluster_numbers.append(label)
        cluster_ids = np.asarray(cluster_ids)
        cluster_zscores = np.asarray(cluster_zscores)
        cluster_numbers = len(np.unique(cluster_numbers))

        if showFig:
            # Plot the dendrogram
            plt.figure(figsize=(10, 5))
            dn = hierarchy.dendrogram(linkage_matrix)
            plt.title('Dendrogram')
            plt.xlabel('Cell Index')
            plt.ylabel('Distance')
            plt.show()
        return cluster_numbers, cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime
    elif method=='DBS':
        silhouette_scores = []
        for eps in np.linspace(0.1, 5, num=10):
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(maskedZscoreArray)
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(maskedZscoreArray, labels))
            else:
                silhouette_scores.append(-1)  # Set a negative value if only one cluster is found

        best_eps_index = np.argmax(silhouette_scores)
        best_eps = np.linspace(0.1, 5, num=10)[best_eps_index]
        best_labels = DBSCAN(eps=best_eps, min_samples=2).fit_predict(maskedZscoreArray)

        cluster_ids = [[] for _ in range(len(np.unique(best_labels)))]
        cluster_zscores = [[] for _ in range(len(np.unique(best_labels)))]

        for i, label in enumerate(best_labels):
            cluster_ids[label].append(cellsId[i])
            cluster_zscores[label].append(np.array(zscoreArray[i]))

        cluster_zscores = np.asarray(cluster_zscores)
        cluster_ids = np.asarray(cluster_ids)

        if showFig:
            plt.plot(np.linspace(0.1, 3.0, num=5), silhouette_scores, marker='o', linestyle='-')
            plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Best Eps for {IDs}')
            plt.xlabel('Eps')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score for Different Eps Values')
            plt.legend()
            plt.show()
            plt.pause(2)
            plt.close()

        return len(np.unique(best_labels)), cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime


def get_matching_indices_and_elements(array1, array2):
    indices = []
    elements = []

    for index, element in enumerate(array1):
        if element in array2:
            indices.append(index)
            elements.append(element)

    return indices, elements


# def getModulatedGlobalcell_Id_count(paw_df, event, condition):
#     varKey = 'modulation_category_'
#
#     if condition!=None:
#         key = (f'{varKey}{event}_{condition}')
#     else:
#         key = (f'{varKey}{event}')
#
#     modCells_Id = {}
#     modCells_count = {}
#     overlapCount=0
#     labels = ['↑↑', '↑-', '↑↓', '-↓', '↓↓', '↓-', '↓↑', '-↑', '--']
#     # get the modulated cell ids and check for overlaps
#     for m, cat in enumerate(labels):
#         # print(m)
#         # print(key)
#         # pdb.set_trace()
#
#         # counts = paw_df[key].value_counts()
#         # paw_df = paw_df.dropna(subset=[key])
#         modCelldf = paw_df[paw_df[key] == cat]  # .groupby("cell_global_Id").mean()
#         modCells_Id[cat] = np.unique(modCelldf['cell_global_Id'])
#     for keys in modCells_Id:
#
#
#     for m in modCells_Id:
#         modCells_count[m] = len(modCells_Id[m])
#     counts = pd.Series(modCells_count)
#     counts = counts.sort_index(ascending=False)
#     pdb.set_trace()
#     # counts = (paw_df[key1].value_counts())
#
#     return modCells_Id, modCells_count, counts

def get_allModulatedCells_nonModulatedCells_Id(paw_df,paw_psth_df,event):
    varKey = 'modulation_category_'
    time = ['before_', 'after_']
    catList = ['↓', '↑', '-']
    keyBefore = varKey + time[0] + event
    keyAfter = varKey + time[1] + event

    modCells_Id_before, modCells_count, counts = getModulatedcell_Id_count(paw_df, catList,time[0], event,condition=None)
    modCells_Id_after, modCells_count, counts = getModulatedcell_Id_count(paw_df, catList,time[1],  event, condition=None)
    # get the id of cells showing modulation or not
    modCell_Id = np.unique(np.concatenate(
        (modCells_Id_before['↑'], modCells_Id_after['↑'], modCells_Id_before['↓'], modCells_Id_after['↓'])))
    non_modCell_Id = np.intersect1d(modCells_Id_before['-'], modCells_Id_after['-'])
    # check overlap
    # print('overlapping cells', np.intersect1d(modCell_Id, non_modCell_Id), 'total cells',
    #       np.sum((len(modCell_Id), len(non_modCell_Id))))
    # get modulated/non_modulated_df
    modulated_mask = paw_psth_df['cell_global_Id'].isin(modCell_Id)
    non_modulated_mask = paw_psth_df['cell_global_Id'].isin(non_modCell_Id)
    modulated_paw_df = paw_psth_df.loc[modulated_mask]
    non_modulated_paw_df = paw_psth_df.loc[non_modulated_mask]
    # compute correlation coefficient
    cells_id = [modCell_Id, non_modCell_Id]
    modCells_df = [modulated_paw_df, non_modulated_paw_df]
    return cells_id, modCells_df
#
# def get_allModulatedCells_nonModulatedCells_Id_both_swing_stance(paw_df,paw_psth_df):
#     varKey = 'modulation_category_'
#     time = ['before_', 'after_']
#     catList = ['↓', '↑', '-']
#     events = ['swing', 'stance']
#     modCell_Id={}
#     non_modCell_Id={}
#     for e in range(2):
#         keyBefore = varKey + time[0] + events[e]
#         keyAfter = varKey + time[1] + events[e]
#
#         modCells_Id_before, modCells_count, counts = getModulatedcell_Id_count(paw_df, catList,time[0], events[e],condition=None)
#         modCells_Id_after, modCells_count, counts = getModulatedcell_Id_count(paw_df, catList,time[1],  events[e], condition=None)
#         # get the id of cells showing modulation or not
#         modCell_Id[events[e]] = np.unique(np.concatenate(
#             (modCells_Id_before['↑'], modCells_Id_after['↑'], modCells_Id_before['↓'], modCells_Id_after['↓'])))
#         non_modCell_Id[events[e]] = np.intersect1d(modCells_Id_before['-'], modCells_Id_after['-'])
#     # check overlap
#     all_cells = np.arange(64) + 1
#     modCell_Id['all']=np.unique(np.concatenate((modCell_Id['swing'], modCell_Id['stance'])))
#     non_modCell_Id['all']=np.setdiff1d(all_cells,modCell_Id['all'])
#     # get modulated/non_modulated_df
#     modulated_mask = paw_psth_df['cell_global_Id'].isin(modCell_Id['all'])
#     non_modulated_mask = paw_psth_df['cell_global_Id'].isin(non_modCell_Id['all'])
#     modulated_paw_df = paw_psth_df.loc[modulated_mask]
#     non_modulated_paw_df = paw_psth_df.loc[non_modulated_mask]
#     # compute correlation coefficient
#     cells_id = [modCell_Id, non_modCell_Id]
#     modCells_df = [modulated_paw_df, non_modulated_paw_df]
#     return cells_id, modCells_df
# def getTrials_PSTH_CorrCoeff_modulatedCells_both_swing_stance(cells_id, modCells_df, event):
#
#     psthKey = 'psth_%sOnsetAligned' % event
#     corr_coeff_all = {}
#     corr_coeff_all['modulated'] = np.array([])
#     corr_coeff_all['non_modulated'] = np.array([])
#
#     for m in range(len(cells_id)):
#
#         for cell in cells_id[m]['all']:
#             trial_psth_modulated = modCells_df[m][modCells_df[m]['cell_global_Id'] == cell][[psthKey]]
#             trial_psth_modulated_Array = [[], []]
#             if len(trial_psth_modulated) > 1:
#                 for trial in range(len(trial_psth_modulated)):
#                     trial_psth_modulated_Array[m].append(trial_psth_modulated.iloc[trial, 0])
#                 trial_psth_modulated_Array[m] = np.array(trial_psth_modulated_Array[m])
#
#                 correlation_matrix = np.corrcoef(trial_psth_modulated_Array[m])
#                 upper_triang_indices = np.triu_indices_from(correlation_matrix, k=1)
#                 corr_coeff_single = correlation_matrix[upper_triang_indices]
#                 if m == 0:
#                     corr_coeff_all['modulated'] = np.concatenate((corr_coeff_all['modulated'], corr_coeff_single))
#                 if m == 1:
#                     corr_coeff_all['non_modulated'] = np.concatenate(
#                         (corr_coeff_all['non_modulated'], corr_coeff_single))
#             # np.correlate(trial_psth_modulated_Array,trial_psth_modulated_Array)
#     modulated = corr_coeff_all['modulated']
#     non_modulated = corr_coeff_all['non_modulated']
#     return modulated, non_modulated
#takes a list of cells and list of paw paramets containing cell id and compute PSTH correlation coef, here it's made for modulated and non modulated cells id
def getTrials_PSTH_CorrCoeff_modulatedCells(cells_id, modCells_df, event):

    psthKey = 'psth_%sOnsetAligned' % event
    corr_coeff_all = {}
    corr_coeff_all['modulated'] = np.array([])
    corr_coeff_all['non_modulated'] = np.array([])

    for m in range(len(cells_id)):

        for cell in cells_id[m]:
            trial_psth_modulated = modCells_df[m][modCells_df[m]['cell_global_Id'] == cell][[psthKey]]
            trial_psth_modulated_Array = [[], []]
            if len(trial_psth_modulated) > 1:
                for trial in range(len(trial_psth_modulated)):
                    trial_psth_modulated_Array[m].append(trial_psth_modulated.iloc[trial, 0])
                trial_psth_modulated_Array[m] = np.array(trial_psth_modulated_Array[m])

                correlation_matrix = np.corrcoef(trial_psth_modulated_Array[m])
                upper_triang_indices = np.triu_indices_from(correlation_matrix, k=1)
                corr_coeff_single = correlation_matrix[upper_triang_indices]
                if m == 0:
                    corr_coeff_all['modulated'] = np.concatenate((corr_coeff_all['modulated'], corr_coeff_single))
                if m == 1:
                    corr_coeff_all['non_modulated'] = np.concatenate(
                        (corr_coeff_all['non_modulated'], corr_coeff_single))
            # np.correlate(trial_psth_modulated_Array,trial_psth_modulated_Array)
    modulated = corr_coeff_all['modulated']
    non_modulated = corr_coeff_all['non_modulated']
    return modulated, non_modulated
import statsmodels.formula.api as smf

def perform_linear_regression(df, dependent_var, independent_var):
    formula = f'{dependent_var} ~ {independent_var}'
    model = smf.ols(formula, data=df, missing='drop')
    results = model.fit()
    return results, results.pvalues[independent_var]

def perform_GLM(df, dependent_var, independent_var):
    formula = f'{dependent_var} ~ {independent_var}'
    family = sm.families.Gaussian()
    link = sm.families.links.identity()
    model = glm.GLM(df[dependent_var], df[[independent_var]], family=family, link=link, missing='drop')
    results = model.fit()
    return results,results.pvalues[independent_var]

def perform_mixedlm(df, dependent_var, independent_var, groups_var):
    formula = f'{dependent_var} ~ {independent_var}'
    model = smf.mixedlm(formula, data=df, groups=groups_var, missing='drop')
    results = model.fit()

    tukey = pairwise_tukeyhsd(df[dependent_var], df[independent_var])
    return results, results.pvalues[independent_var], tukey

def perform_GLM_and_Tukey(df, dependent_var, independent_var):
    formula = f'{dependent_var} ~ {independent_var}'
    family = sm.families.Gaussian()
    link = sm.families.links.identity()
    model = glm.GLM(df[dependent_var], df[[independent_var]], family=family, link=link, missing='drop')
    results = model.fit()

    tukey = pairwise_tukeyhsd(df[dependent_var], df[independent_var])
    return results, results.pvalues[independent_var], tukey


def perform_GLM_group_and_Tukey(df, dependent_var, independent_var, group_var):
    # Create interaction term
    df["interaction"] = df[independent_var] * df[group_var]

    # Fit the GLM model
    formula = f'{dependent_var} ~ {independent_var} + C({group_var}) + {independent_var}:C({group_var})'
    family = sm.families.Gaussian()
    link = sm.families.links.identity()
    model = glm.GLM(df[dependent_var], df[["interaction", independent_var, group_var]], family=family, link=link,
                    missing='drop')
    results = model.fit()
    tukey = pairwise_tukeyhsd(df[dependent_var], df[group_var])
    return results, tukey


def collectModulatedCells_multiple_conditions(mice, PSTHAll, conditionList, recordings):
    pawList = ['FL', 'FR', 'HL', 'HR']

    modCell = 0
    modCellSwing = 0
    cellNb = []
    miceCells_allTrialsPar = []
    mice_allTrialsPar = {}
    micePSTHParameterDict = {}
    nMice = len(mice)
    variablesList = []
    psth_cellList = []
    cell_glob_Id = 0


    for m in range(nMice):
        # c is a cell [c][1] is the recording day it is from
        # count cell number per animal
        nCell = len(PSTHAll[m][2])
        cellNb.append(nCell)
        mouseId = mice[m]
        dayNb=0
        mouseCells_allTrialsPar = []
        mice_allTrialsPar[mouseId] = []
        for c in range(nCell):  # look at a single cell
            date = PSTHAll[m][2][c][1]
            nTrial = len(PSTHAll[m][2][c][3])

            if (c>=1) & (PSTHAll[m][2][c-1][1] !=PSTHAll[m][2][c][1]):
                dayNb+=1
            else:
                dayNb=dayNb
            # pdb.set_trace()
            for r in range(nTrial):
                # psth of a single cell


                psth = PSTHAll[m][2][c][3][r]['allSteps']

                if r == 0:
                    cell_glob_Id += 1
                if (len(psth[0]) != 0) & (len(psth[1]) != 0) & (len(psth[2]) != 0) & (len(psth[3]) != 0):
                    swingOnsetArray=[psth[0]['swingOnset'],psth[1]['swingOnset'],psth[2]['swingOnset'],psth[3]['swingOnset']]
                    swingOffsetArray = [psth[0]['stanceOnset'], psth[1]['stanceOnset'], psth[2]['stanceOnset'],psth[3]['stanceOnset']]
                    (countsProb, counts,stanceOnStd,stanceOnMedian)=analyseCoordination(swingOnsetArray,swingOffsetArray)
                else:
                    stanceOnStd=[np.nan,np.nan,np.nan,np.nan]
                    stanceOnMedian = [np.nan, np.nan, np.nan, np.nan]
                for i in range(4):
                    # cPawPos = singleCellList[r][2]
                    # spikes = singleCellList[r][3]
                    # swingStanceDict = singleCellList[r][4]
                    # pawSpeed = singleCellList[r][5]
                    # strideProps = singleCellList[r][6]

                    psth_cell = {}
                    psth_cell['paw'] = pawList[i]
                    psth_cell['trial'] = r + 1
                    psth_cell['cell'] = c
                    psth_cell['cell_global_Id'] = cell_glob_Id
                    psth_cell['cell_trial_Nb'] = nTrial
                    psth_cell['date'] = date
                    psth_cell['dayNb'] = dayNb
                    psth_cell['mouse'] = mouseId
                    variables = {}
                    variables['paw'] = pawList[i]
                    variables['trial'] = r + 1
                    variables['cell'] = c
                    variables['cell_global_Id'] = cell_glob_Id
                    variables['cell_trial_Nb'] = nTrial
                    variables['date'] = date
                    variables['dayNb'] = dayNb
                    variables['mouse'] = mouseId
                    if nTrial>1 and r==0:
                        variables['trial_category']='first'
                        variables['trial_category_bin'] = 0
                    elif nTrial>1 and r==nTrial-1:
                        variables['trial_category'] = 'last'
                        variables['trial_category_bin'] = 1
                    elif nTrial==1 :
                        variables['trial_category'] = 'single'
                        variables['trial_category_bin'] =2
                    else:
                        variables['trial_category'] = 'intermediate'
                        variables['trial_category_bin'] = 3
                    if c < nCell/2:
                        variables['day_category']='early'
                        variables['day_category_bin'] = 0
                    elif c >=nCell/2:
                        variables['day_category'] = 'late'
                        variables['day_category_bin'] = 1
                    for l, condition in enumerate(conditionList):
                        psth_cond = PSTHAll[m][2][c][3][r][condition]
                        strideProp = PSTHAll[m][2][c][2][r][6][i]
                        PawPos = PSTHAll[m][2][c][2][r][2][i]
                        # swingOnInd=np.argwhere(psth_cond[i]['swingOnset']==PawPos[:,0])
                        # swingOffInd=np.argwhere(psth_cond[i]['stanceOnset']==PawPos[:,0])
                        # print(len(psth_cond[i]['swingOnset']))
                        # print(len(strideProp['swingLength']))
                        # pdb.set_trace()
                        parameters = [f'psth_stanceOnsetAligned_{condition}', f'psth_swingOnsetAligned_{condition}',
                                      f'modulation_category_before_swingOnset_{condition}',
                                      f'modulation_category_before_stanceOnset_{condition}',
                                      f'modulation_category_after_swingOnset_{condition}',
                                      f'modulation_category_after_stanceOnset_{condition}', f'stride_nb_{condition}',f'swingDuration_{condition}',f'swingLength_{condition}',f'swingSpeed_{condition}', f'swingLengthLinear_{condition}']
                        if len(psth_cond[i]) != 0:
                            psth_cell[parameters[0]] = psth_cond[i]['psth_stanceOnsetAligned'][1]
                            psth_cell[parameters[1]] = psth_cond[i]['psth_swingOnsetAligned'][1]

                        if len(psth_cond[i]) != 0:
                            variables[parameters[2]] =  psth_cond[i]['modulation_category_swingOnset'][1][0]
                            variables[parameters[3]] =   psth_cond[i]['modulation_category_stanceOnset'][1][0]
                            variables[parameters[4]] =    psth_cond[i]['modulation_category_swingOnset'][1][1]
                            variables[parameters[5]] =   psth_cond[i]['modulation_category_stanceOnset'][1][1]
                            variables[parameters[6]]=   len(psth_cond[i]['swingStart'])
                            variables[parameters[7]]=   np.mean(psth_cond[i]['swingDuration'])
                            variables[parameters[8]]=   np.mean(psth_cond[i]['swingLength'])

                            variables[parameters[9]] = np.mean(psth_cond[i]['swingSpeed'])
                            variables[parameters[10]] = np.mean(psth_cond[i]['swingLengthLinear'])

                    variables['stanceOnsetStd']=stanceOnStd[i]
                    variables['stanceOnsetMedian'] = stanceOnMedian[i]

                    if len(psth_cond[i]) != 0:
                        timeIntervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                        # timeIntervals = [0.1, 0.2, 0.3]
                        for t in range(len(timeIntervals)):
                            parameters_intervals = ['before_swingOnset_z-score_AUC_%s' % timeIntervals[t],
                                                    'after_swingOnset_z-score_AUC_%s' % timeIntervals[t],
                                                    'before_stanceOnset_z-score_AUC_%s' % timeIntervals[t],
                                                    'after_stanceOnset_z-score_AUC_%s' % timeIntervals[t],
                                                    'before_swingOnset_z-score_peak_%s' % timeIntervals[t],
                                                    'after_swingOnset_z-score_peak_%s' % timeIntervals[t],
                                                    'before_stanceOnset_z-score_peak_%s' % timeIntervals[t],
                                                    'after_stanceOnset_z-score_peak_%s' % timeIntervals[t]]
                            for l, condition in enumerate(conditionList):
                                psth_cond = PSTHAll[m][2][c][3][r][condition]
                                parameters_intervals_condition = [f'before_swingOnset_z-score_AUC_{timeIntervals[t]}_{condition}',
                                                        f'after_swingOnset_z-score_AUC_{timeIntervals[t]}_{condition}',
                                                        f'before_stanceOnset_z-score_AUC_{timeIntervals[t]}_{condition}',
                                                        f'after_stanceOnset_z-score_AUC_{timeIntervals[t]}_{condition}',
                                                        f'before_swingOnset_z-score_peak_{timeIntervals[t]}_{condition}',
                                                        f'after_swingOnset_z-score_peak_{timeIntervals[t]}_{condition}',
                                                        f'before_stanceOnset_z-score_peak_{timeIntervals[t]}_{condition}',
                                                        f'after_stanceOnset_z-score_peak_{timeIntervals[t]}_{condition}']
                                for y in range(len(parameters_intervals_condition)):
                                    variables[parameters_intervals_condition[y]] = psth_cond[i][parameters_intervals[y]]

                    variablesList.append(variables)
                    psth_cellList.append(psth_cell)
    df = pd.DataFrame(variablesList)
    df_psth = pd.DataFrame(psth_cellList)
    for l, condition in enumerate(conditionList):
        psth_keys = [f'psth_stanceOnsetAligned_{condition}', f'psth_swingOnsetAligned_{condition}']
        for key in psth_keys:
            try:
                df_psth[key] = df_psth[key].apply(lambda r: tuple(r)).apply(np.asarray)
            except TypeError:
                pass
    conditionListString = "_".join(conditionList)
    df_psth.to_csv(groupAnalysisDir + 'cells_psth_zscore_%s_%s.csv' % (conditionListString, recordings))
    df.to_csv(groupAnalysisDir + 'psth_multi_%s_%s.csv' % (conditionListString, recordings))
    # pdb.set_trace()

    return df, df_psth  # mice_allTrialsPar.append

def analyseCoordination(swingOnset, swingOffset):
    # define reference paw
    refPaw=0

    refPawSwingOn = swingOnset[refPaw]
    refPawSwingOff = swingOffset[refPaw]
    dt = 0.02
    # counts = np.zeros((8, int(1 / dt) + 1))
    counts = np.zeros((4, int(1 / dt) + 1))
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
                    counts[x, int(iOn / dt):int(iOff / dt)] += 1
                elif (0 <= iOn < 1) and (iOff > 1):
                    counts[x, int(iOn / dt):] += 1
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
                        counts[x, :int(iOff / dt)] += 1
                        iOffArray[x].append(iOff)

    countsProb = counts / np.max(counts)
    stanceOnStd = []
    stanceOnMedian = []
    # pdb.set_trace()
    for c in range(4):
        # iqr.append(stats.iqr(countsProb[c], rng=[70, 90]))
        stanceOnStd.append(np.std(iOffArray[c]))
        # stanceOnMedian.append(np.median(iOffArray[c]))
        try:
            stanceOnMedian.append(np.percentile(iOffArray[c], 50))
        except IndexError:
            # print(c)
            stanceOnMedian.append(np.nan)
    return (countsProb, counts,stanceOnStd,stanceOnMedian)

def psthGroupGenerateClusterDic (cellType, df_psth, df_cells, condition, pawNb):

    import pickle
    pawList = ['FL', 'FR', 'HL', 'HR']
    modCells = {}
    events = ['swing', 'stance']
    cellsDic = {}
    pawNb = [0, 1, 2, 3]
    if cellType=='MLI':
        all_cells = np.arange(64) + 1
    else:
        all_cells = np.arange(34) + 1
    for cell in all_cells:
        cellsDic[cell] = {}
        cellsDic[cell]['psth'] = {}
    for cell in all_cells:
        for p in pawList:
            cellsDic[cell]['psth'][p] = {}
    for cell in all_cells:
        for p in pawList:
            for ev in events:
                cellsDic[cell]['psth'][p][ev] = {}

    for i in pawNb:
        modCells[pawList[i]] = {}
        modCells[pawList[i]]['all'] = np.empty(0)
        modCells[pawList[i]]['swing'] = {}
        modCells[pawList[i]]['stance'] = {}
        for e in reversed(range(2)):
            event = events[e]

            # looking at specific paw values in the df that contains single values
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            # looking at specific paw values in the df that contains psth and zscore
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]
            # detect and remove empty cells, happens when there's no stride in a condition
            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)
            # big panels for each paw

            z_scoreKey = f'psth_{event}OnsetAligned_zscore'
            z_scoreTimeKey = f'psth_{event}OnsetAligned_time'


            for cell in all_cells:

                cellsDic[cell]['mouse'] = np.unique(paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)]['mouse'])[0]


                nDaysMouse=np.max(paw_psth_df[(paw_psth_df['mouse'] == cellsDic[cell]['mouse'] )]['dayNb'])
                cellsDic[cell]['recDate']=np.unique(paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)]['date'])[0]
                cellsDic[cell]['recDateNb'] = [np.unique(paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)]['dayNb'])[0]+1,nDaysMouse+1]
                cellsDic[cell]['trials']=np.unique(paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)]['trial'])
                zscoreArray=paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)][z_scoreKey].values
                zscoreTimeArray = paw_psth_df[(paw_psth_df['cell_global_Id'] == cell)][z_scoreTimeKey].values
                cellsDic[cell]['psth'][pawList[i]][event]['traces']=[]
                for r in range(len(cellsDic[cell]['trials'])):
                    cellsDic[cell]['psth'][pawList[i]][event]['traces'].append([zscoreTimeArray[r],zscoreArray[r][1]])
                      #zscoreSingle = modulated_paw_psth_df[(modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    #zscoreSingleTime = modulated_paw_psth_df[(modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values


            # time around event for keys
            times = ['before_', 'after_']
            # iterate through time (before/after)


            for t in range(2):
                # modCells[pawList[i]]['before']=np.empty(0)
                # modCells[pawList[i]]['after']=np.empty(0)
                # define modulation catgories
                catList = ['↓', '↑', '-']
                # time to look at
                time = times[t]
                # change position of plot if looking at more than one paw

                # ax2 = plt.subplot(gssub5[1])

                # get Id and counts of modulated cells
                modCells_Id, modCells_count, counts = getModulatedcell_Id_count(paw_df, catList, time, event, condition=None)
                if t == 0:
                    modCells[pawList[i]][event]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                    modCells[pawList[i]][event]['not_before'] = modCells_Id['-']
                else:
                    modCells[pawList[i]][event]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                    modCells[pawList[i]][event]['not_after'] = modCells_Id['-']
            # pdb.set_trace()
            modCells[pawList[i]][event]['all_mod'] = np.unique(
                np.concatenate((modCells[pawList[i]][event]['before'], modCells[pawList[i]][event]['after'])))
            # modCells[pawList[i]]['all_non']=np.unique(np.concatenate((modCells[pawList[i]]['not_before'], modCells[pawList[i]]['not_after'])))
            # modCells[pawList[i]]['all_non']=np.setdiff1d(modCells[pawList[i]]['all_non'],np.intersect1d(modCells[pawList[i]]['all_mod'],modCells[pawList[i]]['all_non']))

            modCells[pawList[i]][event]['all_non'] = np.setdiff1d(all_cells, modCells[pawList[i]][event]['all_mod'])
            # print(len(modCells[pawList[i]]['all_mod'] )+len(modCells[pawList[i]]['all_non'] ))
        # pdb.set_trace()
        bottom_pos = 0
        twoCol = ['0.2', '0.7']
        alpha_e = [0.4, 0.8]

    for i in pawNb:
        for event in events:    # bar_width = len(modCells[pawList[i]]['all_mod']) / len(all_cells) * 100
            # bar_left = bottom_pos
            modNb=0
            for cell in all_cells:
                try:
                    if cell in modCells[pawList[i]][event]['all_mod']:
                            cellsDic[cell]['psth'][pawList[i]][event]['mod'] = True
                            modNb+=1
                    else:
                        cellsDic[cell]['psth'][pawList[i]][event]['mod'] = False
                except:
                    pdb.set_trace()

                print(cellsDic[cell]['psth'][pawList[i]][event]['mod'])
            print('modulated number', modNb, 'modulated fraction', modNb*100/(64 if cellType=='MLI' else 34))
    # pdb.set_trace()

    groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/'
    pickleFileName = groupAnalysisDir +f'all_{cellType}_psth_zscore_dic_{condition}.p'
    pickle.dump(cellsDic, open(pickleFileName, 'wb'))