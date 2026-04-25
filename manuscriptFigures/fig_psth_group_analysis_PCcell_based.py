import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createPublicationVisualizations as createPublicationVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import pandas as pd
import tools.groupAnalysis_psth as groupAnalysis_psth


figVersion = '0.1'
mouse ='220211_f38'#,'220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28']
expDate = 'allPC'  #  'some', 'all', 'allMLI' or 'allPC'
recordings = 'allPC' # 'some', 'all', 'allMLI' or 'allPC'
spikeType = 'simple'
expDateForFig = '220503'
recsForFig = [4]
eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
(foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation,mouse)

groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
#date='23jan25'#
date ='23feb21'#

pawNb = 0
pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s' % (recordings)
ephysPSTHAnalysisFile  = eSD.analysisLocation + '/ephysPSTHSummary23feb21_%s_%s.p' % (recordings, spikeType)
ephysPSTHData = pickle.load(open(ephysPSTHAnalysisFile, 'rb'))


possibleConditions = ['allSteps'] #swingDuration_lastRec_20_80'] #['swingLengthLinear_lastRec_20_80']#'allSteps']#,'swingDurationLastRec20-80','swingLengthLastRec20-80', 'indecisiveSteps', 'not_indecisiveSteps']


for l, condition in enumerate(possibleConditions):
    for f in range(len(foldersRecordings)):
        found = False
        if foldersRecordings[f][1] == expDateForFig:
            for r in range(len(foldersRecordings[f][2])):
                for nDay in range(len(ephysPSTHData)):
                    if ephysPSTHData[nDay][1] == expDateForFig:
                        for nRec in recsForFig:
                            if ephysPSTHData[nDay][2][nRec][1][1] == foldersRecordings[f][2][r]:
                                ephysPSTHDict = ephysPSTHData[nDay][3][nRec][condition]
                                ephysPSTHDict_day=ephysPSTHData[nDay][3]
                                swingStanceD = ephysPSTHData[nDay][2][nRec][4]
                                ephys = ephysPSTHData[nDay][2][nRec][3]
                                pawPos = ephysPSTHData[nDay][2][nRec][2]
                                print('recordings for visualization:', foldersRecordings[f][0],
                                      foldersRecordings[f][2][r])
                                found = True
    if condition == 'allSteps':
        pickleFileName1 = groupAnalysisDir + '/cells_psth_zscore_%s_%s_%s' % (condition, recordings,date)
        pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s_%s' % (condition, recordings,date)
    else:
        pickleFileName1 = groupAnalysisDir + '/cells_psth_zscore_%s_%s_%s' % (condition, recordings,date)
        pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s_%s' % (condition, recordings, date)

    variable= 'swingLengthLinear'
    # variable='swingDuration'
    pickleFileName3 = groupAnalysisDir + f'/{recordings}{variable}Response.p'
    print(pickleFileName1)
    print(pickleFileName2)
    df_psth = pickle.load(open(pickleFileName1, 'rb'))
    df_cells = pickle.load(open(pickleFileName2, 'rb'))
    allModTraces = pickle.load(open(pickleFileName3, 'rb'))

    #print(df_psth.head())
    #print(df_cells.head())
    #pdb.set_trace()
    # cPV.psthGroupFigure_early_vs_late(figVersion,recordings[3:], df_psth, df_cells,condition,event, pawNb)
    #cPV.psthGroupFigure_cell_based(figVersion,recordings[3:], df_psth, df_cells,condition, pawNb)
    cPV.psthGroupFigure_cell_basedV2(figVersion,recordings[3:], df_psth, df_cells,condition, pawNb,allModTraces,variable)