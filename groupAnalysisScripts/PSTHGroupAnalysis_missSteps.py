# from oauth2client import tools
# tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
# args = tools.argparser.parse_args()
# import sys
# sys.path.append('~/Analysis/LocoRungs/')
import tools.createGroupVisualizations as createGroupVisualizations
import tools.groupAnalysis as groupAnalysis
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_psth as dataAnalysis_psth
import tools.groupAnalysis_psth as groupAnalysis_psth
import tools.extractSaveData as extractSaveData
import tools.createVisualizations as createVisualizations
import tools.createPresentationVisualizations as createPresentationVisualizations
import tools.parameters as par
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import pickle
import os
groupFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary'
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
cGV = createGroupVisualizations.createGroupVisualizations(groupFigDir)


# collecting analyzed ephys information from each animal
readDataAgain=False
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
recordings = 'allPC'
spikeType = 'complex'
cellType=recordings[3:]

pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s' % recordings

PSTHSummaryAllAnimals = []
date='23feb2'
# date='23feb21'
if os.path.isfile(pickleFileName) and not readDataAgain:
     PSTHSummaryAllAnimals = pickle.load(open(pickleFileName, 'rb'))
else:
     for n in range(len(mice)):
         eSD         = extractSaveData.extractSaveData(mice[n],recStruc='simplexEphy')

         # PSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary23Jan25_%s_%s.p' % (recordings, spikeType)
         PSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary%s_%s_%s.p' % (date,recordings, spikeType)
         PSTHData = pickle.load(open(PSTHAnalysisFile, 'rb'))  # eSD.analysisLocation,
         PSTHSummaryAllAnimals.append([n,mice[n],PSTHData])
         del eSD
     pickle.dump(PSTHSummaryAllAnimals, open(pickleFileName, 'wb'))

analyzeAgain=False
if spikeType=='simple':
    pickleFileName = groupAnalysisDir + f'/cells_psth_zscore_allSteps_{recordings}_{date}'
    pickleFileName0 = groupAnalysisDir + f'/cell_psth_allSteps_{recordings}_{date}'
    pickleFileName1 = groupAnalysisDir + f'/cells_psth_zscore_indecisiveSteps_{recordings}_{date}'
    pickleFileName4 = groupAnalysisDir + f'/cell_psth_indecisiveSteps_{recordings}_{date}'
    pickleFileName2 = groupAnalysisDir + f'/cells_psth_zscore_certainSteps_{recordings}_{date}'
    pickleFileName3 = groupAnalysisDir + f'/cell_psth_certainSteps_{recordings}_{date}'
else: 
    pickleFileName = groupAnalysisDir + f'/cells_psth_zscore_allSteps_{recordings}_{date}_complex'
    pickleFileName0 = groupAnalysisDir + f'/cell_psth_allSteps_{recordings}_{date}_complex'
    pickleFileName1 = groupAnalysisDir + f'/cells_psth_zscore_indecisiveSteps_{recordings}_{date}_complex'
    pickleFileName4 = groupAnalysisDir + f'/cell_psth_indecisiveSteps_{recordings}_{date}_complex'
    pickleFileName2 = groupAnalysisDir + f'/cells_psth_zscore_certainSteps_{recordings}_{date}_complex'
    pickleFileName3 = groupAnalysisDir + f'/cell_psth_certainSteps_{recordings}_{date}_complex'



dfs={}
dfs['indecisive'] = pickle.load(open(pickleFileName4, 'rb'))
dfs['not_indecisive'] =pickle.load(open(pickleFileName3, 'rb'))

dfs['psth_indecisive']=pickle.load(open(pickleFileName1, 'rb'))
dfs['psth_not_indecisive']=pickle.load(open(pickleFileName2, 'rb'))

dfs['allSteps'] =pickle.load(open(pickleFileName0, 'rb'))

dfs['psth_allSteps']=pickle.load(open(pickleFileName, 'rb'))
    #cGV.PSTHGroupFigure_before_after_event_modulation(recordings, df_psth, df,condition,event)
    # cGV.PSTHGroupFigure_before_after_event_early_vs_late(recordings, df_psth, df,condition,event)
    # cGV.psthGroupFigure_early_vs_late_short(recordings, df_psth, df,condition,event)
    # cGV.fractionOfCellsModulated(recordings,df_psth, df,condition)
    # cGV.PSTHGroupFigure_all_event_modulation(df, condition)
cGV.psthGroupFigure_missteps(cellType, dfs, spikeType)
# cGV.psthGroupFigure_cell_based_summary(cellType, dfs)
    # cGV.PSTH_correlation_figure(df, df_psth,condition, recordings)
    # cGV.behavioralParameterDistribution_figure(df)