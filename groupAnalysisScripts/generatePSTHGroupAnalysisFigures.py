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
readDataAgain=True
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
recordings = 'allMLI'

versionDate='23Jan25'
pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s_simple' % recordings

PSTHSummaryAllAnimals = []
if os.path.isfile(pickleFileName) and not readDataAgain:
     PSTHSummaryAllAnimals = pickle.load(open(pickleFileName, 'rb'))
else:
     for n in range(len(mice)):
         eSD         = extractSaveData.extractSaveData(mice[n],recStruc='simplexEphy')
         PSTHAnalysisFile = eSD.analysisLocation + f'/ephysPSTHSummary{versionDate}_{recordings}_simple.p'
         PSTHData = pickle.load(open(PSTHAnalysisFile, 'rb'))  # eSD.analysisLocation,
         PSTHSummaryAllAnimals.append([n,mice[n],PSTHData])
         del eSD
     pickle.dump(PSTHSummaryAllAnimals, open(pickleFileName, 'wb'))

analyzeAgain=True
possibleConditions = ['allSteps']#,'swingDurationLastRec20-80','swingLengthLastRec20-80']#,'stanceDurationLastRec20-80', 'indecisiveSteps', 'not_indecisiveSteps']
# possibleConditions=[ 'swingLengthLastRec0-20',
#         'swingLengthLastRec20-40',
#         'swingLengthLastRec40-60',
#          'swingLengthLastRec60-80',
#         'swingLengthLastRec80-100'
# ]
for l, condition in enumerate(possibleConditions):
    pickleFileName1 = groupAnalysisDir + '/cells_psth_zscore_%s_%s' % (condition, recordings)
    pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s' % (condition, recordings)
    if analyzeAgain:
        df,df_psth =groupAnalysis_psth.collectModulatedCells(mice, PSTHSummaryAllAnimals, condition, recordings)
        pickle.dump(df_psth, open(pickleFileName1, 'wb'))
        pickle.dump(df, open(pickleFileName2, 'wb'))
    else:
        df_psth = pickle.load(open(pickleFileName1, 'rb'))
        df = pickle.load(open(pickleFileName2, 'rb'))
    for event in ['swingOnset','stanceOnset']:
        #cGV.PSTHGroupFigure_before_after_event_modulation(recordings, df_psth, df,condition,event)
        # cGV.PSTHGroupFigure_before_after_event_early_vs_late(recordings, df_psth, df,condition,event)
        # cGV.fractionOfCellsModulated(recordings,df_psth, df,condition)
        # cGV.PSTHGroupFigure_all_event_modulation(df, condition)

        cGV.PSTH_correlation_figure(df, df_psth,condition, recordings)
    # cGV.behavioralParameterDistribution_figure(df)