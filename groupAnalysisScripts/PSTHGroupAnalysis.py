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
recordings = 'allMLI'
spikeType = 'simple'
cellType=recordings[3:]

if spikeType =='simple':
    pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s' % recordings
else:
    pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s_complex' % recordings

PSTHSummaryAllAnimals = []
date='23feb21'
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

analyzeAgain=True

#['swingLengthLinear_allRecs_percentiles_60_80']# ['indecisiveSteps','certainSteps']#,'swingDurationLastRec20-80','swingLengthLastRec20-80']#,'stanceDurationLastRec20-80', 'indecisiveSteps', 'not_indecisiveSteps']
# possibleConditions=[ 'allSteps','swingLengthLinear_allRecs_percentiles_0_20',
#         'swingLengthLinear_allRecs_percentiles_20_40',
#         'swingLengthLinear_allRecs_percentiles_40_60',
#          'swingLengthLinear_allRecs_percentiles_60_80',
#         'swingLengthLinear_allRecs_percentiles_80_100'
# ]
possibleConditions=['swingLength_allRecs_percentiles_0_20',
        'swingLength_allRecs_percentiles_20_40',
        'swingLength_allRecs_percentiles_40_60',
         'swingLength_allRecs_percentiles_60_80',
        'swingLength_allRecs_percentiles_80_100'
]
# possibleConditions=['swingDuration_allRecs_percentiles_0_20',
#         'swingDuration_allRecs_percentiles_20_40',
#         'swingDuration_allRecs_percentiles_40_60',
#          'swingDuration_allRecs_percentiles_60_80',
#         'swingDuration_allRecs_percentiles_80_100'
# ]
# possibleConditions=['swingSpeed_allRecs_percentiles_0_20',
#         'swingSpeed_allRecs_percentiles_20_40',
#         'swingSpeed_allRecs_percentiles_40_60',
#          'swingSpeed_allRecs_percentiles_60_80',
#         'swingSpeed_allRecs_percentiles_80_100'
# ]
# possibleConditions = ['indecisiveSteps','certainSteps']
pawNb=1
for l, condition in enumerate(possibleConditions):
    if spikeType=='simple':
        pickleFileName1 = groupAnalysisDir + '/cells_psth_zscore_%s_%s_%s' % (condition, recordings,date)
        pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s_%s' % (condition, recordings, date)
        pickleFileNameMLI1 = groupAnalysisDir + '/cells_psth_zscore_%s_allMLI_%s' % (condition,date)
        pickleFileNameMLI2 = groupAnalysisDir + '/cell_psth_%s_allMLI_%s' % (condition, date)
        pickleFileNamePC1 = groupAnalysisDir + '/cells_psth_zscore_%s_allPC_%s' % (condition,date)
        pickleFileNamePC2 = groupAnalysisDir + '/cell_psth_%s_allPC_%s' % (condition, date)
    else:
        pickleFileName1 = groupAnalysisDir + '/cells_psth_zscore_%s_%s_%s_complex' % (condition, recordings,date)
        pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s_%s_complex' % (condition, recordings, date)
    # pickleFileName2 = groupAnalysisDir + '/cell_psth_%s_%s' % (condition, recordings)
    if analyzeAgain:
        df,df_psth =groupAnalysis_psth.collectModulatedCells(mice, PSTHSummaryAllAnimals, condition, recordings)

        pickle.dump(df_psth, open(pickleFileName1, 'wb'))
        pickle.dump(df, open(pickleFileName2, 'wb'))
    else:
    # df_psth = pd.read_csv(groupAnalysisDir + '/cells_psth_zscore_%s_%s.csv' % (condition, recordings))
    #     df = pd.read_csv(groupAnalysisDir + '/psth_multi_%s_%s.csv' % (condition, recordings))
        df_psth = pickle.load(open(pickleFileName1, 'rb'))
        df = pickle.load(open(pickleFileName2, 'rb'))
    groupAnalysis_psth.psthGroupGenerateClusterDic(cellType, df_psth, df, condition, pawNb)
    #cGV.PSTHGroupFigure_before_after_event_modulation(recordings, df_psth, df,condition,event)
    # cGV.PSTHGroupFigure_before_after_event_early_vs_late(recordings, df_psth, df,condition,event)
    # cGV.psthGroupFigure_early_vs_late_short(recordings, df_psth, df,condition,event)
    # cGV.fractionOfCellsModulated(recordings,df_psth, df,condition)
    # cGV.PSTHGroupFigure_all_event_modulation(df, condition)
    #cGV.psthGroupFigure_cell_based_single_paw(cellType, df_psth, df, condition,  pawNb)
    # cGV.psthGroupFigure_cell_based_summary(cellType, df, condition,  pawNb, spikeType)
    # cGV.PSTH_correlation_figure(df, df_psth,condition, recordings)
    # cGV.behavioralParameterDistribution_figure(df)
    # cGV.psthGroupFigure_cell_based_single_paw__all_modulated_zscore(cellType, df_psth, df, condition, pawNb)

    # MLI_PC=True
    # #
    # if MLI_PC:
    #     df_psth_MLI = pickle.load(open(pickleFileNameMLI1, 'rb'))
    #     df_MLI = pickle.load(open(pickleFileNameMLI2, 'rb'))
    #     df_psth_PC = pickle.load(open(pickleFileNamePC1, 'rb'))
    #     df_PC = pickle.load(open(pickleFileNamePC2, 'rb'))
    # #
    # #     # cGV.averageZscoreMLI_PC(df_psth_MLI, df_MLI, df_psth_PC, df_PC)
    #     cGV.classifyAverageZscoreMLI_PC(df_psth_MLI, df_MLI, df_psth_PC, df_PC)
        # cGV.compareMLI_PC(df_psth_MLI,df_MLI,df_psth_PC,df_PC)