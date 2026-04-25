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

if spikeType =='simple':
    pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s' % recordings
else:
    pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s_complex' % recordings

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
     print('dumped !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

possibleConditions=[ 'allSteps']
#['swingLengthLinear_allRecs_percentiles_60_80']# ['indecisiveSteps','certainSteps']#,'swingDurationLastRec20-80','swingLengthLastRec20-80']#,'stanceDurationLastRec20-80', 'indecisiveSteps', 'not_indecisiveSteps']
# possibleConditions=[ 'allSteps','swingLengthLinear_allRecs_percentiles_0_20',
#         'swingLengthLinear_allRecs_percentiles_20_40',
#         'swingLengthLinear_allRecs_percentiles_40_60',
#          'swingLengthLinear_allRecs_percentiles_60_80',
#         'swingLengthLinear_allRecs_percentiles_80_100'
# ]
# possibleConditions=['stepLength_allRecs_percentiles_0_20',
#         'stepLength_allRecs_percentiles_20_40',
#         'stepLength_allRecs_percentiles_40_60',
#          'stepLength_allRecs_percentiles_60_80',
#         'stepLength_allRecs_percentiles_80_100'
# ]

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
analyzeAgain=False
pawNb=0
for l, condition in enumerate(possibleConditions):
    pickleFileName1 = groupAnalysisDir + '/cells_CS_%s_%s' % (condition, recordings)
    pickleFileName2 = groupAnalysisDir + '/cells_CS_Loco_%s_%s' % (condition, recordings)
    if analyzeAgain:
        df_CS, compCS_df =groupAnalysis_psth.collectComplexSpikes(mice, PSTHSummaryAllAnimals, condition, recordings)

        pickle.dump(df_CS, open(pickleFileName1, 'wb'))
        pickle.dump(compCS_df, open(pickleFileName2, 'wb'))
    else:
        df_CS = pickle.load(open(pickleFileName1, 'rb'))
        compCS_df = pickle.load(open(pickleFileName2, 'rb'))
cGV.createComplexSpikeAnalysisFigure(df_CS,compCS_df)