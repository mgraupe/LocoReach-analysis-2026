# from oauth2client import tools
# tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
# args = tools.argparser.parse_args()
import sys
sys.path.append('.')
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
import sys

groupFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary'
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
cGV = createGroupVisualizations.createGroupVisualizations(groupFigDir)


# collecting analyzed ephys information from each animal
readDataAgain=False
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']

spikeType = 'simple'
date='25Mar20' #'23feb21'

#cellType=recordings[3:]

recordings = ['allMLI','allPC']

PSTHSummaryAllAnimals = {}

for r in recordings:
    PSTHSummaryAllAnimals[r] = {}
    if spikeType =='simple':
        pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s_%s' % (r,date)
    else:
        pickleFileName = groupAnalysisDir + '/PSTHSummaryAllAnimals_%s_complex' % (r,date)

    # date='23feb21'

    if os.path.isfile(pickleFileName) and not readDataAgain:
        #print(pickleFileName)
        PSTHSummaryAllAnimals[r] = pickle.load(open(pickleFileName, 'rb'))
    else:
        for n in range(len(mice)):
            eSD         = extractSaveData.extractSaveData(mice[n],recStruc='simplexEphy')

            # PSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary23Jan25_%s_%s.p' % (recordings, spikeType)
            PSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary%s_%s_%s.p' % (date,r, spikeType)
            PSTHData = pickle.load(open(PSTHAnalysisFile, 'rb'))  # eSD.analysisLocation,
            PSTHSummaryAllAnimals[r][n] = {}
            PSTHSummaryAllAnimals[r][n]['mouse'] = mice[n]
            PSTHSummaryAllAnimals[r][n]['PSTHdata'] = PSTHData
            del eSD
        pickle.dump(PSTHSummaryAllAnimals[r], open(pickleFileName, 'wb'))




cGV.PSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=True,alignment='stance')
cGV.PSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=False,alignment='stance')
cGV.PSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=False,alignment='swing')
#cGV.PSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=False)

cGV.speedPSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=True,alignment='Stance')
cGV.speedPSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=False,alignment='Stance')
cGV.speedPSTHGroupFigure_rescaledTime(date, recordings, spikeType, PSTHSummaryAllAnimals,rescaled=False,alignment='Swing')

cGV.PSTHGroupFigure_missStep(date, recordings, spikeType, PSTHSummaryAllAnimals)
cGV.PSTHGroupFigure_missStep(date, recordings, spikeType, PSTHSummaryAllAnimals,cellType='allPC')
#cGV.SpeedGroupFigure_missStep(date, recordings, spikeType, PSTHSummaryAllAnimals)

cGV.PSTHGroupFigure_swingOnset(date, recordings, spikeType, PSTHSummaryAllAnimals)
cGV.PSTHGroupFigure_swingOnset(date, recordings, spikeType, PSTHSummaryAllAnimals,cellType='allPC')
#cGV.SpeedGroupFigure_swingOnset(date, recordings, spikeType, PSTHSummaryAllAnimals)