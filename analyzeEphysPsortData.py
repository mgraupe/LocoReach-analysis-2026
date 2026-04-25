from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os

mouseD = '220205_f61' #'220211_f38' #220206_m16' #220525_m28'# '220525_m19' #'220205_f61' #220507_m90' # 220507_m81  # 220525_m28  220211_f38
expDateD = 'allPC'  #  'some', 'all', 'allMLI' or 'allPC'
recordingsD= 'allPC' # 'some', 'all', 'allMLI' or 'allPC'
analyzeDataAgain = True

if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    try:
        expDate = expDateD
    except :
        expDate = 'all'
else:
    expDate = args.date

if args.recordings == None:
    try:
        recordings = recordingsD
    except:
        recordings = 'all'
else:
    recordings = args.recordings

eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
(foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
print('used previously saved DLC instance : ', DLCinstance)
#print(foldersRecordings)
#print(listOfRecordings)
#pdb.set_trace()
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)
# loop over all recording folders

ephysAnalysisFile  = eSD.analysisLocation + '/ephysSummary_%s.p' % recordings

if os.path.isfile(ephysAnalysisFile) and not analyzeDataAgain:
    print('file %s exists already ' % ephysAnalysisFile)
    ephysData = pickle.load(open(ephysAnalysisFile, 'rb'))
else:
    print('analysis will be performed ...')
    ephysData = []
    for f in range(len(foldersRecordings)):
        if not (listOfRecordings[f][4][0] == 'Beh'):  # only include MLIs and PCs in list
            singleCellList = []
            for r in range(len(foldersRecordings[f][2])):
                print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                (existenceRot, fileHandleRot) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
                # eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r], 'RotaryEncoder')
                (existenceEphys1, fileHandleEphys1) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1')
                (existenceEphys2, fileHandleEphys2) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'AxoPatch200_2')
                #(existenceDAQ, fileHandleDAQ) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'DaqDevice')
                (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance)
                if existenceEphys1:
                    (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                    ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                elif existenceEphys2:
                    (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                    ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                if existenceRot:
                    (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                if existenceEphys1 or existenceEphys2:
                    ephysDict = dataAnalysis.analyzeSpikingActivity(ephys,sTimes,linearSpeed,[mouse,foldersRecordings[f][0],foldersRecordings[f][2][r]])
                    eSD.saveEphysData([foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephysDict)
                    #dd = eSD.readEphysData([foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'])
                singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephysDict])
            ephysData.append([foldersRecordings[f][0],foldersRecordings[f][1],singleCellList,listOfRecordings[f][2],listOfRecordings[f][3],listOfRecordings[f][4]])
    pickle.dump(ephysData, open(ephysAnalysisFile, 'wb'))  # eSD.analysisLocation,

cV.showSpikeTimesWaveformsOfOneAnimal(ephysData,listOfRecordings,recordings)
cV.showEphysSummaryOfOneAnimal(ephysData,listOfRecordings,recordings)
