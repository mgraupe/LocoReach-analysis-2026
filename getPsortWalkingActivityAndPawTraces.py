from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_psth as dataAnalysis_psth
import tools.createVisualizations as createVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os

#mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
#for m in range(len(mice)):

mouseD = '221213_f4'
expDateD = 'allMLI'  #  'some', 'all', 'allMLI' or 'allPC'
recordingsD= 'allMLI' # 'some', 'all', 'allMLI' or 'allPC'
analyzeDataAgain = True
spikeType = 'simple'
flatSurface=True
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
# (foldersRecordings,listOfRecordings) = eSD.combineDifferentCategorysOnSameDay(foldersRecordings,listOfRecordings)

DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
print('used previously saved DLC instance : ', DLCinstance)
#print(foldersRecordings)
#print(listOfRecordings)
#pdb.set_trace()
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

if spikeType == 'complex':
    spikeIdx = 1
    ephysPSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary_%s_%s.p' % (recordings,spikeType)
else:
    spikeIdx = 0
    ephysPSTHAnalysisFile  = eSD.analysisLocation + '/ephysPSTHSummary_%s.p' % recordings

if os.path.isfile(ephysPSTHAnalysisFile) and not analyzeDataAgain:
    print('file %s exists already ' % ephysPSTHAnalysisFile)
    ephysData = pickle.load(open(ephysPSTHAnalysisFile, 'rb'))
else:
    print('analysis will be performed ...')
    ephysData = []
    # loop over all recording folders
    for f in range(len(foldersRecordings)):
        singleCellList = []
        for r in range(len(foldersRecordings[f][2])):
            print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
            (existenceRot, fileHandleRot,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
            # eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r], 'RotaryEncoder')
            (existenceEphys1, fileHandleEphys1,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1')
            (existenceEphys2, fileHandleEphys2,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'AxoPatch200_2')
            #(existenceDAQ, fileHandleDAQ) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'DaqDevice')
            (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance)
            if existenceEphys1:
                (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
            elif existenceEphys2:
                (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
            #pdb.set_trace()
            #if existenceDAQ:
            #    (daqData, daqTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice', fileHandleDAQ)
            if existenceRot:
                (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
            if existencePawPos:
                (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters,jointNames) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate)
                swingStanceDict = eSD.readSwingStanceData(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r])

        #cV.createEphysFigureForPresentation(foldersRecordings[f][0],foldersRecordings[f][2][r], cPawPos,ephys,swingStanceDict,sTimes,linearSpeed)
        #pdb.set_trace()
        if existencePawPos and (existenceEphys1 or existenceEphys2):
            # ephysPSTHDict = dataAnalysis.calculateStridebasedPSTH(cPawPos,ephys[spikeIdx],swingStanceDict)
            print('calculate PSTH ... ',end='')
            if not flatSurface:
                strideProps = dataAnalysis_psth.calculateStrideProperties(swingStanceDict, cPawPos, pawSpeed)
            else:
                strideProps = dataAnalysis_psth.calculateStridePropertiesFlatSurface(swingStanceDict, cPawPos, pawSpeed)
            #ephysPSTHDict = dataAnalysis.calculateStridebasedPSTH_condition(cPawPos,ephys[spikeIdx],swingStanceDict,pawSpeed)
            ephysPSTHDict = dataAnalysis_psth.calculatePSTH(cPawPos, pawSpeed, ephys[spikeIdx], swingStanceDict, strideProps,cond='allSteps')

            print('done')
            #print('calcualte shuffles ... ',end='')
            #dataAnalysis.calculateShuffledStridebasedPSTH_condition(cPawPos, ephys[spikeIdx], swingStanceDict,pawSpeed,ephysPSTHDict,30) # normally 300 shuffles
            #print('done')

        singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephysPSTHDict])
        #cV.createEphysPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],cPawPos,ephys,swingStanceDict,ephysPSTHDict,recordings[3:],simpleSorComplexS=spikeType)
        cV.createEphysPSTHPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],cPawPos,ephys,swingStanceDict,ephysPSTHDict['allSteps'],recordings[3:],simpleSorComplexS=spikeType)

    dataAnalysis.calculatePSTH_change(singleCellList)
    ephysData.append([foldersRecordings[f][0],foldersRecordings[f][1],singleCellList])

pickle.dump(ephysData, open(ephysPSTHAnalysisFile, 'wb'))  # eSD.analysisLocation,

# dataAnalysis.PSTHcorrelation(ephysData)
# pickle.dump(ephysData, open(ephysPSTHAnalysisFile, 'wb'))