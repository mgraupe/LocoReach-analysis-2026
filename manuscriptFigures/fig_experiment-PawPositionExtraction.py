import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createPublicationVisualizations as createPublicationVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

mouseList = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28', '220716_f65', '220716_f67']
#mouseList = ['220211_f38','220716_f67']
readSwingStanceData = False

figVersion = '0.3'
mouse = '220211_f38'
expDateForFig = '220503'
recForFig = 4

expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings = 'all' # 'some', 'all', 'allMLI' or 'allPC'

# read correlation coefficients
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
wheelPawCorrCoeffs = np.load(groupAnalysisDir+'/allWheel-PawCorrelationCoefficients.npy')
print(np.shape(wheelPawCorrCoeffs))

# read data for all animals and extract swing-stance phase durations
swingStanceFile = 'swing-stance-duration.p'
if readSwingStanceData:
    allMiceSwingStanceDur = []
    for m in range(len(mouseList)):
        singleMouse = []
        eSD         = extractSaveData.extractSaveData(mouseList[m],recStruc='simplexEphy')
        (foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings)
        print(foldersRecordings)
        for f in range(len(foldersRecordings)):
            singleRecDicts = {}
            for r in range(len(foldersRecordings[f][2])):
                print('getting swing-stance durations for : ',foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate)
                swingStanceDict = eSD.readSwingStanceData(mouseList[m], foldersRecordings[f][0], foldersRecordings[f][2][r])
                singleRecDicts[r] = dataAnalysis.getSwingStanceDurations(cPawPos,swingStanceDict)
            singleMouse.append([[foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]],singleRecDicts])
        allMiceSwingStanceDur.append([m,mouseList[m],singleMouse])
        del eSD
    pickle.dump(allMiceSwingStanceDur,open(swingStanceFile,'wb'))
else:
    allMiceSwingStanceDur = pickle.load(open(swingStanceFile,'rb'))

#pdb.set_trace()
expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings = 'all' # 'some', 'all', 'allMLI' or 'allPC'
# read data for one animal features in figure
eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
(foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
print('used previously saved DLC instance : ', DLCinstance)
print(foldersRecordings)
#pdb.set_trace()
cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation,mouse)
# loop over all recording folders
ephysData = []
for f in range(len(foldersRecordings)):
    if foldersRecordings[f][1] == expDateForFig:
        for r in range(len(foldersRecordings[f][2])):
            #print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
            if int(foldersRecordings[f][2][r][-3:]) == recForFig:
                print('used for fig : ',foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                (existenceFrames, fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
                (existenceRot, fileHandleRot) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
                (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance)
                #if existenceDAQ:
                #    (daqData, daqTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice', fileHandleDAQ)
                if existenceRot:
                    (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                if existencePawPos:
                    #(frames, softFrameTimes, imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior',fileHandleFrames)
                    frame = np.load('frames_6004.npy')
                    #pdb.set_trace()
                    (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate)
                    swingStanceDict = eSD.readSwingStanceData(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r])
                cPV.experimentWalkingFig(figVersion, sTimes,linearSpeed,cPawPos,angluarSpeed,angleTimes,frame,pawSpeed,swingStanceDict,allMiceSwingStanceDur,wheelPawCorrCoeffs)
                #pdb.set_trace()

