import sys
sys.path.append('./')
import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createPublicationVisualizations as createPublicationVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

figVersion = '0.3'
mouse = '220525_m27'
expDateForFig = '220729'
recsForFig = [0,1,2,3,4]
exampleTrace = 4 # 3!

expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings= 'all' # 'some', 'all', 'allMLI' or 'allPC'

eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
(foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
print('used previously saved DLC instance : ', DLCinstance)
print(foldersRecordings)
#pdb.set_trace()
cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation,mouse)
# loop over all recording folders
ephysWalkingData = []
for f in range(len(foldersRecordings)):
    if foldersRecordings[f][1] == expDateForFig:
        for r in range(len(foldersRecordings[f][2])):
            #print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
            if int(foldersRecordings[f][2][r][-3:]) in recsForFig:
                print('used for fig : ',foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                #(existenceFrames, fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
                (existenceRot, fileHandleRot) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
                (existenceEphys1, fileHandleEphys1) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1')
                (existenceEphys2, fileHandleEphys2) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2')
                #(existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance)
                #if existenceDAQ:
                #    (daqData, daqTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice', fileHandleDAQ)
                if existenceEphys1:
                    ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                elif existenceEphys2:
                    ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                if existenceRot:
                    (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                if existenceEphys1 or existenceEphys2:
                    ephysDict = eSD.readEphysData([foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'])
                ephysWalkingData.append([[foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]],[sTimes,linearSpeed],ephysDict,ephys])


#############################
# collecting analyzed ephys information from each animal
readDataAgain=True
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
recordings = 'allMLI'


groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'

#wheelPawCorrCoeffs = np.load(groupAnalysisDir+'/allWheel-PawCorrelationCoefficients.npy')
pickleFileName = groupAnalysisDir + '/ephysSummaryAllAnimals_%s'%recordings
ephysSummaryAllAnimals = []
if os.path.isfile(pickleFileName) and not readDataAgain:
    ephysSummaryAllAnimals = pickle.load(open(pickleFileName, 'rb'))
else:
    for n in range(len(mice)):
        eSD         = extractSaveData.extractSaveData(mice[n],recStruc='simplexEphy')
        ephysAnalysisFile = eSD.analysisLocation + '/ephysSummary_%s.p' % recordings
        ephysData = pickle.load(open(ephysAnalysisFile, 'rb'))  # eSD.analysisLocation,
        ephysSummaryAllAnimals.append([n,mice[n],ephysData])
        del eSD
    pickle.dump(ephysSummaryAllAnimals, open(pickleFileName, 'wb'))


cPV.ephysWalkingFig(figVersion, ephysWalkingData, exampleTrace,ephysSummaryAllAnimals)

