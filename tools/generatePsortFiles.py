import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle


mouse = '221213_f4'
expDate = 'some'
recordings= 'some'

eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
(foldersRecordings,dataFolders) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)
# loop over all recording folders
ephysData = []
for f in range(len(foldersRecordings)):

    for r in range(len(foldersRecordings[f][2])):

        print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
        (existenceRot, fileHandleRot,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
        # eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r], 'RotaryEncoder')
        (existenceEphys, fileHandleEphys,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'AxoPatch200_2')
        if existenceEphys==False:
            (existenceEphys, fileHandleEphys,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'AxoPatch200_1')
        (existenceDAQ, fileHandleDAQ,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'DaqDeviceEphys')
        #print existenceRot, existenceEphys
        #tracks = []
        if existenceEphys:
            #(angles, aTimes,timeStamp,monitor) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1][r],'RotaryEncoder',fileHandleRot)
            #(angularSpeed, linearSpeed, sTimes)  = dataAnalysis.getSpeed(angles,aTimes,wheelCircumsphere)
            try:
                (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys)

            except:
                (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys)
            dt = np.mean(ephysTimes[1:] - ephysTimes[:-1])
            rate = 1. / dt
            print('rate is ', rate)
            fileName='/EphysDataPerSession_%s_%s' % (foldersRecordings[f][0], foldersRecordings[f][2][r])
            eSD.createPsorth5(current, ephysTimes, rate,eSD.analysisLocation, fileName)
