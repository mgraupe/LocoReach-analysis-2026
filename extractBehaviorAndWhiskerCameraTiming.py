from oauth2client import tools
from prompt_toolkit.key_binding.bindings.named_commands import end_of_line

tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify date of recording to analyze", required=False)
tools.argparser.add_argument("-s","--start", help="specify the start recording to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_videoTiming as dataAnalysisVT
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import numpy as np
import sys
import datetime


mouseD = '250121_m01' #'250121_m01' #'230219_m23' #'240129_m97'

expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'

recStructure ='simplexNPX' #'shank3Behavior' # #None# 'simplexBehavior' # specify here 'simplexBehavior', None otherwise

routine = True #precise if you run the script on routine or if you want to specify parameters

# further analaysis parameter
if routine == False: #Adjust the parameters as you wish
    assumePerfectRecording = True  # that means a recording without any flash-back - or double frames : can only be set to True on the new machines
    startRecordingD = None #None #None #3 #None # each session/day per animal is composed of 5 recordings, this index allows chose with which recording to start, default is 0
    endRecording = 1 # in case only specify recording will be analyzed, otherwise set to None
    DetermineAgainLEDcoordinates = False # whether or not to determine LED coordinates even though they exist already for current or previous recording
    DetermineAgainErronousFrames = False # False#True#False# False# True#alse# True#False# True# True# True #False#True # whether or not to determine errnonous frames even though the are already exist for current recording
    recordingWithTail = False #True# True
    whiskerVideoTime = True
    animalVideoTime = False
    manThreshold = False # allows to set the thresholds manually---
    optimize=True
    AutoDetermineLEDcoordinnates=False
else: # parameters that never changes - to use the script on animals with no problem
    assumePerfectRecording = True
    startRecordingD = None
    endRecording = None
    DetermineAgainLEDcoordinates = False
    DetermineAgainErronousFrames = False
    recordingWithTail = False
    whiskerVideoTime = True
    animalVideoTime = True
    manThreshold = False
    optimize=True
    AutoDetermineLEDcoordinnates=False

# in case mouse, and date were specified as input arguments
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

if args.start == None:
    startRecording = startRecordingD
else:
    startRecording = int(args.start)

#print mouse, expDate
#sys.exit(0) #pdb.set_trace()
eSD         = extractSaveData.extractSaveData(mouse,recStruc=recStructure)  # find data folder of specific mouse, create data folder, and hdf5 handle

(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDateD,recordings=recordings) # get recordings for specific mouse and date
# (foldersRecordings, dataFolders,recInputIdx) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
openCVtools  = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
print('folderRecordings: ',foldersRecordings)

# pdb.set_trace()
failedAnimal=[]
successAnimal=[]
failedWhisker=[]
successWhisker=[]
# pdb.set_trace()
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range((0 if startRecording is None else startRecording),(len(foldersRecordings[f][2]) if endRecording is None else endRecording)):
        videoId=f'{mouseD}/{foldersRecordings[f][0]}/{foldersRecordings[f][2][r]}'
        print(); print('Checking recording : ', foldersRecordings[f][0], foldersRecordings[f][2][r])
        # pdb.set_trace()
        (existenceDAQdevice, fileHandleDAQ, config) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice')
        (existenceFTimes, fileHandleFTimes, _) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'frameTimes')
        #print(config['.'])
        #pdb.set_trace()

        recStartTime = [config['.']['startTime'], config['DaqDevice.ma']['__timestamp__'], config['RotaryEncoder.ma']['__timestamp__']] # config['.']['startTime']


        #recEndTime = (config['.']['startTime']  + config['.']['protocol']['conf']['duration'])
        create_date = datetime.datetime.fromtimestamp(recStartTime[0])
        #print('Created on:', create_date)
        (existenceAniBonsaiFrames,fileHandleAniBonsaiFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'GigEAnimalBonsai',startTime=recStartTime[0])
        (existenceWhisBonsaiFrames, fileHandleWhisBonsaiFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'Cham3WhiskerBonsai',startTime=recStartTime[0])
        (existenceAniACQ4Frames, fileHandleAniACQ4Frames, _) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')

        vidRecDict = eSD.videoRecordingImplementation(existenceAniBonsaiFrames,existenceWhisBonsaiFrames,existenceAniACQ4Frames)
        print(vidRecDict)
        #(existenceAniFTimes,fileHandleAniFTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'GigEframeTimes')
        #(existenceWhisFTimes, fileHandleWhisFTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'Cham3WhsikerframeTimes')
        print('animal video : ', end='')
        (currentAniCoodinatesExist, SavedAniLEDcoordinates) = eSD.checkForLEDPositionCoordinates(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r, key=('LEDinAniVideo' if vidRecDict['vidRecSoftware']=='Bonsai' else 'LEDinVideo'))
        if (vidRecDict['vidRecSoftware'] == 'ACQ4') and not currentAniCoodinatesExist:
            (currentAniCoodinatesExist, SavedAniLEDcoordinates) = eSD.checkForLEDPositionCoordinates(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r,key='LEDinAniVideo')
        print('whisker video : ',end='')
        (currentWhisCoodinatesExist, SavedWhisLEDcoordinates) = eSD.checkForLEDPositionCoordinates(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r, key='LEDinWhisVideo')
        (erroneousFramesExist,idxToExcludeAni,canBeUsed) = eSD.checkForErroneousFramesIdx(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r,determineAgain=DetermineAgainErronousFrames)
        #pdb.set_trace()
        # 1. get video stream for Bonsai recording or the frames as np.array for ACQ4 recording #######
        if vidRecDict['animalVideo']:
            if vidRecDict['vidRecSoftware'] == 'Bonsai':
                (aniVideo,softAniFrameIDs,softAniFrameTimes) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'GigEAnimalBonsai',fileHandleAniBonsaiFrames)
            elif vidRecDict['vidRecSoftware'] == 'ACQ4': # here the return argument aniVideo was called `frames` before
                (aniVideo, softAniFrameTimes, imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior',fileHandleAniACQ4Frames)
        if vidRecDict['whiskerVideo']:
            (whisVideo, softWhisFrameIDs, softWhisFrameTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'Cham3WhiskerBonsai',fileHandleWhisBonsaiFrames)

        # 2. determine the LED coordinates in existing video streams ######################################
        # first in animal camera
        if vidRecDict['animalVideo']:
            if (SavedAniLEDcoordinates is None) or DetermineAgainLEDcoordinates :
                aniLEDcoordinates = openCVtools.findLEDNumberArea(aniVideo,'LEDinAniVideo', coordinates=SavedAniLEDcoordinates,currentCoordExist=currentAniCoodinatesExist,determineAgain=True,videoRec=vidRecDict,auto=AutoDetermineLEDcoordinnates)
                eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinAniVideo'],aniLEDcoordinates)
            else:
                aniLEDcoordinates = SavedAniLEDcoordinates
                eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinAniVideo'], aniLEDcoordinates)
        # then in whisker camera
        if vidRecDict['whiskerVideo']:
            if (SavedWhisLEDcoordinates is None) or DetermineAgainLEDcoordinates :
                whisLEDcoordinates = openCVtools.findLEDNumberArea(whisVideo,'LEDinWhisVideo',coordinates=SavedWhisLEDcoordinates,currentCoordExist=currentWhisCoodinatesExist,determineAgain=True,videoRec=vidRecDict,auto=AutoDetermineLEDcoordinnates) #videoType='Cham3WhiskerBonsai'
                eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinWhisVideo'],whisLEDcoordinates)
            else:
                whisLEDcoordinates = SavedWhisLEDcoordinates
                eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinWhisVideo'], whisLEDcoordinates)

        # 3. extract LED traces, or read from file is traces exist already #################################
        try:
            if vidRecDict['animalVideo']:  LEDAnitraces = eSD.readgGenericData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDtracesInAnimalVideo'], 'LEDtraces')
            if vidRecDict['whiskerVideo']:  LEDWhistraces = eSD.readgGenericData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDtracesInWhiskerVideo'], 'LEDtraces')
            if DetermineAgainLEDcoordinates:
                test = makeFailHere # makeFailHere does not exist : will make sure that LED traces are extracted again with new LED coordinates
        except:
            print('Extracting intensity traces of LEDs from videos (takes a while) ... ')
            if vidRecDict['animalVideo']:
                LEDAnitraces = openCVtools.extractLEDtraces(aniVideo,aniLEDcoordinates)#,verbose=False)
                eSD.saveGenericData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDtracesInAnimalVideo'], LEDAnitraces, 'LEDtraces')
            if vidRecDict['whiskerVideo']:
                LEDWhistraces = openCVtools.extractLEDtraces(whisVideo, whisLEDcoordinates)#, verbose=False)
                eSD.saveGenericData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDtracesInWhiskerVideo'],LEDWhistraces, 'LEDtraces')

        # read binary traces from DAQ recording #############################################################
        if existenceDAQdevice:
            (daqValues, daqValuesTimes, fileHandleDAQ) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice', fileHandleDAQ)
            (ledDAQControlArray, exposureWhisDAQArray, exposureAniDAQArray) = eSD.attributeDAQTraces(daqValues, foldersRecordings[f][1],vidRecDict)
            if vidRecDict['vidRecSoftware'] == 'ACQ4': # a camera had it's own DAQ device to record exposure
                (exposureAniDAQArray,exposureAniDAQArrayTimes,startTime) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes',fileHandleFTimes)
                exposureAniDAQArray = exposureAniDAQArray[0]
        #pdb.set_trace()
        # in case of dropped and/or flash-back frames ##########################################################
        if assumePerfectRecording:
            idxToExcludeAni = np.array([], dtype=np.int64)
            idxToExcludeWhis = np.array([], dtype=np.int64)
            canBeUsed = True
        else:
            idxToExcludeWhis = np.array([], dtype=np.int64)
            if (not erroneousFramesExist):# and canBeUsed:
                frames = openCVtools.extractFramesFromVideoFile(aniVideo)
                (idxToExcludeAni,canBeUsed) = dataAnalysis.determineErroneousFrames(frames)
                eSD.saveErroneousFramesIdx([foldersRecordings[f][0], foldersRecordings[f][2][r], 'erroneousAnimalVideoFrames'],idxToExcludeAni,canBeUsed=canBeUsed)
            else:
                print('already determined indicies to exclude : ', idxToExcludeAni)
                print('can be used : ', canBeUsed)

        # finally : determine time of each frame ###############################################################
        # first for animal video
        if animalVideoTime:
            if vidRecDict['animalVideo'] and existenceDAQdevice and canBeUsed:
                # (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, frameDuration) = dataAnalysisVT.determineFrameTimesBasedOnLED(
                #         [LEDAnitraces, aniLEDcoordinates, aniVideo, softAniFrameTimes, recStartTime, idxToExcludeAni], [[exposureAniDAQArray, ], daqValuesTimes],
                #         [[ledDAQControlArray, ], daqValuesTimes], eSD.recordingMachine, verbose=True, tail=recordingWithTail, manualThreshold=manThreshold)
                # print('after determine frame time')
                # pdb.set_trace()
                try :
                    (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, frameDuration) = dataAnalysisVT.determineFrameTimesBasedOnLED(
                        [LEDAnitraces, aniLEDcoordinates, aniVideo, softAniFrameTimes, recStartTime[0], idxToExcludeAni], [[exposureAniDAQArray, ], daqValuesTimes],
                        [[ledDAQControlArray, ], daqValuesTimes], eSD.recordingMachine, verbose=True, tail=recordingWithTail, manualThreshold=manThreshold)
                    #(idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary) = dataAnalysis.determineFrameTimesBasedOnLED(
                    #    [LEDtraces, LEDcoordinates, frames, softFrameTimes, imageMetaInfo, idxToExclude], [exposureDAQArray, exposureDAQArrayTimes],
                    #    [ledDAQControlArray, ledDAQControlArrayTimes], eSD.recordingMachine, verbose=True, tail=recordingWithTail, manualThreshold=manThreshold)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
                    print('FAIL (automatic thresholds for animal)')
                    print('Automatic threshold determination for animal video did NOT work. Try looping thresholds')
                    try:
                        (idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary) = dataAnalysisVT.determineFrameTimesBasedOnLEDLoop([LEDAnitraces,aniLEDcoordinates,aniVideo,softAniFrameTimes,recStartTime[0],idxToExcludeAni],[[exposureAniDAQArray,],daqValuesTimes],[[ledDAQControlArray,], daqValuesTimes],eSD.recordingMachine,videoId,verbose=True,tail=recordingWithTail,manualThreshold=manThreshold)
                    except:
                        animalWorked = False
                        print('FAIL (loop)')
                        failedAnimal.append(videoId)
                    else:
                        animalWorked = True
                        print('SUCCESS with loop')
                        successAnimal.append(videoId)
                else:
                    animalWorked = True
                    print('SUCCESS with automatic thresholds')
                    successAnimal.append(videoId)
                if animalWorked :
                    eSD.saveBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'animalVideo'],idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary, recStartTime[0])

        ######################################################################
        # then for whisker video
        if whiskerVideoTime:
            if vidRecDict['whiskerVideo'] and existenceDAQdevice and canBeUsed:
                try:
                    (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary) = dataAnalysisVT.determineFrameTimesBasedOnLED(
                        [LEDWhistraces, whisLEDcoordinates, whisVideo, softWhisFrameTimes, recStartTime[0], idxToExcludeWhis], [[exposureWhisDAQArray, ], daqValuesTimes],
                        [[ledDAQControlArray, ], daqValuesTimes], eSD.recordingMachine, verbose=True, tail=recordingWithTail, manualThreshold=manThreshold, whisker=True)
                except:
                    print('FAIL (automatic thresholds for whisker)')
                    print('Automatic threshold determination for whisker video did NOT work. Try looping thresholds')
                    try:
                        (idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary) = dataAnalysisVT.determineFrameTimesBasedOnLEDLoop([LEDWhistraces,whisLEDcoordinates,whisVideo,softWhisFrameTimes,recStartTime[0],idxToExcludeWhis],[[exposureWhisDAQArray,],daqValuesTimes],[[ledDAQControlArray,], daqValuesTimes],eSD.recordingMachine,videoId,verbose=True,tail=recordingWithTail,manualThreshold=manThreshold,whisker=True)
                        #framesDuringRecording = frames[recordedFramesIdx]
                        #eSD.saveBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'whiskerVideo'],idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary,recStartTime)
                    except:
                        print('FAIL (loop)')
                        whiskerWorked = False
                        failedWhisker.append(videoId)
                    else:
                        print('SUCCESS with loop')
                        whiskerWorked = True
                        successWhisker.append(videoId)
                else:
                    print('SUCCESS with automatic thresholds')
                    whiskerWorked = True
                    successWhisker.append(videoId)
                if whiskerWorked:
                    eSD.saveBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'whiskerVideo'],idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary,recStartTime[0])
                #eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
print(); print()
if animalVideoTime:
    print('failed list animal',failedAnimal)
    print('success list animal',successAnimal)
if whiskerVideoTime:
    print('failed list whisker', failedWhisker)
    print('success list whisker', successWhisker)

