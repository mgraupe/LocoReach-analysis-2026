from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-s","--start", help="specify the start recording to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import numpy as np

mouseD = '230219_f17'


expDateD = 'some' # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some' # 'all or 'some'
recStructure = None #'simplexNew' #'simplexNPX' #'simplexBehavior'#None# specify here 'simplexBehavior', None otherwise
recBatch = None
reRun= True
obstacleVideo=[None] #['animal'], ['animalObstacle'], ['normal'] (for regular videos without obstacle) or [None] (for old videos ending with _raw_behavior)

startRecordingD = None #None #3 #None # each session/day per animal is composed of 5 recordings, this index allows chose with which recording to start, default is 0
endRecording = None # in case only specify recording will be analyzed, otherwise set to None

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

eSD         = extractSaveData.extractSaveData(mouse,recStruc=recStructure)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=False)
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)) :
    # loop over all recordings in that folder
    for r in range((0 if startRecording is None else startRecording),(len(foldersRecordings[f][2]) if endRecording is None else endRecording)):
        #print foldersRecordings[f][2][r]
        if obstacleVideo[0] is None:

            (existence,fileHandle, _) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
            if existence:
                (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo) = eSD.readBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo'])
                (rungPositions,diffs,alignResults) = cv2Tools.trackRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],videoIdx,startEndExposureTime,obstacleVideo,defineROI=True,recStruc=recStructure,recBatch=recBatch)
                #pdb.set_trace()
                eSD.saveRungMotionData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],rungPositions)
                #cv2Tools.trackPawsAndRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
            #pdb.set_trace()
        else :
            (existenceDAQdevice, fileHandleDAQ, config) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice')
            recStartTime = config['.']['startTime']
            ### animal video
            (existenceAniFrames, fileHandleAniFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'GigEAnimalBonsai', startTime=recStartTime)
            (existenceAniFrameTimes, fileHandleAniFrameTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'animalVideoFrameTimes')
            if existenceAniFrames and existenceAniFrameTimes:

                (idxTimePointsAni, startEndExposureTimeAni, startEndExposurepIdxAni, videoIdxAni, frameSummaryAni, recStartTimeAni) =  eSD.readBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'animalVideo'])
                # if obstacleVideo[0] !='normal':
                if obstacleVideo[0] =='animal' or obstacleVideo[0] =='animalObstacle':
                    (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes, absoluteSignal, obstacle1, obstacle2) = eSD.getObstacleWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
                    (obsVideoTimesArr, obsVideoIdxArr, obsNumber, obsID, obstacleUPdic, obsAngleArr)= dataAnalysis.determineObstacleFrames(startEndExposureTimeAni, videoIdxAni,  angleTimes, absoluteSignal, obstacle1,obstacle2,angleRange = [-8, 55])
                if obstacleVideo[0]=='animal':
                    videoIdx=videoIdxAni
                    timeArray= np.average(startEndExposureTimeAni, axis=1)
                    obstacleVideo=[obstacleVideo[0],obstacleUPdic]
                elif obstacleVideo[0]=='animalObstacle':
                    videoIdx = obsVideoIdxArr
                    timeArray=obsVideoTimesArr
                    obstacleVideo = [obstacleVideo[0], obstacleUPdic]
                else :
                    videoIdx = videoIdxAni
                    timeArray = np.average(startEndExposureTimeAni, axis=1)

                (existenceRungs) = eSD.checkRungMotionData(mouse,foldersRecordings[f][0],  foldersRecordings[f][2][r])
                if existenceRungs and not reRun:
                    print('rung location already extracted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',existenceRungs )
                else:
                    (rungPositions,diffs,alignResults) = cv2Tools.trackRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],videoIdx,timeArray,obstacleVideo,defineROI=False,recStruc=recStructure,recBatch=recBatch)
                    #pdb.set_trace()
                    eSD.saveRungMotionData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],rungPositions)
                #cv2Tools.trackPawsAndRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
            else:
                pass