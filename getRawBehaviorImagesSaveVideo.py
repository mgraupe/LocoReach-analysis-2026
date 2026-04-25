import numpy as np
from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-s","--start", help="specify the start recording to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import sys




mouseD = '230219_m23' # id of the mouse to analyze

#mouseD = '190108_m24'
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'
recStructure = None #'simplexNPX'#'simplexBehavior'# 'simplexBehavior' # specify here 'simplexBehavior', None otherwise
obstacle = False

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

#print mouse, expDate
#sys.exit(0) #pdb.set_trace()
eSD         = extractSaveData.extractSaveData(mouse,recStruc=recStructure)  # find data folder of specific mouse, create data folder, and hdf5 handle
#pdb.set_trace()
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
# (foldersRecordings,dataFolder, recInputIdx) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
#print(len(foldersRecordings))

#pdb.set_trace()
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range((0 if startRecording is None else startRecording),(len(foldersRecordings[f][2]) if endRecording is None else endRecording)):
        #pdb.set_trace()
        (existenceFrames,fileHandleFrames,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (existenceFTimes,fileHandleFTimes,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes')
        (existenceLEDControl, fileHandleLED,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput')
        (existenceRotaryEnc, fileHandleRotEnc,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'RotaryEncoder')
        # if camera was recorded
        if existenceFrames:
            #print('exists',foldersRecordings[f][0],foldersRecordings[f][2][r])
            (frames,softFrameTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior',fileHandleFrames)
        # use frame drop/miss and timing information to save video
        if existenceFrames and existenceFTimes and existenceLEDControl:
            if (eSD.recStructure == 'simplex') or (eSD.recStructure == 'simplexBehavior'):
                (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfoCopy) = eSD.readBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo'])
            elif eSD.recStructure == 'vermis':
                startEndExposureTime = np.column_stack((softFrameTimes,softFrameTimes))
                videoIdx = np.arange(len(frames))
            eSD.saveBehaviorVideoFrames([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideoFirstLastFrames'], frames, videoIdx)
            eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], frames, startEndExposureTime, videoIdx, exportIndFrames=False)
        if obstacle and existenceRotaryEnc:
            (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes, absoluteSignal, obstacle1, obstacle2) = eSD.getObstacleWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
            (startEndExposureTimeObstacle, videoIdxObstacle) = dataAnalysis.determineObstacleFrames(startEndExposurepIdx, videoIdx, wTimes, angleTimes,absoluteSignal, obstacle1, obstacle2)
            # eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], frames, startEndExposureTime, videoIdx, exportIndFrames=False, obstacle=obstacle)