from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recs", help="specify index of the specify recording on that day", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import pickle
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.stats as stats
import datetime
import sys


mouseD = '250121_m01'
# mouseD = '220525_m27' # id of the mouse to analyze
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordingsD='some'     # 'all or 'some'
recStructure = 'simplexNPX'#'simplexEphy' # specify here 'simplexBehavior', None otherwise
recType = 'flat' #'rung' if not on flat surface

if recType == 'rung':
    DLCinstance = 'DLC_resnet50_2025_neuropixelApr24shuffle6_340000'
elif recType == 'flat':
    DLCinstance = 'DLC_resnet50_2025_neuropixelApr24shuffle8_330000'

#file:///media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/210122_f84/210122_f84_2021.03.30_001_locomotionTriggerSIAndMotor60sec_001_raw_behaviorDLC_resnet_50_2021Jun_PawExtraction_m13_m14_f99_f70Jun23shuffle7_200000_meta.pickle

bonsai=False
readDataAgain = True

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

if args.recs == None:
    try:
        recordings = recordingsD
    except :
        recordings = 'all'
else:
    recordings = args.recs


eSD         = extractSaveData.extractSaveData(mouse,recStruc=recStructure)
# (foldersRecordings,dataFolder,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings)  # get recordings for specific mouse and date
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDateD,recordings=recordings)
# (foldersRecordings,listOfRecordings) = eSD.combineDifferentCategorysOnSameDay(foldersRecordings,listOfRecordings)
#(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings)  # get recordings for specific mouse and date
cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

#if expDateD == 'all910' or expDateD == 'all820':
pickleFileName = eSD.analysisLocation + '/allSingStanceDataPerSession_%s.p' % (expDate)
pickleFileName1 = eSD.analysisLocation + '/pawCoordinationDataPerSession_test_%s.p' % (expDate)

# if eSD.analysisConfig['pawTrajectories']['DLCinstance'] is None:
#     DLCinstance = DLCinstanceInput
#     eSD.analysisConfig['pawTrajectories']['DLCinstance'] = DLCinstance
# else:
#     DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
#     print('used previously saved DLC instance : ', DLCinstance)

#########################################################
# Get paws coordinates
if os.path.isfile(pickleFileName) and not readDataAgain:
    recordingsM = pickle.load( open( pickleFileName, 'rb' ) )

else:
    recordingsM = []
    for f in range(len(foldersRecordings)):
        # loop over all recordings in that folder
        wheelMovement = []
        pawTracks = []
        rungMotion = []
        swingPhases = []
        #if foldersRecordings[f][0] == '2021.04.13_000':
        #    startIdx = 1
        #else:
        startIdx = 0
        for r in range(startIdx,len(foldersRecordings[f][2])):
            if foldersRecordings[f][0][:-3] !=foldersRecordings[f][2][r][:-3]:
                (existenceDAQdevice, fileHandleDAQ, config) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice')
                recStartTime = config['.']['startTime']
                # (existenceDAQdevice, fileHandleDAQ, config) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'DaqDevice')
                # recStartTime = config['.']['startTime']
                #(existenceAniFrameTimes, fileHandleAniFrameTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'behaviorVideoFrameTimes')
                # (existenceAniFrameTimes, fileHandleAniFrameTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r],'behaviorVideoFrameTimes')
                (existenceAniFrames, fileHandleAniFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'GigEAnimalBonsai', startTime=recStartTime)
                # read rotary encoder data for wheel speed
                (existenceRotaryEnc, rotFileHandle,_) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
                if existenceRotaryEnc:
                    print('  extracting walking_activity')
                    (angluarSpeed, linearSpeed, sTimes, timeStamp, monitor, angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
                    wheelMovement.append([angluarSpeed, linearSpeed, sTimes, timeStamp, monitor, angleTimes, foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r]])

                videoId=f'{mouseD}/{foldersRecordings[f][0]}/{foldersRecordings[f][2][r]}'

                # check whether paw data exists
                recStartTime = config['.']['startTime']
                # pdb.set_trace()
                # (existencePawPos,PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance, videoPrefix='raw_video')
                (existencePawPos,PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance,videoPrefix='processed-animal-video')
                # read paw data

                #recEndTime = (config['.']['startTime']  + config['.']['protocol']['conf']['duration'])
                create_date = datetime.datetime.fromtimestamp(recStartTime)
                (existenceAniBonsaiFrames,fileHandleAniBonsaiFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'GigEAnimalBonsai',startTime=recStartTime)
                (existenceWhisBonsaiFrames, fileHandleWhisBonsaiFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'Cham3WhiskerBonsai',startTime=recStartTime)
                (existenceAniACQ4Frames, fileHandleAniACQ4Frames, _) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')

                vidRecDict = eSD.videoRecordingImplementation(existenceAniBonsaiFrames,existenceWhisBonsaiFrames,existenceAniACQ4Frames)
                print(vidRecDict)

                # read paw traces and positions of rungs
                if vidRecDict['animalVideo'] and existencePawPos and existenceRotaryEnc:
                    #(rawPawPositionsFromDLCBot, pawTrackingOutliersBot, jointNamesFramesInfoBot, pawSpeedBot, startEndExposureTimeAniBot, rawPawSpeedBot, pawPosBot, croppingParametersBot,jointNamesBot) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, obstacle='bot', obstacleVideo=False)
                    #(rawPawPositionsFromDLCAll, pawTrackingOutliersAll, jointNamesFramesInfoAll, pawSpeedAll, startEndExposureTimeAniAll, rawPawSpeedAll, pawPosAll, croppingParametersAll,jointNamesAll) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, obstacle='all', obstacleVideo=False)
                    #(_, _, _, _, _, _, _, _, jointNames) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, DLCinstance, obstacle=None, obstacleVideo=False, returnData='oldList')
                    #pawIndicesSwingStance = eSD.findIdxForBottomPaws(vidRecDict, jointNames) # determine which paw trajectories to use for swing-stance extraction
                    (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, pawPos, croppingParameter,jointNames)= eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, DLCinstance, obstacle=None, obstacleVideo=False,returnData='oldList') #pawIndicesSwingStance
                    pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameter,jointNames])
                    if recType == 'rung':
                        rungPositions = eSD.readRungMotionData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
                        rungMotion.append([mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],rungPositions])
                #print(jointNames)
                #pdb.set_trace()
                # subdivide trajectories in swing and stance phases
                if existenceRotaryEnc and vidRecDict['animalVideo']:
                    if recType == 'rung':
                        (swingP,forFit) = dataAnalysis.findStancePhases(wheelMovement[-1],pawTracks[-1],rungMotion[-1],eSD.analysisConfig,pawTracks[-1],showFigFit=False,showFigPaw=True,verbose=False,redefineStanceDistances=True)
                    elif recType == 'flat':
                        (swingP,forFit) = dataAnalysis.findSwingPhasesFlat(wheelMovement[-1],pawTracks[-1],eSD.analysisConfig,pawTracks[-1],showFigFit=False,showFigPaw=False,verbose=False,jointID=jointNames)
                    eSD.saveSwingStanceData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],swingP,forFit, pawPos)

                    #print(len(swingP[0][1]),len(swingP[1][1]),len(swingP[2][1]),len(swingP[3][1]))
                    swingPhases.append([mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],swingP,forFit, pawPos])

        recordingsM.append([foldersRecordings[f][0],wheelMovement,pawTracks,rungMotion,swingPhases])

    pickle.dump(recordingsM, open(pickleFileName, 'wb'))

sys.exit(0)

#############################################################

#pdb.set_trace()
First_Analyze_of_Hildebrand_Plot=True
refPaw = 0
if recStructure=='simplexBehavior' or recStructure=='simplexNPX':
    listOfRecordings=foldersRecordings
if os.path.isfile(pickleFileName1) and not First_Analyze_of_Hildebrand_Plot:
    pawCoordData = pickle.load( open( pickleFileName1, 'rb' ) )
else:
    (countPawSeq_mouse, pawSeqProb_mouse,iqr_mouse,median_mouse,swingOn_mouse,swingOff_mouse)= dataAnalysis.analyzeHildebrandPlot(recordingsM, refPaw,listOfRecordings)
    pawCoordData=[countPawSeq_mouse, pawSeqProb_mouse,iqr_mouse,median_mouse,swingOn_mouse,swingOff_mouse]
    pickle.dump(pawCoordData, open(pickleFileName1, 'wb'))

    # pdb.set_trace()
    eSD.writeConfigFile()
    cV.createSwingStanceFigure(recordingsM,expDate)
    # cV.createSwingTraceFigure(recordingsM,linear=False)
    #cV.createSwingTraceFigure(recordingsM,linear=True)
    cV.createSwingSpeedProfileFigure(recordingsM,expDate,linear=False)
    # cV.createSwingTrajectoryProfileFigure(recordingsM, expDate, linear=True)
    if recType == 'rung':
        cV.createRungCrossingFigure(recordingsM,expDate)
    cV.createHildebrandPlotFigure(pawCoordData,refPaw,expDate)


