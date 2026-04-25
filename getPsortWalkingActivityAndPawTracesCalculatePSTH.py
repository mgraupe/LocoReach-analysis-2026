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
#mice = ['220211_f38']
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
#mice = ['220525_m28','220716_f65','220716_f67']
# mice=['220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
for m in range(len(mice)):
    mouseD = mice[m]
    expDateD = 'allPC'  #  'some', 'all', 'allMLI' or 'allPC'
    recordingsD= 'allPC' # 'some', 'all', 'allMLI' or 'allPC'
    analyzeDataAgain = True
    spikeType = 'simple'
    #control=True
    flatSurface=False
    #control_var = 'swingSpeed' # 'swingDuration', 'swingLength'

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
    print('doing analysis for mouse ',mouseD)
    print('used previously saved DLC instance : ', DLCinstance)
    #print(foldersRecordings)
    #print(listOfRecordings)
    #pdb.set_trace()
    cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

    if spikeType == 'complex':
        spikeIdx = 1
        ephysPSTHAnalysisFile = eSD.analysisLocation + '/ephysPSTHSummary25Mar20_%s_%s.p' % (recordings,spikeType)
    else:
        spikeIdx = 0
        #if control:
        ephysPSTHAnalysisFile  = eSD.analysisLocation + '/ephysPSTHSummary25Mar20_%s_%s.p' % (recordings, spikeType)

    if os.path.isfile(ephysPSTHAnalysisFile) and not analyzeDataAgain:
        print('file %s exists already ' % ephysPSTHAnalysisFile)
        ephysDataPerCell = pickle.load(open(ephysPSTHAnalysisFile, 'rb'))
    else:
        print('analysis will be performed ...')
        ephysDataPerCell = []
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
                    (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters,joints) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, DLCinstance)
                    swingStanceDict = eSD.readSwingStanceData(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r])

                #cV.createEphysFigureForPresentation(foldersRecordings[f][0],foldersRecordings[f][2][r], cPawPos,ephys,swingStanceDict,sTimes,linearSpeed)
                #pdb.set_trace()
                if existencePawPos and (existenceEphys1 or existenceEphys2):
                    # ephysPSTHDict = dataAnalysis.calculateStridebasedPSTH(cPawPos,ephys[spikeIdx],swingStanceDict)
                    #print('calculate PSTH ... ',end='')
                    #ephysPSTHDict = dataAnalysis.calculateStridebasedPSTH_condition(cPawPos,ephys[spikeIdx],swingStanceDict,pawSpeed)
                    if not flatSurface:
                        strideProps = dataAnalysis_psth.calculateStrideProperties(swingStanceDict, cPawPos, pawSpeed)
                    else:
                        strideProps = dataAnalysis_psth.calculateStridePropertiesFlatSurface(swingStanceDict, cPawPos, pawSpeed)
                    #strideProperties = dataAnalysis_psth.calculatePSTH(cPawPos,ephys[spikeIdx],swingStanceDict,pawSpeed)
                    #print('done')
                    #print('calcualte shuffles ... ',end='')
                    #dataAnalysis.calculateShuffledStridebasedPSTH_condition(cPawPos, ephys[spikeIdx], swingStanceDict,pawSpeed,ephysPSTHDict,30) # normally 300 shuffles
                    #print('done')
                singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],cPawPos, ephys[spikeIdx], swingStanceDict, pawSpeed,strideProps])
                # cV.createEphysPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],cPawPos,ephys,swingStanceDict,ephysPSTHDict,recordings[3:],simpleSorComplexS=spikeType)
            if not (len(foldersRecordings[f][2])==len(singleCellList)): raise Exception('Not the same length: recording folders and singleCellList')

            conditions = [
                ['allSteps'],
                # ['indecisiveSteps'],
                # ['certainSteps'],
                # ['swingDuration', 'lastRec', 20, 80],
                # ['swingLength', 'lastRec', 20, 80],
                # # ['stepLength', 'lastRec', 20, 80],
                # ['swingLengthLinear', 'lastRec', 20, 80],
                # ['swingSpeed', 'lastRec', 20, 80],
                # ['rungCrossed'],
                #              ['swingDuration','allRecs','percentiles'],
                #              ['swingLength','allRecs','percentiles'],
                #              # ['stepLength','allRecs','percentiles'],
                #
                #              # ['stepDuration','allRecs','percentiles'],
                #              # ['stepMeanSpeed','allRecs','percentiles'],
                #             ['swingLengthLinear', 'allRecs', 'percentiles'],
                #              ['swingSpeed', 'allRecs', 'percentiles'],



                             ]

            condList = []
            for m in range(len(conditions)):
                cL = dataAnalysis_psth.defineCondition(conditions[m],singleCellList)


                # pdb.set_trace()

                if cL[0]=='allSteps' or cL[0]=='indecisiveSteps' or cL[0]=='certainSteps' or cL[0][-5:]=='20_80' :
                    condList.append(cL)
                else:
                    for k in range(len(cL)):
                        condList.append(cL[k])

            # print(condList)
            # print(len(condList))
            # pdb.set_trace()


            # for m in range(len(condList)):
            #     print(condList[m])
            #pdb.set_trace()
            #condition1 = ['allSteps',[None,None]]
            #condition2 = ['swingDuration']

            # decide for appropriate condition
            # if control and control_var!='rungCrossed':
            #
            #     condition1 = ['allSteps',[None,None]]
            #     condition2 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs0-20',singleCellList,control_var)
            #     condition3 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs20-40', singleCellList,control_var)
            #     condition4= dataAnalysis_psth.defineCondition(f'{control_var}AllRecs40-60', singleCellList,control_var)
            #     condition5 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs60-80', singleCellList,control_var)
            #     condition6 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs80-100', singleCellList,control_var)
            #     condition7= dataAnalysis_psth.defineCondition(f'{control_var}LastRec20-80', singleCellList,control_var)
            #
            #     condition8 = dataAnalysis_psth.defineCondition('indecisiveSteps', singleCellList, control_var)
            #     condition9 = dataAnalysis_psth.defineCondition('not_indecisiveSteps', singleCellList, control_var)
            #     # condition6 = dataAnalysis_psth.defineCondition('not_indecisiveSteps', singleCellList)
            #     condList = [condition1,condition2,condition3,condition4, condition5,condition6,condition7]
            #     # pdb.set_trace()
            # elif control and  control_var=='rungCrossed':
            #     condition1 = ['allSteps',[None,None]]
            #     condition2 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs0',singleCellList,control_var)
            #     condition3 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs1', singleCellList,control_var)
            #     condition4= dataAnalysis_psth.defineCondition(f'{control_var}AllRecs2', singleCellList,control_var)
            #     condition5 = dataAnalysis_psth.defineCondition(f'{control_var}AllRecs3', singleCellList,control_var)
            #
            #
            #     condition6= dataAnalysis_psth.defineCondition('indecisiveSteps', singleCellList, control_var)
            #     condition7 = dataAnalysis_psth.defineCondition('not_indecisiveSteps', singleCellList, control_var)
            #     # condition6 = dataAnalysis_psth.defineCondition('not_indecisiveSteps', singleCellList)
            #     condList = [condition1,condition2,condition3,condition4, condition5,condition6,condition7]
            # else:
            #     control_var=None
            #     condition1 = ['allSteps', [None, None]]
            #     condition2 = dataAnalysis_psth.defineCondition('swingDurationLastRec20-80', singleCellList,control_var)
            #     condition3 = dataAnalysis_psth.defineCondition('swingLengthLastRec20-80', singleCellList,control_var)
            #     condition4 = dataAnalysis_psth.defineCondition('indecisiveSteps', singleCellList,control_var)
            #     condition5 = dataAnalysis_psth.defineCondition('not_indecisiveSteps', singleCellList,control_var)
            #     condList = [condition1,condition2,condition3,condition4, condition5]
            # pdb.set_trace()
            # calculate PSTH with the appropriate conditional parameters
            ephysPSTHDict = {}
            for r in range(len(foldersRecordings[f][2])):
                print('rec :',r)
                cPawPos = singleCellList[r][2]
                spikes = singleCellList[r][3]
                swingStanceDict = singleCellList[r][4]
                pawSpeed = singleCellList[r][5]
                strideProps = singleCellList[r][6]
                ephysPSTHDict[r] = {}

                for condition in condList:
                    print('calculate PSTH for %s... ' % condition[0] ,end='')
                    #pdb.set_trace()
                    ephysPSTHDict[r][condition[0]] = dataAnalysis_psth.calculatePSTH(cPawPos, pawSpeed, spikes, swingStanceDict, strideProps, condition)#L, control_var)
                    print('done')
                    cV.createEphysPSTHPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],cPawPos,ephys,swingStanceDict,ephysPSTHDict[r][condition[0]],recordings[3:],condition,simpleSorComplexS=spikeType)
            ephysDataPerCell.append([foldersRecordings[f][0],foldersRecordings[f][1],singleCellList,ephysPSTHDict])
        pickle.dump(ephysDataPerCell, open(ephysPSTHAnalysisFile, 'wb'))

    condListKeys = ['swingLength-lastRec-20-80', 'swingLength-allRecs-percentiles-0-20',
                    'swingLength-allRecs-percentiles-20-40',
                    'swingLength-allRecs-percentiles-40-60',
                    'swingLength-allRecs-percentiles-60-80',
                    'swingLength-allRecs-percentiles-80-100']
    # pdb.set_trace()  # eSD.analysisLocation,
    # cV.createEphysPSTHAcrossTrial(ephysDataPerCell, recordings[3:], condListKeys, simpleSorComplexS=spikeType)
    # dataAnalysis.PSTHcorrelation(ephysDataPerCell)
    # pickle.dump(ephysDataPerCell, open(ephysPSTHAnalysisFile, 'wb'))