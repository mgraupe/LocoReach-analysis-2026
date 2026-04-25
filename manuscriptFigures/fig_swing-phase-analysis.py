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

figVersion = '0.2'
mouse = '220211_f38'
expDateForFig = '220503'
recForFig = 4

expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings = 'all' # 'some', 'all', 'allMLI' or 'allPC'

# read correlation coefficients
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
#wheelPawCorrCoeffs = np.load(groupAnalysisDir+'/allWheel-PawCorrelationCoefficients.npy')
#print(np.shape(wheelPawCorrCoeffs))

# read data for all animals
allSwingStanceFile = groupAnalysisDir + '/all-swing-stance-data.p'
allSwingStanceDict = {}
if readSwingStanceData:
    for m in range(len(mouseList)):
        mouse = mouseList[m]
        allSwingStanceDict[mouse] = {}
        eSD         = extractSaveData.extractSaveData(mouseList[m],recStruc='simplexEphy')
        (foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings)
        try:
            cPV
        except :  # create only for first mouse
            cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation,mouseList[m])
        DLCinstance = eSD.analysisConfig['pawTrajectories']['DLCinstance']
        print(foldersRecordings)
        for f in range(len(foldersRecordings)):
            allSwingStanceDict[mouse][f] = {} # singleRecDicts = {}
            for r in range(len(foldersRecordings[f][2])):
                allSwingStanceDict[mouse][f][r] = {}
                print('getting swing-stance durations for : ',foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters, jointNames) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], expDate, DLCinstance)
                allSwingStanceDict[mouse][f][r]['cPawPos'] = cPawPos
                allSwingStanceDict[mouse][f][r]['pawSpeed'] = pawSpeed
                (angluarSpeed, linearSpeed, sTimes, timeStamp, monitor, angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
                allSwingStanceDict[mouse][f][r]['sTimes'] = sTimes
                allSwingStanceDict[mouse][f][r]['linearSpeed'] = linearSpeed
                allSwingStanceDict[mouse][f][r]['swingStanceDict'] = eSD.readSwingStanceData(mouseList[m], foldersRecordings[f][0], foldersRecordings[f][2][r])
                allSwingStanceDict[mouse][f][r]['rungMotion'] = eSD.readRungMotionData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
                #rungMotion.append([mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],rungPositions])
                #,linearSpeed,cPawPos,angluarSpeed,angleTimes,frame,pawSpeed)
                #allSwingStanceDict[m][f]singleRecDicts[r] = dataAnalysis.getSwingStanceDurations(cPawPos,swingStanceDict)
            #singleMouse.append([[foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]],singleRecDicts])
        #allMiceSwingStanceDur.append([m,mouseList[m],singleMouse])
        del eSD
    pickle.dump(allSwingStanceDict,open(allSwingStanceFile,'wb'))
else:
    allSwingStanceDict = pickle.load(open(allSwingStanceFile,'rb'))
    eSD   = extractSaveData.extractSaveData(mouseList[0],recStruc='simplexEphy')
    cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation, mouseList[0])

#pdb.set_trace()

cPV.swingAnalysisFig(figVersion,mouseList, allSwingStanceDict)


