from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_cellClustering as cluster
import tools.createVisualizations as createVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os

mouseList = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28', '220716_f65', '220716_f67']
#mouseList = ['220525_m19','220525_m27']
expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings= 'all' # 'some', 'all', 'allMLI' or 'allPC'

ephysFile = 'ephysDataCollected.p'
generateCollectedDataFile = False

#print(foldersRecordings)
#print(listOfRecordings)
#pdb.set_trace()
#cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

if generateCollectedDataFile:
    allEphysData = {}
    nCell = 0
    for n in range(len(mouseList)):
        mouse = mouseList[n]
        print(mouse)
        eSD         = extractSaveData.extractSaveData(mouse,recStruc='simplexEphy')
        head, tail = os.path.split(eSD.analysisLocation[:-1]) # get path without the last level of the mouse specific folder
        #print(head, tail, eSD.analysisLocation)
        folder = head + '/simplexSummary/'
        #print(folder)
        (foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
        #(foldersRecordings,listOfRecordings) = eSD.combineDifferentCategorysOnSameDay(foldersRecordings,listOfRecordings)

        #print(listOfRecordings)
        #pdb.set_trace()
        # loop over all recording folders - equivalent to recorded cells
        print('number of cells: ', len(foldersRecordings))
        for f in range(len(foldersRecordings)):
            #singleCellList = []
            if  not (listOfRecordings[f][4][0] == 'Beh'): # only include MLIs and PCs in list
                allEphysData[nCell] = {}
                allEphysData[nCell]['cellType'] = listOfRecordings[f][4][0]
                allEphysData[nCell]['visuallyGuided'] = listOfRecordings[f][4][1]
                allEphysData[nCell]['mouse-cell-recs'] = [mouse,foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2]]
                # loop over all recordings of a given cell
                for r in range(len(foldersRecordings[f][2])):
                    print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
                    (existenceEphys1, fileHandleEphys1) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1')
                    (existenceEphys2, fileHandleEphys2) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'AxoPatch200_2')
                    if existenceEphys1:
                        (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                        ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_1', fileHandleEphys1)
                    elif existenceEphys2:
                        (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                        ephys = eSD.readPSortAnalyzedData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'AxoPatch200_2', fileHandleEphys2)
                    #print(ephys[2].keys())
                    #pdb.set_trace()
                    ephysAnalyzed = eSD.readEphysData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'ephysDataAnalyzed'])

                    allEphysData[nCell][r] = {}
                    allEphysData[nCell][r]['recording'] = foldersRecordings[f][2][r]
                    print('# of SSs: ', len(ephys[2]['ss_wave']))
                    allEphysData[nCell][r]['ss_number'] = len(ephys[2]['ss_wave'])
                    allEphysData[nCell][r]['ss_fr'] = ephysAnalyzed['ss_firingRate']
                    allEphysData[nCell][r]['ss_wave'] = np.mean(ephys[2]['ss_wave'],axis=0)
                    allEphysData[nCell][r]['ss_wave_span'] = ephys[2]['ss_wave_span'][0]
                    allEphysData[nCell][r]['ss_xprob'] = ephys[2]['ss_xprob']
                    allEphysData[nCell][r]['ss_xprob_span'] = ephys[2]['ss_xprob_span']
                    allEphysData[nCell][r]['ss_avgSpikeParams'] = ephysAnalyzed['ss_avgSpikeParams']
                    # [0.05,0.5,1.,5.]
                    allEphysData[nCell][r]['ss_spike-count_CVs'] = [ephysAnalyzed['ss_spike-count_cv_0.05'],ephysAnalyzed['ss_spike-count_cv_0.5'],ephysAnalyzed['ss_spike-count_cv_1.0'],ephysAnalyzed['ss_spike-count_cv_5.0']]
                    #
                    allEphysData[nCell][r]['cs_number'] = len(ephys[2]['cs_wave'])
                    if len(ephys[2]['cs_wave']) > 0:
                        print('# of CSs: ', len(ephys[2]['cs_wave']))
                        allEphysData[nCell][r]['cs_fr'] = 1. / (np.mean(np.diff(ephys[1])))
                        allEphysData[nCell][r]['cs_wave'] = np.mean(ephys[2]['cs_wave'], axis=0)
                        allEphysData[nCell][r]['cs_wave_span'] = ephys[2]['cs_wave_span'][0]
                        allEphysData[nCell][r]['cs_xprob'] = ephys[2]['cs_xprob']
                        allEphysData[nCell][r]['cs_xprob_span'] = ephys[2]['cs_xprob_span']
                        allEphysData[nCell][r]['cs_time_to_prev_ss'] = ephys[2]['cs_time_to_prev_ss']
                        allEphysData[nCell][r]['cs_time_to_next_ss'] = ephys[2]['cs_time_to_next_ss']
                nCell+=1

                #singleCellList.append([r,[foldersRecordings[f][0],foldersRecordings[f][2][r],'ephysDataAnalyzed'],ephys])
                #cV.createEphysPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],cPawPos,ephys,swingStanceDict,ephysPSTHDict,recordings[3:],simpleSorComplexS=spikeType)
            #ephysData.append([foldersRecordings[f][0],foldersRecordings[f][1],singleCellList])
        #pickle.dump(ephysData, open(ephysPSTHAnalysisFile, 'wb'))  # eSD.analysisLocation,
    pickle.dump(allEphysData, open(folder+ephysFile, 'wb'))  # eSD.analysisLocation,
else:
    eSD = extractSaveData.extractSaveData(mouseList[0], recStruc='simplexEphy')
    head, tail = os.path.split(eSD.analysisLocation[:-1])  # get path without the last level of the mouse specific folder
    print(head, tail, eSD.analysisLocation)
    folder = head + '/simplexSummary/'
    allEphysData = pickle.load(open(folder + ephysFile, 'rb'))  # eSD.analysisLocation,

#pdb.set_trace()
cluster.clusterCellTypes(allEphysData,folder)