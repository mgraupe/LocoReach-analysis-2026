import sys
sys.path.append('./')
import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_psth as dApsth
import tools.createPublicationVisualizations as createPublicationVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import pickle

def computeSuccessDictPerSession(m,session):
    successRateSession = {}  # np.zeros(4)
    successRateSession['normStepCycleBins'] = np.zeros(201)
    successRateSession['normTimeStepCycle'] = np.linspace(0, 2, 201)
    successRateSession['totalNumberOfSteps'] = 0  # len(idxSwings)
    successRateSession['swingWithLaserActivation'] = 0
    successRateSession['swingNoLaserActivation'] = 0
    successRateSession['stanceWithLaserActivation'] = 0
    successRateSession['swingWithLaserIxd'] = []
    for rec in range(5):
        print('recording : ', m, session, rec)
        pickleFileName = analysisDir + m + f'/{m}_{session}_locomotionEMGAndMotor60sec_Bonsai_000-00{rec}_swingStancePhases.p'
        # pickleFileNameTomato=analysisDir+miceForFig[1]+f'/{miceForFig[1]}_{datesForFig[1]}_locomotionEMGAndMotor60sec_Bonsai_000-00{rec}_swingStancePhases.p'

        print('loading pickle files...')
        try:
            swingStanceD = pickle.load(open(pickleFileName, 'rb'))
        except:
            print(pickleFileName, 'has not been recorded')
            recorded = False
        else:
            # triggerDict[rec] = {}
            # triggerDict[rec]['swingStance'] = swingStanceDic
            recorded = True
        ###########
        EMG_dir = '/media/invivodata2/altair_data/dataMichael/' + session + f'/locomotionEMGAndMotor60sec_Bonsai_000/00{rec}/EMGsignals.ma'
        EMG = dataAnalysis.readEMG(EMG_dir)
        # triggerDict[rec]['Trigger']=EMG['current_chan2']
        # triggerDict[rec]['time']=EMG['time']
        if recorded:
            t_emg = EMG['time']
            pawPos = swingStanceD['pawPos']
            laserDict = dApsth.getLaserActivationStartAndEnd(t_emg, EMG['current_chan2'])
            pawID = 0

            xSpeed = swingStanceD['forFit'][pawID][1]
            wSpeed = swingStanceD['forFit'][pawID][i][0]

            linearPawPos = swingStanceD['forFit'][pawID][5]
            rungNumbers = swingStanceD['swingP'][pawID][2]
            stepCharacter = recs[n][4][j][3][i][3]

            rungCrossed = np.diff(rungNumbers)
            idxSwings = swingStanceD['swingP'][pawID][1]
            indecisiveSteps = swingStanceD['swingP'][i][3]
            recTimes = swingStanceD['forFit'][pawID][2]

            successRateSession['totalNumberOfSteps'] += len(idxSwings)
            dApsth.determineLaserActivationDistribution(successRateSession, idxSwings, recTimes, pawPos, pawID, laserDict)

    return successRateSession

figVersion = '4.2'
mouseOpsin ='220214_f43'#,'220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28']
expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings = 'all' # 'some', 'all', 'allMLI' or 'allPC'
startRecording = 4 #None #3 #None # each session/day per animal is composed of 5 recordings, this index allows chose with which recording to start, default is 0
endRecording = None # in case only specify recording will be analyzed, otherwise set to None
experiment='opto'
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
analysisDir= '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'



pickleFileNameEphy = analysisDir + f'210318_f82/allEphysDataPerSession_2021.06.11_002_E2_VC_LEDtrigger_train_002.p'
pickleFileName2 = groupAnalysisDir + f'/behavior_data_{experiment}'
pickleFileName1 = groupAnalysisDir + f'/behavior_data_trajectories_{experiment}'

pickleFileOpsinSummary = groupAnalysisDir + f'/opsinMiceSummary_{experiment}.p'
pickleFileTomSummary = groupAnalysisDir + f'/tomMiceSummary_{experiment}.p'

ephyData=pickle.load( open( pickleFileNameEphy, 'rb' ) )
stridePar=pickle.load( open( pickleFileName2, 'rb' ) )
strideTraj=pickle.load( open( pickleFileName1, 'rb' ) )
miceForFig=['230226_m10','230402_m5']
datesForFig=['2023.07.28_000','2023.07.28_004']
#rec='4'
computeLaserSuccess = False
eSD1 = extractSaveData.extractSaveData(mouseOpsin, recStruc='simplexEphy')
#(foldersRecordings, dataFolders, listOfRecordings) = eSD1.getEphysRecordingsList(expDate=expDate, recordings=recordings)  # get recordings for specific mouse and date
cPV = createPublicationVisualizations.createVisualizations(eSD1.publicationFigLocation,mouseOpsin)

pawID = 0

if computeLaserSuccess:
    opsinMice = {'230226_m10': {
        'recordings': ['2023.07.24_000', '2023.07.25_000', '2023.07.25_008', '2023.07.26_000', '2023.07.26_008', '2023.07.27_000', '2023.07.27_008', '2023.07.28_000', '2023.07.28_008']},
                 '230402_m3': {'recordings': ['2023.07.24_005', '2023.07.25_005', '2023.07.25_013', '2023.07.26_002', '2023.07.26_010', '2023.07.27_002', '2023.07.27_010', '2023.07.28_002',
                                              '2023.07.28_010']}, '230402_m4': {
            'recordings': ['2023.07.24_006', '2023.07.25_006', '2023.07.25_014', '2023.07.26_003', '2023.07.26_011', '2023.07.27_003', '2023.07.27_011', '2023.07.28_003', '2023.07.28_011']},
                 '230403_f3': {'recordings': ['2023.07.24_002', '2023.07.25_002', '2023.07.25_010', '2023.07.26_006', '2023.07.26_014', '2023.07.27_006', '2023.07.27_014', '2023.07.28_006',
                                              '2023.07.28_014']}, '230226_f88': {
            'recordings': ['2023.07.24_003', '2023.07.25_003', '2023.07.25_011', '2023.07.26_007', '2023.07.26_015', '2023.07.27_007', '2023.07.27_015', '2023.07.28_007', '2023.07.28_015']}}
    tomMice = {'230405_m1': {
        'recordings': ['2023.07.24_004', '2023.07.25_004', '2023.07.25_012', '2023.07.26_001', '2023.07.26_009', '2023.07.27_001', '2023.07.27_009', '2023.07.28_001', '2023.07.28_009']},
               '230402_m5': {'recordings': ['2023.07.24_007','2023.07.25_007','2023.07.25_015','2023.07.26_004','2023.07.26_012','2023.07.27_004','2023.07.27_012','2023.07.28_004','2023.07.28_012']}, '230403_f2': {
            'recordings': ['2023.07.24_001', '2023.07.25_001', '2023.07.25_009', '2023.07.26_005', '2023.07.26_013', '2023.07.27_005', '2023.07.27_013', '2023.07.28_005', '2023.07.28_013']}}

    for m in opsinMice:
        for session in opsinMice[m]['recordings']:
            opsinMice[m][session] = {}
            #for key,values in cases.items():
            successRateSession = computeSuccessDictPerSession(m,session)
            opsinMice[m][session]['scucessRateSession'] = successRateSession

    for m in tomMice:
        for session in tomMice[m]['recordings']:
            tomMice[m][session] = {}
            #for key,values in cases.items():
            successRateSession = computeSuccessDictPerSession(m,session)
            tomMice[m][session]['scucessRateSession'] = successRateSession
    pickle.dump(opsinMice, open(pickleFileOpsinSummary, 'wb')) # pickle.dump(df_psth, open(pickleFileName1, 'wb'))
    pickle.dump(tomMice,open( pickleFileTomSummary, 'wb' ))

else:
    pass
    opsinMice = pickle.load(open(pickleFileOpsinSummary, 'rb'))
    tomMice=pickle.load( open( pickleFileTomSummary, 'rb' ))

(opsValues, tomValues) = cPV.fig_real_time_experimentOverview(figVersion,opsinMice,tomMice)

#exit(0)
#pdb.set_trace()
##########################################################
# now collect the data for the figure
triggerDict = {}
for rec in range(5):
    print('recording : ', rec)
    triggerDict[rec] = {}
    triggerDict[rec]['ops'] = {}
    triggerDict[rec]['tom'] = {}
    pickleFileNameOpsin=analysisDir+miceForFig[0]+f'/{miceForFig[0]}_{datesForFig[0]}_locomotionEMGAndMotor60sec_Bonsai_000-00{rec}_swingStancePhases.p'
    pickleFileNameTomato=analysisDir+miceForFig[1]+f'/{miceForFig[1]}_{datesForFig[1]}_locomotionEMGAndMotor60sec_Bonsai_000-00{rec}_swingStancePhases.p'

    print('loading pickle files...')
    try:
        opsSwingStanceDic=pickle.load( open( pickleFileNameOpsin, 'rb' ) )
    except:
        print(pickleFileNameOpsin, 'has not been recorded')
    else:
        triggerDict[rec]['ops']['swingStance'] = opsSwingStanceDic
    ###########
    try:
        tomSwingStanceDic=pickle.load( open( pickleFileNameTomato, 'rb' ) )
    except:
        print(pickleFileNameTomato, 'has not been recorded')
    else:
        triggerDict[rec]['tom']['swingStance'] = tomSwingStanceDic

    # pdb.set_trace()

    EMG_dir_ops='/media/invivodata2/altair_data/dataMichael/'+datesForFig[0]+f'/locomotionEMGAndMotor60sec_Bonsai_000/00{rec}/EMGsignals.ma'
    EMG_dir_tom='/media/invivodata2/altair_data/dataMichael/'+datesForFig[1]+f'/locomotionEMGAndMotor60sec_Bonsai_000/00{rec}/EMGsignals.ma'
    EMG_ops=dataAnalysis.readEMG(EMG_dir_ops)
    EMG_tom=dataAnalysis.readEMG(EMG_dir_tom)
    # EMG_opss=pd.read_csv(EMG_dir_ops)
    # EMG_tom=pd.read_csv(EMG_dir_tom)

    triggerDict[rec]['ops']['Trigger']=EMG_ops['current_chan2']
    triggerDict[rec]['tom']['Trigger']=EMG_tom['current_chan2']


    triggerDict[rec]['ops']['time']=EMG_ops['time']
    triggerDict[rec]['tom']['time'] = EMG_tom['time']

#pdb.set_trace()
#cPV.fig_real_time_experiment(figVersion,stridePar, strideTraj,musSwingStanceDic,salineSwingStanceDic, trigger)

# cPV.fig_real_time_experiment(figVersion,stridePar, strideTraj, triggerDict,opsValues,tomValues) #,salineSwingStanceDic, trigger)

cPV.fig_real_time_experiment(figVersion,stridePar, strideTraj, triggerDict,opsValues,tomValues,ephyData) #,salineSwingStanceDic, trigger)

