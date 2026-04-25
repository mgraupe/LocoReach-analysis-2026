import platform
import h5py
from scipy.interpolate import interp1d
from skimage import io
import tifffile as tiff
import numpy as np
import glob
# import sima
# import sima.segment
import time
import pdb
import cv2
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
#import deepdish as dd
#import sys
#import os

from psort.dependencies import deepdish_package
from psort.utils import signals_lib
from copy import deepcopy
from psort.utils import dictionaries
from psort.utils.database import PsortDataBase

from tools.h5pyTools import h5pyTools
import tools.googleDocsAccess as googleDocsAccess
from tools.pyqtgraph.configfile import *
from ScanImageTiffReader import ScanImageTiffReader

analysisParams= OrderedDict([
    ('swingStanceExtraction',{
        'stanceDistances': None,
        }),
    ('pawTrajectories' ,{
        'DLCinstance': {},
        }),
    ('projectionInTimeParameters', {
        'horizontalCuts' : None ,
        }),
    ('caEphysParameters', {
        'leaveOut' : None,
        }),
    ])

class extractSaveData:
    def __init__(self, mouse , recStruc=None, mountNPXfolder=False):
        self.mouse = mouse[:10]
        self.h5pyTools = h5pyTools()


        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' or platform.node() == 'thinkpadX1B' or platform.node() == 'thinkpadX1C':
            laptop = True
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/mgraupe/.anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'otillo':
            laptop = False
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/mgraupe/anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'yamal' or platform.node() == 'cerebellum-HP' or platform.node() == 'andry-ThinkPad-X1-Carbon-2nd' or platform.node() == 'OptiPlex-7070'  or platform.node() == 'studentPC3':
            laptop = False
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/andry/anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'studentPC2':
            laptop = False
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/margaux/anaconda3/envs/locorungs39/bin/python'
        elif platform.node() == 'optiplex-7040-student':
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/opt/anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'bs-analysis':  # andry-ThinkPad-X1-Carbon-2nd
            laptop = False
            self.analysisBase = '/home/mgraupe/nyc_data/'
        elif platform.node() == 'analysispc':
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/opt/anaconda/anaconda3/envs/locorungs39/bin/python'
        elif platform.node() == 'analysisRack1':
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/opt/anaconda3/envs/locorungs39/bin/python'
        else:
            print('Run this script on a server or laptop. Otherwise, adapt directory locations.')
            sys.exit(1)

        # define in which structure the recording took place : based on data or on optional input argument
        if  int(self.mouse[:6]) < 200300:
            self.recStructure = 'vermis'
        else:
            self.recStructure = 'simplex'
        if recStruc == 'optogenetics':
            self.recStructure = 'optogenetics'
        elif recStruc == 'simplexBehavior':
            self.recStructure = 'simplexBehavior'
        elif recStruc == 'simplexEphy':
            self.recStructure = 'simplexEphy'
        elif recStruc == 'simplexObstacleImaging':
            self.recStructure = 'simplexObstacleImaging'
        elif recStruc == 'simplexNPX':
            self.recStructure = 'simplexNPX'
        elif recStruc == 'shank3Behavior':
            self.recStructure = 'shank3Behavior'
        print('The %s was recorded in mouse %s' % (self.recStructure,self.mouse))

        # read google doc with the list of all experiments performed in the specific structure
        self.listOfAllExpts = googleDocsAccess.getExperimentSpreadsheet(self.recStructure)
        #print(self.listOfAllExpts)

        # read vermis or simplex files containing list of mice and list of relevant recordings
        if self.recStructure == 'vermis':
            self.config = readConfigFile('vermisAnimals.config')
        elif self.recStructure == 'simplex':
            self.config = readConfigFile('simplexAnimals.config')
        elif self.recStructure == 'simplexBehavior':
            self.config = readConfigFile('simplexAnimalsBehavior.config')
        elif self.recStructure == 'optogenetics':
            self.config = readConfigFile('simplexAnimalEphy.config')
        elif self.recStructure == 'simplexEphy':
            self.config = readConfigFile('simplexAnimalEphy.config')
        elif self.recStructure == 'simplexObstacleImaging':
            self.config = readConfigFile('simplexObstacleImaging.config')
        elif self.recStructure == 'simplexNPX':
            self.config = readConfigFile('simplexNPXRecordings.config')
        elif self.recStructure == 'shank3Behavior':
            self.config = readConfigFile('shank3Behavior.config')
        else:
            print('specify recording structure')

        # print(self.listOfAllExpts['201007_t00'])
        # extract recording dates of the specific animal
        dates = []
        for d in self.listOfAllExpts[self.mouse]['dates']:
            dates.append(d)

        if dates[0] >= '181018':
            self.dataBase = '/media/invivodata2/'
        else:
            self.dataBase = '/media/invivodata/'

        # check if directory is mounted
        if not os.listdir(self.dataBase):
            os.system('mount %s' % self.dataBase)

        if not os.listdir(self.analysisBase):
            os.system('mount %s' % self.analysisBase)

        # mount npx folders
        if mountNPXfolder:
            self.npxDataBase = '/media//invivodata2/' #'/media/desdemona_native/'
            if not os.listdir(self.npxDataBase):
                os.system('mount %s' % self.npxDataBase)

        self.dataPCLocation = OrderedDict(
            [('behaviorPC', 'behaviorPC_data/dataMichael/'), ('2photonPC', ('altair_data/dataMichael/' if int(self.mouse[:6]) >= 170829 else 'altair_data/experiments/data_Michael/acq4/')),
                ('', ('altair_data/dataMichael/' if int(self.mouse[:6]) >= 170829 else 'altair_data/experiments/data_Michael/acq4/')), ('npxPC', 'neuropixelPC_data/DataMichael/')])
        # pdb.set_trace()
        self.computerDict = OrderedDict(
            [('behaviorPC', 'behaviorPC'), ('2photonPC', '2photonPC'),('', '2photonPC'),('npxPC', 'npxPC')])

        self.analysisLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % mouse
        self.analysisLocationtot = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/'
        self.figureLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/%s/' % self.mouse
        self.publicationFigLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/publicationFigures/'
        self.presentationFigLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/presentationFigures/'
        if mountNPXfolder:
            self.NPXrawData = self.npxDataBase + 'neuropixelPC_data/DataMichael/' #'NPX/Data/' #/media/desdemona_native/NPX/Data/2023/20231213/11_09_00
        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.analysisLocation)
        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.figureLocation)

        fName = self.analysisLocation + 'analysis.hdf5'
        # if os.path.isfile(fName):
        try:
            self.f = h5py.File(fName, 'a')
        except FileExistsError:
            print('hdf5 file is used by another process and cannot be accessed.')
            sys.exit(1)
        else:
            pass

        # read animal specific config file with analysis parameters
        self.analysisConfigFile = self.analysisLocation + '%s.config' % self.mouse
        if os.path.isfile(self.analysisConfigFile):
            print('analysis parameter exist already')
            self.analysisConfig = readConfigFile(self.analysisConfigFile)
        else:
            self.analysisConfig = analysisParams
            writeConfigFile(self.analysisConfig, self.analysisConfigFile)
            print(self.analysisConfig)

        # experiments stored under this name were recorded as sequence with a certain number of repetitions
        self.listOfSequenceExperiments = ['locomotionTriggerSIAndMotor', 'locomotionTriggerSIAndMotorJin', 'locomotionTriggerSIAndMotor60sec', 'locomotion_recording_setup2','E2_VC_LEDtrigger','locomotionAndMotor60sec_Bonsai','E2_VC_LEDtrigger_train','locomotionTriggerSIAndMotorEMG60sec','locomotion_recording_setupBonsai','locomotionTriggerSIAndMotor60sec_Bonsai', 'locomotionEMGAndMotor60sec_Bonsai','optoPulseTrain_Bonsai']

    ############################################################
    def __del__(self):
        try:
            self.f.flush()
        except:
            pass
        else:
            print('hdf5 file flushed!')
        # self.f.close()
        print('on exit')

    ############################################################
    def writeConfigFile(self):
        print('analysis parameters : ', self.analysisConfig)
        print('analysis parameters writting to file : ', self.analysisConfigFile)
        writeConfigFile(self.analysisConfig, self.analysisConfigFile)

    ############################################################
    # dataConsistencyAndTimeCheck(self, foldersRecordings[f],wheel,paws,caimg):
    def dataConsistencyAndTimeCheck(self, foldersRecordings,wheel,paws,caimg):
        # check whether the same number of recordings from ACQ4 and Scanimage exist
        #pdb.set_trace()
        if len(foldersRecordings[2]) == len(caimg['specificTiffLists'][0][0]):
            print('PASS: Same number of ACQ4 recordings as ScanImage tiff files!', len(caimg['specificTiffLists'][0][0]))
            nRecs = len(foldersRecordings[2])
        else:
            print('Problem in consistency check!')
            pdb.set_trace()

        # check whether all recordings started at the same time, down to deltaStart
        siRecIDs = np.unique(caimg['timeStamps'][:, 1])
        if len(siRecIDs) != nRecs:
            print('ScanImage time-stamp IDs do not correspond to recoding number!')
            print('SI IDs, nRecs :', siRecIDs, nRecs)

        deltaStart = 1. # delay between recording starts in sec
        for i in range(nRecs):
            if np.abs(wheel[i]['timeStamp']-paws[i]['recStartTime']) > deltaStart: # comparision between rotary encoder and camera recording
                print('Rotary encoder and Video recording not at the same time.')
                print('wheel, paw, difference : ',wheel[i]['timeStamp'],wheel[i]['timeStamp'],wheel[i]['timeStamp']-paws[i]['recStartTime'])
                pdb.set_trace()
            # check now between rotary encoder and ScanImage recording
            correction = 0.
            #print(i)
            if (foldersRecordings[0] == '2021.05.26_000') and (self.mouse=='210214_m17'):
                correction = 3600.
                print('ATTENTION : weird correction applied')
            siMask = (caimg['timeStamps'][:,1] == siRecIDs[i])
            if np.abs(wheel[i]['timeStamp'] - caimg['timeStamps'][siMask][0,3] - correction) > deltaStart:
                print('Rotary encoder and ScanImage recording not at the same time.')
                print('wheel, ScanImage, difference : ', wheel[i]['timeStamp'], caimg['timeStamps'][siMask][0,3], wheel[i]['timeStamp'] - caimg['timeStamps'][siMask][0,3])
                pdb.set_trace()

        print('PASS : Time consistency!')
        #pdb.set_trace()
    ############################################################
    def getMotioncorrectedStack(self, folder, rec, suffix):
        allFiles = []
        for file in glob.glob(self.analysisLocation + '%s_%s_%s*_%s.tif' % (self.mouse, folder, rec, suffix)):
            allFiles.append(file)
            print(file)
        if len(allFiles) > 1:
            print('more than one matching image file')
            sys.exit(1)
        else:
            motionCoor = np.loadtxt(allFiles[0][:-3] + 'csv', delimiter=',', skiprows=1)
            imStack = io.imread(allFiles[0])
        return (imStack, motionCoor, allFiles[0])

    ############################################################
    def saveRungMotionData(self, mouse, date, rec, rungPositions):
        rec = rec.replace('/', '-')
        pickle.dump(rungPositions, open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb'))

    def checkRungMotionData(self, mouse, date, rec):
        rec = rec.replace('/', '-')
        videoFileName=  self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec)
        if not os.path.isfile(videoFileName):

            return False
        else:
            return True
    ############################################################
    def readRungMotionData(self, mouse, date, rec):
        rec = rec.replace('/', '-')
        rungPositions = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'rb'))
        return rungPositions

    ############################################################
    def saveSwingStanceData(self, mouse, date, rec, swingP,forFit, pawPos):
        rec = rec.replace('/', '-')
        swingStanceDict = {}
        swingStanceDict['swingP'] = swingP
        swingStanceDict['forFit'] = forFit
        swingStanceDict['pawPos'] = pawPos
        pickle.dump(swingStanceDict, open(self.analysisLocation + '%s_%s_%s_swingStancePhases.p' % (mouse, date, rec), 'wb'))

    ############################################################
    def readSwingStanceData(self, mouse, date, rec):
        rec = rec.replace('/', '-')
        swingStanceDict = pickle.load(open(self.analysisLocation + '%s_%s_%s_swingStancePhases.p' % (mouse[:10], date, rec), 'rb'))
        # pdb.set_trace()
        return swingStanceDict

    ############################################################
    def saveDLCInstanceForRecording(self,dateFolder,rec,DLCinstance):
        if not (dateFolder in self.analysisConfig['pawTrajectories']['DLCinstance']):
            # pdb.set_trace()
            if not isinstance(self.analysisConfig['pawTrajectories'].get('DLCinstance'), dict):
                self.analysisConfig['pawTrajectories']['DLCinstance'] = {}
            self.analysisConfig['pawTrajectories']['DLCinstance'][dateFolder] = {}
        self.analysisConfig['pawTrajectories']['DLCinstance'][dateFolder][rec] = DLCinstance

    ############################################################
    def readDLCInstanceForRecording(self,dateFolder,rec):
        if not (dateFolder in self.analysisConfig['pawTrajectories']['DLCinstance']):
            raise Exception('Entry %s does not exist in config file' % dateFolder)
        if not(rec in self.analysisConfig['pawTrajectories']['DLCinstance'][dateFolder]):
            raise Exception('Entry %s / %s does not exist in config file' % (dateFolder,rec))
        DLCinstance = self.analysisConfig['pawTrajectories']['DLCinstance'][dateFolder][rec]
        return DLCinstance

    ############################################################
    def transfer_data_from_psortDataBase_to_workingDataBase(self):
        psortDataBase_currentSlot = self.psortDataBase.get_currentSlotDataBase()
        psortDataBase_topLevel = self.psortDataBase.get_topLevelDataBase()
        self._workingDataBase['isAnalyzed'] = psortDataBase_currentSlot['isAnalyzed']
        index_start_on_ch_data = psortDataBase_currentSlot['index_start_on_ch_data'][0]
        index_end_on_ch_data = psortDataBase_currentSlot['index_end_on_ch_data'][0]
        self._workingDataBase['index_start_on_ch_data'][0] = index_start_on_ch_data
        self._workingDataBase['index_end_on_ch_data'][0] = index_end_on_ch_data
        self._workingDataBase['ch_data'] = psortDataBase_topLevel['ch_data'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['ch_time'] = psortDataBase_topLevel['ch_time'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['ss_index'] = psortDataBase_topLevel['ss_index'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['cs_index'] = psortDataBase_topLevel['cs_index'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['cs_index_slow'] = psortDataBase_topLevel['cs_index_slow'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['sample_rate'][0] = psortDataBase_topLevel['sample_rate'][0]
        if psortDataBase_topLevel['isLfpSideloaded'][0]:
            self._workingDataBase['ch_lfp'] = psortDataBase_topLevel['ch_lfp'][index_start_on_ch_data:index_end_on_ch_data]
        self._workingDataBase['isLfpSideloaded'][0] = psortDataBase_topLevel['isLfpSideloaded'][0]
        # if the SLOT is already analyzed then transfer the data over,
        # otherwise, do not transfer and use the current values for the new slot
        if self._workingDataBase['isAnalyzed'][0]:
            for key in dictionaries._singleSlotDataBase.keys():
                self._workingDataBase[key] = psortDataBase_currentSlot[key]
            self._workingDataBase['flag_index_detection'][0] = False
        else:
            self._workingDataBase['flag_index_detection'][0] = True
        return 0

    ##################################################################################
    def readAnalyzedPSortData(self,file_fullPath):
        #file_fullPath = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/220211_f38/EphysDataPerSession_2022.05.03_000_locomotionEphys2Motor60sec_004_psorted.psort'
        #_fileDataBase = deepcopy(dictionaries._fileDataBase)
        # file loading
        #_fileDataBase['load_file_fullPath'] = file_fullPath
        #_fileDataBase['isMainSignal'][0] = True
        #_fileDataBase['isCommonAverage'][0] = False
        #_fileDataBase['isLfpSignal'][0] = False

        # init workingDataBase
        self._workingDataBase = deepcopy(dictionaries._workingDataBase)

        grandDataBase = deepdish_package.io.load(file_fullPath)
        self.psortDataBase = PsortDataBase()
        self.psortDataBase.load_dataBase(file_fullPath, grandDataBase=grandDataBase, isCommonAverage=False)

        self.transfer_data_from_psortDataBase_to_workingDataBase()
        # analyzing signals
        # print('grandDataBase :',grandDataBase)
        #print('workingDataBase:', self._workingDataBase)
        print('_workingDataBase sample rate:', self._workingDataBase['sample_rate'][0])
        print('   applying filters ... ',end='')
        signals_lib.filter_data(self._workingDataBase)
        print('done')
        if self._workingDataBase['flag_index_detection'][0]:
            signals_lib.detect_ss_index(self._workingDataBase)
            signals_lib.detect_cs_index_slow(self._workingDataBase)
            signals_lib.align_cs(self._workingDataBase)  # self.undoRedo_add()
        else:
            self._workingDataBase['flag_index_detection'][0] = True
        print('   update _workingDataBase ... ',end='')
        signals_lib.reset_cs_ROI(self._workingDataBase)
        print(' 0 ', end='')
        signals_lib.reset_ss_ROI(self._workingDataBase)
        print(' 1 ', end='')
        signals_lib.extract_ss_peak(self._workingDataBase)
        print(' 2 ', end='')
        signals_lib.extract_cs_peak(self._workingDataBase)
        print(' 3 ', end='')
        signals_lib.extract_ss_waveform(self._workingDataBase)
        print(' 4 ', end='')
        signals_lib.extract_cs_waveform(self._workingDataBase)
        print(' 5 ', end='')
        signals_lib.extract_ss_similarity(self._workingDataBase)
        print(' 6 ', end='')
        signals_lib.extract_cs_similarity(self._workingDataBase)
        print(' 7 ', end='')
        signals_lib.extract_ss_ifr(self._workingDataBase)
        print(' 8 ', end='')
        signals_lib.extract_cs_ifr(self._workingDataBase)
        print(' 9 ', end='')
        signals_lib.extract_ss_time(self._workingDataBase)
        print(' 10 ', end='')
        signals_lib.extract_cs_time(self._workingDataBase)
        print(' 11 ', end='')
        signals_lib.extract_ss_xprob(self._workingDataBase)
        print(' 12 ', end='')
        signals_lib.extract_cs_xprob(self._workingDataBase)
        print(' 13 ', end='')
        #signals_lib.extract_ss_pca(self._workingDataBase)
        print(' 14 ', end='')
        #signals_lib.extract_cs_pca(self._workingDataBase)
        print(' 15 ', end='')
        #signals_lib.extract_ss_scatter(self._workingDataBase)
        print(' 16 ', end='')
        #signals_lib.extract_cs_scatter(self._workingDataBase)
        print('done ')
        return grandDataBase

    ############################################################
    def readPSortAnalyzedData(self, fold, eD, recording, device, fData, readRawData=True):
        print(fold,eD,recording,device,fData)
        pSortFileName = 'EphysDataPerSession_%s_%s_psorted.psort' % (fold,recording)
        if os.path.isfile(self.analysisLocation+pSortFileName):
            print('Ephys data has been p-sorted and file %s exists.' % pSortFileName)
            # self.pSortF = h5py.File(self.analysisLocation+pSortFileName, 'a')
            fullFilePath = self.analysisLocation+pSortFileName
            grandDataBase = self.readAnalyzedPSortData(fullFilePath)
            #grandDataBase = dd.io.load(self.analysisLocation+pSortFileName)

            topLevelDataBase = grandDataBase[-1]
            ch_time = topLevelDataBase['ch_time']
            ss_index = topLevelDataBase['ss_index']
            cs_index = topLevelDataBase['cs_index']
            ss_time = ch_time[ss_index]
            cs_time = ch_time[cs_index]
            return ([ss_time,cs_time,self._workingDataBase,grandDataBase])
        else:
            print('p-sorted file does not exist! Ephys data needs to be p-sorted to generate %s' % pSortFileName)
            pdb.set_trace()
            return None

    ############################################################
    def extractRoiSignals(self, folder, rec, tifFile):

        self.simaPath = self.analysisLocation + '%s_%s_%s' % (self.mouse, folder, rec)
        print(self.simaPath)
        if os.path.isdir(self.simaPath + '.sima'):
            print('sima dir exists')
            dataSet = sima.ImagingDataset.load(self.simaPath + '.sima')
        else:
            print('create sima dir')
            sequences = [sima.Sequence.create('TIFF', tifFile)]
            dataSet = sima.ImagingDataset(sequences, self.simaPath, channel_names=['GCaMP6F'])

        img = dataSet.time_averages[0][:, :, 0]

        self.simaPath = self.simaPath + '.sima/'
        overwrite = True
        if os.path.isfile(self.simaPath + 'rois.pkl'):
            print('rois already exist')
            input_ = raw_input('rois traces exist already, do you want to overwrite? (type \'y\' to overwrite, any character if not) : ')
            if input_ != 'y':
                overwrite = False
        if overwrite:
            print('create rois with roibuddy')
            segmentation_approach = sima.segment.STICA(channel='GCaMP6F', components=1, mu=0.9,  # weighting between spatial - 1 - and temporal - 0 - information
                # spatial_sep=0.8
            )
            print('segmenting calcium image ... ', end=" ")
            dataSet.segment(segmentation_approach, 'GCaMP6F_signals', planes=[0])
            print('done')
            while True:
                input_ = raw_input('Please check ROIs in \'roibuddy\' (type \'exit\' to halt) : ')
                if input_ == 'exit':
                    sys.exit(1)
                else:
                    break
        dataSet = sima.ImagingDataset.load(self.simaPath)
        rois = dataSet.ROIs['GCaMP6F_signals']

        # Extract the signals.
        dataSet.extract(rois, signal_channel='GCaMP6F', label='GCaMP6F_signals')
        raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        # dataSet.export_signals('example_signals.csv', channel='GCaMP6F',signals_label='GCaMP6F_signals')
        # pdb.set_trace()

        # dataSet.signals('GCaMP6F')['GCaMP6F_signals']['mean_frame']

        # pdb.set_trace()
        roiLabels = []
        # extrace labels
        for n in range(len(rois)):
            roiLabels.append(rois[n].label)  # print 'ROI label', n, rois[n].label

        return (img, rois, raw_signals)

    ############################################################
    def getRecordingsList(self, expDate='all', recordings='all'):
        if recordings == 'all910':
            self.recID = 'recs910'
        elif recordings == 'all820':
            self.recID = 'recs820'
        elif recordings == 'all':
            self.recID = ['recs910','recs820']

        #print(self.listOfAllExpts['231012_m54'])
        #pdb.set_trace()
        for i in range(len(self.config)):
            if self.config['%s' % i]['mouse'] == self.mouse:
                print('experiment dictionary of mouse exists')
                self.expDict = self.config['%s' % i]['days']
                dictExists = True
        # pdb.set_trace()
        folderRec = []
        if self.mouse in self.listOfAllExpts:
            # print mouse
            # print expDate, self.listOfAllExpts[mouse]['dates']
            expDateList = []
            # pdb.set_trace()
            # provide choice of which days to include in analysis
            if expDate == 'some':
                print('Dates when experiments where performed with animal %s :' % self.mouse)
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    print('  %s %d' % (d, didx))
                    didx += 1
                print('Choose the dates for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\', or \'0-5\' : ', end='')
                daysInput = input()
                if '-' in daysInput:
                    startEnd = [int(i) for i in daysInput.split('-')]
                    daysInputIdx = [i for i in range(startEnd[0],startEnd[1]+1)]
                else:
                    daysInputIdx = [int(i) for i in daysInput.split(',')]  # print(daysInputIdx,daysInputIdx[0],type(daysInputIdx))
            elif expDate == 'all910' or expDate == 'all820':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        if self.recID in self.expDict[d]:
                            daysInputIdx.append(didx)
                    didx+=1

            elif expDate == 'all':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        daysInputIdx.append(didx)
                    didx+=1
            ########################################################
            # generate list of days to analyze
            if expDate == 'some' or expDate=='all910' or expDate=='all820' or expDate == 'all':
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if didx in daysInputIdx:
                        # print(d)
                        expDateList.append(d)
                    didx += 1
            else:
                expDateList.append(expDate)
            print('Selected dates :', expDateList)
            #pdb.set_trace()
            #####################################################
            # chose recordings
            if recordings == 'all910':
                print('All 910 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:

                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs910' in self.expDict[eD]:
                        idx910recs = list(self.expDict[eD]['recs910'].keys())
                    else:
                        idx910recs = None
                    for fold in dataFolders:
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)

                        if idx910recs is not None:
                            idx910recs = [int(i) for i in idx910recs]
                            # pdb.set_trace()
                            for n in range(len(idx910recs)):
                                trials = self.expDict[eD]['recs910'][str(idx910recs[n])]['trials']

                                if trials == 'all':
                                    trials = [i for i in range(5)] # assume to include all five trials if 'trials' subfield does not exist in the config file
                                elif type(trials)==int:
                                    trials=[trials]
                                recInputIdx.append([recIdx+idx910recs[n],idx910recs[n],trials])
                        recIdx+=len(recList)
            elif recordings == 'all820':
                print('All 820 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs820' in self.expDict[eD]:
                        idx820recs = list(self.expDict[eD]['recs820'].keys())
                    else:
                        idx820recs = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx820recs is not None:
                            idx820recs = [int(i) for i in idx820recs]
                            for n in range(len(idx820recs)):
                                trials = self.expDict[eD]['recs820'][str(idx820recs[n])]['trials']
                                if trials == 'all':
                                    trials = [i for i in range(1)]  # assume to include all five trials if 'trials' subfield does not exist in the config file
                                recInputIdx.append([recIdx+idx820recs[n],idx820recs[n],trials])
                        recIdx+=len(recList)
                #pdb.set_trace()
            elif recordings == 'some':
                # first show all recordings for a given date
                print('Choose recording to analyze')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    for fold in dataFolders:
                        print(' ', fold)

                        # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')


                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                        if '_bis' in fold and dataFolders[fold]['recComputer']:
                            self.dataLocation=self.dataLocation.replace('_bis','')
                        if not os.path.exists(self.dataLocation):
                            #    print('experiment %s exists' % fold)
                            # else:
                            print('Problem, experiment does not exist')
                        # recList = OrderedDict()
                        recList = self.getDirectories(self.dataLocation)
                        for r in recList:
                            print('    %s  %s' % (r, recIdx))
                            recIdx += 1
                print('Choose the recordings for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\', or \'0-5\' : ', end='')
                recInput = input()
                if '-' in recInput:
                    startEnd = [int(i) for i in recInput.split('-')]
                    recInputIdxRaw = [i for i in range(startEnd[0],startEnd[1]+1)]
                else:
                    recInputIdxRaw = [int(i) for i in recInput.split(',')]
                #TODO : allow for 'some' recordings of a given day to be picked
                recIdx = 0
                recIdx2 = 0
                #rec820Idx = 0
                for eD in expDateList:
                    for fold in dataFolders:
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        for r in range(len(recList)):
                            if recIdx in recInputIdxRaw:
                                dirs = self.getDirectories(self.dataLocation + '/' + recList[r])
                                if (len(dirs) == 1)  and (dirs[0]=='CameraGigEBehavior') : # 820 recording
                                    recInputIdx.append([recIdx2 + r, r, [i for i in range(1)]])
                                else:
                                    # recInputIdx.append([recIdx2 + r, r, [i for i in range(len(dirs))]])
                                    recInputIdx.append([recIdx2 + r,r, [i for i in range(len(dirs))]])
                            # pdb.set_trace()
                            recIdx+=1
                        recIdx2 += len(recList)
                print(recInputIdx)
                # pdb.set_trace()
            elif recordings == 'all':
                print('All 910 and 820 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs910' in self.expDict[eD]:
                        #idx910 = self.expDict[eD]['recs910']['recordings']
                        idx910recs = list(self.expDict[eD]['recs910'].keys())
                    else:
                        idx910recs = None
                    if 'recs820' in self.expDict[eD]:
                        #idx820 = self.expDict[eD]['recs820']['recordings']
                        idx820recs = list(self.expDict[eD]['recs820'].keys())
                    else:
                        idx820recs = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx910recs is not None:
                            idx910recs = [int(i) for i in idx910recs]
                            for n in range(len(idx910recs)):
                                trials = self.expDict[eD]['recs910'][str(idx910recs[n])]['trials']
                                if trials=='all':
                                    trials = [i for i in range(5)] # assume to include all five trials if 'trials' subfield does not exist in the config file
                                recInputIdx.append([recIdx+idx910recs[n],idx910recs[n],trials])
                        if idx820recs is not None:
                            idx820recs = [int(i) for i in idx820recs]
                            for n in range(len(idx820recs)):
                                trials = self.expDict[eD]['recs820'][str(idx820recs[n])]['trials']
                                if trials=='all':
                                    trials = [i for i in range(1)]  # assume to include all five trials if 'trials' subfield does not exist in the config file
                                recInputIdx.append([recIdx+idx820recs[n],idx820recs[n],trials])
                        recIdx+=len(recList)
            else:
                recInputIdx = [int(i) for i in recordings.split(',')]
            # pdb.set_trace()
            print('list of recordings : ', recInputIdx)
            # pdb.set_trace()
            # print(recordings)
            #
            # pdb.set_trace()
            # then compile a list the selected recordings
            recIdx = 0
            for eD in expDateList:
                # print(expDateList, self.listOfAllExpts[mouse]['dates'], len(self.listOfAllExpts[mouse]['dates']))
                dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                #print(eD, self.listOfAllExpts[self.mouse]['dates'][eD], dataFolders)
                for fold in dataFolders:
                    #print('fold :', fold)
                    # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                    # pdb.set_trace()
                    self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                    self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                    print(self.dataLocation)
                    if not os.path.exists(self.dataLocation):
                        # print('experiment %s exists' % fold)
                        # else:
                        print('Problem, experiment does not exist')
                    recList = self.getDirectories(self.dataLocation)
                    if recordings == 'all':
                        recInputIdx = []
                        tempRecList = []
                        for r in recList:
                            # if r[:-4] == 'locomotionTriggerSIAndMotor' or r[:-4] == 'locomotionTriggerSIAndMotorJin' or r[:-4] == 'locomotionTriggerSIAndMotor60sec' :
                            if r[:-4] in self.listOfSequenceExperiments:
                                subFolders = self.getDirectories(self.dataLocation + '/' + r)
                                # pdb.set_trace()
                                for i in range(len(subFolders)):
                                    if subFolders[i][0] == '0':
                                        tempRecList.append(r + '/' + subFolders[i])
                                    else:
                                        tempRecList.append(r)
                                        break
                            else:
                                tempRecList.append(r)
                        folderRec.append([fold, eD, tempRecList])  # folderRec.append([fold,eD,recList])
                    else: # recordings == 'some':
                        for r in recList:
                            # pdb.set_trace()
                            # only add recordings which were previously selected
                            tempRecList = []
                            # pdb.set_trace()
                            # for f in recInputIdx:
                            #     print(f[0])
                            # pdb.set_trace()
                            if any([recIdx == f[0] for f in recInputIdx]): # check whether recIdx matches any of the first indicies of the recInputIdx list
                                # if r[:-4] == 'locomotionTriggerSIAndMotor' or r[:-4] == 'locomotionTriggerSIAndMotorJin' or r[:-4] == 'locomotionTriggerSIAndMotor60sec':
                                # pdb.set_trace()
                                for i in range(len(recInputIdx)):
                                    if recIdx == recInputIdx[i][0] :
                                        recInfo = recInputIdx[i]
                                # consistency check
                                print(r, recInfo)
                                if 'g' not in r[-3:]: # this check is not performed for npx files
                                    if recInfo[1] != int(r[-3:]):
                                        print('Something wrong with the recording number!')
                                        print(r, recInfo)
                                        # pdb.set_trace()
                                # pdb.set_trace()
                                if r[:-4] in self.listOfSequenceExperiments:
                                    subFolders = self.getDirectories(self.dataLocation + '/' + r)
                                    # pdb.set_trace()
                                    for i in range(len(subFolders)):
                                        if subFolders[i][0] == '0': # check if subfolders are numbered of the format 000, 001, etc.
                                            if int(subFolders[i]) in recInfo[2] :  # check of the trial folder is a valid recording (otherwise, exclude from list)
                                                tempRecList.append(r + '/' + subFolders[i])
                                        else:
                                            tempRecList.append(r)
                                            break
                                else:
                                    tempRecList.append(r)
                                    # pdb.set_trace()
                                folderRec.append([fold, eD, tempRecList])
                                #print('folderRec : ', folderRec)
                            recIdx += 1
                        # pdb.set_trace()
        print('Data was recorded on %s' % self.recordingMachine)
        # pdb.set_trace()
        return (folderRec, dataFolders)
    ############################################################
    def getEphysRecordingsList(self, expDate='all', recordings='all'):
        if recordings == 'allMLI':
            self.recID = 'recsMLI'
        elif recordings == 'allPC':
            self.recID = 'recsPC'
        elif recordings == 'allBehavior':
            self.recID = 'recsBehavior'
        elif recordings == 'all':
            self.recID = ['recsMLI','recsPC','recsBehavior']

        #pdb.set_trace()
        for i in range(len(self.config)):
            if self.config['%s' % i]['mouse'] == self.mouse:
                print('experiment dictionary of mouse exists')
                self.expDict = self.config['%s' % i]['days']
                dictExists = True
        #pdb.set_trace()
        folderRec = []
        if self.mouse in self.listOfAllExpts:
            # print mouse
            # print expDate, self.listOfAllExpts[mouse]['dates']
            expDateList = []
            # pdb.set_trace()
            # provide choice of which days to include in analysis
            if expDate == 'some':
                print('Dates when experiments where performed with animal %s :' % self.mouse)
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    print('  %s %d' % (d, didx))
                    didx += 1
                print('Choose the dates for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\', or \'0-5\' : ', end='')
                daysInput = input()
                if '-' in daysInput:
                    startEnd = [int(i) for i in daysInput.split('-')]
                    daysInputIdx = [i for i in range(startEnd[0],startEnd[1]+1)]
                else:
                    daysInputIdx = [int(i) for i in daysInput.split(',')]  # print(daysInputIdx,daysInputIdx[0],type(daysInputIdx))
            elif expDate == 'allPC' or expDate == 'allMLI':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        #print(self.expDict[d])
                        if self.recID in self.expDict[d]:
                            daysInputIdx.append(didx)
                    didx+=1
            elif expDate == 'all':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        if any(key in self.expDict[d] for key in self.recID):
                            daysInputIdx.append(didx)
                    didx+=1
            ########################################################
            # generate list of days to analyze
            if expDate == 'some' or expDate=='allMLI' or expDate=='allPC' or expDate == 'all':
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if didx in daysInputIdx:
                        # print(d)
                        expDateList.append(d)
                    didx += 1
            else:
                expDateList.append(expDate)
            print('Selected dates :', expDateList)
            #pdb.set_trace()
            #####################################################
            # chose recordings
            if recordings == 'allMLI':
                print('All MLI recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recsMLI' in self.expDict[eD]:
                        idx910recs = list(self.expDict[eD]['recsMLI'].keys())
                    else:
                        idx910recs = None
                    for fold in dataFolders:
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        if idx910recs is not None:
                            idx910recs = [int(i) for i in idx910recs]
                            for n in range(len(idx910recs)):
                                #print(idx910recs,n)
                                trials = self.expDict[eD]['recsMLI'][str(idx910recs[n])]['trials']
                                if 'visuallyGuided' in self.expDict[eD]['recsMLI'][str(idx910recs[n])].keys():
                                    visGuided = ['MLI',True]
                                else:
                                    visGuided = ['MLI',False]
                                #if trials == 'all':
                                #    trials = [i for i in range(5)] # assume to include all five trials if 'trials' subfield does not exist in the config file
                                #elif type(trials)==int:
                                #print(idx910recs, n,trials,recIdx)
                                trials=[trials]
                                #recInputIdx.append([recIdx+idx910recs[n],idx910recs[n],trials])
                                recInputIdx.append([recIdx,eD, idx910recs[n], trials,visGuided])
                                #nRecs = len(trials[0])
                        recIdx+=len(recList)
                #pdb.set_trace()
            elif recordings == 'allPC':
                print('All PC recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recsPC' in self.expDict[eD]:
                        idx820recs = list(self.expDict[eD]['recsPC'].keys())
                    else:
                        idx820recs = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx820recs is not None:
                            idx820recs = [int(i) for i in idx820recs]
                            for n in range(len(idx820recs)):
                                trials = self.expDict[eD]['recsPC'][str(idx820recs[n])]['trials']
                                if 'visuallyGuided' in self.expDict[eD]['recsPC'][str(idx820recs[n])].keys():
                                    visGuided = ['PC',True]
                                else:
                                    visGuided = ['PC',False]
                                #if trials == 'all':
                                #    trials = [i for i in range(1)]  # assume to include all five trials if 'trials' subfield does not exist in the config file
                                trials = [trials]
                                #recInputIdx.append([recIdx+idx820recs[n],idx820recs[n],trials])
                                recInputIdx.append([recIdx,eD,idx820recs[n],trials,visGuided])
                        recIdx+=len(recList)
                #pdb.set_trace()
            elif recordings == 'some':
                # first show all recordings for a given date
                print('Choose recording to analyze')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    for fold in dataFolders:
                        print(' ', fold)
                        # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        #self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                        #print(self.dataLocation)
                        if not os.path.exists(self.dataLocation):
                            #    print('experiment %s exists' % fold)
                            # else:
                            print('Problem, experiment does not exist')
                        # recList = OrderedDict()
                        recList = self.getDirectories(self.dataLocation)
                        for r in recList:
                            print('    %s  %s' % (r, recIdx))
                            recIdx += 1
                print('Choose the recordings for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\', or \'0-5\' : ', end='')
                recInput = input()
                if '-' in recInput:
                    startEnd = [int(i) for i in recInput.split('-')]
                    recInputIdxRaw = [i for i in range(startEnd[0],startEnd[1]+1)]
                else:
                    recInputIdxRaw = [int(i) for i in recInput.split(',')]
                #TODO : allow for 'some' recordings of a given day to be picked
                recIdx = 0
                recIdx2 = 0
                #rec820Idx = 0
                for eD in expDateList:
                    for fold in dataFolders:
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        for r in range(len(recList)):
                            if recIdx in recInputIdxRaw:
                                dirs = self.getDirectories(self.dataLocation + '/' + recList[r])
                                if (len(dirs) == 1)  and (dirs[0]=='CameraGigEBehavior') : # 820 recording
                                    recInputIdx.append([recIdx2 + r, eD, r, [i for i in range(1)]])
                                else:
                                    recInputIdx.append([recIdx2 + r,eD, r, [i for i in range(len(dirs))]])
                            #pdb.set_trace()
                            recIdx+=1
                        recIdx2 += len(recList)
                #print(recInputIdx)
                #pdb.set_trace()
            elif recordings == 'all':
                print('All MLI and PC recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recsMLI' in self.expDict[eD]:
                        #idx910 = self.expDict[eD]['recs910']['recordings']
                        idx910recs = list(self.expDict[eD]['recsMLI'].keys())
                    else:
                        idx910recs = None
                    if 'recsPC' in self.expDict[eD]:
                        #idx820 = self.expDict[eD]['recs820']['recordings']
                        idx820recs = list(self.expDict[eD]['recsPC'].keys())
                    else:
                        idx820recs = None
                    if 'recsBehavior' in self.expDict[eD]:
                        #idx820 = self.expDict[eD]['recs820']['recordings']
                        idxBehrecs = list(self.expDict[eD]['recsBehavior'].keys())
                    else:
                        idxBehrecs = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx910recs is not None:
                            idx910recs = [int(i) for i in idx910recs]
                            for n in range(len(idx910recs)):
                                trials = self.expDict[eD]['recsMLI'][str(idx910recs[n])]['trials']
                                if 'visuallyGuided' in self.expDict[eD]['recsMLI'][str(idx910recs[n])].keys():
                                    visGuided = ['MLI',True]
                                else:
                                    visGuided = ['MLI',False]
                                #if trials=='all':
                                #    trials = [i for i in range(5)] # assume to include all five trials if 'trials' subfield does not exist in the config file
                                trials = [trials]
                                #recInputIdx.append([recIdx+idx910recs[n],idx910recs[n],trials])
                                recInputIdx.append([recIdx,eD, idx910recs[n], trials, visGuided])
                        if idx820recs is not None:
                            idx820recs = [int(i) for i in idx820recs]
                            for n in range(len(idx820recs)):
                                trials = self.expDict[eD]['recsPC'][str(idx820recs[n])]['trials']
                                if 'visuallyGuided' in self.expDict[eD]['recsPC'][str(idx820recs[n])].keys():
                                    visGuided = ['PC',True]
                                else:
                                    visGuided = ['PC',False]
                                #if trials=='all':
                                #    trials = [i for i in range(1)]  # assume to include all five trials if 'trials' subfield does not exist in the config file
                                trials = [trials]
                                #recInputIdx.append([recIdx+idx820recs[n],idx820recs[n],trials])
                                recInputIdx.append([recIdx,eD, idx820recs[n], trials, visGuided])
                        if idxBehrecs is not None:
                            idxBehrecs = [int(i) for i in idxBehrecs]
                            for n in range(len(idxBehrecs)):
                                trials = self.expDict[eD]['recsBehavior'][str(idxBehrecs[n])]['trials']
                                if 'visuallyGuided' in self.expDict[eD]['recsBehavior'][str(idxBehrecs[n])].keys():
                                    visGuided = ['Beh',True]
                                else:
                                    visGuided = ['Beh',False]
                                #if trials=='all':
                                #    trials = [i for i in range(1)]  # assume to include all five trials if 'trials' subfield does not exist in the config file
                                trials = [trials]
                                #recInputIdx.append([recIdx+idx820recs[n],idx820recs[n],trials])
                                recInputIdx.append([recIdx,eD, idxBehrecs[n], trials, visGuided])
                        recIdx+=len(recList)
            else:
                recInputIdx = [int(i) for i in recordings.split(',')]
            print('list of recordings : ', recInputIdx)
            # print(recordings)
            #

            # then compile a list the selected recordings
            recIdx = 0
            for eD in expDateList:
                # print(expDateList, self.listOfAllExpts[mouse]['dates'], len(self.listOfAllExpts[mouse]['dates']))
                dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                # print(eD, self.listOfAllExpts[mouse]['dates'],dataFolders)
                for fold in dataFolders:
                    # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                    # pdb.set_trace()
                    self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                    self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                    #print(self.dataLocation)
                    if not os.path.exists(self.dataLocation):
                        # print('experiment %s exists' % fold)
                        # else:
                        print('Problem, experiment does not exist')
                    recList = self.getDirectories(self.dataLocation)
                    #pdb.set_trace()
                    for j in range(len(recInputIdx)):
                        if recInputIdx[j][1] == eD:
                            #print(recInputIdx[j])
                            if j>0:
                                if recInputIdx[j][1] == recInputIdx[j-1][1]: # restart to count the recording index if two cells where recorded at the same day
                                    recIdx -= len(recList)
                            tempRecList = []
                            for r in recList: # loop over list of actually saved recordings on the backup machine
                                #print(fold, eD, r, j, recIdx,(recInputIdx[j][0] + np.array(recInputIdx[j][3])))
                                # only add recordings which were previously selected
                                #print([recIdx == (f[0] + np.arange(2)) for f in recInputIdx])
                                #if np.any(np.hstack([recIdx == (f[0]+np.array(f[3])) for f in recInputIdx[j]])): # check whether recIdx matches any of the first indicies of the recInputIdx list
                                if np.any((recIdx == (recInputIdx[j][0] + np.array(recInputIdx[j][3])))): # for f in recInputIdx[j]])):
                                    # if r[:-4] == 'locomotionTriggerSIAndMotor' or r[:-4] == 'locomotionTriggerSIAndMotorJin' or r[:-4] == 'locomotionTriggerSIAndMotor60sec':
                                    # for i in range(len(recInputIdx)):
                                    #   if recIdx == recInputIdx[i][0] :
                                    #       recInfo = recInputIdx[i]
                                    # consistency check
                                    #print('r and recinfo : ',r, recInfo, recInfo[3], int(r[-3:]))
                                    #if np.all(np.array(recInfo[3]) != int(r[-3:])):
                                    if recordings == 'some':
                                        tempRecList.append(r)
                                    else:
                                        if np.all(np.array(recInputIdx[j][3]) != int(r[-3:])):
                                            print('Something wrong with the recording number!')
                                            print(recInputIdx[j][3], r)
                                            #pdb.set_trace()
                                        else:
                                            tempRecList.append(r)
                                recIdx += 1

                            folderRec.append([fold, eD, tempRecList])
        #pdb.set_trace()
        print('Data was recorded on %s' % self.recordingMachine)
        return (folderRec, dataFolders,recInputIdx)

    ############################################################
    def combineDifferentCategorysOnSameDay(self, foldersRecs,listOfRecs):
        newfolderRecs = []
        newListOfRecs = []
        idx = 0
        #for f in range(len(foldersRecs)):
        while idx<len(foldersRecs):
            trials = foldersRecs[idx][2]
            #indicies = list(listOfRecs[idx][3]) # if type(listOfRecs[idx][3])==tuple else (listOfRecs[idx][3])
            if (idx+1)<len(foldersRecs):
                p = 1
                while foldersRecs[idx][1] == foldersRecs[idx+p][1]: # for the same day
                    print(p,idx+p,len(foldersRecs),foldersRecs[idx][1],foldersRecs[idx+p][1])
                    trials.extend(foldersRecs[idx+p][2])
                    #indicies.extend(list(listOfRecs[idx+p][3]))
                    if (idx+p+1)<len(foldersRecs):
                        p+=1
                    else:
                        p+=1
                        break
            else:
                p=1
            trials.sort()
            newfolderRecs.append([foldersRecs[idx][0],foldersRecs[idx][1],trials])
            newListOfRecs.append([listOfRecs[idx][0],listOfRecs[idx][1],listOfRecs[idx][2],idx])
            idx+=p
        return (newfolderRecs,newListOfRecs)

    ############################################################
    def getDirectories(self, location):
        # seqFolder = self.dataLocation + '/' + r
        if '_bis' in location:
            location=location.replace('_bis','')
            # pdb.set_trace()

        subFolders = [os.path.join(o) for o in os.listdir(location) if os.path.isdir(os.path.join(location, o))]
        # Create a list of subfolders to remove ('2023' or '2022')
        subfolders_to_remove = ['2023', '2022']

        # Remove the subfolders to_remove from the subFolders list
        subFolders = [subfolder for subfolder in subFolders if
                      not any(item in subfolder for item in subfolders_to_remove)]
        subFolders.sort()
        return subFolders
    ############################################################
    def  videoRecordingImplementation(self,existenceAniBonsaiFrames,existenceWhisBonsaiFrames,existenceAniACQ4Frames):
        vidRecDict = {}
        print('===========')
        print((' Animal' if (existenceAniBonsaiFrames or existenceAniACQ4Frames) else ''),end=' ')
        print(('and Whisker' if existenceWhisBonsaiFrames else ''),end=' ')
        #print(('NO' if (existenceAniBonsaiFrames or existenceAniACQ4Frames or existenceWhisBonsaiFrames)),end_of_line=' ')
        print('videos recorded', end=' ')
        print('with', ('Bonsai' if (existenceAniBonsaiFrames or existenceWhisBonsaiFrames) else ' ACQ4'))
        print('===========')
        vidRecDict['animalVideo'] = True if (existenceAniBonsaiFrames or existenceAniACQ4Frames) else  False
        vidRecDict['whiskerVideo'] = True if existenceWhisBonsaiFrames else False
        vidRecDict['vidRecSoftware'] = 'Bonsai' if (existenceAniBonsaiFrames or existenceWhisBonsaiFrames) else 'ACQ4'
        return vidRecDict
    ############################################################
    def checkIfDeviceWasRecorded(self, fold, eD, recording, device, startTime = None, npxFolders=[None,None,False]):
        # recLocation =  (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
        # pdb.set_trace()
        #print(recLocation,eD,int(eD))

        if (device == 'RotaryEncoder') : # rotary encoder file is called with number '2' on the behavior setup
            device = 'RotaryEncoder'

        if os.path.exists(recLocation):
            print('%s contains %s for device: %s, ' % (fold, recording, device), end=" ")
        else:
            print('Problem, recording does not exist')
        if device in ['CameraGigEBehavior', 'CameraPixelfly']:
            if int(eD) >= 191104:
                pathToFile = recLocation + device + '/' + 'video_000.ma'
            else:
                pathToFile = recLocation + device + '/' + 'frames.ma'
        elif device == 'PreAmpInput' or device == 'DaqDevice':
            pathToFile = recLocation + '%s.ma' % 'DaqDevice'
        elif device == 'frameTimes':
            pathToFile = recLocation + '%s/%s.ma' % ('CameraGigEBehavior', 'daqResult')
        elif device=='rungLocation':
            pathToFile = self.analysisLocation + '%s_%s_%s_rungPositions.p' % (self.mouse, eD, recording)
        elif device == 'SICaImaging':
            recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            print('  ',recLocation)
            tiffList = glob.glob(recLocation + '*tif')
            tiffList.sort()
            print(tiffList)
            if len(tiffList) > 0:
                print('  YES Ca imaging was acquired with ScanImage')
                return (True, tiffList, recLocation)
            else:
                print('  No Ca imaging with ScanImage here.')
                return (False, [], None)
        elif device in ['GigEAnimalBonsai','Cham3WhiskerBonsai']:
            # recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
            #if self.recStructure != 'simplexNPX' or self.recStructure != 'shank3Behavior':
            #    recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            if '/' in recording: # Use split to separate the string based on '/', this is the condition for repeated recordings
                recordingSplit = recording.split('/')
                if self.recStructure == 'simplexNPX' or self.recStructure == 'shank3Behavior':
                    recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recordingSplit[0] + '/'
                    if int(eD) <= 231201:
                        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
                else:
                    recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            #else:
            #    recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
            print('  check %s data in :' % device , recLocation)
            if device == 'GigEAnimalBonsai':
                fNames = ['videoAnimal_','animalCounterStamp_','animalTimeStamp_']
                # fNames = ['videoAnimal_', 'animalCounterStamp_', 'animalFrameID_']
            elif device == 'Cham3WhiskerBonsai':
                fNames = ['videoWhisker','frameID','timeStamp']
            correspondingFiles = []
            vidFiles = glob.glob(recLocation+ fNames[0]+'*')
            frameIDFiles = glob.glob(recLocation+ fNames[1]+'*')
            timeStampFiles = glob.glob(recLocation+ fNames[2]+'*')
            #print('  ',vidFiles)
            #pdb.set_trace()
            if vidFiles:
                vidF = self.findFileWithMatchingTimeStamp(vidFiles,startTime,fType=fNames[0])
                frameIDF = self.findFileWithMatchingTimeStamp(frameIDFiles,startTime,fType=fNames[1])
                timeStampF = self.findFileWithMatchingTimeStamp(timeStampFiles,startTime,fType=fNames[2])
            else:
                vidF = frameIDF = timeStampF = []
            #pdb.set_trace()
            if len(vidF)==1 and len(frameIDF)==1 and len(timeStampF)==1:
                print(' YES %s was recorded, file %s' % (device,vidF[0]))
                return (True,[vidF[0],frameIDF[0],timeStampF[0]])
            else:
                print('  videoFile, frameIDFile, timeStampFile : ', vidF, frameIDF, timeStampF)
                print('  NO %s or files cannot be found' % device)
                return (False,[])
            #pdb.set_trace()
        elif device == 'EMGsignal':
            recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            bonsaiFiles = glob.glob(recLocation + device + '*')
            #pdb.set_trace()
            bF = self.findFileWithMatchingTimeStamp(bonsaiFiles, startTime)
            if len(bF)==1:
                print('  YES %s was recorded, file %s' % (device,bF[0]))
                return (True,bF[0],None)
            else:
                print(' bonsaiFile : ', bF)
                print('  NO %s or files cannot be found' % device)
                return (False,[],None)
        elif  (device == 'behaviorVideoFrameTimes') or (device == 'animalVideoFrameTimes') or (device=='whiskerVideoFrameTimes'):
            names = [fold,recording,device[:-10]]
            print(names)
            #pdb.set_trace()
            try:
                self.readBehaviorVideoTimeData(names)
            except:
                print(' NO frame times for %s !' % device)
                return (False,[])
            else:
                print(' YES %s was recorded ! ' % device)
                return (True,[])
        elif device == 'NPX':
            recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            #print(fold, eD, recording,npxFolders)
            # pdb.set_trace()
            if npxFolders[2]: # in case of chronic NPX recording
                # pathToFolder = recLocation + npxFolders[1] + '_g0/'
                pathToFolder = recLocation + npxFolders[1] + '/'
                # pathToFile = pathToFolder + npxFolders[1] + '_g0_imec0/' + npxFolders[1] + '_g0_t0.imec0.ap.bin'
                pathToFile = pathToFolder + npxFolders[1] + '_imec0/' + npxFolders[1] + '_t0.imec0.ap.bin'
                # pathToFolder =  recLocation +  npxFolders[1] + '_g0/'
                # pathToFile = pathToFolder +  npxFolders[1] + '_g0_imec0/' + npxFolders[1] + '_g0_t0.imec0.ap.bin'    #'2023/20231213/11_09_00'
            else:
                pathToFolder = self.NPXrawData + fold[:4] + '/' + fold[:10].replace('.','') + '/' + npxFolders[0] + '/'
                pathToFile = pathToFolder +  npxFolders[1]  + '/' +  npxFolders[1] + '_t0.nidq.bin'
        else:
            pathToFile = recLocation + '%s.ma' % device
        print(pathToFile)

        if os.path.isfile(pathToFile):
            if device=='rungLocation':
                return (True, pathToFile)
            elif device == 'NPX':
                print('  YES device %s was acquired' % device)
                return (True, pathToFile, pathToFolder)
            #pdb.set_trace()
            fData = h5py.File(pathToFile, 'r')
            mainFolder=recLocation[:-4]
            config = self.readRecordingMetaInformation(recLocation)
            try:
                kk = fData['data']
            except KeyError:
                print('  NO data exists but device %s was acquired' % device)
                return (False, None,None)
            else:
                print('  YES device %s was acquired' % device)
                return (True, fData, config)
        else:
            print('  NO %s device was acquired' % device)
            return (False, None,None)

    ############################################################
    def attributeDAQTraces(self,daqValues,date,vidRecDict):
        #pdb.set_trace()
        if vidRecDict['vidRecSoftware'] == 'ACQ4':
            # len(daqValues)=5; 0 - record on signal; 1,2 - motorizaton trigger (1 - down, 2 - up); 3 - motorization on signal; 4 - LEDcontrolSignal
            ledDAQControlArray = daqValues[4]  #  control of blinking LED 25 ms on, 25 ms off, period 50 ms
            # daqValues[2] - record signal, on for the entire recording
            # exposureAniDAQArray = daqValues[2]
            return (ledDAQControlArray,None,None)
        elif vidRecDict['vidRecSoftware'] == 'Bonsai':
            if self.recStructure == 'simplexNPX':
                if len(daqValues) == 5:
                    ledDAQControlArray = daqValues[1]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                    exposureWhisDAQArray = daqValues[2]
                    exposureAniDAQArray = daqValues[0]
                    npxSyncSignal = daqValues[4]
                elif len(daqValues) == 7:
                    ledDAQControlArray = daqValues[3]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                    exposureWhisDAQArray = daqValues[4]
                    exposureAniDAQArray = daqValues[2]
                    npxSyncSignal = daqValues[6]
                else:
                    Exception('Different type of experiment! Check the identity and attribution of the DAQ channels!')
            elif self.recStructure == 'shank3Behavior':
                if len(daqValues) == 5:
                    ledDAQControlArray = daqValues[1]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                    exposureWhisDAQArray = daqValues[2]
                    exposureAniDAQArray = daqValues[0]
                    # npxSyncSignal = daqValues[4]
                elif len(daqValues) == 7:
                    ledDAQControlArray = daqValues[3]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                    exposureWhisDAQArray = daqValues[4]
                    exposureAniDAQArray = daqValues[2]
                elif len(daqValues) == 6:
                    ledDAQControlArray = daqValues[3]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                    exposureWhisDAQArray = daqValues[5]
                    exposureAniDAQArray = daqValues[4]
                else:
                    Exception('Different type of experiment! Check the identity and attribution of the DAQ channels for %s experiments!' % self.recStructure )
            elif int(date) >= 230424 and int(date) < 230512:  # foldersRecordings[f][0][:-4]!='2023.04.21':
                ledDAQControlArray = daqValues[3]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                exposureWhisDAQArray = daqValues[4]
                # daqValues[2] - record signal, on for the entire recording
                exposureAniDAQArray = daqValues[2]
            elif int(date) > 230512 :
                ledDAQControlArray = daqValues[3]
                exposureAniDAQArray = daqValues[4]
                exposureWhisDAQArray = daqValues[5]  # pdb.set_trace()
            else:
                ledDAQControlArray = daqValues[0]  # control of blinking LED 25 ms on, 25 ms off, period 50 ms
                exposureWhisDAQArray = daqValues[1]
                # daqValues[2] - record signal, on for the entire recording
                exposureAniDAQArray = daqValues[3]
            return (ledDAQControlArray,exposureWhisDAQArray,exposureAniDAQArray)
    ############################################################
    def checkIfPawPositionWasExtracted(self, fold, eD, recording,DLCinst, videoPrefix='raw_behavior'):

        rec = recording.replace('/', '-')
        if videoPrefix == 'resized':
            fName = self.analysisLocation + '%s_%s_%s*_resized%s.h5' % (self.mouse, fold, rec, DLCinst)
        elif videoPrefix == 'raw_behavior':
            fName = self.analysisLocation + '%s_%s_%s_raw_behavior%s.h5' % (self.mouse, fold, rec, DLCinst)
        elif videoPrefix == 'processed-animal-video':
            fName = self.analysisLocation + '%s_%s_%s_processed-animal-video%s.h5' % (self.mouse, fold, rec, DLCinst)
        elif videoPrefix == 'processed-animalObstacle-video':
            fName = self.analysisLocation + '%s_%s_%s_processed-animalObstacle-video%s.h5' % (self.mouse, fold, rec, DLCinst)
        elif videoPrefix == 'processed-whiskerObstacle-video':
            DegNetwork = DLCinst
            if DegNetwork == 'Obstacle_strategy_2_deepethogram' or 'New_obstacle_strategy_deepethogram' or 'Miss_Obstacle_deepethogram':
                videoName = 'processed-animalObstacle-video'
            else:
                videoName = 'processed-whiskerObstacle-video'
            fName = f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/deg_projects/{DegNetwork}/DATA/{self.mouse}_{fold}_{rec}_{videoName}/{self.mouse}_{fold}_{rec}_{videoName}_labels.csv' #_predictions.csv
        print('Checking whether PAW trajectories were extracted and saved in : ', fName)
        fList = glob.glob(fName)
        #print(fName)
        # pdb.set_trace()
        if len(fList) > 1:
            print('  more than one file exist matching the file pattern %s' % fName)
            return (False, None)
        elif len(fList) == 1:
            print('  YES %s PAW data extraced and saved in %s' % (videoPrefix,fList[0]))
            return (True, fList[0])
        else:
            print('  no extraced PAW data found for %s' % fName)
            return (False, None)

    ############################################################
    def readRawData(self, fold, eD, recording, device, fData, readRawData=True, obstacle=False):
        # recLocation = (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        # pdb.set_trace()
        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
        #print('recording',recording)
        if device == 'RotaryEncoder':
            if int(eD)>230421 and int(eD)<260126:
                idxChanA = 0
                idxChanB = 1
                idxAbs = 2
                idxObs1 = 3
                idxObs2 = 4
                startTimeKey = 'info/2/DAQ/DIC2ChannelA'
                python3 = True
            else:
                idxChanA = 1
                idxChanB = 0
                idxAbs = 2
                idxObs1 = 4
                idxObs2 = 3
                startTimeKey = 'info/2/DAQ/DIC2ChannelA'
                python3 = False
            # data from activity monitor
            #print(len(fData['data']))
            #pdb.set_trace()
            if len(fData['data']) == 1:
                angles = fData['data'][()][0]
            # data during high-res recording
            ppu = 1024
            chanA = fData['data'][idxChanA]
            chanB = fData['data'][idxChanB]
            chanAB = chanA.astype(bool)
            chanBB = chanB.astype(bool)

            bitSequence = (chanAB ^ chanBB) | chanBB << 1
            delta = (bitSequence[1:] - bitSequence[:-1]) % 4
            delta = np.concatenate((np.array([0]), delta))
            delta[delta == 3] = -1
            angles = np.cumsum(-delta) * 360. / (2. * 4. * ppu)

            if obstacle:
                absoluteSignal = fData['data'][idxAbs]
                obstacle1 = fData['data'][idxObs1]
                obstacle2 = fData['data'][idxObs2]
            times = fData['info/1/values'][()]
            try:
                startTime = fData[startTimeKey].attrs['startTime']
            except:
                startTime = os.path.getctime(recLocation)
                monitor = True
            else:
                monitor = False
            if obstacle:
                difference = np.diff(absoluteSignal)  # calculate difference
                obs1UP_idx = np.arange(len(absoluteSignal))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
                #anglesUP = angles[obs1UP_idx]
                #print(anglesUP,np.diff(anglesUP))
                #pdb.set_trace()
                return (angles, times, startTime, monitor,absoluteSignal, obstacle1, obstacle2)
            else:
                return (angles, times, startTime, monitor)

        elif device == 'AxoPatch200_2':
            current = fData['data'][()][0]
            if current.sum()==0:
                current = fData['data'][()][1]
            # pdb.set_trace()
            ephysTimes = fData['info/1/values'][()]
            # imageMetaInfo = self.readMetaInformation(recLocation)
            return (current, ephysTimes)

        elif device == 'AxoPatch200_1':
            current = fData['data'][()][0]
            # pdb.set_trace()
            ephysTimes = fData['info/1/values'][()]
            # imageMetaInfo = self.readMetaInformation(recLocation)
            return (current, ephysTimes)

        elif device == 'EMGsignals' or device== 'DaqDeviceEMG':
            EMG={}
            EMG['nChan']=len(fData['data'][()])
            if EMG['nChan']>0:
                # for i in range(EMG['nChan']):
                #     currentKey=f'current_chan{i}'
                #     EMG[currentKey]=0
                for n in range(EMG['nChan']):
                    currentKey = f'current_chan{n}'
                    EMG[currentKey]=fData['data'][()][n]
            else:
                EMG['current_chan0']=fData['data'][()][0]

            EMG['time'] = fData['info/1/values'][()]
            # imageMetaInfo = self.readMetaInformation(recLocation)

            return EMG

        elif device == 'Imaging':
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = self.readMetaInformation(recLocation)
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'CameraGigEBehavior':
            print('reading raw GigE data ...')
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = self.readRecordingMetaInformation(recLocation)
            print('done')
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'CameraPixelfly':
            print('reading raw Pixelfly data ...', end=" ")
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            xPixelSize = fData['info/3/pixelSize'].attrs['0']
            yPixelSize = fData['info/3/pixelSize'].attrs['1']
            xSize = fData['info/3/region'].attrs['2']
            ySize = fData['info/3/region'].attrs['3']
            imageMetaInfo = np.array([xSize * xPixelSize, ySize * yPixelSize, xPixelSize, yPixelSize])
            print('done')
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'PreAmpInput' or device == 'frameTimes' or device == 'DaqDeviceEphys' or device == 'DaqDevice' or device=='DaqDeviceEMG':
            # pdb.set_trace()
            values = fData['data'][()]
            valueTimes = fData['info/1/values'][()]
            sT = []
            for key, val in fData['info/2/DAQ'].items():
                try :
                    sT.append(fData['info/2/DAQ/'+key].attrs['startTime'])
                except:
                    pass
                #print(fData['info/2/DAQ/'+key].attrs['startTime']
                #pdb.set_trace()
                #sT.append(fData['info/2/DAQ'][key]['startTime'])
            unique_elements = np.unique(np.asarray(sT))
            # pdb.set_trace()
            if len(unique_elements) == 1:
                print('all start times are the SAME')
                startTime = unique_elements
            else:
                print('start times are DIFFERENT')
                pdb.set_trace()

            #startTimes = fData['info/2/DAQ']
            return (values, valueTimes, startTime)

        elif device == 'pawTraces':
            pawF = h5py.File(fData, 'a')
            pawTraces = pawF['df_with_missing']['table'][()]
            pawTracesA = pawTraces.view((float, (len(pawTraces[0][1]) + 1)))
            pawTracesA[:, 0] = np.arange(len(pawTraces))

            pFileName = '%s*.pickle' % fData[:-3]
            pfList = glob.glob(pFileName)
            print(pfList)
            if len(pfList) == 1:
                #pdb.set_trace()
                pawMetaData = pickle.load(open(pfList[0], 'rb'))
            else:
                pawMetaData = None
            return (pawTracesA, pawMetaData)
        elif device in ['GigEAnimalBonsai','Cham3WhiskerBonsai']: # ['videoAnimal_','animalCounterStamp_','animalTimeStamp_']
            videoFile = fData[0]
            counterStamp = np.loadtxt(fData[1])
            timeStamp = np.loadtxt(fData[2])
            return (videoFile,counterStamp,timeStamp)
        elif device == 'EMGsignal':
            df = pd.read_csv(fData)
            return df
        elif device == 'WhiskerTouch':
            print(fData)
            # pdb.set_trace()
            whisker_touch = pd.read_csv(fData)

            return whisker_touch

    ############################################################
    def videoToArray(self,videoFileName):
        # get properties
        self.video = cv2.VideoCapture(videoFileName)
        self.Vlength = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.Vwidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Vheight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Vfps = self.video.get(cv2.CAP_PROP_FPS)
        #if outputProps:
        print('Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (self.Vlength, self.Vwidth, self.Vheight, self.Vfps))

        if not self.video.isOpened():
            print('Could not open video')
            sys.exit()
        frames = np.empty((self.Vlength, self.Vheight,self.Vwidth))
        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        frames[0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nFrame = 1
        while True:
            ok, img = self.video.read()
            if ok:
                frames[nFrame] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                nFrame += 1
            else:
                break
        print(nFrame, ' where read from file ',videoFileName)
        return frames
    ############################################################
    def findFileWithMatchingTimeStamp(self,fileList,timeStamp,fType=None):
        #jitter = 75. # difference in sec
        match = []
        timeDiffs = []
        # pdb.set_trace()
        for i in range(len(fileList)):
            #ti_b = os.stat(fileList[i]).st_birthtime
            #print(fileList[i][-21:-13],fileList[i][-12:-4])
            #(fileList[i][-21:-13].split('-'))
            date = [int(d) for d in fileList[i][-23:-13].split('-')]
            tt = [int(t) for t in fileList[i][-12:-4].split('_')]

            date_time = datetime.datetime(date[0],date[1],date[2],tt[0],tt[1],tt[2])
            creationTime = time.mktime(date_time.timetuple())
            # print(creationTime,timeStamp,np.abs(creationTime-timeStamp))
            #ti_c = os.path.getctime(fileList[i])
            #ti_m = os.path.getmtime(fileList[i])
            # pdb.set_trace()
            # timeDiffs.append(np.abs(creationTime-timeStamp))
            timeDiffs.append(creationTime - timeStamp)
            #print('  Time difference between ACQ4 and video file time-stamps : ', creationTime-timeStamp,creationTime,timeStamp)
            #if ((creationTime-timeStamp)<jitter) and ((creationTime-timeStamp)>0.):
            #    print('  Added : Time difference between ACQ4 and video file time-stamps : ', creationTime-timeStamp,creationTime,timeStamp)
            #    match.append(fileList[i])
        smallest_positive_timeDiff = min(dd for dd in timeDiffs if dd >= 0.)
        indexSmallest = timeDiffs.index(smallest_positive_timeDiff)
        match.append(fileList[indexSmallest])
        # pdb.set_trace()
        if fType[:5] == 'video':
            print(' Chosen time difference between ACQ4 and video file time-stamps : ', timeDiffs[indexSmallest], ' All time differences : ', timeDiffs)
        # DO MORE if match list is empty
        if len(match)==0:
            # Find the smallest value in the list
            min_value = min(timeDiffs)
            # Get the index of the smallest value
            min_index = timeDiffs.index(min_value)
            for i in range(len(fileList)):
                print('index, time-difference, file name : %s %s %s ' % (i,timeDiffs[i],fileList[i]))
            print('file with shortest time time difference (%s s) is the video file %s' % ((creationTime-timeStamp), fileList[min_index]))
            user_input = input('do you want to include that file? (y/n) otherwise specify index of the file to include (e.g. 0, 1, ...)')
            if user_input == 'y':
                match.append(fileList[min_index])
            elif user_input.isdigit():
                match.append(fileList[int(user_input)])

        return match
    ############################################################
    def readMetaInformation(self, filePath):
        # convert to um
        conversion = 1.E6
        config = readConfigFile(filePath + '.index')
        pixWidth = config['.']['Scanner']['program'][0]['scanInfo']['pixelWidth'][0] * conversion
        pixHeight = config['.']['Scanner']['program'][0]['scanInfo']['pixelHeight'][0] * conversion
        dimensionXY = np.array(config['.']['Scanner']['program'][0]['roi']['size']) * conversion
        position = np.array(config['.']['Scanner']['program'][0]['roi']['pos']) * conversion

        if pixWidth == pixHeight:
            deltaPix = pixWidth
        else:
            print('Pixel height and width are not equal.')
            sys.exit(1)

        # print r'dimensions (x,y, pixelsize in um) : ', np.hstack((dimensionXY,deltaPix))
        return np.hstack((position, dimensionXY, deltaPix))  # self.h5pyTools.createOverwriteDS(dataGroup,dataSetName,hstack((dimensionXY,deltaPix)))

    ############################################################
    def readRecordingMetaInformation(self, filePath):
        config = readConfigFile(filePath + '.index')
        #starttime = config['.']['startTime']
        # pdb.set_trace()
        return config
    ############################################################
    def readRecordingConfig(self, fold, eD, recording):
        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
        found = any(s in recording for s in self.listOfSequenceExperiments)
        if found:
            mainFolder = recLocation[:-4]
        else:
            mainFolder = recLocation
        #pdb.set_trace()
        config = readConfigFile(mainFolder + '.index')
        return config
    ############################################################
    def saveLEDPositionCoordinates(self, groupNames, coordinates, auto=False):
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # pdb.set_trace()
        if not auto:
            self.h5pyTools.createOverwriteDS(grpHandle, 'LEDcoordinates', np.column_stack((coordinates[1], coordinates[2])), [['nLED', coordinates[0]], ['circleRadius', coordinates[3]], ['spacing', coordinates[4]], ['theta', coordinates[5]]])
        else:
            self.h5pyTools.createOverwriteDS(grpHandle, 'LEDcoordinates', np.column_stack((coordinates[1], coordinates[2])), [['nLED', coordinates[0]], ['circleRadius', coordinates[3]]])
    ############################################################
    def saveGenericData(self, groupNames, data, entryName):
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, entryName , data)

    ############################################################
    def readgGenericData(self, groupNames, entryName):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        #pdb.set_trace()
        data = self.f[grpName + '/%s' % entryName][()]
        return data

    ############################################################
    def readLEDPositionCoordinates(self, currentGroupName, auto=False):

        temp = self.f[currentGroupName + '/LEDcoordinates'][()]
        posX = temp[:, 0]
        posY = temp[:, 1]
        nLED = self.f[currentGroupName + '/LEDcoordinates'].attrs['nLED']
        circleRadius = self.f[currentGroupName + '/LEDcoordinates'].attrs['circleRadius']
        if not auto:
            spacing = self.f[currentGroupName + '/LEDcoordinates'].attrs['spacing']
            theta = self.f[currentGroupName + '/LEDcoordinates'].attrs['theta']
            coordinates = np.array([nLED, posX, posY, circleRadius,spacing,theta], dtype=object)  # type is object to allow for different data structure of the entries, i.e., single number and array
        else:
            coordinates = np.array([nLED, posX, posY, circleRadius])
        return coordinates

    ############################################################
    # foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r
    def checkForLEDPositionCoordinates(self, date, folder, recordings, r, key=None, auto=False):
        if key is None:
            dictKey = 'LEDinVideo'
        else:
            dictKey = key
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        currentGroupNames = [date, recordings[r], dictKey]
        #print(currentGroupNames)
        #pdb.set_trace()
        (currentGroupName, currentGrpHandle) = self.h5pyTools.getH5GroupName(self.f, currentGroupNames)
        # check if coordinates for current recording exist already
        try:
            currentLEDcoordinates = self.readLEDPositionCoordinates(currentGroupName, auto)
        except KeyError:
            currentCoodinatesExist = False
        else:
            print('%s LED roi coordinates for current recording exist' % dictKey)
            currentCoodinatesExist = True
            return (currentCoodinatesExist, currentLEDcoordinates)

        if r > 0:
            previousGroupNames = [date, recordings[r - 1], dictKey]
            (previousGroupName, previousGrpHandle) = self.h5pyTools.getH5GroupName(self.f, previousGroupNames)
            try:
                previousLEDcoordinates = self.readLEDPositionCoordinates(previousGroupName)  # self.f[previousGroupName+'/LEDcoordinates'][()]
            except KeyError:
                pass
            else:
                print('LED roi coordinates for previous recording exist')

                return (currentCoodinatesExist, previousLEDcoordinates)
        #use previous recording LED coordinates for the same date
        elif len(recordings)==1 and '000' not in recordings[0]:
            # pdb.set_trace()
            preRecName=recordings[0][:len(recordings[0])-1]+str(int(recordings[0][len(recordings[0])-1:])-1)
            previousGroupNames = [date, preRecName, dictKey]
            (previousGroupName, previousGrpHandle) = self.h5pyTools.getH5GroupName(self.f, previousGroupNames)
            try:
                previousLEDcoordinates = self.readLEDPositionCoordinates(previousGroupName)  # self.f[previousGroupName+'/LEDcoordinates'][()]
            except KeyError:
                pass
            else:
                print('LED roi coordinates for previous recording exist')

                return (currentCoodinatesExist, previousLEDcoordinates)
        print('NO %s LED roi coordinates exist' % dictKey)
        # pdb.set_trace()
        return (currentCoodinatesExist, None)

    ############################################################
    def checkForErroneousFramesIdx(self, date, folder, recordings, r, determineAgain=False):
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        currentGroupNames = [date, recordings[r], 'erroneousAnimalVideoFrames']
        #print(currentGroupNames)
        #pdb.set_trace()
        try:
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, currentGroupNames)
            idxExc = self.f[grpName + '/idxToExclude'][()]
            # print(type(idxExc))
            idxExc = np.array(idxExc, dtype=int)  # check if coordinates for current recording exist already
        except KeyError:
            print('idx of of erroneous frames DOES NOT exist')
            excludeIdxExist = False
            canBeUsed = True
            return (excludeIdxExist, None, canBeUsed)
        else:
            print('idx of erroneous frames exist')
            excludeIdxExist = True
            try:
                canBeUsed = self.f[grpName + '/canBeUsed'][()][0]
            except KeyError:
                canBeUsed = True
            if np.array_equal(idxExc, np.array([-1])):
                idxExc = np.array([], dtype=int)
            if determineAgain:
                return (False, idxExc,canBeUsed)
            else:
                return (excludeIdxExist, idxExc,canBeUsed)

    ############################################################
    def saveErroneousFramesIdx(self, groupNames, idxToExclude,canBeUsed=True):
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        # print(groupNames)
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # print(grpName,grpHandle)
        if len(idxToExclude) == 0:
            idxToExclude = np.array([-1])  # pdb.set_trace(
        self.h5pyTools.createOverwriteDS(grpHandle, 'idxToExclude', idxToExclude)
        self.h5pyTools.createOverwriteDS(grpHandle, 'canBeUsed', np.array([canBeUsed]))
        self.f.flush()
        print('saved successfully', idxToExclude)

    ############################################################
    # idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo
    def saveBehaviorVideoTimeData(self, groupNames, idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo):
        # framesDuringRecording, startEndFrameTime, startEndFrameIdx, imageMetaInfo)
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # self.h5pyTools.createOverwriteDS(grpHandle,'behaviorFrames',len(frames))
        #pdb.set_trace()
        self.h5pyTools.createOverwriteDS(grpHandle, 'indexVideo', videoIdx, ['startTime', imageMetaInfo])
        self.h5pyTools.createOverwriteDS(grpHandle, 'indexTimePoints', idxTimePoints)
        self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposureTime', startEndExposureTime)
        self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposurepIndex', startEndExposurepIdx)
        self.h5pyTools.createOverwriteDS(grpHandle, 'frameDropExcludeSummary', frameSummary)

    ############################################################
    def readBehaviorVideoTimeData(self, groupNames):
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        #pdb.set_trace()
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # pdb.set_trace()
        videoIdx = self.f[grpName + '/indexVideo'][()]
        imageMetaInfo = self.f[grpName + '/indexVideo'].attrs['startTime']
        idxTimePoints = self.f[grpName + '/indexTimePoints'][()]
        startEndExposureTime = self.f[grpName + '/startEndExposureTime'][()]
        startEndExposurepIdx = self.f[grpName + '/startEndExposurepIndex'][()]
        frameSummary = self.f[grpName + '/frameDropExcludeSummary'][()]
        return (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo)

    ############################################################
    def getBehaviorVideoFrames(self, groupNames):
        (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        print(grpName)
        #pdb.set_trace()  [date,rec,'behavior_video']
        try:
            firstLastRecordedFrame = self.f[grpName + '/firstLastRecordedFrame'][()]
        except:  # before, first and last name was stored under a different name
            groupNames[2] = 'behavior_video'
            (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
            firstLastRecordedFrame = self.f[grpName + '/firstLastRecordedFrame'][()]
        else:
            pass

        return firstLastRecordedFrame

    ############################################################
    # self.saveBehaviorVideoData([date,rec,'behavior_video'],  framesRaw,videoIdx,startEndExposureTime, imageMetaInfo)
    # [foldersRecordings[f][0], foldersRecordings[f][2][r],'behavior_video']
    def saveBehaviorVideoFrames(self, groupNames,  framesRaw, videoIdx ):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        #print(grpName)
        #self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposureTime', startEndExposureTime,['imageMetaInfo', imageMetaInfo])
        self.h5pyTools.createOverwriteDS(grpHandle, 'firstLastRecordedFrame', np.array((framesRaw[videoIdx[0]],framesRaw[videoIdx[-1]])))

    ############################################################
    def saveImageStack(self, frames, fTimes, imageMetaInfo, groupNames, motionCorrection=[]):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImaging', frames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImagingTime', fTimes)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImagingField', imageMetaInfo)
        if len(motionCorrection) > 1:
            self.h5pyTools.createOverwriteDS(grpHandle, 'motionCoordinates', motionCorrection)

    ############################################################
    def readImageStack(self, groupNames):
        (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        frames = self.f[grpName + '/caImaging'][()]
        fTimes = self.f[grpName + '/caImagingTime'][()]
        imageMetaInfo = self.f[grpName + '/caImagingField'][()]
        return (frames, fTimes, imageMetaInfo)
    ############################################################
    def saveEphysData(self, groupNames, ephysDict):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        #pdb.set_trace()
        #print(grpName)
        for key, value in ephysDict.items():
            #print('saving %s' % key, value)
            self.h5pyTools.createOverwriteDS(grpHandle, key, value)

    ############################################################
    def readEphysData(self, groupNames):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        ephysDict = {}
        keys = self.f[grpName].keys()
        #pdb.set_trace()
        for key in keys:
            #print('loading %s' % key)
            ephysDict[key] = self.f[grpName + '/' + key][()]

        return ephysDict

    ############################################################
    def saveWalkingActivity(self, angularSpeed, linearSpeed, wTimes, angles, aTimes, startTime, monitor, groupNames):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'angularSpeed', angularSpeed, ['monitor', monitor])
        self.h5pyTools.createOverwriteDS(grpHandle, 'linearSpeed', linearSpeed)
        self.h5pyTools.createOverwriteDS(grpHandle, 'walkingTimes', wTimes, ['startTime', startTime])
        self.h5pyTools.createOverwriteDS(grpHandle, 'anglesTimes', np.column_stack((aTimes, angles)), ['startTime', startTime])

    ############################################################
    def saveObstacleWalkingActivity(self, angularSpeed, linearSpeed, wTimes, angles, aTimes, startTime, monitor, absoluteSignal, obstacle1, obstacle2,groupNames):
        self.saveWalkingActivity(angularSpeed, linearSpeed, wTimes, angles, aTimes, startTime, monitor, groupNames)
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'absoluteSignal', np.column_stack((aTimes, absoluteSignal)), ['startTime', startTime])
        self.h5pyTools.createOverwriteDS(grpHandle, 'obstacle1', np.column_stack((aTimes, obstacle1)), ['startTime', startTime])
        self.h5pyTools.createOverwriteDS(grpHandle, 'obstacle2', np.column_stack((aTimes, obstacle2)), ['startTime', startTime])

    ############################################################
    def getWalkingActivity(self, groupNames, saveToDict = [False,None]):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        #print(grpName)
        if saveToDict[0]:
            saveToDict[1][groupNames[-1]] = self.hdf5_to_dict(self.f[grpName])
            return
            #print(dd)
            #pdb.set_trace()
        else:
            angularSpeed = self.f[grpName + '/angularSpeed'][()]
            #pdb.set_trace()
            monitor = self.f[grpName + '/angularSpeed'].attrs['monitor']
            linearSpeed = self.f[grpName + '/linearSpeed'][()]
            wTimes = self.f[grpName + '/walkingTimes'][()]
            startTime = self.f[grpName + '/walkingTimes'].attrs['startTime']
            angleTimes = self.f[grpName + '/anglesTimes'][()]
            return (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes)
    ############################################################
    def hdf5_to_dict(self,group):
        result = {}
        for key, item in group.items():
            #print(key, item)
            if isinstance(item, h5py.Group):
                print('found group')
                result[key] = self.hdf5_to_dict(item)
            else:
                #print('found data-set',)
                result[key] = np.array(item[()])
                #print('saved to dict')
        #print(result)
        return result

    ############################################################
    def getObstacleWalkingActivity(self, groupNames, returnData = 'oldList'):
        if returnData == 'newDict':
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
            walkingDict = self.hdf5_to_dict(self.f[grpName])
            return walkingDict
        elif returnData == 'oldList':
            (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes) = self.getWalkingActivity(groupNames)
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
            absoluteSignal = self.f[grpName + '/absoluteSignal'][()]
            obstacle1 = self.f[grpName + '/obstacle1'][()]
            obstacle2 = self.f[grpName + '/obstacle2'][()]
            return (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes, absoluteSignal, obstacle1, obstacle2)

    ############################################################
    def getPawRungPickleData(self, date, rec):
        frontpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (self.mouse, date, rec), 'rb'))
        hindpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (self.mouse, date, rec), 'rb'))
        rungs = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (self.mouse, date, rec), 'rb'))
        return (frontpawPos, hindpawPos, rungs)

    ############################################################
    def readMetaDataFileAndReadSetting(self, tiffFile, keyWord):
        metData = ScanImageTiffReader(tiffFile).metadata()
        #pdb.set_trace()
        keyWordIdx = metData.find(keyWord)
        splitString = re.split('=|\n', metData[keyWordIdx:])
        keyWordParameter = float(splitString[1])
        return keyWordParameter

    ############################################################
    def readTimeStampOfFrame(self, tiffFileObj, nFrame):
        # frameNumberAcqMode
        # numeric, number of frame, counted from beginning of acquisition mode
        #
        # frameNumberAcq
        # numeric, number of frame in current acquisition
        #
        # acqNumber
        # numeric, number of current acquisition
        #
        # epochAcqMode
        # string, time of the acquisition of the acquisiton of the first pixel in the current acqMode; format: output of datestr(now) '25-Jul-2014 12:55:21'
        #
        # frameTimestamp
        # [s] time of the first pixel in the frame passed since acqModeEpoch
        #
        # acqStartTriggerTimestamp
        # [s] time of the acq start trigger for the current acquisition
        #
        # nextFileMarkerTimestamp
        # [s] time of the last nextFileMarker recorded. NaN if no nextFileMarker was recorded

        desc = tiffFileObj.description(nFrame)
        keyWordIdx = desc.find('epoch')  # string, time of the acquisition of the acquisiton of the first pixel in the current acqMode; format: output of datestr(now) '25-Jul-2014 12:55:21'
        dateString = re.split('\[|\]', desc[keyWordIdx:])
        dateIdv = dateString[1].split()
        # print(dateIdv)
        unixStartTime = int(datetime.datetime(int(dateIdv[0]), int(dateIdv[1]), int(dateIdv[2]), int(dateIdv[3]), int(dateIdv[4]), int(float(dateIdv[5]))).strftime('%s'))
        #
        keyWordIdx = desc.find('frameTimestamps_sec')  # [s] time of the first pixel in the frame passed since acqModeEpoch
        splitString = re.split('=|\n', desc[keyWordIdx:])
        frameTimestamps = float(splitString[1])

        keyWordIdx = desc.find('acqTriggerTimestamps_sec')  # [s] time of the acq start trigger for the current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        acqTriggerTimestamps = float(splitString[1])

        keyWordIdx = desc.find('frameNumberAcquisition')  # numeric, number of frame in current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        frameNumberAcquisition = int(splitString[1])

        keyWordIdx = desc.find('acquisitionNumbers')  # numeric, number of current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        acquisitionNumbers = int(splitString[1])

        unixFrameTime = unixStartTime + frameTimestamps
        # print(tiffFile,unixTime)
        return ([frameNumberAcquisition, acquisitionNumbers, unixStartTime, unixFrameTime, frameTimestamps, acqTriggerTimestamps])
    ############################################################
    def getRawCalciumImagingData(self, tiffList,saveDir):
        imagingData = []
        timeStamps = []
        for i in range(len(tiffList)):
            # data = ScanImageTiffReader(tiffPaths[i]).data()
            tiffFileObject = ScanImageTiffReader(tiffList[i])
            data = tiffFileObject.data()
            # pdb.set_trace()
            fN = np.shape(data)[0]
            # frameNumbers.append(fN)
            for n in range(fN):
                timeStamps.append(self.readTimeStampOfFrame(tiffFileObject, n))
            timeStampsASingle = np.asarray(timeStamps)
            imagingData.append([i, data, timeStampsASingle])
        timeStampsA = np.asarray(timeStamps)
        return (imagingData, timeStampsA)  # np.save(saveDir+'/suite2p/plane0/timeStamps.npy',timeStampsA)

    ############################################################
    def getAnalyzedCaImagingData(self, analysisLocation, tiffList, specificTiffList):

        timeStamps = []
        for i in range(len(tiffList)):
            zF = self.readMetaDataFileAndReadSetting(tiffList[i], 'scanZoomFactor')
            if i == 0:
                zFold = zF
            else:
                if zF != zFold:
                    print('scanZoomFactor is not the same between recordings!')
            tiffFileObject = ScanImageTiffReader(tiffList[i])
            timeS = self.readTimeStampOfFrame(tiffFileObject, 0)
            timeStamps.append(timeS[3])
        #
        #pdb.set_trace()
        nDirs = 0
        for name in glob.glob(analysisLocation+'_suite2p_%s' % specificTiffList[0][2]):
            caAnalysisLocation = name
            print(name)
            nDirs+=1
        if nDirs > 1:
            print('There are more than one matching directory!')
            pdb.set_trace()
        #caAnalysisLocation = eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p/'
        if os.path.isdir(caAnalysisLocation):
            ops = np.load(caAnalysisLocation + '/suite2p/plane0/ops.npy',allow_pickle=True)
            ops = ops.item()
            nframes = ops['nframes']
            meanImg = ops['meanImg']
            try:
                meanImgE = ops['meanImgE']
            except KeyError:
                print('no enhanced image available ... using normal image')
                meanImgE = np.copy(meanImg)
            # pdb.set_trace()
            return (nframes, meanImg, meanImgE, zF, timeStamps)
        else:
            print('Ca imaging data has not been analyzed with suite2p yet!')

    ############################################################
    def extractAndSaveCaTimeStamps(self, dataDir, saveDir, tiffPaths):
        (_, timeStampsA) = self.getRawCalciumImagingData(tiffPaths, saveDir)
        # if gcampVersion == 'gcamp7f':

        np.save(saveDir + '/suite2p/plane0/timeStamps.npy', timeStampsA)
        # elif gcampVersion == 'jrgeco':
        #     np.save(saveDir + '/chan2/suite2p/plane0/timeStamps.npy', timeStampsA)

    ############################################################
    def getCaImagingRoiData(self, caAnalysisLocation, tiffList,gcampVersion):
        # frameNumbers = []
        # timeStamps = []
        # for i in range(len(tiffList)):
        #    data = ScanImageTiffReader(tiffList[i]).data()
        #    fN = np.shape(data)[0]
        #    frameNumbers.append(fN)
        #    for n in range(fN):
        #        timeStamps.append(self.readTimeStampOfRecording(tiffList[i],n))

        # pdb.set_trace()

        if os.path.isdir(caAnalysisLocation):
            timeStamps = np.load(caAnalysisLocation + '/suite2p/plane0/timeStamps.npy')
            F = np.load(caAnalysisLocation + '/suite2p/plane0/F.npy')
            if len(timeStamps) == 2*np.shape(F)[1]:
                timeStamps = timeStamps[::2]
            Fneu = np.load(caAnalysisLocation + '/suite2p/plane0/Fneu.npy')
            ops = np.load(caAnalysisLocation + '/suite2p/plane0/ops.npy',allow_pickle=True)
            ops = ops.item()
            iscell = np.load(caAnalysisLocation + '/suite2p/plane0/iscell.npy')
            stat = np.load(caAnalysisLocation + '/suite2p/plane0/stat.npy',allow_pickle=True)
            spks = np.load(caAnalysisLocation + '/suite2p/plane0/spks.npy',allow_pickle=True)

            # pdb.set_trace()
            nRois = np.arange(len(F))
            realCells = (iscell[:, 0] == 1)
            nRois = nRois[realCells]
            Fluo = F[realCells] - 0.7 * Fneu[realCells]  # substract neuropil data
            stat = stat[realCells]
            # pdb.set_trace()
            if gcampVersion=='jrgeco':
                return (Fluo, nRois, ops, timeStamps, stat, spks, Fneu,iscell,F)
            else:
                return (Fluo, nRois, ops, timeStamps, stat)
    ############################################################
    def saveTif(self, frames, mouse, date, rec, norm=None):
        img_stack_uint8 = np.array(frames, dtype=np.uint8)
        if norm:
            tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack%s.tif' % (mouse, date, rec, norm), img_stack_uint8)
        else:
            tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def readTif(self, frames, mouse, date, rec):
        img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def readPawTrackingData(self, date, rec,expDate, DLCinstance, obstacle=None,obstacleVideo=False,returnData='oldList'):
        if DLCinstance == None:
            DLCinstance = self.analysisConfig['pawTrajectories']['DLCinstance']#[expDate]
        print(DLCinstance)
        if DLCinstance is None:
            print('PROBLEM, DLCinstance has not been defined yet. Run outlier extraction script before.')
            pdb.set_trace()
        rec = rec.replace('/', '-')
        if obstacleVideo:
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'obstacleTrackingData', DLCinstance])
        else:
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'pawTrackingData',DLCinstance])
        #pdb.set_trace()
        if returnData == 'newDict':
            pawTrackDict = self.hdf5_to_dict(self.f[grpName])
            #pdb.set_trace()
            return pawTrackDict
        elif returnData == 'oldList':
            # pdb.set_trace()
            rawPawPositionsFromDLC = self.f[grpName + '/rawPawPositionsFromDLC'][()]
            # self.h5pyTools.createOverwriteDS(grpHandle, 'croppingParameters', np.array(cropping))
            # pdb.set_trace()
            croppingParameters = self.f[grpName + '/croppingParameters'][()]
            #if obstacle!=None:
            #pdb.set_trace()
            if 'jointNames' in self.f[grpName]:
                labelsName=self.f[grpName + '/jointNames'][()]
            else : 
                labelsName = ['FL','FR','HL','HR']
            pawTrackingOutliers = []
            jointNamesFramesInfo = []
            pawSpeed = []
            rawPawSpeed = []
            cPawPos = []
            jointNames=[]
            if obstacle==None:
                # pawIdx=[0,1,2,3]
                # pawIdx = [1,2,3,4]
                pawIdx = [7,8,9,10]
            elif obstacle=='bot':
                pawIdx = [4,5,6,7]
                #pawIdx=[8,9,10,11]
            elif obstacle == 'all':
                pawIdx = range(len(labelsName))
            elif isinstance(obstacle, list):
                pawIdx = obstacle
            #pdb.set_trace()
            for i in pawIdx:
                pTTemp = self.f[grpName + '/pawTrackingOutliers%s' % i][()]
                pawTrackingOutliers.append(pTTemp)
                jNTemp = 'paw %s' % i #self.f[grpName + '/pawTrackingOutliers%s' % i].attrs['PawID']
                jointNamesFramesInfo.append(jNTemp)
                pStemp = self.f[grpName + '/clearedPawSpeed%s' % i][()]
                pawSpeed.append(pStemp)
                pPPtemp = self.f[grpName + '/clearedXYPos%s' % i][()]
                cPawPos.append(pPPtemp)
                rPStemp = self.f[grpName + '/rawPawSpeed%s' % i][()]
                rawPawSpeed.append(rPStemp)
                #if obstacle:
                jointNames.append(labelsName[i])
                # print(jointNames)
                #if obstacle!=None and i == 0:
                 #   recStartTime = self.f[grpName + '/clearedPawSpeed%s' % i].attrs['recStartTime']
                #elif obstacle!=None and i==8:
                recStartTime = self.f[grpName + '/clearedPawSpeed%s' % i].attrs['recStartTime']
            return (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters,jointNames)
            #if obstacle!=None:
            #    return (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters,jointNames)
            #else:
            #    return (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters,jointNames)
    ############################################################
    def findIdxForBottomPaws(self, vidRecDict, jointNames):
        requiredForSwingStance = [[b'front',b'left'],[b'front',b'right'],[b'hind',b'left'],[b'hind',b'right']]
        pawIdx = []
        try :
            index = next(i for i, s in enumerate(jointNames)  if b'bottom' in s)
        except :
            print('  Animal video ONLY contains bottom view.')
        else:
            for n in range(len(requiredForSwingStance)):
                index = next(i for i, s in enumerate(jointNames) if b'bottom' in s and requiredForSwingStance[n][0] in s and requiredForSwingStance[n][1] in s)
                pawIdx.append(index)

        return (pawIdx)


    ############################################################
    # savePawTrackingData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],DLCinstance,pawTrackingOutliers,pawMetaData,startEndExposureTime,imageMetaInfo,generateVideo=False)
    def savePawTrackingData(self, mouse, date, rec, DLCinstance, pawPositions, pawTrackingOutliers, pawMetaData, startEndExposureTime, startTime, generateVideo=True):
        # pdb.set_trace()
        jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
        jointIdx = pawMetaData['data']['DLC-model-config file']['all_joints']
        cropping = pawMetaData['data']['cropping_parameters']
        print('cropping parameters', cropping)
        nJoints = len(jointNames)
        # pdb.set_trace()
        rec = rec.replace('/', '-')
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'pawTrackingData', DLCinstance])
        self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawPositionsFromDLC', pawPositions)
        self.h5pyTools.createOverwriteDS(grpHandle, 'croppingParameters', np.array(cropping))
        self.h5pyTools.createOverwriteDS(grpHandle, 'jointNames', np.array(jointNames))
        self.h5pyTools.createOverwriteDS(grpHandle, 'jointIndicies', np.array(jointIdx))
        timeArray = np.average(startEndExposureTime,axis=1) # use the 'middle' of the exposure time as time-point of the frame
        for i in range(nJoints):
            pawMask = pawTrackingOutliers[i][3]
            #pdb.set_trace()
            rawPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)])) ** 2) / np.diff(timeArray)
            rawSpeedTime = (timeArray[:-1] + timeArray[1:]) / 2.
            clearedPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)][pawMask])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)][pawMask])) ** 2) / np.diff(timeArray[pawMask])
            clearedPawXSpeed = np.diff(pawPositions[:, (i * 3 + 1)][pawMask]) / np.diff(timeArray[pawMask])
            clearedPawYSpeed = np.diff(pawPositions[:, (i * 3 + 2)][pawMask]) / np.diff(timeArray[pawMask])
            clearedSpeedTime = (timeArray[pawMask][:-1] + timeArray[pawMask][1:]) / 2.
            clearedPosIdx = np.arange(len(pawPositions))[pawMask]
            clearedXYPos = np.column_stack((timeArray[pawMask], pawPositions[:, (i * 3 + 1)][pawMask], pawPositions[:, (i * 3 + 2)][pawMask], clearedPosIdx))
            clearedSpeedIdx = np.array((clearedPosIdx[:-1] + clearedPosIdx[1:]) / 2., dtype=int)
            #pdb.set_trace()
            self.h5pyTools.createOverwriteDS(grpHandle, 'pawTrackingOutliers%s' % i, pawTrackingOutliers[i][3],[['PawID', jointNames[i]],['pawTrackingOutliers',np.array([ pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]])]])#, pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
            # pawTrackingOutliers.append([i, tot, correct, correctIndicies, jointNames[i], onePawData, onePawDataTmp, frDispl, frDisplOrig])
            #['PawID', [jointNames[i], pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
            self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawSpeed%s' % i, np.column_stack((rawSpeedTime, rawPawSpeed)), ['recStartTime', startTime])
            self.h5pyTools.createOverwriteDS(grpHandle, 'clearedPawSpeed%s' % i, np.column_stack((clearedSpeedTime, clearedPawSpeed, clearedPawXSpeed, clearedPawYSpeed, clearedSpeedIdx)),
                                             ['recStartTime', startTime])
            self.h5pyTools.createOverwriteDS(grpHandle, 'clearedXYPos%s' % i, clearedXYPos)  # pdb.set_trace()
        if generateVideo:
            colors = [(180, 119,31 ),(14, 127, 255),(44, 160, 44),(40, 39,214)]#[44, 160, 44],[214,39,40]]
            fps = 80
            width = 800
            heigth = 600
            #colors = [(255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
            indicatorPositions = [(270, 15), (270, 35), (240, 35), (240, 15)]
            sourceVideoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
            outputVideoFileName = self.analysisLocation + '%s_%s_%s_paw_tracking.avi' % (mouse, date, rec)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
            # print('1',fourcc,fps,width,heigth)
            out = cv2.VideoWriter(outputVideoFileName, fourcc, fps, (width, heigth))
            source = cv2.VideoCapture(sourceVideoFileName)
            nFrame = 0
            pos = [[],[],[],[]]
            maxLength = 20
            value = 30
            alpha=0.1
            while (source.isOpened()):
                ret, frame = source.read()
                grey_new = np.where((255 - frame) < value,255,frame+value)
                if ret == True:
                    # cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i],4), (10,20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
                    for i in range(4):
                        # print(int(pawPositions[nFrame, 3 * i + 1]+0.5),int(pawPositions[nFrame, 3 * i + 2]+0.5))
                        pos[i].append([cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5)])
                        #overlay = grey_new.copy()
                        for k in range(len(pos[i])-1):
                            cv2.line(grey_new, pos[i][k], pos[i][k+1], colors[i], thickness=2)
                            #overlay = cv2.addWeighted(overlay, 0.1 , grey_new, 1 - alpha, 0)
                            #grey_new = overlay.copy()
                        #cv2.drawMarker(frame, (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5)), colors[i],
                        #               cv2.MARKER_CROSS, 20, 2)
                        cv2.circle(grey_new, (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5)),4, colors[i],-1)
                        #if nFrame in pawTrackingOutliers[i][3]:
                        #    cv2.circle(frame, indicatorPositions[i], 7, (0, 255, 0), -1)
                        #else:
                        #    cv2.circle(frame, indicatorPositions[i], 7, (0, 0, 255), -1)
                        while len(pos[i])>maxLength:
                            pos[i].pop(0)
                    out.write(grey_new)
                else:
                    break
                # print(nFrame)
                nFrame += 1
            out.release()
            source.release()
    ################################################################
    def savePawObstacleTrackingData(self, mouse, date, rec, DLCinstance, pawPositions, pawTrackingOutliers, pawMetaData, startEndExposureTime, startTime,generateVideo=True, obstacleVideo=True):

        jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
        jointIdx = pawMetaData['data']['DLC-model-config file']['all_joints']
        cropping = pawMetaData['data']['cropping_parameters']
        print('cropping parameters', cropping)
        # pdb.set_trace()
        rec = rec.replace('/', '-')
        if not obstacleVideo:
            (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'pawTrackingData', DLCinstance])
            # use the 'middle' of the exposure time as time-point of the frame
            timeArray = np.average(startEndExposureTime, axis=1)
        else:
            (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'obsTrackingData', DLCinstance])
            timeArray = startEndExposureTime
        self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawPositionsFromDLC', pawPositions)
        self.h5pyTools.createOverwriteDS(grpHandle, 'jointNames', np.asarray(jointNames))
        self.h5pyTools.createOverwriteDS(grpHandle, 'croppingParameters', np.array(cropping))

        #pdb.set_trace()

        for i in range(len(jointNames)):
            pawMask = pawTrackingOutliers[i][3]
            # pdb.set_trace()
            rawPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)])) ** 2) / np.diff(timeArray)
            rawSpeedTime = (timeArray[:-1] + timeArray[1:]) / 2.
            clearedPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)][pawMask])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)][pawMask])) ** 2) / np.diff(timeArray[pawMask])
            clearedPawXSpeed = np.diff(pawPositions[:, (i * 3 + 1)][pawMask]) / np.diff(timeArray[pawMask])
            clearedPawYSpeed = np.diff(pawPositions[:, (i * 3 + 2)][pawMask]) / np.diff(timeArray[pawMask])
            clearedSpeedTime = (timeArray[pawMask][:-1] + timeArray[pawMask][1:]) / 2.
            clearedPosIdx = np.arange(len(pawPositions))[pawMask]
            clearedXYPos = np.column_stack((timeArray[pawMask], pawPositions[:, (i * 3 + 1)][pawMask], pawPositions[:, (i * 3 + 2)][pawMask], clearedPosIdx))
            clearedSpeedIdx = np.array((clearedPosIdx[:-1] + clearedPosIdx[1:]) / 2., dtype=int)
            #pdb.set_trace()
            self.h5pyTools.createOverwriteDS(grpHandle, 'pawTrackingOutliers%s' % i, pawTrackingOutliers[i][3],[['PawID', jointNames[i]],['pawTrackingOutliers',np.array([ pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]])]])#, pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
            # pawTrackingOutliers.append([i, tot, correct, correctIndicies, jointNames[i], onePawData, onePawDataTmp, frDispl, frDisplOrig])
            #['PawID', [jointNames[i], pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
            try:
                self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawSpeed%s' % i, np.column_stack((rawSpeedTime, rawPawSpeed)), ['recStartTime', startTime])
                self.h5pyTools.createOverwriteDS(grpHandle, 'clearedPawSpeed%s' % i, np.column_stack((clearedSpeedTime, clearedPawSpeed, clearedPawXSpeed, clearedPawYSpeed, clearedSpeedIdx)),['recStartTime', startTime])
            except ValueError:
                self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawSpeed%s' % i, np.column_stack((rawSpeedTime, rawPawSpeed[:-1])), ['recStartTime', startTime])
                self.h5pyTools.createOverwriteDS(grpHandle, 'clearedPawSpeed%s' % i, np.column_stack((clearedSpeedTime, clearedPawSpeed[:-1], clearedPawXSpeed[:-1], clearedPawYSpeed[:-1], clearedSpeedIdx)),['recStartTime', startTime])
            self.h5pyTools.createOverwriteDS(grpHandle, 'clearedXYPos%s' % i, clearedXYPos)  # pdb.set_trace()
        if generateVideo:
            fps = 80
            width = 800
            heigth = 600
            colors = [(255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
            indicatorPositions = [(270, 15), (270, 35), (240, 35), (240, 15)]
            sourceVideoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
            outputVideoFileName = self.analysisLocation + '%s_%s_%s_paw_tracking.avi' % (mouse, date, rec)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
            # print('1',fourcc,fps,width,heigth)
            out = cv2.VideoWriter(outputVideoFileName, fourcc, fps, (width, heigth))
            source = cv2.VideoCapture(sourceVideoFileName)
            nFrame = 0
            while (source.isOpened()):
                ret, frame = source.read()
                if ret == True:
                    # cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i],4), (10,20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
                    for i in range(4):
                        # print(int(pawPositions[nFrame, 3 * i + 1]+0.5),int(pawPositions[nFrame, 3 * i + 2]+0.5))
                        cv2.drawMarker(frame, (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5)), colors[i],
                                       cv2.MARKER_CROSS, 20, 2)
                        if nFrame in pawTrackingOutliers[i][3]:
                            cv2.circle(frame, indicatorPositions[i], 7, (0, 255, 0), -1)
                        else:
                            cv2.circle(frame, indicatorPositions[i], 7, (0, 0, 255), -1)
                    out.write(frame)
                else:
                    break
                # print(nFrame)
                nFrame += 1
            out.release()
            source.release()

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def saveBehaviorVideo(self, mouse, date, rec, framesRaw, startEndExposureTime, videoIdx,exportIndFrames=False,obstacle=False):
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
        midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.
        #pdb.set_trace()
        # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/', '-')
        if obstacle:
            videoFileName = self.analysisLocation + '%s_%s_%s_raw_obstacle.avi' % (mouse, date, rec)
        else:
            videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))

        vLength = np.shape(framesRaw)[0]
        width = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        print('number of frames :', vLength, width, heigth)
        fps = 80
        # pdb.set_trace()
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width, heigth))
        # # pdb.set_trace()
        nMultipleSaveImages = 2000
        for i in np.arange(len(videoIdx)):
            frame8bit = np.array(np.transpose(framesRaw[videoIdx[i]]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (videoIdx[i], (vLength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            out.write(frame)
            if exportIndFrames:
                if not (i % nMultipleSaveImages):
                    status = cv2.imwrite(videoFileName[:-4]+'_#%s.png' % str(i), frame)
        print('video length:', len(videoIdx))
        # # Release everything if job is finished
        # # cap.release()
        out.release()
        # cv2.destroyAllWindows()

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def regenerateBehaviorVideoAtDeterminedFrameTimes(self, mouse, date, rec, videoName, startEndExposureTime, videoIdx,videoId,exportIndFrames,obstacle=False,name=None,obsInfo=None, modifyContrast = True, laser=None, regen=False):
        # if modifyContrast != None:
        #     alpha = modifyContrast[0]
        #     beta = modifyContrast[1]

        if laser is not None:
            optoEMG = True
            EMG = laser[0]
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']

        if obstacle:
            midFrameTimes = startEndExposureTime
        else:
            midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.

        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_processed-%s-video.avi' % (mouse, date, rec, name)
        if not os.path.isfile(videoFileName) or regen:


            video = cv2.VideoCapture(videoName)
            Vlength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            Vwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            Vheigth = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            Vfps = video.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
            fps = 80
            out = cv2.VideoWriter(videoFileName, fourcc, fps, (Vwidth, Vheigth),0
                                  )

            nFrame = 0
            nFramesInNewVid = 0
            nMultipleSaveImages = 2000
            start_time = time.time()
            # pdb.set_trace()
            while True:
                ok, img = video.read()
                progress = (nFrame / Vlength * 100)  # Calculate the loading progress percentage
                elapsed_time = time.time() - start_time
                # Calculate remaining time
                if nFrame>1:
                    remaining_frames = Vlength - nFrame
                    frames_per_second = nFrame / elapsed_time
                    estimated_remaining_time = remaining_frames / frames_per_second
                    sys.stdout.write('\r')
                    sys.stdout.write(
                        f'{videoId} | writing  frames: {nFrame}/{Vlength} - {progress:.2f}% complete | Elapsed Time: {elapsed_time:.2f} s | Estimated Remaining Time: {estimated_remaining_time:.2f} s')
                    sys.stdout.flush()
                if ok:
                    if nFrame in videoIdx:
                        frame = np.array(img, dtype=np.uint8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        frame = clahe.apply(frame)
                        cv2.putText(frame, 'time %s sec' % round(midFrameTimes[nFramesInNewVid], 4), (10, 40),
                                    cv2.QT_FONT_NORMAL, 0.6, color=250)
                        if exportIndFrames:
                            if not (nFrame % nMultipleSaveImages):
                                print(nFrame)
                                # np.save(open(videoFileName[:-4] + '_#%s.npy' % str(nFrame), 'wb'), frame)
                                cv2.imwrite(videoFileName[:-4] + '_#%s.png' % str(nFrame), frame)
                        if obstacle:
                            ttext = 'frame %04d / %s, Obs. %02d, Obs. ID %s, Obs. angle %s' % (
                                nFrame, (Vlength - 1), obsInfo[0][nFramesInNewVid], int(obsInfo[1][nFramesInNewVid]),
                                round(obsInfo[2][nFramesInNewVid], 2))
                            cv2.putText(frame, ttext, (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=250)
                            if name == 'whiskerObstacle' and modifyContrast:
                                # pdb.set_trace()
                                frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                                # frame = cv2.equalizeHist(gray)


                        else:
                            cv2.putText(frame, 'frame %04d / %s' % (nFrame, (Vlength - 1)), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=250)
                        # if optoEMG:
                        #     idxFrame = np.argmin(np.abs(EMG['time']-midFrameTimes[nFramesInNewVid]))
                        #     if EMG['current_chan2'][idxFrame]:
                        #         cv2.circle(frame,(20,60),15,color=(255,0,0),thickness=-1)
                                #cv2.putText(frame, 'laser : frame %04d / %s' % (nFrame, (Vlength - 1)), (10, 60), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
                        out.write(frame)
                        nFramesInNewVid += 1
                    nFrame += 1
                else:
                    print('breaking at nFrame : ', nFrame)
                    break
            print('raw video length : ', nFrame)
            print('new video length : ', len(videoIdx))
            video.release()
            out.release()
            # pdb.set_trace()
            if not nFramesInNewVid == len(videoIdx):
                print(nFramesInNewVid, len(videoIdx))
                print('The number of frames in the new video does not match!')
        else:
            print('\n video file already generated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    # def regenerateBehaviorVideoAtDeterminedFrameTimes(self, mouse, date, rec, videoName, startEndExposureTime, videoIdx,exportIndFrames,obstacle=False,name=None,obsInfo=None, modifyContrast = True):
    #     # if modifyContrast != None:
    #     #     alpha = modifyContrast[0]
    #     #     beta = modifyContrast[1]
    #
    #     # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
    #     if obstacle:
    #         midFrameTimes = startEndExposureTime
    #     else:
    #         midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.
    #     #pdb.set_trace()
    #     # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
    #     # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
    #     # replace possible backslashes from subdirectory structure and
    #     rec = rec.replace('/', '-')
    #     #if obstacle:
    #     #    videoFileName = self.analysisLocation + '%s_%s_%s_raw_obstacle.avi' % (mouse, date, rec)
    #
    #
    #     # if name == 'whiskerObstacle' and modifyContrast is not None:
    #     #     videoFileName = self.analysisLocation + '%s_%s_%s_processed-%s-video_enhanced.avi' % (mouse, date, rec,name)
    #     # if obstacle:
    #     videoFileName = self.analysisLocation + '%s_%s_%s_processed-%s-video.avi' % (mouse, date, rec, name)
    #     # else:
    #     #     videoFileName = self.analysisLocation + '%s_%s_%s_processed-video.avi' % (mouse, date, rec)
    #     # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))
    #
    #     #vLength = np.shape(framesRaw)[0]
    #     #width = np.shape(framesRaw)[1]
    #     #heigth = np.shape(framesRaw)[2]
    #
    #     # video = frames
    #     video = cv2.VideoCapture(videoName)
    #     Vlength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     Vwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     Vheigth = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     Vfps = video.get(cv2.CAP_PROP_FPS)
    #
    #     # Define the codec and create VideoWriter object
    #     # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
    #     # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
    #     # M J P G is working great !! 184 MB per video
    #     # H F Y U is working great !! 2.7 GB per video
    #     # M P E G has issues !! DON'T USE (frames are missing)
    #     # X V I D : frame 3001 missing and last nine frames are screwed
    #     # 0 (no compression) : frame 3001 missing last 2 frames are the same
    #     # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
    #     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
    #     fps = 80
    #     out = cv2.VideoWriter(videoFileName, fourcc, fps, (Vwidth, Vheigth))
    #     # if outputProps:
    #     #print('  Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (Vlength, Vwidth, Vheight, Vfps))
    #     #maskGrid = np.indices((Vheight, Vwidth))
    #     # read first video frame
    #
    #     nFrame = 0
    #     nFramesInNewVid = 0
    #     nMultipleSaveImages = 100
    #     while True:
    #         ok, img = video.read()
    #         if ok:
    #             if nFrame in videoIdx:
    #                 frame = np.array(img, dtype=np.uint8) #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #                 cv2.putText(frame, 'time %s sec' % round(midFrameTimes[nFramesInNewVid], 4), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
    #                 if exportIndFrames:
    #                     if not (nFrame % nMultipleSaveImages):
    #                         print(nFrame)
    #                         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #                         print(np.shape(gray))
    #                         np.save(open(videoFileName[:-4] + '_#%s.npy' % str(nFrame),'wb'),gray )
    #                         cv2.imwrite(videoFileName[:-4] + '_#%s.png' % str(nFrame), frame)
    #                 if obstacle:
    #                     ttext = 'frame %04d / %s, Obs. %02d, Obs. ID %s, Obs. angle %s' % (nFrame, (Vlength - 1),obsInfo[0][nFramesInNewVid],int(obsInfo[1][nFramesInNewVid]),round(obsInfo[2][nFramesInNewVid],2))
    #                     cv2.putText(frame, ttext, (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
    #                     if name == 'whiskerObstacle' and modifyContrast:
    #            		        # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    #                         # Convert the frame to grayscale
    #                         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #                         # Enhance the contrast using histogram equalization
    #                         gray_eq = cv2.equalizeHist(gray)
    #
    #                         # Convert grayscale to BGR
    #                         frame = cv2.cvtColor(cv2.convertScaleAbs(gray_eq), cv2.COLOR_GRAY2BGR)
    #                 else:
    #                     cv2.putText(frame, 'frame %04d / %s' % (nFrame, (Vlength - 1)), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
    #                 out.write(frame)
    #                 nFramesInNewVid+=1
    #             nFrame += 1
    #         else:
    #             print('breaking at nFrame : ', nFrame)
    #             break
    #     print('raw video length : ', nFrame)
    #     print('new video length : ', len(videoIdx))
    #     video.release()
    #     out.release()
    #     if not nFramesInNewVid==len(videoIdx):
    #         print(nFramesInNewVid,len(videoIdx))
    #         print('The number of frames in the new video does not match!')


        #cv2.destroyAllWindows()


        # # # pdb.set_trace()
        # nMultipleSaveImages = 1000
        # for i in np.arange(len(videoIdx)):
        #     frame8bit = np.array(np.transpose(frame[videoIdx[i]]), dtype=np.uint8)
        #     frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
        #     cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
        #     cv2.putText(frame, 'frame %04d / %s' % (videoIdx[i], (Vlength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
        #     out.write(frame)
        #     if exportIndFrames:
        #         if not (i % nMultipleSaveImages):
        #             status = cv2.imwrite(videoFileName[:-4]+'_#%s.png' % str(i), frame)
        # print('video length:', len(videoIdx))
        # # Release everything if job is finished
        # # cap.release()
        # out.release()



    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def saveBehaviorVideoWithCa(self, mouse, date, rec, framesRaw, expStartTime, expEndTime, imageMetaInfo, angles, aTimes):
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        midFrameTimes = (expStartTime + expEndTime) / 2.
        # pdb.set_trace()
        # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior_withCa.avi' % (mouse, date, rec)
        # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))
        caImg = io.imread('/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/190101_f15/2019.03.21_000_suite2p_reg/suite2p/plane0/reg_tif/AVG2_output_1-902C.tif')
        # pdb.set_trace()
        caImg = (caImg - np.min(caImg)) * 255. / (np.max(caImg) - np.min(caImg))
        # caImg = (caImg - 15.)*255./(30.-15.)
        vLength = np.shape(framesRaw)[0]
        width = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        print('number of frames :', vLength, width, heigth)
        fps = 250
        fBehavior = 200
        fCa = 15
        afterEvery = fBehavior / fCa
        # pdb.set_trace()
        ##########################################################
        # create walking dynamics figure instance
        fig0 = plt.figure(figsize=(6.56, 3.5))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.23)
        # x1 = 0.
        # y1 = 0.
        ax0 = fig0.add_subplot(1, 1, 1)
        ax0 = plt.axes(xlim=(0, 30), ylim=(-10, 130))
        line0, = ax0.plot([], [], 'k-', lw=2)
        ax0.set_ylabel('position (cm)', fontsize=18)
        ax0.set_xlabel('time (s)', fontsize=18)

        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        #
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.xaxis.set_ticks_position('bottom')
        #
        # if xyInvisible[1]:
        # ax.spines['left'].set_visible(False)
        # ax.yaxis.set_visible(False)
        # else:
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')

        ##########################################################
        # create calcium dynamics figure instance
        fig = plt.figure(figsize=(6.56, 3.5))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.23)
        # x1 = 0.
        # y1 = 0.
        ax = fig.add_subplot(1, 1, 1)
        ax = plt.axes(xlim=(0, 30), ylim=(45, 170))
        line1, = ax.plot([], [], 'g-', lw=2)
        line2, = ax.plot([], [], 'b-', lw=2)
        line3, = ax.plot([], [], 'r-', lw=2)
        ax.set_ylabel('fluorescence (a.u.)', fontsize=18)
        ax.set_xlabel('time (s)', fontsize=18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        #
        # if xyInvisible[1]:
        # ax.spines['left'].set_visible(False)
        # ax.yaxis.set_visible(False)
        # else:
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        # ax.set_ylabel('fluorescence (a.u.)')
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width + 512, heigth + 350))
        nCa = 1
        ttime = []
        ffluo = [[], [], []]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        rois = [[464, 452, 11], [329, 99, 11], [318, 235, 11],  # [446,389,11]
                ]
        nPos = 0
        for i in np.arange(len(framesRaw)):
            output = np.zeros((heigth + 350, width + 512, 3), dtype="uint8")
            frame8bit = np.array(np.transpose(framesRaw[i]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            # cv2.circle(frame, (100, 100), 50, (0, 255, 0), -1)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (i, (vLength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            output[0:heigth, 0:width, :] = frame
            ttime.append(midFrameTimes[i])

            #####################################
            mask0 = aTimes < midFrameTimes[i]
            line0.set_data(aTimes[mask0], angles[mask0] * 80. / 360.)
            fig0.canvas.draw()
            img0 = np.fromstring(fig0.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img0 = img0.reshape(fig0.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            output[600:, :656, :] = img0
            ####################################
            # treatment of ca image
            frameCa8bit = np.array(caImg[nCa - 1], dtype=np.uint8)
            frameCa = cv2.cvtColor(frameCa8bit, cv2.COLOR_GRAY2BGR)

            for n in range(len(rois)):
                cv2.circle(frameCa, (rois[n][0], rois[n][1]), rois[n][2], colors[n], 2)
            cv2.putText(frameCa, '50 um', (35, 480), cv2.QT_FONT_NORMAL, 0.55, color=(220, 220, 220))
            cv2.line(frameCa, (30, 490), (94, 490), (220, 220, 220), 4)
            output[44:(44 + 512), (width):(width + 512), :] = frameCa
            # plot roi and add to movie
            for n in range(len(rois)):
                mask = np.zeros(shape=frameCa.shape, dtype="uint8")
                cv2.circle(mask, (rois[n][0], rois[n][1]), rois[n][2], (255, 255, 255), -1)
                maskedImg = cv2.bitwise_and(src1=frameCa, src2=mask)
                ffluo[n].append(np.mean(maskedImg[maskedImg > 0]))

            #########################################
            line1.set_data(ttime, ffluo[0])
            line2.set_data(ttime, ffluo[1])
            line3.set_data(ttime, ffluo[2])
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ##########################################
            output[600:, 656:(656 + 656), :] = img
            # pdb.set_trace()
            #
            out.write(output)
            if (i > nCa * afterEvery) and nCa < (len(caImg)):
                nCa += 1
                print(i, nCa, nCa * afterEvery)
        # Release everything if job is finished
        # cap.release()
        out.release()
        cv2.destroyAllWindows()

    ######################################################################################
    def createPsorth5(self,current, ephysTimes, rate, saveDir,fileName):

        dd = current
        tt = ephysTimes

        interpData = interp1d(tt, dd)
        ttNew = np.linspace(tt[0], tt[-1], 2 * len(tt), endpoint=True)
        ddNew = interpData(ttNew)
        print(int(1. / ttNew[1]))
        newSampleRate=np.array([int(1./ttNew[1])])


        with h5py.File(saveDir+fileName+'_psort.h5', "w") as f:

            dset = f.create_dataset("ch_data", data=ddNew*1E12)

            dset = f.create_dataset("ch_time", data=ttNew)

            dset = f.create_dataset("sample_rate", data=newSampleRate, dtype='int')
    def saveBehaviorVideoSwingStance(self, mouse, date, rec, framesRaw, startEndExposureTime, videoIdx,recs, day,trial, exportIndFrames):
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']

        # recTimes = swingStanceD['forFit'][i][2]
        midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.
        #pdb.set_trace()
        # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior_swing_stance.avi' % (mouse, date, rec)
        # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))

        vLength = np.shape(framesRaw)[0]
        width = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        # print('number of frames :', vLength, width, heigth)
        fps = 80
        # pdb.set_trace()
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width, heigth))

        # pdb.set_trace()
        nMultipleSaveImages = 1000
        for i in np.arange(len(videoIdx)):
            frame8bit = np.array(np.transpose(framesRaw[videoIdx[i]]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (videoIdx[i], (vLength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))

            for p in range(4):
                idxSwings = np.array(recs[day][4][trial][3][p][1])
                stepCharacter = recs[day][4][trial][3][p][3]
                recTimes = recs[day][4][trial][4][p][2]
                linearPawPos = recs[day][4][trial][4][p][5]
                forFit = recs[day][4][trial][4][p][4]
                pawSwing =idxSwings
                indecisive=stepCharacter
                frameForFit=forFit

                # swingOn=pawSwing[:,0]
                # swingOff = pawSwing[:, 1]
                # swingMask=(videoIdx==swingOn) and (videoIdx<swingOff)

                for k in range(len(pawSwing)-1):
                    swingLen=linearPawPos[pawSwing[k,1]][1]-linearPawPos[pawSwing[k,0]][1]
                    if swingLen<0:
                   #colors in BGR
                        color = (50, 215, 50)
                    else:
                        color = (0, 255, 0)
                    stanceColor=(71, 99, 255)
                    missColor=(0, 165, 255)
                    if (videoIdx[i]>=frameForFit[pawSwing[k,0]] and videoIdx[i]<frameForFit[pawSwing[k,1]] ) :
                        if p==0:
                            cv2.putText(frame, f'swing [{frameForFit[pawSwing[k,0]]}, {frameForFit[pawSwing[k,1]]}] ({swingLen:.2f})' , (520, 240), cv2.QT_FONT_NORMAL, 0.6,color=color)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, 'miss', (570, 260),cv2.QT_FONT_NORMAL, 0.6, color=missColor)
                        if p==1:
                            cv2.putText(frame, f'swing [{frameForFit[pawSwing[k,0]]}, {frameForFit[pawSwing[k,1]]}] ({swingLen:.2f})' , (520, 480), cv2.QT_FONT_NORMAL, 0.6, color=color)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, 'miss', (570, 500),cv2.QT_FONT_NORMAL, 0.6, color=missColor)
                        if p==2:
                            cv2.putText(frame, f'swing [{frameForFit[pawSwing[k,0]]}, {frameForFit[pawSwing[k,1]]}] ({swingLen:.2f})'  , (200, 240), cv2.QT_FONT_NORMAL, 0.6,color=color)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, 'miss', (250, 260),cv2.QT_FONT_NORMAL, 0.6, color=missColor)
                        if p==3:
                            cv2.putText(frame, f'swing [{frameForFit[pawSwing[k,0]]}, {frameForFit[pawSwing[k,1]]}] ({swingLen:.2f})' , (200, 480), cv2.QT_FONT_NORMAL, 0.6, color=color)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, 'miss', (250, 500),cv2.QT_FONT_NORMAL, 0.6, color=missColor)
                    elif (videoIdx[i]>=frameForFit[pawSwing[k,1]] and videoIdx[i]<frameForFit[pawSwing[k+1,0]]) :
                        if p==0:
                            cv2.putText(frame,f'stance [{frameForFit[pawSwing[k,1]]}, {frameForFit[pawSwing[k+1,0]]}]' , (520, 240), cv2.QT_FONT_NORMAL, 0.6,color=stanceColor)
                            if indecisive[k][3] == True:
                                cv2.putText(frame, '', (570, 260), cv2.QT_FONT_NORMAL, 0.6, color=stanceColor)
                        if p==1:
                            cv2.putText(frame, f'stance [{frameForFit[pawSwing[k,1]]}, {frameForFit[pawSwing[k+1,0]]}]', (520, 480), cv2.QT_FONT_NORMAL, 0.6, color=stanceColor)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, '', (570, 260),cv2.QT_FONT_NORMAL, 0.6, color=stanceColor)
                        if p==2:
                            cv2.putText(frame, f'stance [{frameForFit[pawSwing[k,1]]}, {frameForFit[pawSwing[k+1,0]]}]', (200, 240), cv2.QT_FONT_NORMAL, 0.6,color=stanceColor)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, '', (570, 260),cv2.QT_FONT_NORMAL, 0.6, color=stanceColor)
                        if p==3:
                            cv2.putText(frame, f'stance [{frameForFit[pawSwing[k,1]]}, {frameForFit[pawSwing[k+1,0]]}]', (200, 480), cv2.QT_FONT_NORMAL, 0.6, color=stanceColor)
                            if indecisive[k][3]==True:
                                cv2.putText(frame, '', (570, 260),cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))

            out.write(frame)
            # if not (i % nMultipleSaveImages):
            #     status = cv2.imwrite(videoFileName[:-4]+'_#%s.png' % str(i), frame)
        print('video length:', len(videoIdx))
        # Release everything if job is finished
        # cap.release()
        out.release()
        cv2.destroyAllWindows()

    ############################################################
    def readNPXdata(self, recFolder,npxFolder,spikeGLXrecording):
        pdb.set_trace()
        # pathToCatGTouptut = self.analysisLocation + recFolder + '-' + spikeGLXrecording + '/catgt_' + spikeGLXrecording + '_g0/' + spikeGLXrecording + '_g0_imec0/'
        pathToCatGTouptut = self.analysisLocation + '/'+ recFolder + '_' + npxFolder + '/catgt_' + spikeGLXrecording + '_g0/'
        print(pathToCatGTouptut)
        imecDict = {}
        metaInfoFileName = pathToCatGTouptut+spikeGLXrecording+'_g0_tcat.imec0.ap.meta'  # f58_recording_g0_tcat.imec0.ap.meta
        imecDict['imecMetaInfo'] = self.readMetaInformation(metaInfoFileName)
        imecDict['imecSyn'] = np.loadtxt(pathToCatGTouptut+spikeGLXrecording+'_g0_tcat.imec0.ap.xd_384_6_500.txt')
        imecDict['nidaqSyn'] = np.loadtxt(pathToCatGTouptut+spikeGLXrecording+'_g0_tcat.nidq.xd_0_1_500.txt')
        imecDict['nidaqRecStart'] = np.loadtxt(pathToCatGTouptut+spikeGLXrecording+'_g0_tcat.nidq.xd_0_5_0.txt')
        imecDict['nidaqRecEnd'] = np.loadtxt(pathToCatGTouptut + spikeGLXrecording + '_g0_tcat.nidq.xid_0_5_0.txt')
        #spike_times, spike_clusters, spike_amplitudes, spike_positions, templates_raw, template_position
        print('reading spike-sorted data ... ',end='')
        (imecDict['spike_times'],imecDict['spike_clusters'],imecDict['spike_amplitudes'],imecDict['spike_positions'],imecDict['templates_raw'],imecDict['template_position'],imecDict['cluster_info']) = self.load_phy_folder(pathToCatGTouptut + 'kilosort3/')
        print('done')
        return imecDict

        ############################################################
    def readChronicNPXdata(self, recFolder, npxFolder, spikeGLXrecording):
        # pdb.set_trace()
        pathToCatGTouptut = self.analysisLocation + recFolder + '-' + spikeGLXrecording + '/catgt_' + spikeGLXrecording + '_g0/' + spikeGLXrecording + '_g0_imec0/'
        pathToCatGTouptutObx = self.analysisLocation + recFolder + '-' + spikeGLXrecording + '/catgt_' + spikeGLXrecording + '_g0/'
        print(pathToCatGTouptut)
        imecDict = {}
        metaInfoFileName = pathToCatGTouptut + spikeGLXrecording + '_g0_tcat.imec0.ap.meta'  # f58_recording_g0_tcat.imec0.ap.meta
        imecDict['imecMetaInfo'] = self.readMetaInformation(metaInfoFileName)
        imecDict['imecSyn'] = np.loadtxt(pathToCatGTouptut + spikeGLXrecording + '_g0_tcat.imec0.ap.xd_384_6_500.txt')
        if os.path.exists(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_1_6_500.txt'):
            imecDict['withOpto'] = False
            imecDict['nidaqSyn'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_1_6_500.txt')
            imecDict['nidaqRecStart'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_0_0_0.txt')
            imecDict['nidaqRecEnd'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xid_0_0_0.txt')
        elif os.path.exists(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_2_6_500.txt'):
            imecDict['withOpto'] = True
            imecDict['nidaqSyn'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_2_6_500.txt')
            imecDict['nidaqRecStart'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_1_0_0.txt')
            imecDict['nidaqRecEnd'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xid_1_0_0.txt')
            imecDict['optoStart'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_1_1_0.txt')
            imecDict['optoEnd'] = np.loadtxt(pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xid_1_1_0.txt')
        else:
            print('file %s was not found ' % (pathToCatGTouptutObx + spikeGLXrecording + '_g0_tcat.obx0.obx.xd_x_6_500.txt'))
            pdb.set_trace()
        if len(imecDict['nidaqRecStart']) != len(imecDict['nidaqRecEnd']):
            print( '  Nidaq Rec On signals do not have the same length : ',imecDict['nidaqRecStart'], imecDict['nidaqRecEnd'])
            pdb.set_trace()
        # spike_times, spike_clusters, spike_amplitudes, spike_positions, templates_raw, template_position
        print('reading spike-sorted data ... ', end='')
        (imecDict['spike_times'], imecDict['spike_clusters'], imecDict['spike_amplitudes'],
         imecDict['spike_positions'], imecDict['templates_raw'], imecDict['template_position'],
         imecDict['cluster_info'], imecDict['cluster_group']) = self.load_phy_folder(pathToCatGTouptut + 'kilosort4/')
        print('done')
        return imecDict
    ############################################################
    def readMetaInformation(self,fileName):
        with open(fileName, 'r') as file:
            # Initialize an empty dictionary to store variable name-value pairs
            variables = {}
            # Read each line from the file
            for line in file:
                # Split each line into variable name and value using '=' delimiter
                parts = line.split('=')
                # Strip whitespace from variable name and value
                var_name = parts[0].strip()
                var_value = parts[1].strip()
                # Store variable name and value in the dictionary
                variables[var_name] = var_value
        return variables

    ############################################################
    def load_phy_folder(self, sortfolder):
        '''
        Phy stores data as .npy and tab separated (.tsv) files in a folder.

        This function reads the spike times and cluster identities from a folder and
        computes the spike amplitudes and approximate spike locations (XY).

        This is an approximate way of computing the spike depths since we don't
        actually read the waveforms (so it is fast); we use the templates instead.

        Example:

        spike_times,spike_clusters,spike_amplitudes,spike_positions,templates_raw,templates_position = load_phy_folder(folder)

        '''
        # load the channel locations
        channel_pos = np.load(sortfolder + 'channel_positions.npy')
        # load each spike cluster number
        spike_clusters = np.load(sortfolder + 'spike_clusters.npy')
        # load spiketimes
        spike_times = np.load(sortfolder + 'spike_times.npy')
        # load overview table
        #cluster_info = np.loadtxt(sortfolder + 'cluster_info.tsv', skiprows=1, dtype=str)
        #cluster_info = np.genfromtxt(sortfolder + 'cluster_info.tsv', dtype= None, skip_header=1, missing_values=None)
        df = pd.read_csv(sortfolder + 'cluster_info.tsv', sep='\t')
        cluster_info = df.values
        #pdb.set_trace()
        dfkilosort = pd.read_csv(sortfolder + 'cluster_group.tsv', sep='\t')
        cluster_group = dfkilosort.values
        # load spike templates (which template was fitted)
        spike_templates = np.load(sortfolder + 'spike_templates.npy')
        # load the templates used to extract the spikes
        templates = np.load(sortfolder + 'templates.npy')
        # Load the amplitudes used to fit the template
        spike_template_amplitudes = np.load(sortfolder + 'amplitudes.npy')
        # load the whitening matrix (to correct for the data having been whitened)
        whitening_matrix = np.load(sortfolder + 'whitening_mat_inv.npy').T
        # the raw templates are the dot product of the templates by the whitening matrix
        templates_raw = np.dot(templates, whitening_matrix)
        # compute the peak to peak of each template
        templates_peak_to_peak = (templates_raw.max(axis=1) - templates_raw.min(axis=1))
        # the amplitude of each template is the max of the peak difference for all channels
        templates_amplitude = templates_peak_to_peak.max(axis=1)
        # compute the center of mass (X,Y) of the templates
        template_position = [templates_peak_to_peak * pos for pos in channel_pos.T]
        template_position = np.vstack([np.sum(t, axis=1) / np.sum(templates_peak_to_peak, axis=1) for t in template_position]).T
        # get the spike positions and amplitudes from the average templates
        spike_amplitudes = templates_amplitude[spike_templates] * spike_template_amplitudes
        spike_positions = template_position[spike_templates, :].squeeze()
        return spike_times, spike_clusters, spike_amplitudes, spike_positions, templates_raw, template_position, cluster_info, cluster_group


