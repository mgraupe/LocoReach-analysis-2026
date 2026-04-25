import sys
sys.path.append('./')
import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createPublicationVisualizations as createPublicationVisualizations
import tools.parameters as par
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mouseList = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28', '220716_f65', '220716_f67']

figVersion = '0.3'
#expDateForFig = '220729'
#recsForFig = [0,1,2,3,4]
#exampleTrace = 4 # 3!

expDate = 'all'  #  'some', 'all', 'allMLI' or 'allPC'
recordings= 'all' # 'some', 'all', 'allMLI' or 'allPC'

eSD         = extractSaveData.extractSaveData(mouseList[0],recStruc='simplexEphy')
#(foldersRecordings,dataFolders,listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

head, tail = os.path.split(eSD.analysisLocation[:-1])  # get path without the last level of the mouse specific folder
# print(head, tail, eSD.analysisLocation)
folder = head + '/simplexSummary/'
umapFile = 'clusterSummaryData.p'

umapDict = pickle.load(open(folder + umapFile, 'rb'))
cPV = createPublicationVisualizations.createVisualizations(eSD.publicationFigLocation,mouseList[0])

cPV.ephysTSNEFig(figVersion, umapDict)
