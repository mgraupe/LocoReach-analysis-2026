import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.extractSaveData as extractSaveData
#This script generate a dictionnary containing all swing and stance related mesures for stride analysis
#make sure that you ran the extractSwingAndStancePhases script
#uptade list of animal and experiments


experiment='opto' # specify the experiment to analyze, ephy, muscimol, calcium, muscimol_batch1, muscimol_batch2, test, opto


treatments=[]
recStruc=[]
#test
# mouseList =  ['220204_m49','220204_f52']
treatmentList=['muscimol','saline']
if experiment=='ephy':
#Ephy Mice
    mouseList =  ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
    recStruc = 'simplexEphy'
elif experiment=='muscimol':
#all muscimol animals
    # mouseList=['201017_m99','210113_m78','210113_m77','201017_m98','201017_m1','201207_f42','201207_f43','210113_f79','220204_m49','220204_m50','220204_m51','220204_f52','220204_f53','220204_f54','220204_f55','220204_f56']
    # treatmentList=['muscimol', 'saline', 'saline','saline','muscimol','muscimol','muscimol','muscimol','muscimol','muscimol','muscimol','saline','saline','saline','saline','muscimol']
#muscimol animals without 220204_m50 because no red signal and cannula too medial
    mouseList = ['201017_m99', '210113_m78', '210113_m77', '201017_m98', '201017_m1', '201207_f42', '201207_f43','210113_f79', '220204_m49', '220204_m51', '220204_f52', '220204_f53', '220204_f54','220204_f55', '220204_f56']
    treatmentList = ['muscimol', 'saline', 'saline', 'saline', 'muscimol', 'muscimol', 'muscimol', 'muscimol','muscimol', 'muscimol', 'saline', 'saline', 'saline', 'saline', 'muscimol']

    treatments==True
    recStruc = 'simplexBehavior'
elif experiment=='muscimol_batch1':
# first muscimol experiment batch
    mouseList =  ['201017_m99','210113_m78','210113_m77','201017_m98','201017_m1','201207_f42','201207_f43','210113_f79']
    treatmentList=['muscimol', 'saline', 'saline','saline','muscimol','muscimol','muscimol','muscimol']
    treatments == True
    recStruc = 'simplexBehavior'
elif experiment=='muscimol_batch2':
# second muscimol experiment batch
    mouseList =  ['220204_m49','220204_m50','220204_m51','220204_f52','220204_f53','220204_f54','220204_f55','220204_f56']
    treatmentList=['muscimol','muscimol','muscimol','saline','saline','saline','saline','muscimol']
    treatments == True
    recStruc = 'simplexBehavior'
elif experiment=='calcium_simplex':
#simplex Mouse
    mouseList = ['210122_f84','210120_m85','210120_m86','210214_m12','210214_m14','210214_m15','210214_m13','210214_m20','210214_m18','210214_m19','210214_m17','210122_f83','210927_f23','210924_f64','210906_f97','210906_f98']
    recStruc = 'simplex'
elif experiment=='test':
    mouseList = ['220211_f38','220214_f43']
    recStruc='simplexEphy'
elif experiment=='test_muscimol':
    mouseList = ['220204_f56','220204_m49']
    treatmentList = ['muscimol', 'saline']
    recStruc='simplexBehavior'
elif experiment=='opto':
    mouseList  = ['230226_m10', '230403_f2','230402_m3', '230402_m4','230402_m5', '230403_f3','230226_f88','230405_m1']
    treatmentList = ['opsin', 'tdTomato','opsin', 'opsin','tdTomato','opsin','opsin','tdTomato']
    recStruc='simplexBehavior'
expDate = 'all910'
recordings = 'all910'
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'

mouseDict = {}
# loop over mice
#MG : this loop is perfect to load all mouse specific data into a dictionary
for a in range(len(mouseList)):
    print(expDate, recordings)

    eSD = extractSaveData.extractSaveData(mouseList[a], recStruc=recStruc)


    if experiment=="ephy" or experiment=='test':
        (foldersRecordings, dataFolder, listOfRecordings) = eSD.getEphysRecordingsList(expDate=expDate, recordings=recordings)
        (foldersRecordings, listOfRecordings) = eSD.combineDifferentCategorysOnSameDay(foldersRecordings,listOfRecordings)
    # else:
        # (foldersRecordings, dataFolder) = eSD.getRecordingsList(expDate=expDate,
        #                                                         recordings=recordings)  # get recordings for specific mouse and date

    pickleFileName = eSD.analysisLocation + '/allSingStanceDataPerSession_%s.p' % expDate
    recordingsM = pickle.load(open(pickleFileName, 'rb'))
#specify content of the dictionary
    if experiment=='muscimol' or experiment=='muscimol_batch1' or experiment=='muscimol_batch2' or experiment=='test_muscimol':
        mouseDict[a] = {'mouseName': mouseList[a], 'treatment':treatmentList[a],
                        'foldersRecordings': foldersRecordings,
                        'pawData': recordingsM}
    else:
        mouseDict[a] = {'mouseName': mouseList[a], 'treatment':treatmentList[a],
                        'pawData': recordingsM}
    del eSD

pickle.dump(mouseDict, open(groupAnalysisDir + '/%s_MiceDictionary.p'%experiment, 'wb'))


