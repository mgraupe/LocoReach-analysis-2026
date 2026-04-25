import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import os
import subprocess
import time
import pickle
import multiprocessing

saveDir = 'scriptRunHistory/'

# mouseList = ['210122_f84',
#              '210120_m85',
#              '210122_f83',
#              '210214_m12',
#              '210214_m14',
#              '210214_m15',
#              '210214_m13',
#              '210214_m20',
#              '210214_m18',
#              '210214_m19',
#              '210214_m17',
#              '210120_m86',
#              '210927_f23',
#              '210924_f64',
#              '210906_f97',
#              '210906_f98',
#              ]

mouseList = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28', '220716_f65', '220716_f67']
#['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']
script = 'runGLManalysisEphys-Behavior' #'analyzeEphysPsortData'#runGLManalysisEphys-Behavior'#getPsortWalkingActivityAndPawTraces' #StepTriggeredCaTraceAverages' #collectWheelPawAndCaData' #.py'analyzeWheelPawAndCaCorrelations' #analyzeStepTriggeredCaTraceAverages'
expDate = 'allPC'
recordings = 'allPC'

commandHist = []

outList = []
# loop over mice
for m in mouseList:
    #comandString = 'python %s.py -m %s -d %s -r %s' % (script,m,expDate,recordings)
    comandString = 'python %s.py -m %s -d %s -r %s' % (script, m, expDate,recordings)
    print(comandString)
    #tp = os.system('pwd')
    #print tp
    (out,err) = subprocess.getstatusoutput(comandString)
    print(m,out,err)
    outList.append(out)
    commandHist.append([comandString,out,err])
    #pdb.set_trace()

print(outList)
ttt = time.strftime("%y-%m-%d")
sname = os.path.basename(__file__)
pickle.dump( commandHist, open( saveDir+"%s_%s_script-%s.p" % (ttt,sname,script), "wb" ) )