# from oauth2client import tools
# tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
# tools.argparser.add_argument("-r","--recordings", help="specify the recordings to analyze", required=False)
# args = tools.argparser.parse_args()
import sys
sys.path.append('.')
import tools.createGroupVisualizations as createGroupVisualizations
import tools.groupAnalysis as groupAnalysis
import tools.dataAnalysis as dataAnalysis
import tools.dataAnalysis_psth as dataAnalysis_psth
import tools.groupAnalysis_psth as groupAnalysis_psth
import tools.extractSaveData as extractSaveData
import tools.createVisualizations as createVisualizations
import tools.createPresentationVisualizations as createPresentationVisualizations
import tools.parameters as par
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import pickle
import os
import sys

from scipy.interpolate import interp1d
import matplotlib as mpl
from matplotlib import rcParams
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

#####################

def crosscorr(deltat, y0, y1, correlationRange=1.5, fast=False):
    """
            home-written routine to calcualte cross-correlation between two contiuous traces
            new version from February 9th, 2011
    """

    if len(y0) != len(y1):
        print('Data to be correlated has different dimensions!')
        sys.exit(1)

    y0mean = y0.mean()
    y1mean = y1.mean()
    y0sd = y0.std()
    y1sd = y1.std()

    if y0sd != 0 and y1sd != 0:
        y0norm = (y0 - y0mean) / y0sd
        y1norm = (y1 - y1mean) / y1sd
    else:
        y0norm = y0 - y0mean
        y1norm = y1 - y1mean

    # defined range calculation of cross-correlation
    # value is specified in main routine
    # deltat = 0.9

    pointnumber1 = len(y0)

    ncorrrange = np.ceil(correlationRange / deltat)
    corrrange = np.arange(2 * ncorrrange + 1) - ncorrrange
    ycorr = np.zeros(len(corrrange))

    # print corrrange
    if fast:
        pass
    else:
        for n in corrrange:
            corrpairs = pointnumber1 - abs(n)
            # ccc = arange(corrpairs)
            # print n
            if n < 0:
                y1mod = np.hstack((y1norm[int(-abs(n)):], y1norm[:-int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n > -10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1)
            elif n == 0:
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1norm)) / (float(pointnumber1))
                # print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], (float(pointnumber1-1))
            elif n > 0:
                y1mod = np.hstack((y1norm[int(abs(n)):], y1norm[:int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n < 10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1-1)
            else:
                print('Problem!')
                exit(1)
            # print n , ycorr[n+ncorrrange]

    float_corrrange = np.array([float(i) for i in corrrange])

    xcorr = float_corrrange * deltat

    normcorr = np.column_stack((xcorr, ycorr))
    return normcorr

#####################
def layoutOfPanel(ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #
    if xyInvisible[0]:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
    else:
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
    #
    if xyInvisible[1]:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')


    if xLabel != None :
        ax.set_xlabel(xLabel)

    if yLabel != None :
        ax.set_ylabel(yLabel)

    if Leg != None :
        ax.legend(loc=Leg[0], frameon=False)
        if len(Leg)>1 :
            legend = ax.get_legend()  # plt.gca().get_legend()
            ltext = legend.get_texts()
            plt.setp(ltext, fontsize=Leg[1])
###############################

groupFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary'
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
#cGV = createGroupVisualizations.createGroupVisualizations(groupFigDir)


# collecting analyzed ephys information from each animal
readDataAgain=False
mice = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']

spikeType = 'simple'
date='25Mar20' #'23feb21'

#cellType=recordings[3:]

recordings = ['allMLI','allPC']
cellType = recordings[0]
PSTHquantifyDict = pickle.load(open('psth-speed-profiles_%s.p' % cellType, 'rb'))
nMice = len(mice)
nCells = 0

psthCases = ['psthSucc','psthMiss','psthExp']
speedCases = ['speedSucc','speedMiss','speedExp']
labels = ['successful','miss-step','expected']

## Figure generation
if cellType == 'allMLI':
    nRowsCols = 8
elif cellType == 'allPC':
    nRowsCols = 6

fig_width = 40  # width in inches
fig_height = 30  # height in inches
fig_size = [fig_width, fig_height]
params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.figsize': fig_size, 'savefig.dpi': 600, 'axes.linewidth': 1.3,
          'ytick.major.size': 4,  # major tick size in points
          'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False  # major tick size in points
          # 'edgecolor' : None
          # 'xtick.major.size' : 2,
          # 'ytick.major.size' : 2,
          }
rcParams.update(params)
fig = plt.figure()
gs = gridspec.GridSpec(nRowsCols, nRowsCols,  # ,
                       # width_ratios=[0.1,1,5]
                       # height_ratios=[2, 1.5]
                       )
# define vertical and horizontal spacing between panels
gs.update(wspace=0.3, hspace=0.3)
# plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
# plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
# plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
# plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
# plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

# possibly change outer margins of the figure
plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
nPanel=0
for n in range(nMice):
    nCellPerAnimal = len(PSTHquantifyDict[mice[n]])
    nCells+=nCellPerAnimal
    # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
    for m in range(nCellPerAnimal):  # cells per animal
        gssub2 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[nPanel], hspace=0.1, wspace=0.3)
        nFig = 0
        nCase = 0
        for i in range(3):
            for j in range(3):
                ax0 = plt.subplot(gssub2[nFig])
                psthT = PSTHquantifyDict[mice[n]][m]['timePSTH']
                psth1 = PSTHquantifyDict[mice[n]][m][psthCases[i]]
                #psth2 = PSTHquantifyDict[mice[n]][m][psthCases[j]]
                speed1 = PSTHquantifyDict[mice[n]][m][speedCases[j]]
                #speed2 = PSTHquantifyDict[mice[n]][m][speedCases[j]]
                speedT = PSTHquantifyDict[mice[n]][m]['timeSpeed']
                interpSpeed1 = interp1d(speedT, speed1, fill_value='extrapolate')  # ,kind='cubic'
                #interpSpeed2 = interp1d(speedT, speed2, fill_value='extrapolate')
                speed1PSTHTime = interpSpeed1(psthT)
                #speed2PSTHTime = interpSpeed2(psthT)
                dT = psthT[1]-psthT[0]
                cc = crosscorr(dT,speed1PSTHTime,psth1,correlationRange=0.39)
                #ccSpeed = crosscorr(dT, speed1PSTHTime, speed2PSTHTime, correlationRange=0.39)
                ax0.axhline(0, ls=':', c='0.6')
                ax0.axvline(0, ls=':', c='0.6')
                ax0.plot(cc[:,0],cc[:,1],c='C0',label='PSTH')
                #ax0.plot(ccSpeed[:, 0], ccSpeed[:, 1], c='C1', label='speed')
                #if nFig != 6:
                layoutOfPanel(ax0, xLabel=(speedCases[j] if nFig>5 else None),yLabel=(psthCases[i] if not nFig%3 else None),Leg=([2] if nFig==0 else None))
                #else:
                #layoutOfPanel(ax0, xLabel=psthCases[j], yLabel=speedCases[i])
                #pdb.set_trace()
                #else:
                #    layoutOfPanel(ax0, xLabel=labels[j], yLabel=labels[i])
                nFig+=1

        nPanel+=1
        #PSTHquantifyDict[mouse][m]['timePSTH'] = tTime
        #PSTHquantifyDict[mouse][m]['psthSucc'] = tempPSTHSucc / nRecs
        #PSTHquantifyDict[mouse][m]['psthMiss'] = tempPSTHMiss / nRecs
        #PSTHquantifyDict[mouse][m]['psthExp'] = tempExpStanceMis / nRecs
        #PSTHquantifyDict[mouse][m]['peakTimeSuccessfulStep'] = tTime[np.argmax(tempPSTHSucc / nRecs)]
        #PSTHquantifyDict[mouse][m]['peakTimeMissStep'] = tTime[np.argmax(tempPSTHMiss / nRecs)]
        #PSTHquantifyDict[mouse][m]['peakTimeExpectedStanceMissStep'] = tTime[np.argmax(tempExpStanceMis / nRecs)]
        #PSTHquantifyDict[mouse][m]['psthStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['stanceStart']

        #PSTHquantifyDict[mouse][m]['timeSpeed'] = tTime
        #PSTHquantifyDict[mouse][m]['speedSucc'] = tempPSTHSucc / nRecs
        #PSTHquantifyDict[mouse][m]['speedMiss'] = tempPSTHMiss / nRecs
        #PSTHquantifyDict[mouse][m]['speedExp'] = tempExpStanceMis / nRecs
        #PSTHquantifyDict[mouse][m]['speedStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['speedCenteredStanceStart']

plt.savefig('cross-correlations_across_%s.pdf' % cellType)
