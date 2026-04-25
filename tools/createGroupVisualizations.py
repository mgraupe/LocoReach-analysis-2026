'''
        Class to provide figures, images and videos concerning a group of animals
        
'''

import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.animation as animation
# from scipy.interpolate import UnivariateSpline
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from statsmodels.regression.linear_model import OLS
# from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
import tools.groupAnalysis_psth as groupAnalysis_psth
import tools.dataAnalysis_psth as dataAnalysis_psth
import pdb
import scipy
# #from pylab import *
# import tifffile as tiff
import matplotlib as mpl
from matplotlib import rcParams
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from collections import OrderedDict
from matplotlib.ticker import MultipleLocator
import tools.groupAnalysis as groupAnalysis
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaResults
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pingouin as pg
import pingouin as pg
# from sklearn.cluster import KMeans
# import umap
# from mpl_toolkits.mplot3d import Axes3D
# #import sima
# #import sima.motion
# #import sima.segment
# from scipy.stats.stats import pearsonr
# from sklearn.preprocessing import minmax_scale
# from scipy.interpolate import interp1d
# #from mtspec import mt_coherence
from scipy import stats
import seaborn as sns
# import matplotlib.ticker as ticker
# from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
# from sklearn.preprocessing import MinMaxScaler
# from yellowbrick.text import UMAPVisualizer
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/'
import pickle

mpl.style.use('default')
params= OrderedDict([
    ('videoParameter',{
        'dpi': 500,
        }),
    ('projectionInTimeParameters', {
        'horizontalCuts' : [5,8,9,10,12] ,
        'verticalCut' : 50. ,
        'stimStart': 5. ,
        'stimLength' : 0.2 ,
        'fitStart' : 5. ,
        'baseLinePeriod' : 5. ,
        'threeDAspectRatio' : 3,
        'stimulationBarLocation' : -0.1,
        }),
    ('caEphysParameters', {
        'leaveOut' : 0.1,
        }),
    ])

import tools.dataAnalysis as dataAnalysis
from tools.pyqtgraph.configfile import *
#mpl.use('Qt5Agg')

class createGroupVisualizations:
    
    ##########################################################################################
    def __init__(self,figureDir):

        self.figureDirectory = figureDir
        if not os.path.isdir(self.figureDirectory):
            os.system('mkdir %s' % self.figureDirectory)

        #MG we don't need the config file for the moment
        # configFile = self.figureDirectory + '%s.config'
        # if os.path.isfile(configFile):
        #     self.config = readConfigFile(configFile)
        # else:
        #     self.config = params
        #     writeConfigFile(self.config,configFile)

        self.pawID = ['FR', 'FL', 'HL', 'HR']

    ##########################################################################################
    def determineFileName(self,reco,what=None,date=None):
        if (what is None) and (date is None):
            ff = self.figureDirectory + '%s' % (reco)
        elif date is None:
            ff = self.figureDirectory + '%s_%s' % (reco,what)
        else:
            ff = self.figureDirectory + '%s_%s_%s' % (date,reco,what)
        return ff

    ##########################################################################################
    def layoutOfPanel(self, ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):

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
    ##########################################################################################
    def BehaviorFigStyle(self,ax,fontsize=5,xLabel=None, yLabel=None):
        self.layoutOfPanel(ax, xLabel=xLabel, yLabel=yLabel, Leg=[1, 9])
        ax.legend(loc="upper right", frameon=False, fontsize=fontsize)
        majorLocator_x = MultipleLocator(1)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    def createPeakCaFigure(self, mouseDict):
        fig_width = 6  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 1,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[1, 2.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        #plt.figtext(0.06, 0.96, animalNames, clip_on=False, color='black',size=10)
        #mouseName=mouseDict[a]["mouseName"]
        #recordingDays=mouseDict[a]['expDateList']

                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        #pdb.set_trace()
        # for i in range(4):
        #     axList[i] = [plt.subplot(gs[2*i]),plt.subplot(gs[2*i+1])]
        #     axList[i][0].axhline(y=0,ls='--',c='0.7')
        #     axList[i][1].axhline(y=0, ls='--', c='0.7')
        # for n in range(nRois):
        #     for i in range(4):
        #         axList[i][0].plot(np.arange(nDays) + 1, peakCaTransients[n, i,:,0], 'o-',ms=3,lw=1,alpha=0.6,color=cmap(n / nRois),clip_on=False)
        pawIdx = 1
        ax0.plot([-0.6,0,0.6],[0.6,0,0.6],ls='--',c='0.5')
        for n in range (len(mouseDict)):
            #dd = mouseDict[n]['caData']
            #pdb.set_trace()
            ax0.plot(np.mean(mouseDict[n]['caData'][:, pawIdx,:,0],axis=1),np.std(mouseDict[n]['caData'][:, pawIdx,:,0],axis=1), 'o',ms=5, alpha=0.4)

        self.layoutOfPanel(ax0, xLabel='mean of amplitude', yLabel='STD of amplitude')#, Leg=[1, 9])

        #majorLocator_x = MultipleLocator(1)
        #ax0.xaxis.set_major_locator(majorLocator_x)
        # #Second subplot

        fname = 'peak-calicum-statistics'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ########################################################################################################
    def generateCaCorrelationPlot(self, allCorrData910,allCorrData820,analysisDate,mouseList,onlyAligned=False):
        # allCorrData.append([i, mouseList[i], correlationData, varExplained])
        # all correlation coefficients during the entire recording period
        # allCorrData910.append([i,mouseList[i],correlationData,varExplained,motorizationCorrelationData,varExplainedMotorization,roiData,allCorrDataPerSession])
        aD910 = []
        for n in range(len(allCorrData910)):
            mouse = allCorrData910[n][1]
            correlationData = allCorrData910[n][2]
            intersectDataTemp = allCorrData910[n][6]
            if onlyAligned:
                (intersectData, nDays, nRois, idxRef, refDay, dayList) = dataAnalysis.choseAlignmentReferenceDay(mouse, intersectDataTemp)
                daysInclude = dayList
            else:
                daysInclude = list(range(len(correlationData)))
            # corrCaTraces[i,t] = np.array([combis[i]figureDirectory[0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1]])
            minMaxCa    = [1,-1]
            minMaxWheel = [1,-1]
            minMaxPaw   = [1,-1]
            allCorrEffs = []
            allEuclDist = []
            allXDist   = []
            allYDist   = []
            allEuclDistU = []
            for nSess in range(len(correlationData)):
                if nSess in daysInclude:
                    nTrials = np.shape(correlationData[nSess][1])[1]
                    for i in range(nTrials):
                        allCorrEffs.extend(correlationData[nSess][1][:,i,2])
                        #if i == 0:
                        allEuclDist.extend(correlationData[nSess][1][:,i,4])
                        allXDist.extend(correlationData[nSess][1][:,i,5])
                        allYDist.extend(correlationData[nSess][1][:,i,6])
                        if i==0:
                            allEuclDistU.extend(correlationData[nSess][1][:, i, 4])
                            #allXDistU.extend(correlationData[nSess][1][:, i, 5])
                            #allYDistU.extend(correlationData[nSess][1][:, i, 6])

            #pdb.set_trace()
            allCorrEffs = np.asarray(allCorrEffs)
            allEuclDist = np.asarray(allEuclDist)
            allXDist = np.asarray(allXDist)
            allYDist = np.asarray(allYDist)
            allEuclDistU = np.asarray(allEuclDistU)

            aD910.append([mouse,allCorrEffs,allEuclDist,allXDist,allYDist,allEuclDistU])
        # all correlation coefficients during the motorization period
        aD910Act = []
        for n in range(len(allCorrData910)):
            mouse = allCorrData910[n][1]
            correlationData = allCorrData910[n][4]
            intersectDataTemp = allCorrData910[n][6]
            if onlyAligned:
                (intersectData, nDays, nRois, idxRef, refDay, dayList) = dataAnalysis.choseAlignmentReferenceDay(mouse, intersectDataTemp)
                daysInclude = dayList
            else:
                daysInclude = list(range(len(correlationData)))
            # corrCaTraces[i,t] = np.array([combis[i]figureDirectory[0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1]])
            minMaxCa    = [1,-1]
            minMaxWheel = [1,-1]
            minMaxPaw   = [1,-1]
            # find maxima and minima
            allCorrEffs = []
            allEuclDist = []
            allXDist   = []
            allYDist   = []
            allEuclDistU = []
            #nTrials = np.shape(correlationData[0][1])[1]
            for nSess in range(len(correlationData)):
                if nSess in daysInclude:
                    nTrials = np.shape(correlationData[nSess][1])[1]
                    for i in range(nTrials):
                        allCorrEffs.extend(correlationData[nSess][1][:,i,2])
                        #if i == 0:
                        allEuclDist.extend(correlationData[nSess][1][:,i,4])
                        allXDist.extend(correlationData[nSess][1][:,i,5])
                        allYDist.extend(correlationData[nSess][1][:,i,6])
                        if i==0:
                            allEuclDistU.extend(correlationData[nSess][1][:, i, 4])
                            #allXDistU.extend(correlationData[nSess][1][:, i, 5])
                            #allYDistU.extend(correlationData[nSess][1][:, i, 6])
            #pdb.set_trace()
            allCorrEffs = np.asarray(allCorrEffs)
            allEuclDist = np.asarray(allEuclDist)
            allXDist = np.asarray(allXDist)
            allYDist = np.asarray(allYDist)
            allEuclDistU = np.asarray(allEuclDistU)

            aD910Act.append([mouse,allCorrEffs,allEuclDist,allXDist,allYDist,allEuclDistU])

        aD820 = []
        for n in range(len(allCorrData820)):
            mouse = allCorrData820[n][1]
            correlationData = allCorrData820[n][2]
            intersectDataTemp = allCorrData910[n][6]
            allCorrDataPerSession = allCorrData820[n][7]
            if len(correlationData) == len(allCorrDataPerSession):
                print('equal lenghts!!')
            if onlyAligned:
                dayList = dataAnalysis.getDayListFor820Recs(mouse, intersectDataTemp, allCorrDataPerSession)
                daysInclude = dayList
            else:
                daysInclude = list(range(len(correlationData)))
            # corrCaTraces[i,t] = np.array([combis[i]figureDirectory[0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1]])
            minMaxCa    = [1,-1]
            minMaxWheel = [1,-1]
            minMaxPaw   = [1,-1]
            allCorrEffs = []
            allEuclDist = []
            allXDist   = []
            allYDist   = []
            allEuclDistU = []
            for nSess in range(len(correlationData)):
                if nSess in daysInclude:
                    nTrials = np.shape(correlationData[nSess][1])[1]
                    for i in range(nTrials):
                        allCorrEffs.extend(correlationData[nSess][1][:,i,2])
                        #if i == 0:
                        allEuclDist.extend(correlationData[nSess][1][:,i,4])
                        allXDist.extend(correlationData[nSess][1][:,i,5])
                        allYDist.extend(correlationData[nSess][1][:,i,6])
                        if i==0:
                            allEuclDistU.extend(correlationData[nSess][1][:, i, 4])
                            #allXDistU.extend(correlationData[nSess][1][:, i, 5])
                            #allYDistU.extend(correlationData[nSess][1][:, i, 6])

            #pdb.set_trace()
            allCorrEffs = np.asarray(allCorrEffs)
            allEuclDist = np.asarray(allEuclDist)
            allXDist = np.asarray(allXDist)
            allYDist = np.asarray(allYDist)
            allEuclDistU = np.asarray(allEuclDistU)

            aD820.append([mouse,allCorrEffs,allEuclDist,allXDist,allYDist,allEuclDistU])
        # all correlation coefficients during the motorization period
        aD820Act = []
        for n in range(len(allCorrData820)):
            mouse = allCorrData820[n][1]
            correlationData = allCorrData820[n][4]
            intersectDataTemp = allCorrData910[n][6]
            allCorrDataPerSession = allCorrData820[n][7]
            if onlyAligned:
                dayList = dataAnalysis.getDayListFor820Recs(mouse, intersectDataTemp, allCorrDataPerSession)
                daysInclude = dayList
            else:
                daysInclude = list(range(len(correlationData)))
            # corrCaTraces[i,t] = np.array([combis[i]figureDirectory[0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist[0],xyDist[1]])
            minMaxCa    = [1,-1]
            minMaxWheel = [1,-1]
            minMaxPaw   = [1,-1]
            # find maxima and minima
            allCorrEffs = []
            allEuclDist = []
            allXDist   = []
            allYDist   = []
            allEuclDistU = []

            for nSess in range(len(correlationData)):
                if nSess in daysInclude:
                    nTrials = np.shape(correlationData[nSess][1])[1]
                    for i in range(nTrials):
                        allCorrEffs.extend(correlationData[nSess][1][:,i,2])
                        #if i == 0:
                        allEuclDist.extend(correlationData[nSess][1][:,i,4])
                        allXDist.extend(correlationData[nSess][1][:,i,5])
                        allYDist.extend(correlationData[nSess][1][:,i,6])
                        if i==0:
                            allEuclDistU.extend(correlationData[nSess][1][:, i, 4])
                            #allXDistU.extend(correlationData[nSess][1][:, i, 5])
                            #allYDistU.extend(correlationData[nSess][1][:, i, 6])
            #pdb.set_trace()
            allCorrEffs = np.asarray(allCorrEffs)
            allEuclDist = np.asarray(allEuclDist)
            allXDist = np.asarray(allXDist)
            allYDist = np.asarray(allYDist)
            allEuclDistU = np.asarray(allEuclDistU)

            aD820Act.append([mouse,allCorrEffs,allEuclDist,allXDist,allYDist,allEuclDistU])

        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3,4,  # ,
                               #width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               #height_ratios=[10, 1, 3])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.1)

        # sub-panel enumerations
        plt.figtext(0.06, 0.97, 'Correlation coefficients between ROIs', clip_on=False, color='black', weight='bold', size=12)

        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])
        ##################################
        for n in range(len(aD910)):
            ax0=plt.subplot(gs[n])
            ax0.text(0.05,0.9,'m: %s' % aD910[n][0],transform=ax0.transAxes,fontsize=9,color='0.5')
            allCorrEffs = aD910[n][1]
            allCorrActEffs = aD910Act[n][1]
            allCorrEffs820 = aD820[n][1]
            allCorrActEffs820 = aD820Act[n][1]
            ax0.hist(allCorrEffs,lw=2,bins=100,histtype='step',color='C0',density=True,label='entire recording')
            #ax0.hist(allCorrEffs820, lw=2, bins=100, histtype='step', color='C0', density=True, alpha=0.2)
            ax0.hist(allCorrActEffs,lw=2, bins=100, histtype='step',color='C1', density=True,label='motorization')
            #ax0.hist(allCorrActEffs820, lw=2, bins=100, histtype='step',color='C1',alpha=0.2, density=True, label='motorization')
            ax0.axvline(x=0,ls='--',lw=2,c='0.7')
            #ax0.axvline(x=np.mean(allCorrEffs),lw=2,color='C1')
            ax0.set_xlim(-1,1)
            print(np.mean(allCorrEffs))
            if n in [0,4] :
                self.layoutOfPanel(ax0, xLabel=None, yLabel='frequency',xyInvisible=[True,False])
            elif n in [8]:
                self.layoutOfPanel(ax0, xLabel='correlation', yLabel='frequency')
            elif n in [9,10]:
                self.layoutOfPanel(ax0, xLabel='correlation', yLabel=None, xyInvisible=[False, True])
            elif n in [11]:
                self.layoutOfPanel(ax0, xLabel='correlation', yLabel=None, xyInvisible=[False, True],Leg=[(0.6,0.8),9])
            else:
                self.layoutOfPanel(ax0, xLabel=None, yLabel=None, xyInvisible=[True, True])


        if onlyAligned:
            fname = 'ca-correlation-coefficient_distribution_algined'
        else:
            fname = 'ca-correlation-coefficient_distribution'
        #fname = 'ca-correlation-coefficient_distribution'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

        plt.clf()
        # figure #################################
        fig_width = 9  # width in inches
        fig_height = 7  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 1,  # ,
                               # width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               # height_ratios=[10, 1, 3])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.22)

        # sub-panel enumerations
        plt.figtext(0.06, 0.97, 'Correlation coefficients between ROIs', clip_on=False, color='black', weight='bold', size=12)

        ax0 = plt.subplot(gs[0])
        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])
        ##################################
        for n in range(len(aD910)):
            #ax0 = plt.subplot(gs[n])
            #ax0.text(0.05, 0.9, 'm: %s' % aD910[n][0], transform=ax0.transAxes, fontsize=9, color='0.5')
            allCorrEffs = aD910[n][1]
            allCorrActEffs = aD910Act[n][1]
            allCorrEffs820 = aD820[n][1]
            allCorrActEffs820 = aD820Act[n][1]
            c = 'C0'
            ax0.plot(([2*n,2*n+1]),([np.median(allCorrEffs),np.median(allCorrActEffs)]),c='black',lw=1)
            ax0.boxplot(allCorrEffs,  positions=[2*n], showfliers=False, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            #flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='black'),
            )
            c2 = 'C1'
            ax0.boxplot(allCorrActEffs, positions=[2*n+1], showfliers=False, patch_artist=True,
            boxprops=dict(facecolor=c2, color=c2),
            capprops=dict(color=c2),
            whiskerprops=dict(color=c2),
            #flierprops=dict(color=c2, markeredgecolor=c2),
            medianprops=dict(color='black'),
            )
            #ax0.hist(allCorrEffs, lw=2, bins=100, histtype='step', color='C0', density=True, label='entire recording')
            #ax0.hist(allCorrActEffs, lw=2, bins=100, histtype='step', color='C1', density=True, label='motorization')
            ax0.axhline(y=0, ls='--', lw=2, c='0.7')
            # ax0.axvline(x=np.mean(allCorrEffs),lw=2,color='C1')
            ax0.set_ylim(-1, 1)
            #print(np.mean(allCorrEffs))
            #if n in [0, 4]:
            self.layoutOfPanel(ax0, xLabel='animal', yLabel='correlation')
            plt.xticks(np.arange(0,2*len(mouseList),2),mouseList, rotation = 45, ha='right')
            #ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
            #elif n in [8]:
            #    self.layoutOfPanel(ax0, xLabel='correlation', yLabel='frequency')
            #elif n in [9, 10]:
            #    self.layoutOfPanel(ax0, xLabel='correlation', yLabel=None, xyInvisible=[False, True])
            #elif n in [11]:
            #    self.layoutOfPanel(ax0, xLabel='correlation', yLabel=None, xyInvisible=[False, True], Leg=[(0.6, 0.8), 9])
            #else:
            #    self.layoutOfPanel(ax0, xLabel=None, yLabel=None, xyInvisible=[True, True])

        if onlyAligned:
            fname = 'ca-correlation-coefficient_boxplot_algined'
        else:
            fname = 'ca-correlation-coefficient_boxplot'
        # fname = 'ca-correlation-coefficient_distribution'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ########################################################################################################
    def generatePCACorrelationPlot(self, allCorrData910, allCorrData820, analysisDate, onlyAligned=False):
        #corrs = np.zeros((len(correlationData), len(varExplained[0]), 5))
        #for nSess in range(len(correlationData)):
        #    corrs[nSess] = np.abs(correlationData[nSess][4][:, [1, 3, 5, 7, 9]])
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        pcaCorrMotor = []
        for n in range(len(allCorrData910)):
            mouse = allCorrData910[n][1]
            if onlyAligned:
                correlationData = allCorrData910[n][4]
                varExplained = allCorrData910[n][5]
            else:
                correlationData = allCorrData910[n][2]
                varExplained = allCorrData910[n][3]
            intersectDataTemp = allCorrData910[n][6]
            if onlyAligned:
                (intersectData, nDays, nRois, idxRef, refDay, dayList) = dataAnalysis.choseAlignmentReferenceDay(mouse, intersectDataTemp)
                daysInclude = dayList
            else:
                daysInclude = list(range(len(correlationData)))

            corrs = np.zeros((len(daysInclude), len(varExplained[0]), 5))
            n=0
            for nSess in range(len(correlationData)):
                if nSess in daysInclude:
                    corrs[n] = np.abs(correlationData[nSess][4][:, [1, 3, 5, 7, 9]])
                    n+=1

            pcaCorrMotor.append([mouse,corrs])

        for n in range(len(pcaCorrMotor)):
            print(pcaCorrMotor[n][0],np.shape(pcaCorrMotor[n][1]))

        #pdb.set_trace()
        # ax1 = plt.subplot(gs[1])
        # barWidth=0.1
        # ddd = 0.8
        # ax1.axhline(y=0,ls='--',color='0.8')
        # ax1.bar(np.arange(len(varExplained[0]))+ddd, np.mean(corrs[:,:,0],axis=0),color='0.3', width=barWidth, edgecolor='white',label='wheel')
        # ax1.bar(np.arange(len(varExplained[0]))+ddd+barWidth, np.mean(corrs[:,:,1],axis=0),color='C0',width=barWidth, edgecolor='white',label='FL paw')
        # ax1.bar(np.arange(len(varExplained[0]))+ddd+2*barWidth, np.mean(corrs[:,:,2],axis=0),color='C1', width=barWidth, edgecolor='white',label='FR paw')
        # ax1.bar(np.arange(len(varExplained[0]))+ddd+3*barWidth, np.mean(corrs[:,:,3],axis=0),color='C2', width=barWidth, edgecolor='white',label='HR paw')
        # ax1.bar(np.arange(len(varExplained[0]))+ddd+4*barWidth, np.mean(corrs[:,:,4],axis=0),color='C3', width=barWidth, edgecolor='white',label='HL paw')
        # ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # self.layoutOfPanel(ax1, yLabel='mean correlation',Leg=[1,9])

        # figure #################################
        fig_width = 5.5  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1,1,  # ,
                               #width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               #height_ratios=[10, 1, 3])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.18, right=0.94, top=0.95, bottom=0.2)

        # sub-panel enumerations
        if onlyAligned:
            plt.figtext(0.06, 0.97, 'Correlations with 1. principal component (motorization period only)', clip_on=False, color='black', size=12)
        else:
            plt.figtext(0.06, 0.97, 'Correlations with 1. principal component (entire rec. period)', clip_on=False, color='black', size=12)

        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])
        ##################################
        ax0 = plt.subplot(gs[0])
        for n in range(len(pcaCorrMotor)):
            #ax0.text(0.05,0.9,'m: %s' % aD910[n][0],transform=ax0.transAxes,fontsize=9,color='0.5')
            #print(pcaCorrMotor[n][0], np.shape(pcaCorrMotor[n][1]))
            #for i in range(2):
            ax0.plot(np.arange(4), np.mean(pcaCorrMotor[n][1][:,0,:],axis=0)[1:],'o-',color=cmap(n/len(pcaCorrMotor)))

        self.layoutOfPanel(ax0, xLabel=None, yLabel='correlation')
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        labels = [item.get_text() for item in ax0.get_xticklabels()]
        labels = ['0', 'FL speed', 'FR speed', 'HR speed', 'HL speed']

        ax0.set_xticklabels(labels,rotation=45,ha='right')

        if onlyAligned:
            fname = 'pca-correlation-coefficient_distribution_algined'
        else:
            fname = 'pca-correlation-coefficient_distribution'
        #fname = 'ca-correlation-coefficient_distribution'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ##########################################################################################
    # def createStepNumberFigure(self,  averageStepsAllMice,sumStepsAllMice,avgValue,sem,sessionStepAverage,AnovaRMRes,AnovaRMSesRes,mdf,conf_int):
    def createStepNumberFigure(self,  strideNumberData, experiment, sessionValues=True):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        cmap2 = cm.get_cmap('Reds')
        nDays = []
        xArray=[]
        animalNames = []
        for i in range(len(strideNumberData[1])):
            animalNames.append(strideNumberData[1][i][0])
        print(animalNames)
        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[1,1],
                               height_ratios=[1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.4, wspace=0.3)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.4, wspace=0.3)
        # plt.figtext(0.35, 0.95, "Average stride number", clip_on=False, color='black',size=14)


                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        #pdb.set_trace()
        for f in range (len(strideNumberData[1])):
            nDays = len(strideNumberData[1][f][1])


            ax0.plot(np.arange(nDays)[:11]+1,strideNumberData[0][f][1][:11], 'o-',ms=2, label=strideNumberData[1][f][0], color=cmap(f/len(strideNumberData[1])),alpha=0.2)

        ax0.legend(loc="upper left", bbox_to_anchor=(10,0.1))
        stepNb11days=strideNumberData[2][0][0:11]
        stepNbSEM11days=strideNumberData[2][1][0:11]
        #ax0.text(0, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        multiplier = 0
        if strideNumberData[2][2][1]<0.001:
            multiplier = 3
        elif strideNumberData[2][2][1]<0.01:
            multiplier = 2
        elif strideNumberData[2][2][1]<0.05:
            multiplier = 1
        ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(strideNumberData[0]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')
        ax0.text(0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' % (strideNumberData[2][2][0], strideNumberData[2][2][1],strideNumberData[2][2][2],strideNumberData[2][2][3]),ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=5,color='0.7')
        ax0.plot(np.arange(10)+1,stepNb11days,'-',label=None, linewidth=2,c='k')
        #ax0.errorbar(np.arange(11),stepNb11days, yerr=stepNbSEM11days, color=cmap(0.01) )
        plt.fill_between(np.arange(len(stepNb11days))+1, stepNb11days - stepNbSEM11days,stepNb11days + stepNbSEM11days, color='0.6', alpha=0.2)
        ax0.set_title('Mean stride number (all paws)')
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride number (avg.)')#, Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        (PawStepNumber, meanPawStepNumber, semPawStepNumber)=groupAnalysis.getAverageSingleGroup(strideNumberData[3])
        (PawStepNumberDf, PawStepNumberPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            strideNumberData[3],sessionValues=True, treatments=False)

        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        for i in range(4):
            (pawStar)=groupAnalysis.starMultiplier(PawStepNumberPValues[i])

            ax1=plt.subplot(gs[1])
            ax1.plot(np.arange(10) + 1, meanPawStepNumber[:,i][:11], '-', label='%s %s'%(pawId[i], pawStar), linewidth=2, c=colors[i])

            plt.fill_between(np.arange(10) + 1, meanPawStepNumber[:,i][:11] - semPawStepNumber[:,i][:11], meanPawStepNumber[:,i][:11] + semPawStepNumber[:,i][:11], color=colors[i], alpha=0.1)

            self.layoutOfPanel(ax1, xLabel=None, yLabel='Stride number (avg.)')#, Leg=[1, 9])

        ax1.set_title('Mean stride number (paw specific)')
        ax1.legend(loc="upper right", frameon=False, fontsize=8)

        ax1.xaxis.set_major_locator(majorLocator_x)


        fname = '%s_mean_steps_number'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')
        if sessionValues==True:
            gs = gridspec.GridSpec(1, 3,  # ,
                                   width_ratios=[1, 1, 1],
                                   height_ratios=[1]
                                   )

            ax3 = plt.subplot(gs[0])

            for m in range (len(strideNumberData[1])):
                nDays=len(strideNumberData[1][m][1])

                for n in range(nDays):
                    xArray= np.repeat(np.arange(nDays), 10)
                    xArray=xArray+np.tile(np.arange(10)/7,nDays)

                    animalSteps= strideNumberData[1][m][1].flatten()
                # pdb.set_trace()
                ax3.plot(xArray,animalSteps+m*90, 'o-', color=cmap(m/len(strideNumberData[1])), label=None)  # if mouseList[m] not in plt.gca().get_legend_handles_labels() [-1] else '')
            #ax3.text(0.7, 0.04, 'AnovaRM: F value=%s, p value=%s'%(round(AnovaRMSesRes.anova_table.iloc[0, 2],4),round(AnovaRMSesRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax1.transAxes, fontsize=9)
            # ax1.text(0.7,0.01, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s , %s)' %(round(mdf.fe_params['sessionNumber'],4), round(mdf.pvalues['sessionNumber'],4),round(conf_int.iloc[2,0],3),round(conf_int.iloc[2,1],3)),ha='left', va='center', transform=ax1.transAxes, fontsize=9)
            # stepAvg =[sessionStepAverage[r][2] for r in range (len(sessionStepAverage))]
            (sessionStepAverageArray, sessionStepAverage) = groupAnalysis.get2DArrayTolerantAverage(strideNumberData[1])
            xArray1=[]
            for n in range(14):
                xArray1 = np.repeat(np.arange(14),10)
                # pdb.set_trace()

                xArray1= xArray1 +np.tile(np.arange(10)/7,14)
                xArray1=xArray1.reshape(14,10)
                sessionStepAverage=np.asarray(sessionStepAverage)
                sessionStepAverage= sessionStepAverage.flatten()
            # pdb.set_trace()
            ax3.plot(xArray1.flatten(), sessionStepAverage+n*50, '-o',color='black', label="average")
            # #plt.legend()
            self.layoutOfPanel(ax3, xLabel='recordings', yLabel='average steps during trial', Leg=[1, 9],xyInvisible=[False,False])
            # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)
        #
        # plt.show()
        # pdb.set_trace()
        # # np.save('testScripts/meanStepNumber.npy',np.asarray(sN))
        fname = '%ssteps_number_trial_values'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')

    ##########################################################################################
    def createStepDurationFigure(self, stepDurationData,experiment):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')
        nDays = []
        xArray = []
        animalNames = []
        for i in range(len(stepDurationData[0])):
            animalNames.append(stepDurationData[0][i][0])
        print(animalNames)
        fig_width = 9  # width in inches
        fig_height = 4  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 3,  # ,
                               width_ratios=[5,1,1],
                               height_ratios=[1,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.6, hspace=0.25)
        plt.figtext(0.30, 0.95, "Median swing/stance duration", clip_on=False, color='black',size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[:,:1])

        for f in range(len(stepDurationData[0])):
            nDays = len(stepDurationData[0][f][1])
            animalSteps = stepDurationData[0][f]
            # sns.relplot(x="recording days", y="average steps", kind="line", data=animalSteps);
            ax0.plot(np.arange(nDays)[:11]+1, stepDurationData[0][f][1][:11], 'o-', ms=2,label=None, color=cmap(f / len(stepDurationData[0])),alpha=0.2)
            ax0.plot(np.arange(nDays)[:11]+1, stepDurationData[2][f][1][:11], 'o-', ms=2,label=None,color=cmap(f / len(stepDurationData[0])), alpha=0.2)
        swingDuration11days = stepDurationData[3][0][0:11]
        swingDurationdaysError = stepDurationData[3][1][0:11]
        stanceDuration11days=stepDurationData[4][0][0:11]
        stanceDuration11daysError=stepDurationData[4][1][0:11]

        ax0.legend(loc="upper left", bbox_to_anchor=(10, 0.1))
        multiplier = 0
        if stepDurationData[4][2][1]<0.001:
            multiplier = 3
        elif stepDurationData[4][2][1]<0.01:
            multiplier = 2
        elif stepDurationData[4][2][1]<0.05:
            multiplier = 1
        #ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=20, color='k')
        ax0.text(0.78, 0.83, '(N=%s)' % len(stepDurationData[0]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')

        ax0.text(0.0, 0, 'Swing--MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[3][2][0], stepDurationData[3][2][1], stepDurationData[3][2][2],stepDurationData[3][2][3]), ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=5,color='0.7')
        ax0.text(0, 0.025, 'Stance--MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[4][2][0], stepDurationData[4][2][1], stepDurationData[4][2][2],stepDurationData[4][2][3]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=5,color='0.8')
        #ax0.text(0.40, 0.04, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResStance.anova_table.iloc[0, 2], 4), round(AnovaRMResStance.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7,color=cmap(0.1))


        ax0.plot(np.arange(len(swingDuration11days))+1, swingDuration11days, '--', label="swing  n.s.", color='k',linewidth=2)

        ax0.plot(np.arange(len(stanceDuration11days))+1, stanceDuration11days, '-', label="stance  %s"%('*'*multiplier), color='0.5', linewidth=2)
        plt.fill_between(np.arange(len(swingDuration11days))+1, swingDuration11days - swingDurationdaysError, swingDuration11days + swingDurationdaysError,color='0.6', alpha=0.2)
        plt.fill_between(np.arange(len(stanceDuration11days))+1, stanceDuration11days - stanceDuration11daysError,stanceDuration11days + stanceDuration11daysError, color='0.7', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Median swing/stance duration (s)', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        
        (PawSwingDuration, meanPawSwingDuration, semPawSwingDuration)=groupAnalysis.getAverageSingleGroup(stepDurationData[6])
        (swingDf, swingDurPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            stepDurationData[6],sessionValues=True, treatments=False)

        (FLSwingStar)=groupAnalysis.starMultiplier(swingDurPValues[0])
        (FRSwingStar) = groupAnalysis.starMultiplier(swingDurPValues[1])
        (HLSwingStar)=groupAnalysis.starMultiplier(swingDurPValues[2])
        (HRSwingStar) = groupAnalysis.starMultiplier(swingDurPValues[3])

        (PawStanceDuration, meanPawStanceDuration, semPawStanceDuration)=groupAnalysis.getAverageSingleGroup(stepDurationData[7])
        (stanceDf, stanceDurPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            stepDurationData[7], sessionValues=True, treatments=False)

        (FLStanceStar) = groupAnalysis.starMultiplier(stanceDurPValues[0])
        (FRStanceStar) = groupAnalysis.starMultiplier(stanceDurPValues[1])
        (HLStanceStar) = groupAnalysis.starMultiplier(stanceDurPValues[2])
        (HRStanceStar) = groupAnalysis.starMultiplier(stanceDurPValues[3])

        ax1=plt.subplot(gs[0,1:3])
        ax1.plot(np.arange(11) + 1, meanPawSwingDuration[:,0][:11], '--', label='FL %s'%FLSwingStar, linewidth=2, c='steelblue')
        ax1.plot(np.arange(11) + 1, meanPawSwingDuration[:, 1][:11], '--', label='FR %s'%FRSwingStar, linewidth=2, c='darkorange')
        plt.fill_between(np.arange(11) + 1, meanPawSwingDuration[:,0][:11] - semPawSwingDuration[:,0][:11], meanPawSwingDuration[:,0][:11] + semPawSwingDuration[:,0][:11], color='steelblue', alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawSwingDuration[:, 1][:11] - semPawSwingDuration[:, 1][:11],meanPawSwingDuration[:, 1][:11] + semPawSwingDuration[:, 1][:11], color='darkorange', alpha=0.1)
        self.layoutOfPanel(ax1, xLabel='Days', yLabel=None, Leg=[1, 2])
        ax1.plot(np.arange(11) + 1, meanPawStanceDuration[:,0][:11], '-', label='FL %s'%FLStanceStar, linewidth=2, c='steelblue')
        ax1.plot(np.arange(11) + 1, meanPawStanceDuration[:, 1][:11], '-', label='FR %s'%FRStanceStar, linewidth=2, c='darkorange')
        plt.fill_between(np.arange(11) + 1, meanPawStanceDuration[:,0][:11] - semPawStanceDuration[:,0][:11], meanPawStanceDuration[:,0][:11] + semPawStanceDuration[:,0][:11], color='steelblue', alpha=0.25)
        plt.fill_between(np.arange(11) + 1, meanPawStanceDuration[:, 1][:11] - semPawStanceDuration[:, 1][:11],meanPawStanceDuration[:, 1][:11] + semPawStanceDuration[:, 1][:11], color='darkorange', alpha=0.25)
        
        
        
        ax2 = plt.subplot(gs[1,1:3])
        ax2.plot(np.arange(11) + 1, meanPawSwingDuration[:,2][:11], '--', label='HL %s'%HLSwingStar, linewidth=2, c='yellowgreen')
        ax2.plot(np.arange(11) + 1, meanPawSwingDuration[:, 3][:11], '--', label='HR %s'%HRSwingStar, linewidth=2, c='salmon')
        plt.fill_between(np.arange(11) + 1, meanPawSwingDuration[:,2][:11] - semPawSwingDuration[:,2][:11], meanPawSwingDuration[:,2][:11] + semPawSwingDuration[:,2][:11], color='yellowgreen', alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawSwingDuration[:, 3][:11] - semPawSwingDuration[:, 3][:11],meanPawSwingDuration[:, 3][:11] + semPawSwingDuration[:, 3][:11], color='salmon', alpha=0.1)
        ax2.plot(np.arange(11) + 1, meanPawStanceDuration[:,2][:11], '-', label='HL %s'%HLStanceStar, linewidth=2, c='yellowgreen')
        ax2.plot(np.arange(11) + 1, meanPawStanceDuration[:, 3][:11], '-', label='HR %s'%HRStanceStar, linewidth=2, c='salmon')
        plt.fill_between(np.arange(11) + 1, meanPawStanceDuration[:,2][:11] - semPawStanceDuration[:,2][:11], meanPawStanceDuration[:,2][:11] + semPawStanceDuration[:,2][:11], color='yellowgreen', alpha=0.25)
        plt.fill_between(np.arange(11) + 1, meanPawStanceDuration[:, 3][:11] - semPawStanceDuration[:, 3][:11],meanPawStanceDuration[:, 3][:11] + semPawStanceDuration[:, 3][:11], color='salmon', alpha=0.25)
        self.layoutOfPanel(ax2, xLabel='Days', yLabel=None, Leg=[1, 2])
        ax1.legend(loc="upper right", frameon=False, fontsize=6)
        ax2.legend(loc="upper right", frameon=False, fontsize=6)
        ax1.xaxis.set_major_locator(majorLocator_x)
        ax2.xaxis.set_major_locator(majorLocator_x)

        #
        # # Second subplot
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4)
        # ax1 = []
        # ax1 = plt.subplot(gs[1])
        # # sessionNb=np.arrange(1,6)
        # for m in range(len(allSessionStepDurationMean)):
        #     nDays = len(allSessionStepDurationMean[m][1])
        #     # print(len(sumStepsAllMice),nDays)
        #     for n in range(nDays):
        #         xArray = np.repeat(np.arange(nDays), 5)
        #         xArray = xArray + np.tile(np.arange(5) / 7, nDays)
        #         # print(xArray)
        #         # sns.relplot(x=nDays, y=sumStepsAllMice.iloc[m,1], kind='line',data=sumStepsAllMice);
        #         # animalSteps=sumStepsAllMiceDf.iloc[m,1].flatten()
        #         animalSteps = allSessionStepDurationMean[m][1].flatten()
        #         # print(animalSteps)
        #         ax1.plot(xArray, animalSteps + m * 0.5, 'o-', color=cmap(m / len(allSessionStepDurationMean)),
        #                  label=None)  # if mouseList[m] not in plt.gca().get_legend_handles_labels() [-1] else '')
        # ax1.text(0.7, 0.04, 'AnovaRM: F value=%s, p value=%s' % (
        # round(AnovaRMSesRes.anova_table.iloc[0, 2], 4), round(AnovaRMSesRes.anova_table.iloc[0, 3], 4)), ha='left',
        #          va='center', transform=ax1.transAxes, fontsize=9)
        # ax1.text(0.7, 0.01, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s , %s)' % (
        # round(mdf.fe_params['sessionNumber'], 4), round(mdf.pvalues['sessionNumber'], 4), round(conf_int.iloc[2, 0], 3),
        # round(conf_int.iloc[2, 1], 3)), ha='left', va='center', transform=ax1.transAxes, fontsize=9)
        # # stepAvg =[sessionStepAverage[r][2] for r in range (len(sessionStepAverage))]
        #
        # for n in range(14):
        #     xArray1 = np.repeat(np.arange(14), 5)
        #     xArray1 = xArray1 + np.tile(np.arange(5) / 7, 14)
        #     # xArray1=xArray1.reshape(14,5)
        #     allSessionStepDurationMeanAverage = np.asarray(allSessionStepDurationMeanAverage)
        #     allSessionStepDurationMeanAverage = allSessionStepDurationMeanAverage.flatten()
        # plt.plot(xArray1, allSessionStepDurationMeanAverage + n * 0.1, '-o', color='black', label="average")
        # # plt.legend()
        # self.layoutOfPanel(ax1, xLabel='recordings', yLabel='mean steps duration during trial', Leg=[1, 9],
        #                    xyInvisible=[False, False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)

        #plt.show()
        #pdb.set_trace()
        # np.save('testScripts/meanStepNumber.npy',np.asarray(sN))
        fname = '%s_steps_duration'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')
   ##########################################################################################
    def createStepSpeedFigure(self,  pawSpeedData, experiment):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        nDays = []
        xArray=[]
        animalNames = []
        for i in range(len(pawSpeedData[1])):
            animalNames.append(pawSpeedData[1][i][0])
        print(animalNames)
        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[1,1],
                               height_ratios=[1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)
        # plt.figtext(0.35, 0.95, "Average swing speed", clip_on=False, color='black',size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        for f in range (len(pawSpeedData[0])):
            nDays = len(pawSpeedData[0][f][1])

            ax0.plot(np.arange(nDays)[:11]+1,pawSpeedData[0][f][1][:11], 'o-', label=None, color=cmap(f/len(pawSpeedData[1])),ms=2,alpha=0.2)
        ax0.legend(bbox_to_anchor=(1, 0.5),loc="upper left")
        plt.tight_layout()
        #ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        ax0.text(0.0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' %(pawSpeedData[2][2][0], pawSpeedData[2][2][1],pawSpeedData[2][2][2],pawSpeedData[2][2][3]),ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=5,color='0.7')
        multiplier = 0
        if pawSpeedData[2][2][1]<0.001:
            multiplier = 3
        elif pawSpeedData[2][2][1]<0.01:
            multiplier = 2
        elif pawSpeedData[2][2][1]<0.05:
            multiplier = 1
        ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(pawSpeedData[1]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')
        ax0.set_title('Mean swing speed (all paws)')
        speed11days = pawSpeedData[2][0][0:11]
        speed11daysError = pawSpeedData[2][1][0:11]

        #ax0.errorbar(np.arange(11), speed11days, yerr=speed11daysError, color=cmap(0.01))

        ax0.plot(np.arange(len(speed11days))+1,speed11days,'-',label=None, color='k',linewidth=2)
        ax0.fill_between(np.arange(len(speed11days))+1,speed11days-speed11daysError,speed11days+speed11daysError, color='0.6', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='mean swing speed (cm/s)', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)


        ax0.xaxis.set_major_locator(majorLocator_x)
        (PawStepNumber, meanPawStepNumber, semPawStepNumber)=groupAnalysis.getAverageSingleGroup(pawSpeedData[3])

        (pawSpeedDataDf, pawSpeedDataPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            pawSpeedData[3],sessionValues=True, treatments=False)
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        for i in range(4):
            (pawStar)=groupAnalysis.starMultiplier(pawSpeedDataPValues[i])
            ax1=plt.subplot(gs[1])
            ax1.plot(np.arange(10) + 1, meanPawStepNumber[:,i][:11], '-', label='%s %s'%(pawId[i], pawStar), linewidth=2, c=colors[i])

            ax1.fill_between(np.arange(10) + 1, meanPawStepNumber[:,i][:11] - semPawStepNumber[:,i][:11], meanPawStepNumber[:,i][:11] + semPawStepNumber[:,i][:11], color=colors[i], alpha=0.1)

            self.layoutOfPanel(ax1, xLabel='Days', yLabel=None, Leg=[1, 2])


        ax1.legend(loc="upper right", frameon=False, fontsize=8)
        ax1.set_title('Mean swing speed (paw specific)')
        ax1.xaxis.set_major_locator(majorLocator_x)


        #  #Second subplot
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4)
        # ax1 = []
        # ax1 = plt.subplot(gs[1])
        # #sessionNb=np.arrange(1,6)
        # for m in range (len(allPawSessionSpeed)):
        #     nDays=len(allPawSessionSpeed[m][1])
        #     #print(len(sumStepsAllMice),nDays)
        #     for n in range(nDays):
        #         xArray= np.repeat(np.arange(nDays), 5)
        #         xArray=xArray+np.tile(np.arange(5)/7,nDays)
        #         #print(xArray)
        #         #sns.relplot(x=nDays, y=sumStepsAllMice.iloc[m,1], kind='line',data=sumStepsAllMice);
        #         #animalSteps=sumStepsAllMiceDf.iloc[m,1].flatten()
        #         animalSteps= allPawSessionSpeed[m][1].flatten()
        #         #print(animalSteps)
        #         ax1.plot(xArray,animalSteps+m*10, 'o-', color=cmap(m/len(allPawSessionSpeed)), label=None)  # if mouseList[m] not in plt.gca().get_legend_handles_labels() [-1] else '')
        # ax1.text(0.7, 0.04, 'AnovaRM: F value=%s, p value=%s'%(round(AnovaRMSesRes.anova_table.iloc[0, 2],4),round(AnovaRMSesRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax1.transAxes, fontsize=9)
        # ax1.text(0.7,0.00, 'MixedLM: Coef. days=%s, p value=%s,  95conf-int=(%s ,%s)' %(round(mdf.fe_params['sessionNumber'],4), round(mdf.pvalues['sessionNumber'],4),round(conf_int.iloc[2,0],3),round(conf_int.iloc[2,1],3)),ha='left', va='center', transform=ax1.transAxes, fontsize=9)
        # stepAvg=[]
        #
        #
        # for n in range(14):
        #     xArray1 = np.repeat(np.arange(14),5)
        #     xArray1= xArray1 +np.tile(np.arange(5)/7,14)
        #     #xArray1=xArray1.reshape(14,5)
        #     stepAvg=np.asarray(allPawSessionSpeedAvg)
        #     stepAvg= stepAvg.flatten()
        # plt.plot(xArray1, stepAvg+n*10, '-o',color='black', label="average")
        # #plt.legend()
        # self.layoutOfPanel(ax1, xLabel='recordings', yLabel='average paw speed during trial', Leg=[1, 9],xyInvisible=[False,False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)
        #
        # ax2= plt.subplot(gs[2])
        # #print("voilà",allSpeedHighNb)
        #
        # #print("look again", allSpeedHighNb[1][1])
        # for q in range (len(all90PercSpeed)):
        #
        #     nDays = len(all90PercSpeed[q][1])
        #
        #     ax2.plot(np.arange(nDays),all90PercSpeed[q][1], 'o-', label=None, color=cmap(q/len(all90PercSpeed)))
        # plt.plot(np.arange(len(all90PercSpeedAvg)), all90PercSpeedAvg, '-', label="average", color='black')
        # ax2.text(0.6, 0.01, 'AnovaRM: F value=%s, p value=%s' % (round(Anova90PercSpeed.anova_table.iloc[0, 2],4), round(Anova90PercSpeed.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax2.transAxes, fontsize=9)
        # self.layoutOfPanel(ax2, xLabel='recordings', yLabel='swings speed 90 percentile', Leg=[1, 9])
        #plt.show()
        #pdb.set_trace()
        # np.save('testScripts/meanStepNumber.npy',np.asarray(sN))
        fname = '%s_swing_speed'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')

    ##########################################################################################
    def createRungCrossedFigure(self, rungCrossedData, experiment):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        nDays = []
        xArray=[]
        animalNames = []
        for i in range(len(rungCrossedData[0])):
            animalNames.append(rungCrossedData[0][i][0])
        print(animalNames)
        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[1,1],
                               height_ratios=[1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)
        # plt.figtext(0.3, 0.95, 'Fraction of ' + r'$\geqq$' +'2 rungs crossed', clip_on=False, color='black',size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        #plt.figtext(0.06, 0.96, animalNames, clip_on=False, color='black',size=10)

                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        nRec = 0

        for f in range (len(rungCrossedData[0])):
            nDays = len(rungCrossedData[0][f][1])

            ax0.plot(np.arange(nDays)[:11]+1,rungCrossedData[0][f][1][:11], 'o-', label=None, color=cmap(f/len(rungCrossedData[0])),alpha=0.2,ms=2)

        ax0.legend(loc="upper left", bbox_to_anchor=(10,0.1))
        #ax0.text(0.4, 0.025, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResAll.anova_table.iloc[0, 2], 4), round(AnovaRMResAll.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7)
        ax0.text(0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (rungCrossedData[2][2][0], rungCrossedData[2][2][1],rungCrossedData[2][2][2], rungCrossedData[2][2][2]), ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=6,color='0.7')
        multiplier = 0
        if rungCrossedData[2][2][1]<0.001:
            multiplier = 3
        elif rungCrossedData[2][2][1]<0.01:
            multiplier = 2
        elif rungCrossedData[2][2][1]<0.05:
            multiplier = 1
        ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(rungCrossedData[0]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')
        rungCross = rungCrossedData[2][0]#[0:11]
        rungCrossError = rungCrossedData[2][1]#[0:11]

        #ax0.errorbar(np.arange(11), rungCrossError, yerr=rungCrossError, color=cmap(0.01))

        ax0.plot(np.arange(len(rungCross))+1,rungCross,'-',label=None, color='k',linewidth=2)
        plt.fill_between(np.arange(len(rungCross))+1,rungCross-rungCrossError,rungCross+rungCrossError, color='0.6', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed', Leg=[1, 9])
        ax0.set_title('Fraction of ' + r'$\geqq$' +'2 rungs crossed (all paws)')
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        #perform the stats
        (pawRungCrossed, meanPawRungCrossed, semPawRungCrossed)=groupAnalysis.getAverageSingleGroup(rungCrossedData[3])
        (swingDf, pawPValues) = groupAnalysis.PawListToPandasDFAndMxLM(rungCrossedData[3],sessionValues=True, treatments=False)



        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        for i in range(4):
            (pawStar)=groupAnalysis.starMultiplier(pawPValues[i])
            ax1=plt.subplot(gs[1])
            ax1.plot(np.arange(10) + 1, meanPawRungCrossed[:,i][:11], '-', label='%s %s'%(pawId[i],pawStar), linewidth=2, c=colors[i])
            ax1.fill_between(np.arange(10) + 1, meanPawRungCrossed[:,i][:11] - semPawRungCrossed[:,i][:11], meanPawRungCrossed[:,i][:11] + semPawRungCrossed[:,i][:11], color=colors[i], alpha=0.1)

            self.layoutOfPanel(ax1, xLabel='Days', yLabel=None)#, Leg=[1, 9])


        ax1.set_title('Fraction of ' + r'$\geqq$' +'2 rungs crossed (paw specific)')
        ax1.legend(loc="upper right", frameon=False, fontsize=8)

        ax1.xaxis.set_major_locator(majorLocator_x)

        # # Second subplot
        # gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.4)
        # ax1 = []
        # ax1 = plt.subplot(gssub1[0])
        # # sessionNb=np.arrange(1,6)
        # for f in range(len(frontPawPercentage)):
        #     nDays = len(frontPawPercentage[f][1])
        #
        #     ax1.plot(np.arange(nDays), frontPawPercentage[f][1], 'o-', label=frontPawPercentage[f][0],color=cmap(f / len(frontPawPercentage)))
        # ax1.text(0.6, 0.01, 'AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResFront.anova_table.iloc[0, 2],4), round(AnovaRMResFront.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax1.transAxes, fontsize=9)
        # plt.plot(np.arange(len(frontPawsAvg)),frontPawsAvg,'-',label="average", color='black')
        # self.layoutOfPanel(ax1, xLabel='recordings', yLabel='Front paws fraction of >2 rungs cross', Leg=[1, 9],xyInvisible=[False, False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)
        #
        # # Third subplot
        # gssub2 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[2], hspace=0.4)
        # ax2 = []
        # ax2 = plt.subplot(gs[2])
        # # sessionNb=np.arrange(1,6)
        # for f in range(len(frontPawPercentage)):
        #     nDays = len(frontPawPercentage[f][1])
        #
        #     ax2.plot(np.arange(nDays), hindPawPercentage[f][1], 'o-', label=hindPawPercentage[f][0],
        #              color=cmap(f / len(hindPawPercentage)))
        # ax2.text(0.6, 0.01, 'AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResHind.anova_table.iloc[0, 2],4), round(AnovaRMResHind.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax2.transAxes, fontsize=9)
        # plt.plot(np.arange(len(hindPawsAvg)), hindPawsAvg, '-', label="average",color='black')
        # self.layoutOfPanel(ax2, xLabel='recordings', yLabel='Hind paws fraction of >2 rungs cross',Leg = [1, 9],xyInvisible = [False, False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)

        #plt.show()
        #pdb.set_trace()
        fname = '%s_Fraction_of_rungs_crossed'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')

    def createStrideLenghtFigure(self, strideLengthData, experiment):
        from matplotlib import cm

        cmap=cm.get_cmap('tab20')


        fig_width = 12  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid':False   # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 2,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[1, 2.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure

        # create figure instance
        fig = plt.figure()
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        #plt.figtext(0.06, 0.96, animalNames, clip_on=False, color='black',size=10)


                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])


        for f in range(len(strideLengthData[0])):
            nDays = len(strideLengthData[0][f][1])

            ax0.plot(np.arange(nDays)[:11] + 1, strideLengthData[0][f][1][:11], 'o-', label=None,
                     color=cmap(f / len(strideLengthData[0])), ms=2, alpha=0.2)
        ax0.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
        plt.tight_layout()
        # ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        ax0.text(0.0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (
        strideLengthData[2][2][0], strideLengthData[2][2][1], strideLengthData[2][2][2], strideLengthData[2][2][3]), ha='left',
                 va='center', transform=ax0.transAxes, style='italic', fontsize=8, color='0.7')
        multiplier = 0
        if strideLengthData[2][2][1] < 0.001:
            multiplier = 3
        elif strideLengthData[2][2][1] < 0.01:
            multiplier = 2
        elif strideLengthData[2][2][1] < 0.05:
            multiplier = 1
        ax0.text(0.9, 0.95, '*' * multiplier, ha='left', va='center', transform=ax0.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(strideLengthData[0]), ha='left', va='center', transform=ax0.transAxes,
                 style='italic', fontsize=10, color='k')
        ax0.set_title('Stride length  (all paws)')
        strideLength11days = strideLengthData[2][0][0:11]
        strideLength11daysError = strideLengthData[2][1][0:11]

        # ax0.errorbar(np.arange(11), speed11days, yerr=speed11daysError, color=cmap(0.01))

        plt.plot(np.arange(len(strideLength11days)) + 1, strideLength11days, '-', label=None, color='k', linewidth=2)
        plt.fill_between(np.arange(len(strideLength11days)) + 1, strideLength11days - strideLength11daysError,
                         strideLength11days + strideLength11daysError, color='0.6', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride Length (cm)', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        (pawStrideLength, meanPawStridestrideLength, semPawIstrideLength) = groupAnalysis.getAverageSingleGroup(
            strideLengthData[3])
        (pawStrideLengthDf, pawStrideLengthPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            strideLengthData[3], sessionValues=True, treatments=False)
        colors = ['C0', 'C1', 'C2', 'C3']
        pawId = ['FL', 'FR', 'HL', 'HR']
        for i in range(4):
            (pawStar) = groupAnalysis.starMultiplier(pawStrideLengthPValues[i])
            # pdb.set_trace()
            ax1 = plt.subplot(gs[1])
            ax1.plot(np.arange(10) + 1, meanPawStridestrideLength[:, i][:11], '-', label='%s %s' % (pawId[i], pawStar),
                     linewidth=2, c=colors[i])

            plt.fill_between(np.arange(10) + 1, meanPawStridestrideLength[:, i][:11] - semPawIstrideLength[:, i][:11],
                             meanPawStridestrideLength[:, i][:11] + semPawIstrideLength[:, i][:11], color=colors[i],
                             alpha=0.1)

            self.layoutOfPanel(ax1, xLabel='Days', yLabel='Stride length (cm)', xyInvisible=[False, False],
                               Leg=[1, 2])
            ax1.legend(loc="upper right", frameon=False, fontsize=8)
            ax1.set_title('Stride length  (paw specific)')
        (swingLenStd_df, swingLenStd_mdf, swingLenStd_md_paw, swingLenStd_pawStars, swingLenStd_Stars_all) = groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideLengthData[5], treatments=False, varName='swing duration Std')
        swingLenStd_df['paw'].replace({0: 'FL', 1: 'FR', 2: 'HL', 3: 'HR'}, inplace=True)
        # pdb.set_trace()
        star_day = groupAnalysis.starMultiplier(swingLenStd_mdf.pvalues['recordingDay'])
        star_trial = groupAnalysis.starMultiplier(swingLenStd_mdf.pvalues['trial'])
        star_day_paw=[]
        for p in range(4):
            star_day_paw.append(groupAnalysis.starMultiplier(swingLenStd_md_paw[p].pvalues['recordingDay']))
        ax2=plt.subplot(gs[2])
        ax3=plt.subplot(gs[3])
        ax2.set_title('Stride length Std (all paw)')
        ax3.set_title('Stride length Std (paw specific)')
        # sns.lineplot(data=swingLenStd_df, x='recordingDay', y='measuredValue', hue=None, errorbar='se',err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax5)
        sns.lineplot(data=swingLenStd_df, x='recordingDay', y='measuredValue', hue=None, err_style='band', err_kws={'alpha':0.07}, color='black', ax=ax2)
        sns.lineplot(data=swingLenStd_df, x='recordingDay', y='measuredValue', hue='mouse', errorbar=None, alpha=0.2, ax=ax2)
        sns.lineplot(data=swingLenStd_df, x='recordingDay', y='measuredValue', hue='paw',palette=sns.color_palette("tab10", n_colors=4), legend='auto', ax=ax3)
        self.layoutOfPanel(ax2, xLabel='Day', yLabel='Swing Length Std', Leg=[1, 9])
        self.layoutOfPanel(ax3, xLabel='Day', yLabel='Swing Length Std', Leg=[1, 9])

        ax2.text(0.9, 0.97, '%s' % (star_day), ha='center', va='center', transform=ax2.transAxes, style='italic',fontfamily='serif', fontsize=12, color='k')
        ax2.text(0.9, 0.92, '%s' % (star_trial.replace('*','#')), ha='center', va='center', transform=ax2.transAxes, style='italic',fontfamily='serif', fontsize=12, color='k')

        ax2.legend(loc="upper right", frameon=False, fontsize=8)


        ax3.legend(loc="best", frameon=False, fontsize=8)

        # ax3.legend('%s %s'%(pawId,star_day_paw),frameon=False, fontsize=8, loc='best')
        ax2.xaxis.set_major_locator(majorLocator_x)
        ax3.xaxis.set_major_locator(majorLocator_x)

        ax4=plt.subplot(gs[4])

        (wheelDistanceDf, wheelDistancemdf, wheelDistancemd_paw, wheelDistancepawPvalues, wheelSpeedstars_all)=groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideLengthData[6], trialValues=True, treatments=False)
        #(wheelDistance_mean, wheelDistance_std, wheelDistance_sem)=groupAnalysis.getMeanStdNan(strideLengthData[6])
        sns.lineplot(data=wheelDistanceDf, x='recordingDay', y='measuredValue', hue=None, err_style='band', err_kws={'alpha':0.07}, color='black', ax=ax4)
        sns.lineplot(data=wheelDistanceDf, x='recordingDay', y='measuredValue', hue='mouse', errorbar=None, alpha=0.2,
                     ax=ax4)
        self.layoutOfPanel(ax4, xLabel='Day', yLabel='Wheel distance (cm)', Leg=[1, 9])
        ax4.xaxis.set_major_locator(majorLocator_x)
        ax4.set_title('Wheel distance')
        
        ax5=plt.subplot(gs[5])

        (wheelSpeedDf, wheelSpeedmdf, wheelSpeedmd_paw, wheelSpeedpawPvalues,wheelSpeedstars_all)=groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideLengthData[7], trialValues=True, treatments=False)
        # (wheelSpeed_mean, wheelSpeed_std, wheelSpeed_sem)=groupAnalysis.getMeanStdNan(strideLengthData[6])
        sns.lineplot(data=wheelSpeedDf, x='recordingDay', y='measuredValue', errorbar='sd', hue=None, err_style='band', err_kws={'alpha':0.07}, color='black', ax=ax5)
        sns.lineplot(data=wheelSpeedDf, x='recordingDay', y='measuredValue',  hue='mouse', errorbar=None, alpha=0.2,
                     ax=ax5)
        self.layoutOfPanel(ax5, xLabel='Day', yLabel='Wheel Speed (cm)', Leg=[1, 9])
        ax5.xaxis.set_major_locator(majorLocator_x)
        ax5.set_title('Wheel Speed')

        # ax1.text(0.5, 0.01, 'AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResFront.anova_table.iloc[0, 2],4), round(AnovaRMResFront.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax1.transAxes, fontsize=9)
        # self.layoutOfPanel(ax1, xLabel='recordings', yLabel='Front paws stride length (cm)', Leg=[1, 9],xyInvisible=[False, False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)
        #
        # # Third subplot
        # gssub2 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[2], hspace=0.4)
        # ax2 = []
        # ax2 = plt.subplot(gs[2])
        # # sessionNb=np.arrange(1,6)
        # for f in range(len(allHindPawStrideLength)):
        #     nDays = len(allHindPawStrideLength[f][1])
        #
        #     ax2.plot(np.arange(nDays), allHindPawStrideLength[f][1], 'o-', label=allHindPawStrideLength[f][0],
        #              color=cmap(f / len(allHindPawStrideLength)))
        #
        # plt.plot(np.arange(len(hindPawsAvg)), hindPawsAvg, '-', label="average",color='black')
        # ax2.text(0.6, 0.01, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResHind.anova_table.iloc[0, 2],4), round(AnovaRMResHind.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax2.transAxes, fontsize=9)
        # self.layoutOfPanel(ax2, xLabel='Days', yLabel='Hind paws stride length (cm)',Leg = [1, 9],xyInvisible = [False, False])
        # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)

        #plt.show()

        fname = '%s_stride_lenght'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')
 ########################################################################################################
    def createStepQualityFigure(self, stepsQualityData, experiment):
            from matplotlib import cm
            cmap=cm.get_cmap('tab20')
            nDays = []
            xArray=[]
            animalNames = []
            for i in range(len(stepsQualityData[1])):
                animalNames.append(stepsQualityData[1][i][0])
            print(animalNames)
            fig_width = 12  # width in inches
            fig_height = 5  # height in inches
            fig_size = [fig_width, fig_height]
            params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                      'ytick.labelsize': 10,
                      'figure.figsize': fig_size, 'savefig.dpi': 600,
                      'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                      'xtick.major.size': 4  # major tick size in points
                      # 'edgecolor' : None
                      # 'xtick.major.size' : 2,
                      # 'ytick.major.size' : 2,
                      }
            rcParams.update(params)

            # set sans-serif font to Arial
            rcParams['font.sans-serif'] = 'Arial'

            # create figure instance
            fig = plt.figure()

            # define sub-panel grid and possibly width and height ratios
            gs = gridspec.GridSpec(1, 2,
                                   width_ratios=[1,1],
                                   height_ratios=[1]
                                   )
            # define vertical and horizontal spacing between panels
            gs.update(wspace=0.4, hspace=0.25)
            # plt.figtext(0.30, 0.95, "Fraction of indecisive strides", clip_on=False, color='black',size=14)
            # possibly change outer margins of the figure
            plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

                        # first sub-plot #######################################################
            ax0 = plt.subplot(gs[0])
            for f in range (len(stepsQualityData[1])):
                nDays = len(stepsQualityData[1][f][1])

                ax0.plot(np.arange(nDays)[:11]+1,stepsQualityData[1][f][1][:11], 'o-', label=None, color=cmap(f/len(stepsQualityData[1][f][1])),ms=2,alpha=0.2)

            ax0.legend(loc="upper left", bbox_to_anchor=(10,0.1))
            ax0.set_title('Indecisive steps fraction (all paws)')
            inDstep11days = stepsQualityData[2][0][0:11]
            inDstepSEM11days = stepsQualityData[2][1][0:11]
            multiplier = 0
            if stepsQualityData[2][2][1] < 0.001:
                multiplier = 3
            elif stepsQualityData[2][2][1]  < 0.01:
                multiplier = 2
            elif stepsQualityData[2][2][1]  < 0.05:
                multiplier = 1
            ax0.text(0.9, 0.95, '*' * multiplier, ha='left', va='center', transform=ax0.transAxes, style='italic',
                     fontfamily='serif', fontsize=15, color='k')
            ax0.text(0.87, 0.9, '(N=%s)' % len(stepsQualityData[1]), ha='left', va='center', transform=ax0.transAxes,
                     style='italic', fontsize=10, color='k')
            #ax0.errorbar(np.arange(11), inDstep11days, yerr=inDstepSEM11days, color=cmap(0.01))

            #ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
            ax0.text(0.0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' %(stepsQualityData[2][2][0] , stepsQualityData[2][2][1] ,stepsQualityData[2][2][2] ,stepsQualityData[2][2][3] ),ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=5,color='0.7')
            ax0.plot(np.arange(len(inDstep11days))+1,inDstep11days,'-',label=None, color='k',linewidth=2)
            plt.fill_between(np.arange(len(inDstep11days))+1,inDstep11days-inDstepSEM11days,inDstep11days+inDstepSEM11days, color='0.6', alpha=0.2)
            self.layoutOfPanel(ax0, xLabel='Days', yLabel='Fraction of indecisive strides', Leg=[1, 9])
            majorLocator_x = MultipleLocator(1)
            ax0.xaxis.set_major_locator(majorLocator_x)

            (pawIndeciveStrides, meanPawIndeciveStrides, semPawIndeciveStrides) = groupAnalysis.getAverageSingleGroup(
                stepsQualityData[3])
            (pawStepQualityDataDf, pawStepQualityDataPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
                stepsQualityData[3], sessionValues=True, treatments=False)
            colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
            pawId = ['FL', 'FR', 'HL', 'HR']
            for i in range(4):
                (pawStar) = groupAnalysis.starMultiplier(pawStepQualityDataPValues[i])
            # pdb.set_trace()
                ax1 = plt.subplot(gs[1])
                ax1.plot(np.arange(10) + 1, meanPawIndeciveStrides[:, i], '-', label='%s %s'%(pawId[i],pawStar), linewidth=2, c=colors[i])

                plt.fill_between(np.arange(10) + 1, meanPawIndeciveStrides[:, i] - semPawIndeciveStrides[:, i],
                                 meanPawIndeciveStrides[:, i] + semPawIndeciveStrides[:, i], color=colors[i], alpha=0.1)

                self.layoutOfPanel(ax1, xLabel='Days', yLabel='Indecisive steps fraction', xyInvisible=[False,False], Leg=[1, 2])


            ax1.legend(loc="upper right", frameon=False, fontsize=8)
            ax1.set_title('Indecisive steps fraction (paw specific)')
            ax1.xaxis.set_major_locator(majorLocator_x)

            #  #Second subplot
            # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4)
            # ax1 = []
            # ax1 = plt.subplot(gs[1])
            # #sessionNb=np.arrange(1,6)
            # for m in range (len(sessIndeciveStepNb)):
            #     nDays=len(sessIndeciveStepNb[m][1])
            #     #print(len(sumStepsAllMice),nDays)
            #     for n in range(nDays):
            #         xArray= np.repeat(np.arange(nDays), 5)
            #         xArray=xArray+np.tile(np.arange(5)/7,nDays)
            #
            #         animalSteps= (np.asarray(sessIndeciveStepNb[m][1])).flatten()
            #         #print(animalSteps)
            #         ax1.plot(xArray,animalSteps+m*0.8, 'o-', color=cmap(m/len(sessIndeciveStepNb)), label=None)  # if mouseList[m] not in plt.gca().get_legend_handles_labels() [-1] else '')
            # ax1.text(0.7, 0.04, 'AnovaRM: F value=%s, p value=%s'%(round(AnovaRMSesRes.anova_table.iloc[0, 2],4),round(AnovaRMSesRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax1.transAxes, fontsize=9)
            # ax1.text(0.7,0.01, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s , %s)' %(round(mdf.fe_params['sessionNumber'],4), round(mdf.pvalues['sessionNumber'],4),round(conf_int.iloc[2,0],3),round(conf_int.iloc[2,1],3)),ha='left', va='center', transform=ax1.transAxes, fontsize=9)
            # #stepAvg =[sessionStepAverage[r][2] for r in range (len(sessionStepAverage))]
            #
            # for n in range(14):
            #     xArray1 = np.repeat(np.arange(14),5)
            #     xArray1= xArray1 +np.tile(np.arange(5)/7,14)
            #
            #     sessionStepAverage0=np.asanyarray(sessIndeciveStepNbAvg)
            #     sessionStepAverage= sessionStepAverage0.flatten()
            # plt.plot(xArray1, sessionStepAverage+n*0.4, '-o',color='black', label="average")
            # #plt.legend()
            # self.layoutOfPanel(ax1, xLabel='recordings', yLabel='fraction of indecisive steps', Leg=[1, 9],xyInvisible=[False,False])
            # plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)

            #plt.show()
            #pdb.set_trace()
            fname = '%s_fraction_of_indecisive_steps'%experiment
            # plt.savefig(fname + '.png')
            plt.savefig(self.figureDirectory + '/' + fname + '.png')

    def generateBehaviorStepNumberFig (self, strideNumberData, treatments, experiment, individualValues=True):
        from matplotlib import cm
        cmap=cm.get_cmap('Greys')#('tab20')
        cmap1=cm.get_cmap('Reds')#('tab20')
        nDays = []
        xArray=[]
        animalNames = []

        fig_width = 25  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 10, 'axes.titlesize': 10, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        fig = plt.figure()
        rcParams.update(params)
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[1.5,1],
                               height_ratios=[1])
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # create figure instance
        gs.update(wspace=0.2, hspace=0.3)
        ax0 = plt.subplot(gs[0])
        # # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        majorLocator_x = MultipleLocator(1)

            #plot averages, average values in strideNumberData[0], trial values in strideNumberData[1]
        (salineAvg, MuscimolAvg, saline, muscimol, nSaline, nMuscimol) = groupAnalysis.getTreatmentAverages1D(strideNumberData[0])
        ax0.plot(np.arange(10) + 1, salineAvg[0], '-', label='saline', linewidth=2, c='k')
        ax0.fill_between(np.arange(10) + 1, salineAvg[0] - salineAvg[2],salineAvg[0] + salineAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg[0], '-', label='muscimol', linewidth=2, c='r')
        ax0.fill_between(np.arange(10) + 1, MuscimolAvg[0] - MuscimolAvg[2],MuscimolAvg[0] +MuscimolAvg[2], color='r', alpha=0.1)

        salineName =[]
        if individualValues:
            #plot individual average values
            for m in range(len(strideNumberData[0])):
                if strideNumberData[0][m][2]=='saline' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="201017_m98":
                    ax0.plot(np.arange(10)+1,strideNumberData[0][m][1], color=cmap(m/len(strideNumberData[0])),alpha=1,label='%s,%s'%(strideNumberData[0][m][0],strideNumberData[0][m][2]))
                if strideNumberData[0][m][2]=='muscimol' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="210113_f79":
                    ax0.plot(np.arange(10)+1,strideNumberData[0][m][1], color=cmap1(m/len(strideNumberData[0])),alpha=1,label='%s,%s'%(strideNumberData[0][m][0],strideNumberData[0][m][2]))


        (salineArray,muscimolArray)=groupAnalysis.splitGroups(strideNumberData[3])
        # pdb.set_trace()
        ax0.text(0.10, 0.90, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')


        salineAvgPaw=np.nanmean(salineArray,axis=0)
        salineSem=stats.sem(salineArray,axis=0,nan_policy='omit')

        muscimolAvgPaw=np.nanmean(muscimolArray,axis=0)
        muscimolSem = stats.sem(muscimolArray, axis=0, nan_policy='omit')

        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.3, wspace=0.7)

        #put the array vertically to extract paw values easily

        salineAvgPlot=np.vstack(salineAvgPaw)
        muscimolAvgPlot = np.vstack(muscimolAvgPaw)
        salineSemVstack = np.vstack(salineSem)
        muscimolSemVstack=np.vstack(muscimolSem)
        pawId=['FL','FR','HL','HR']
        for i in range(4):
            ax1 = plt.subplot(gssub0[i])
            ax1.plot(np.arange(10) + 1, salineAvgPlot[:, i], '-', label='%s sal'%pawId[i], linewidth=2, c='k')
            ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, i], '-', label='%s mus'%pawId[i], linewidth=2, c='salmon')
            ax1.fill_between(np.arange(10) + 1, salineAvgPlot[:, i] - salineSemVstack[:, i],
                             salineAvgPlot[:, i] + salineSemVstack[:, i], color='k', alpha=0.1)
            ax1.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, i] - muscimolSemVstack[:, i],
                             muscimolAvgPlot[:, i] + muscimolSemVstack[:, i], color='salmon', alpha=0.1)
            ax1.legend(bbox_transform='transAxes', loc="upper left", frameon=False, fontsize=5)
            ax1.set_xlabel('', fontsize=6)
            ax1.set_ylabel('', fontsize=6)
            ax1.tick_params('both',labelsize=5)

            self.layoutOfPanel(ax1, xLabel='Days', yLabel='swing number', Leg=[1, 9])

            ax1.xaxis.set_major_locator(majorLocator_x)
        # ax1.plot(np.arange(10) + 1, salineAvgPlot[:,0], '-', label='FL-sal', linewidth=2, c='grey')
        # ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, 0], '-', label='FL-mus', linewidth=2, c='crimson')
        # plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,0] - salineSemVstack[:,0], salineAvgPlot[:,0] + salineSemVstack[:,0], color='grey', alpha=0.1)
        # plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 0] - muscimolSemVstack[:, 0],muscimolAvgPlot[:, 0] + muscimolSemVstack[:, 0], color='crimson', alpha=0.1)
        #
        # ax2 = plt.subplot(gssub0[1])
        # ax2.plot(np.arange(10) + 1, salineAvgPlot[:, 1], '-', label='FR-sal', linewidth=2, c='k')
        # ax2.plot(np.arange(10) + 1, muscimolAvgPlot[:, 1], '-', label='FR-mus', linewidth=2, c='coral')
        # plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,1] - salineSemVstack[:,1], salineAvgPlot[:,1] + salineSemVstack[:,1], color='grey', alpha=0.1)
        # plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 1] - muscimolSemVstack[:, 1],muscimolAvgPlot[:, 1] + muscimolSemVstack[:, 1], color='coral', alpha=0.1)
        #
        # ax3 = plt.subplot(gssub0[2])
        # ax3.plot(np.arange(10) + 1, salineAvgPlot[:,2], '-', label='HL-sal', linewidth=2, c='grey')
        # ax3.plot(np.arange(10) + 1, muscimolAvgPlot[:,2], '-', label='HL-mus', linewidth=2, c='crimson')
        # plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,2] - salineSemVstack[:,2], salineAvgPlot[:,2] + salineSemVstack[:,2], color='grey', alpha=0.1)
        # plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 2] - muscimolSemVstack[:, 2],muscimolAvgPlot[:, 2] + muscimolSemVstack[:, 2], color='crimson', alpha=0.1)
        #
        # ax4 = plt.subplot(gssub0[3])
        # ax4.plot(np.arange(10) + 1, salineAvgPlot[:, 3], '-', label='HR-sal', linewidth=2, c='k')
        # ax4.plot(np.arange(10) + 1, muscimolAvgPlot[:, 3], '-', label='HR-mus', linewidth=2, c='coral')
        # plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,3] - salineSemVstack[:,3], salineAvgPlot[:,3] + salineSemVstack[:,3], color='grey', alpha=0.1)
        # plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 3] - muscimolSemVstack[:, 3],muscimolAvgPlot[:, 3] + muscimolSemVstack[:, 3], color='coral', alpha=0.1)

        self.layoutOfPanel(ax0, xLabel='Days', yLabel='swing number', Leg=[1, 9])

        # self.layoutOfPanel(ax2, xLabel='Days', yLabel='swing number', Leg=[1, 9])
        # self.layoutOfPanel(ax3, xLabel='Days', yLabel='swing number', Leg=[1, 9])
        # self.layoutOfPanel(ax4, xLabel='Days', yLabel='swing number', Leg=[1, 9])
        #remove duplicate labels
        handles, labels = ax0.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)


        ax0.legend(handle_list, label_list,loc="upper right", frameon=False, fontsize=4)

        # ax2.legend(loc="upper right", frameon=False, fontsize=6)
        # ax3.legend(loc="upper right", frameon=False, fontsize=6)
        # ax4.legend(loc="upper right", frameon=False, fontsize=6)

        ax0.xaxis.set_major_locator(majorLocator_x)

        # ax2.xaxis.set_major_locator(majorLocator_x)
        # ax3.xaxis.set_major_locator(majorLocator_x)
        # ax4.xaxis.set_major_locator(majorLocator_x)
        if treatments and individualValues:
            #fname = 'step_number_Muscimol_B1B2_figure_individual_values'
            fname = '%s_wing_number_figure_individual_values'%experiment
        elif individualValues==False:
            #fname='step_number_Muscimol_B1B2_figure'
            fname = '%s_swing_number_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')


    def generateBehaviorStepDurationFig (self, stepDurationData,experiment,treatments, individualValues):

        from matplotlib import cm
        cmap=cm.get_cmap('Greys')#('tab20')
        cmap1=cm.get_cmap('Reds')#('tab20')

        fig_width = 12  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        grid = plt.GridSpec(4, 4, wspace=.6, hspace=.25)
        ax0 = plt.subplot(grid[:2, :2])
        #get and plot average swing duration
        (salineSwingDurAvg, MuscimolSwingDurAvg, salineSwingDur, muscimolSwingDur,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[0])
        ax0.plot(np.arange(10) + 1, salineSwingDurAvg[0], '-', label='saline swing', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineSwingDurAvg[0] - salineSwingDurAvg[2],salineSwingDurAvg[0] + salineSwingDurAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolSwingDurAvg[0], '-', label='muscimol swing', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolSwingDurAvg[0] - MuscimolSwingDurAvg[2],MuscimolSwingDurAvg[0] +MuscimolSwingDurAvg[2], color='r', alpha=0.1)


        # pdb.set_trace()

        ax1 = plt.subplot(grid[2:4, :2])
        #get and plot average stance duration
        (salineStanceDurAvg, MuscimolStanceDurAvg, salineStanceDur, muscimolStanceDur,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[2])
        ax1.plot(np.arange(10) + 1, salineStanceDurAvg[0], '-', label='saline stance', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineStanceDurAvg[0] - salineStanceDurAvg[2],salineStanceDurAvg[0] + salineStanceDurAvg[2], color='grey', alpha=0.1)
        ax1.plot(np.arange(10) + 1, MuscimolStanceDurAvg[0], '-', label='muscimol stance', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolStanceDurAvg[0] - MuscimolStanceDurAvg[2],MuscimolStanceDurAvg[0] +MuscimolStanceDurAvg[2], color='r', alpha=0.1)
        ax1.text(0.60, 0.83, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')

        if individualValues:
            #plot individual average values for stance
            for m in range(len(stepDurationData[0])):
                if stepDurationData[2][m][2]=='saline' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="201017_m98":
                    ax1.plot(np.arange(10)+1,stepDurationData[2][m][1], color=cmap(m/len(stepDurationData[2])),alpha=1,label='%s,%s'%(stepDurationData[2][m][0],stepDurationData[2][m][2]))
                if stepDurationData[2][m][2]=='muscimol' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="210113_f79":
                    ax1.plot(np.arange(10)+1,stepDurationData[2][m][1], color=cmap1(m/len(stepDurationData[2])),alpha=1,label='%s,%s'%(stepDurationData[2][m][0],stepDurationData[2][m][2]))
            #plot individual average values for swing
            for m in range(len(stepDurationData[0])):
                if stepDurationData[0][m][2]=='saline' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="201017_m98":
                    ax0.plot(np.arange(10)+1,stepDurationData[0][m][1], color=cmap(m/len(stepDurationData[0])),alpha=1,label='%s,%s'%(stepDurationData[0][m][0],stepDurationData[0][m][2]))
                if stepDurationData[0][m][2]=='muscimol' :#and 'f' in strideNumberData[0][m][0]:#and strideNumberData[0][m][0]=="210113_f79":
                    ax0.plot(np.arange(10)+1,stepDurationData[0][m][1], color=cmap1(m/len(stepDurationData[0])),alpha=1,label='%s,%s'%(stepDurationData[0][m][0],stepDurationData[0][m][2]))


        #extract swing duration average for each paw
        (salineSwingArray,muscimolSwingArray)=groupAnalysis.splitGroups(stepDurationData[6])
        salineSwingAvg=np.nanmean(salineSwingArray,axis=0)
        salineSwingSem=stats.sem(salineSwingArray,axis=0,nan_policy='omit')
        muscimolSwingAvg=np.nanmean(muscimolSwingArray,axis=0)
        muscimolSwingSem = stats.sem(muscimolSwingArray, axis=0, nan_policy='omit')

        #extract stance duration average for each paw
        (salineStanceArray,muscimolStanceArray)=groupAnalysis.splitGroups(stepDurationData[7])
        salineStanceAvg=np.nanmean(salineStanceArray,axis=0)
        salineStanceSem=stats.sem(salineStanceArray,axis=0,nan_policy='omit')
        muscimolStanceAvg=np.nanmean(muscimolStanceArray,axis=0)
        muscimolStanceSem = stats.sem(muscimolStanceArray, axis=0, nan_policy='omit')

        
        #vstack swing duration paw values

        salineSwingAvgPlot=np.vstack(salineSwingAvg)
        muscimolSwingAvgPlot = np.vstack(muscimolSwingAvg)
        salineSwingSemVstack = np.vstack(salineSwingSem)
        muscimolSwingSemVstack=np.vstack(muscimolSwingSem)
        #vstack stance duration paw values
        salineStanceAvgPlot=np.vstack(salineStanceAvg)
        muscimolStanceAvgPlot = np.vstack(muscimolStanceAvg)
        salineStanceSemVstack = np.vstack(salineStanceSem)
        muscimolStanceSemVstack=np.vstack(muscimolStanceSem)

#plot FL swing  duration for saline vs muscimol
        ax2 = plt.subplot(grid[0, 2])
        ax2.plot(np.arange(10) + 1, salineSwingAvgPlot[:,0], '-', label='FL-sal-swing', linewidth=2, c='grey')
        ax2.plot(np.arange(10) + 1, muscimolSwingAvgPlot[:, 0], '-', label='FL-mus-swing', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineSwingAvgPlot[:,0] - salineSwingSemVstack[:,0], salineSwingAvgPlot[:,0] + salineSwingSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolSwingAvgPlot[:, 0] - muscimolSwingSemVstack[:, 0],muscimolSwingAvgPlot[:, 0] + muscimolSwingSemVstack[:, 0], color='crimson', alpha=0.1)
        # plot FL stance duration for salune vs muscimol
        ax3 = plt.subplot(grid[2, 2])
        ax3.plot(np.arange(10) + 1, salineStanceAvgPlot[:,0], '-', label='FL-sal-stance', linewidth=2, c='grey')
        ax3.plot(np.arange(10) + 1, muscimolStanceAvgPlot[:, 0], '-', label='FL-mus-stance', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineStanceAvgPlot[:,0] - salineStanceSemVstack[:,0], salineStanceAvgPlot[:,0] + salineStanceSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolStanceAvgPlot[:, 0] - muscimolStanceSemVstack[:, 0],muscimolStanceAvgPlot[:, 0] + muscimolStanceSemVstack[:, 0], color='crimson', alpha=0.1)

        # plot FR swing  duration for saline vs muscimol
        ax4 = plt.subplot(grid[1, 2])
        ax4.plot(np.arange(10) + 1, salineSwingAvgPlot[:, 1], '-', label='FR-sal-swing', linewidth=2, c='k')
        ax4.plot(np.arange(10) + 1, muscimolSwingAvgPlot[:, 1], '-', label='FR-mus-swing', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineSwingAvgPlot[:,1] - salineSwingSemVstack[:,1], salineSwingAvgPlot[:,1] + salineSwingSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolSwingAvgPlot[:, 1] - muscimolSwingSemVstack[:, 1],muscimolSwingAvgPlot[:, 1] + muscimolSwingSemVstack[:, 1], color='coral', alpha=0.1)
        # plot FR stance  duration for saline vs muscimol
        ax5 = plt.subplot(grid[3, 2])
        ax5.plot(np.arange(10) + 1, salineStanceAvgPlot[:, 1], '-', label='FR-sal-stance', linewidth=2, c='k')
        ax5.plot(np.arange(10) + 1, muscimolStanceAvgPlot[:, 1], '-', label='FR-mus-stance', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineStanceAvgPlot[:,1] - salineStanceSemVstack[:,1], salineStanceAvgPlot[:,1] + salineStanceSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolStanceAvgPlot[:, 1] - muscimolStanceSemVstack[:, 1],muscimolStanceAvgPlot[:, 1] + muscimolStanceSemVstack[:, 1], color='coral', alpha=0.1)
        # plot HL swing  duration for saline vs muscimol
        ax6 = plt.subplot(grid[0, 3])
        ax6.plot(np.arange(10) + 1, salineSwingAvgPlot[:,2], '-', label='HL-sal-swing', linewidth=2, c='grey')
        ax6.plot(np.arange(10) + 1, muscimolSwingAvgPlot[:,2], '-', label='HL-mus-swing', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineSwingAvgPlot[:,2] - salineSwingSemVstack[:,2], salineSwingAvgPlot[:,2] + salineSwingSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolSwingAvgPlot[:, 2] - muscimolSwingSemVstack[:, 2],muscimolSwingAvgPlot[:, 2] + muscimolSwingSemVstack[:, 2], color='crimson', alpha=0.1)
        # plot HL stance  duration for saline vs muscimol
        ax7 = plt.subplot(grid[2, 3])
        ax7.plot(np.arange(10) + 1, salineStanceAvgPlot[:,2], '-', label='HL-sal-stance', linewidth=2, c='grey')
        ax7.plot(np.arange(10) + 1, muscimolStanceAvgPlot[:,2], '-', label='HL-mus-stance', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineStanceAvgPlot[:,2] - salineStanceSemVstack[:,2], salineStanceAvgPlot[:,2] + salineStanceSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolStanceAvgPlot[:, 2] - muscimolStanceSemVstack[:, 2],muscimolStanceAvgPlot[:, 2] + muscimolStanceSemVstack[:, 2], color='crimson', alpha=0.1)
        # plot HR wing  duration for saline vs muscimol
        ax8 = plt.subplot(grid[1, 3])
        ax8.plot(np.arange(10) + 1, salineSwingAvgPlot[:, 3], '-', label='HR-sal-swing', linewidth=2, c='k')
        ax8.plot(np.arange(10) + 1, muscimolSwingAvgPlot[:, 3], '-', label='HR-mus-swing', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineSwingAvgPlot[:,3] - salineSwingSemVstack[:,3], salineSwingAvgPlot[:,3] + salineSwingSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolSwingAvgPlot[:, 3] - muscimolSwingSemVstack[:, 3],muscimolSwingAvgPlot[:, 3] + muscimolSwingSemVstack[:, 3], color='coral', alpha=0.1)
        # plot HR stance  duration for saline vs muscimol
        ax9 = plt.subplot(grid[3, 3])
        ax9.plot(np.arange(10) + 1, salineStanceAvgPlot[:, 3], '-', label='HR-sal-stance', linewidth=2, c='k')
        ax9.plot(np.arange(10) + 1, muscimolStanceAvgPlot[:, 3], '-', label='HR-mus-stance', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineStanceAvgPlot[:,3] - salineStanceSemVstack[:,3], salineStanceAvgPlot[:,3] + salineStanceSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolStanceAvgPlot[:, 3] - muscimolStanceSemVstack[:, 3],muscimolStanceAvgPlot[:, 3] + muscimolStanceSemVstack[:, 3], color='coral', alpha=0.1)
        self.BehaviorFigStyle(ax0,fontsize=5,yLabel="Median Swing Duration (s)")
        self.BehaviorFigStyle(ax1,fontsize=5,xLabel="Days",yLabel="Median Stance Duration (s)")
        self.BehaviorFigStyle(ax2)
        self.BehaviorFigStyle(ax3)
        self.BehaviorFigStyle(ax4)
        self.BehaviorFigStyle(ax5)
        self.BehaviorFigStyle(ax6)
        self.BehaviorFigStyle(ax7)
        self.BehaviorFigStyle(ax8)
        self.BehaviorFigStyle(ax9)
        if treatments and individualValues:
            fname = '%s_step_duration_figure_individual_values'%experiment
        elif individualValues==False:
            fname='%s_step_duration_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')

    def generateBehaviorSpeedFig (self, pawSpeedData, experiment, individualValues):
        from matplotlib import cm
        cmap=cm.get_cmap('Greys')#('tab20')
        cmap1=cm.get_cmap('Reds')#('tab20')
        nDays = []
        xArray=[]
        animalNames = []

        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 7, 'axes.titlesize': 7, 'font.size': 7, 'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        grid = plt.GridSpec(2, 4, wspace=.6, hspace=.25)

        #plot average speed values saline vs muscimol
        ax0 = plt.subplot(grid[:, :2])
        (salineAvg, MuscimolAvg, saline, muscimol, nSaline, nMuscimol) = groupAnalysis.getTreatmentAverages1D(pawSpeedData[0])
        ax0.plot(np.arange(10) + 1, salineAvg[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvg[0] - salineAvg[2],salineAvg[0] + salineAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolAvg[0] - MuscimolAvg[2],MuscimolAvg[0] +MuscimolAvg[2], color='r', alpha=0.1)
        ax0.text(0.10, 0.90, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')

        #plot individual average values
        if individualValues:
            for m in range(len(pawSpeedData[0])):
                if pawSpeedData[0][m][2]=='saline' :#and strideNumberData[0][m][0]=="201017_m98":
                    ax0.plot(np.arange(10)+1,pawSpeedData[1][m][1], color=cmap(m/len(pawSpeedData[0])),alpha=1,label='%s,%s'%(pawSpeedData[1][m][0],pawSpeedData[1][m][2]))
                if pawSpeedData[0][m][2]=='muscimol' :#and strideNumberData[0][m][0]=="210113_f79":
                    ax0.plot(np.arange(10)+1,pawSpeedData[1][m][1], color=cmap1(m/len(pawSpeedData[0])),alpha=1,label='%s,%s'%(pawSpeedData[1][m][0],pawSpeedData[1][m][2]))
        ax0.text(0.10, 0.90, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')
        (salineArray,muscimolArray)=groupAnalysis.splitGroups(pawSpeedData[3])
        salineAvg=np.nanmean(salineArray,axis=0)
        salineSem=stats.sem(salineArray,axis=0,nan_policy='omit')
        muscimolAvg=np.nanmean(muscimolArray,axis=0)
        muscimolSem = stats.sem(muscimolArray, axis=0, nan_policy='omit')

        # pdb.set_trace()
        ax1 = plt.subplot(grid[0, 2])
        salineAvgPlot=np.vstack(salineAvg)
        muscimolAvgPlot = np.vstack(muscimolAvg)
        salineSemVstack = np.vstack(salineSem)
        muscimolSemVstack=np.vstack(muscimolSem)
        #pdb.set_trace()
        ax1.plot(np.arange(10) + 1, salineAvgPlot[:,0], '-', label='FL-sal', linewidth=2, c='grey')
        ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, 0], '-', label='FL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,0] - salineSemVstack[:,0], salineAvgPlot[:,0] + salineSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 0] - muscimolSemVstack[:, 0],muscimolAvgPlot[:, 0] + muscimolSemVstack[:, 0], color='crimson', alpha=0.1)

        ax2 = plt.subplot(grid[1, 2])
        ax2.plot(np.arange(10) + 1, salineAvgPlot[:, 1], '-', label='FR-sal', linewidth=2, c='k')
        ax2.plot(np.arange(10) + 1, muscimolAvgPlot[:, 1], '-', label='FR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,1] - salineSemVstack[:,1], salineAvgPlot[:,1] + salineSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 1] - muscimolSemVstack[:, 1],muscimolAvgPlot[:, 1] + muscimolSemVstack[:, 1], color='coral', alpha=0.1)

        ax3 = plt.subplot(grid[0, 3])
        ax3.plot(np.arange(10) + 1, salineAvgPlot[:,2], '-', label='HL-sal', linewidth=2, c='grey')
        ax3.plot(np.arange(10) + 1, muscimolAvgPlot[:,2], '-', label='HL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,2] - salineSemVstack[:,2], salineAvgPlot[:,2] + salineSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 2] - muscimolSemVstack[:, 2],muscimolAvgPlot[:, 2] + muscimolSemVstack[:, 2], color='crimson', alpha=0.1)

        ax4 = plt.subplot(grid[1, 3])
        ax4.plot(np.arange(10) + 1, salineAvgPlot[:, 3], '-', label='HR-sal', linewidth=2, c='k')
        ax4.plot(np.arange(10) + 1, muscimolAvgPlot[:, 3], '-', label='HR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,3] - salineSemVstack[:,3], salineAvgPlot[:,3] + salineSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 3] - muscimolSemVstack[:, 3],muscimolAvgPlot[:, 3] + muscimolSemVstack[:, 3], color='coral', alpha=0.1)


        self.BehaviorFigStyle(ax0, xLabel='Days', yLabel='swing speed (cm/s)')
        ax0.legend(loc="upper right", frameon=False, fontsize=4)
        self.BehaviorFigStyle(ax1, xLabel='Days', yLabel='swing speed (cm/s)')
        self.BehaviorFigStyle(ax2)
        self.BehaviorFigStyle(ax3)
        self.BehaviorFigStyle(ax4)
        if individualValues:
            fname = '%s_swing_speed_individual values'%experiment
        else:
            fname = '%s_swing_speed_figure'%experiment
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')

    def generateBehaviorRungCrossedFig (self, rungCrossedData, experiment, individualValues):
        from matplotlib import cm
        cmap=cm.get_cmap('Greys')#('tab20')
        cmap1=cm.get_cmap('Reds')#('tab20')
        nDays = []
        xArray=[]
        animalNames = []

        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 7, 'axes.titlesize': 7, 'font.size': 7, 'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        grid = plt.GridSpec(2, 4, wspace=.6, hspace=.25)

        #plot average speed values saline vs muscimol
        ax0 = plt.subplot(grid[:, :2])
        (salineAvg, MuscimolAvg, saline, muscimol, nSaline, nMuscimol) = groupAnalysis.getTreatmentAverages1D(rungCrossedData[0])
        ax0.plot(np.arange(10) + 1, salineAvg[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvg[0] - salineAvg[2],salineAvg[0] + salineAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolAvg[0] - MuscimolAvg[2],MuscimolAvg[0] +MuscimolAvg[2], color='r', alpha=0.1)
        ax0.text(0.75, 0.87, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')

        #plot individual average values
        if individualValues:
            for m in range(len(rungCrossedData[0])):
                if rungCrossedData[0][m][2]=='saline' :#and strideNumberData[0][m][0]=="201017_m98":
                    ax0.plot(np.arange(10)+1,rungCrossedData[0][m][1], color=cmap(m/len(rungCrossedData[0])),alpha=1,label='%s,%s'%(rungCrossedData[0][m][0],rungCrossedData[1][m][2]))
                if rungCrossedData[0][m][2]=='muscimol' :#and strideNumberData[0][m][0]=="210113_f79":
                    ax0.plot(np.arange(10)+1,rungCrossedData[0][m][1], color=cmap1(m/len(rungCrossedData[0])),alpha=1,label='%s,%s'%(rungCrossedData[0][m][0],rungCrossedData[1][m][2]))

        (salineArray,muscimolArray)=groupAnalysis.splitGroups(rungCrossedData[3])
        salineAvg=np.nanmean(salineArray,axis=0)
        salineSem=stats.sem(salineArray,axis=0,nan_policy='omit')
        muscimolAvg=np.nanmean(muscimolArray,axis=0)
        muscimolSem = stats.sem(muscimolArray, axis=0, nan_policy='omit')


        ax1 = plt.subplot(grid[0, 2])
        salineAvgPlot=np.vstack(salineAvg)
        muscimolAvgPlot = np.vstack(muscimolAvg)
        salineSemVstack = np.vstack(salineSem)
        muscimolSemVstack=np.vstack(muscimolSem)
        #pdb.set_trace()
        ax1.plot(np.arange(10) + 1, salineAvgPlot[:,0], '-', label='FL-sal', linewidth=2, c='grey')
        ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, 0], '-', label='FL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,0] - salineSemVstack[:,0], salineAvgPlot[:,0] + salineSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 0] - muscimolSemVstack[:, 0],muscimolAvgPlot[:, 0] + muscimolSemVstack[:, 0], color='crimson', alpha=0.1)

        ax2 = plt.subplot(grid[1, 2])
        ax2.plot(np.arange(10) + 1, salineAvgPlot[:, 1], '-', label='FR-sal', linewidth=2, c='k')
        ax2.plot(np.arange(10) + 1, muscimolAvgPlot[:, 1], '-', label='FR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,1] - salineSemVstack[:,1], salineAvgPlot[:,1] + salineSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 1] - muscimolSemVstack[:, 1],muscimolAvgPlot[:, 1] + muscimolSemVstack[:, 1], color='coral', alpha=0.1)

        ax3 = plt.subplot(grid[0, 3])
        ax3.plot(np.arange(10) + 1, salineAvgPlot[:,2], '-', label='HL-sal', linewidth=2, c='grey')
        ax3.plot(np.arange(10) + 1, muscimolAvgPlot[:,2], '-', label='HL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,2] - salineSemVstack[:,2], salineAvgPlot[:,2] + salineSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 2] - muscimolSemVstack[:, 2],muscimolAvgPlot[:, 2] + muscimolSemVstack[:, 2], color='crimson', alpha=0.1)

        ax4 = plt.subplot(grid[1, 3])
        ax4.plot(np.arange(10) + 1, salineAvgPlot[:, 3], '-', label='HR-sal', linewidth=2, c='k')
        ax4.plot(np.arange(10) + 1, muscimolAvgPlot[:, 3], '-', label='HR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,3] - salineSemVstack[:,3], salineAvgPlot[:,3] + salineSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 3] - muscimolSemVstack[:, 3],muscimolAvgPlot[:, 3] + muscimolSemVstack[:, 3], color='coral', alpha=0.1)

        self.BehaviorFigStyle(ax0, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed')
        self.BehaviorFigStyle(ax1, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed')
        self.BehaviorFigStyle(ax2)
        self.BehaviorFigStyle(ax3)
        self.BehaviorFigStyle(ax4)

        if individualValues:
            fname = '%s_Fraction_rungs_crossed_figure_individual values'%experiment
        else:
            fname = '%s_Fraction_rungs_crossed_figure'%experiment

        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')

    def generateBehaviorStepQualityFig (self, stepsQualityData, experiment, individualValues):
        from matplotlib import cm
        cmap=cm.get_cmap('Greys')#('tab20')
        cmap1=cm.get_cmap('Reds')#('tab20')
        nDays = []
        xArray=[]
        animalNames = []

        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 7, 'axes.titlesize': 7, 'font.size': 7, 'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        grid = plt.GridSpec(2, 4, wspace=.6, hspace=.25)

        #plot average speed values saline vs muscimol
        ax0 = plt.subplot(grid[:, :2])

        (salineAvg, MuscimolAvg, saline, muscimol, nSaline, nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepsQualityData[0])
        # pdb.set_trace()
        ax0.plot(np.arange(10) + 1, salineAvg[0], '-', label='saline', linewidth=2, c='k')
        ax0.fill_between(np.arange(10) + 1, salineAvg[0] - salineAvg[2],salineAvg[0] + salineAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg[0], '-', label='muscimol', linewidth=2, c='r')
        ax0.fill_between(np.arange(10) + 1, MuscimolAvg[0] - MuscimolAvg[2],MuscimolAvg[0] +MuscimolAvg[2], color='r', alpha=0.1)
        ax0.text(0.75, 0.87, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')

        #plot individual average values
        if individualValues:
            for m in range(len(stepsQualityData[1])):
                if stepsQualityData[0][m][2]=='saline' :#and strideNumberData[0][m][0]=="201017_m98":
                    ax0.plot(np.arange(10)+1,stepsQualityData[0][m][1], color=cmap(m/len(stepsQualityData[0])),alpha=1,label='%s,%s'%(stepsQualityData[0][m][0],stepsQualityData[0][m][2]))
                if stepsQualityData[0][m][2]=='muscimol' :#and strideNumberData[0][m][0]=="210113_f79":
                    ax0.plot(np.arange(10)+1,stepsQualityData[0][m][1], color=cmap1(m/len(stepsQualityData[0])),alpha=1,label='%s,%s'%(stepsQualityData[0][m][0],stepsQualityData[0][m][2]))

        (salineArray,muscimolArray)=groupAnalysis.splitGroups(stepsQualityData[3])
        salineAvg=np.nanmean(salineArray,axis=0)
        salineSem=stats.sem(salineArray,axis=0,nan_policy='omit')
        muscimolAvg=np.nanmean(muscimolArray,axis=0)
        muscimolSem = stats.sem(muscimolArray, axis=0, nan_policy='omit')


        ax1 = plt.subplot(grid[0, 2])
        salineAvgPlot=np.vstack(salineAvg)
        muscimolAvgPlot = np.vstack(muscimolAvg)
        salineSemVstack = np.vstack(salineSem)
        muscimolSemVstack=np.vstack(muscimolSem)
        #pdb.set_trace()
        ax1.plot(np.arange(10) + 1, salineAvgPlot[:,0], '-', label='FL-sal', linewidth=2, c='grey')
        ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, 0], '-', label='FL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,0] - salineSemVstack[:,0], salineAvgPlot[:,0] + salineSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 0] - muscimolSemVstack[:, 0],muscimolAvgPlot[:, 0] + muscimolSemVstack[:, 0], color='crimson', alpha=0.1)

        ax2 = plt.subplot(grid[1, 2])
        ax2.plot(np.arange(10) + 1, salineAvgPlot[:, 1], '-', label='FR-sal', linewidth=2, c='k')
        ax2.plot(np.arange(10) + 1, muscimolAvgPlot[:, 1], '-', label='FR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,1] - salineSemVstack[:,1], salineAvgPlot[:,1] + salineSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 1] - muscimolSemVstack[:, 1],muscimolAvgPlot[:, 1] + muscimolSemVstack[:, 1], color='coral', alpha=0.1)

        ax3 = plt.subplot(grid[0, 3])
        ax3.plot(np.arange(10) + 1, salineAvgPlot[:,2], '-', label='HL-sal', linewidth=2, c='grey')
        ax3.plot(np.arange(10) + 1, muscimolAvgPlot[:,2], '-', label='HL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,2] - salineSemVstack[:,2], salineAvgPlot[:,2] + salineSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 2] - muscimolSemVstack[:, 2],muscimolAvgPlot[:, 2] + muscimolSemVstack[:, 2], color='crimson', alpha=0.1)

        ax4 = plt.subplot(grid[1, 3])
        ax4.plot(np.arange(10) + 1, salineAvgPlot[:, 3], '-', label='HR-sal', linewidth=2, c='k')
        ax4.plot(np.arange(10) + 1, muscimolAvgPlot[:, 3], '-', label='HR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,3] - salineSemVstack[:,3], salineAvgPlot[:,3] + salineSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 3] - muscimolSemVstack[:, 3],muscimolAvgPlot[:, 3] + muscimolSemVstack[:, 3], color='coral', alpha=0.1)

        self.BehaviorFigStyle(ax0, xLabel='Days', yLabel='Fraction of indecisive stride')
        self.BehaviorFigStyle(ax1)
        self.BehaviorFigStyle(ax2)
        self.layoutOfPanel(ax3)
        self.layoutOfPanel(ax4)
        if individualValues:
            fname = '%s_Fraction_of_indecisive stride_figure_individual values'%experiment
        else:
            fname = '%s_Fraction of indecisive stride_figure'%experiment


        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')

    def generateBehaviorStrideLengthFig (self, strideLengthData,experiment):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        nDays = []
        xArray=[]
        animalNames = []

        fig_width = 12  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 7, 'axes.titlesize': 7, 'font.size': 7, 'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        grid = plt.GridSpec(2, 4, wspace=.6, hspace=.25)

        #plot average speed values saline vs muscimol
        ax0 = plt.subplot(grid[:, :2])
        (salineAvg, MuscimolAvg, saline, muscimol, nSaline, nMuscimol) = groupAnalysis.getTreatmentAverages1D(strideLengthData[0])
        ax0.plot(np.arange(10) + 1, salineAvg[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvg[0] - salineAvg[2],salineAvg[0] + salineAvg[2], color='grey', alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolAvg[0] - MuscimolAvg[2],MuscimolAvg[0] +MuscimolAvg[2], color='r', alpha=0.1)
        ax0.text(0.75, 0.87, '(N=%s saline, %s muscimol)' % (nSaline,nMuscimol), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='k')
        #
        (salineArray,muscimolArray)=groupAnalysis.splitGroups(strideLengthData[3])
        #pdb.set_trace()
        salineAvg=np.nanmean(salineArray,axis=0)
        salineSem=stats.sem(salineArray,axis=0,nan_policy='omit')
        muscimolAvg=np.nanmean(muscimolArray,axis=0)
        muscimolSem = stats.sem(muscimolArray, axis=0, nan_policy='omit')


        ax1 = plt.subplot(grid[0, 2])
        salineAvgPlot=np.vstack(salineAvg)
        muscimolAvgPlot = np.vstack(muscimolAvg)
        salineSemVstack = np.vstack(salineSem)
        muscimolSemVstack=np.vstack(muscimolSem)
        #pdb.set_trace()
        ax1.plot(np.arange(10) + 1, salineAvgPlot[:,0], '-', label='FL-sal', linewidth=2, c='grey')
        ax1.plot(np.arange(10) + 1, muscimolAvgPlot[:, 0], '-', label='FL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,0] - salineSemVstack[:,0], salineAvgPlot[:,0] + salineSemVstack[:,0], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 0] - muscimolSemVstack[:, 0],muscimolAvgPlot[:, 0] + muscimolSemVstack[:, 0], color='crimson', alpha=0.1)

        ax2 = plt.subplot(grid[1, 2])
        ax2.plot(np.arange(10) + 1, salineAvgPlot[:, 1], '-', label='FR-sal', linewidth=2, c='k')
        ax2.plot(np.arange(10) + 1, muscimolAvgPlot[:, 1], '-', label='FR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,1] - salineSemVstack[:,1], salineAvgPlot[:,1] + salineSemVstack[:,1], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 1] - muscimolSemVstack[:, 1],muscimolAvgPlot[:, 1] + muscimolSemVstack[:, 1], color='coral', alpha=0.1)

        ax3 = plt.subplot(grid[0, 3])
        ax3.plot(np.arange(10) + 1, salineAvgPlot[:,2], '-', label='HL-sal', linewidth=2, c='grey')
        ax3.plot(np.arange(10) + 1, muscimolAvgPlot[:,2], '-', label='HL-mus', linewidth=2, c='crimson')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,2] - salineSemVstack[:,2], salineAvgPlot[:,2] + salineSemVstack[:,2], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 2] - muscimolSemVstack[:, 2],muscimolAvgPlot[:, 2] + muscimolSemVstack[:, 2], color='crimson', alpha=0.1)

        ax4 = plt.subplot(grid[1, 3])
        ax4.plot(np.arange(10) + 1, salineAvgPlot[:, 3], '-', label='HR-sal', linewidth=2, c='k')
        ax4.plot(np.arange(10) + 1, muscimolAvgPlot[:, 3], '-', label='HR-mus', linewidth=2, c='coral')
        plt.fill_between(np.arange(10) + 1, salineAvgPlot[:,3] - salineSemVstack[:,3], salineAvgPlot[:,3] + salineSemVstack[:,3], color='grey', alpha=0.1)
        plt.fill_between(np.arange(10) + 1, muscimolAvgPlot[:, 3] - muscimolSemVstack[:, 3],muscimolAvgPlot[:, 3] + muscimolSemVstack[:, 3], color='coral', alpha=0.1)

        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride Length (cm)',Leg=[1, 9])
        self.layoutOfPanel(ax1, xLabel='Days', yLabel='Stride Length (cm)', Leg=[1, 9])
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='Stride Length (cm)', Leg=[1, 9])
        self.layoutOfPanel(ax3, xLabel='Days', yLabel='Stride Length (cm)', Leg=[1, 9])
        self.layoutOfPanel(ax4, xLabel='Days', yLabel='Stride Length (cm)', Leg=[1, 9])
        ax1.legend(loc="upper right", frameon=False, fontsize=6)
        ax2.legend(loc="upper right", frameon=False, fontsize=6)
        ax3.legend(loc="upper right", frameon=False, fontsize=6)
        ax4.legend(loc="upper right", frameon=False, fontsize=6)
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        ax1.xaxis.set_major_locator(majorLocator_x)
        ax2.xaxis.set_major_locator(majorLocator_x)
        ax3.xaxis.set_major_locator(majorLocator_x)
        ax4.xaxis.set_major_locator(majorLocator_x)
        fname = '%s_Stride Length_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.png')

    def createBehaviorStepTimingFigure (self, stepDurationData, experiment):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')

        fig_width = 15  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 3,  # ,
                               width_ratios=[1,1,1],
                               height_ratios=[1,1,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.6, hspace=0.25)
        plt.figtext(0.30, 0.95, "Limb duty factor, temporal symmetry", clip_on=False, color='black', size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])

        (salineArray, muscimolArray) = groupAnalysis.splitGroups(
            stepDurationData[8])
        salineAvg=np.mean(np.nanmean(salineArray,axis=2),axis=0)
        salineSem=stats.sem((np.nanmean(salineArray,axis=2)),axis=0,nan_policy='omit')
        MuscimolAvg = np.mean(np.nanmean(muscimolArray,axis=2),axis=0)
        MuscimolSem = stats.sem((np.nanmean(muscimolArray,axis=2)),axis=0,nan_policy='omit')
        #pdb.set_trace()


        ax0.plot(np.arange(10) + 1, salineAvg, '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvg - salineSem, salineAvg+ salineSem, color='grey',alpha=0.1)
        ax0.plot(np.arange(10) + 1, MuscimolAvg, '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, MuscimolAvg - MuscimolSem, MuscimolAvg + MuscimolSem, color='r',alpha=0.1)
        # ax0.text(0.75, 0.87, '(N=%s saline, %s muscimol)' % (nSaline, nMuscimol), ha='left', va='center',
        #          transform=ax0.transAxes, style='italic', fontsize=6, color='k')
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Limb duty factor', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        ax1 = plt.subplot(gs[1])

        (salineAvgTempHind,muscimolAvgTempHind, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[9])
        # pdb.set_trace()
        ax1.plot(np.arange(10) + 1, salineAvgTempHind[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempHind[0] - salineAvgTempHind[2],salineAvgTempHind[0] + salineAvgTempHind[2], color='grey', alpha=0.1)
        ax1.plot(np.arange(10) + 1, muscimolAvgTempHind[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempHind[0] - muscimolAvgTempHind[2],muscimolAvgTempHind[0] +muscimolAvgTempHind[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax1, xLabel='Days', yLabel='Hind Limbs Temporal symmetry', Leg=[1, 9])
        
        ax2 = plt.subplot(gs[2])

        (salineAvgTempFront,muscimolAvgTempFront, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[10])
        # pdb.set_trace()
        ax2.plot(np.arange(10) + 1, salineAvgTempFront[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempFront[0] - salineAvgTempFront[2],salineAvgTempFront[0] + salineAvgTempFront[2], color='grey', alpha=0.1)
        ax2.plot(np.arange(10) + 1, muscimolAvgTempFront[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempFront[0] - muscimolAvgTempFront[2],muscimolAvgTempFront[0] +muscimolAvgTempFront[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax2.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='Front Limbs Temporal symmetry', Leg=[1, 9])
        
        ax3 = plt.subplot(gs[3])

        (salineAvgTempFR_HL,muscimolAvgTempFR_HL, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[11])
        # pdb.set_trace()
        ax3.plot(np.arange(10) + 1, salineAvgTempFR_HL[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempFR_HL[0] - salineAvgTempFR_HL[2],salineAvgTempFR_HL[0] + salineAvgTempFR_HL[2], color='grey', alpha=0.1)
        ax3.plot(np.arange(10) + 1, muscimolAvgTempFR_HL[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempFR_HL[0] - muscimolAvgTempFR_HL[2],muscimolAvgTempFR_HL[0] +muscimolAvgTempFR_HL[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax3.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax3, xLabel='Days', yLabel='FR_HL Limbs Temporal symmetry', Leg=[1, 9])
        
        ax4 = plt.subplot(gs[4])

        (salineAvgTempFL_HR,muscimolAvgTempFL_HR, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[12])
        # pdb.set_trace()
        ax4.plot(np.arange(10) + 1, salineAvgTempFL_HR[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempFL_HR[0] - salineAvgTempFL_HR[2],salineAvgTempFL_HR[0] + salineAvgTempFL_HR[2], color='grey', alpha=0.1)
        ax4.plot(np.arange(10) + 1, muscimolAvgTempFL_HR[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempFL_HR[0] - muscimolAvgTempFL_HR[2],muscimolAvgTempFL_HR[0] +muscimolAvgTempFL_HR[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax4.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax4, xLabel='Days', yLabel='FL_HR Limbs Temporal symmetry', Leg=[1, 9])

        ax5 = plt.subplot(gs[5])

        (salineAvgTempFR_HR,muscimolAvgTempFR_HR, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[13])
        # pdb.set_trace()
        ax5.plot(np.arange(10) + 1, salineAvgTempFR_HR[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempFR_HR[0] - salineAvgTempFR_HR[2],salineAvgTempFR_HR[0] + salineAvgTempFR_HR[2], color='grey', alpha=0.1)
        ax5.plot(np.arange(10) + 1, muscimolAvgTempFR_HR[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempFR_HR[0] - muscimolAvgTempFR_HR[2],muscimolAvgTempFR_HR[0] +muscimolAvgTempFR_HR[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax5.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax5, xLabel='Days', yLabel='FR_HR Limbs Temporal symmetry', Leg=[1, 9])
        
        ax6 = plt.subplot(gs[6])

        (salineAvgTempFL_HL,muscimolAvgTempFL_HL, saline, muscimol,nSaline,nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[14])
        # pdb.set_trace()
        ax6.plot(np.arange(10) + 1, salineAvgTempFL_HL[0], '-', label='saline', linewidth=2, c='k')
        plt.fill_between(np.arange(10) + 1, salineAvgTempFL_HL[0] - salineAvgTempFL_HL[2],salineAvgTempFL_HL[0] + salineAvgTempFL_HL[2], color='grey', alpha=0.1)
        ax6.plot(np.arange(10) + 1, muscimolAvgTempFL_HL[0], '-', label='muscimol', linewidth=2, c='r')
        plt.fill_between(np.arange(10) + 1, muscimolAvgTempFL_HL[0] - muscimolAvgTempFL_HL[2],muscimolAvgTempFL_HL[0] +muscimolAvgTempFL_HL[2], color='r', alpha=0.1)
        majorLocator_x = MultipleLocator(1)
        ax6.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax6, xLabel='Days', yLabel='FL_HL Limbs Temporal symmetry', Leg=[1, 9])
        
        fname = '%s_Temp_parameters_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')


    def createStepDurationFigureSeparated(self, stepDurationData,experiment):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')
        nDays = []
        xArray = []
        animalNames = []
        for i in range(len(stepDurationData[0])):
            animalNames.append(stepDurationData[0][i][0])
        print(animalNames)
        fig_width = 12  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid':False  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(4, 2,  # ,
                               width_ratios=[1, 1,1],
                               height_ratios=[1,1,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.4)
        # plt.figtext(0.30, 0.95, "Median swing/stance duration", clip_on=False, color='black', size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        ax2=plt.subplot(gs[2])

        #plot average values with individual mice
        for f in range(len(stepDurationData[0])):
            nDays = len(stepDurationData[0][f][1])

            
            #swing values
            ax0.plot(np.arange(nDays)[:11] + 1, stepDurationData[0][f][1][:11], 'o-', ms=2, label=stepDurationData[0][f][0],
                     color=cmap(f / len(stepDurationData[0])), alpha=0.2)
            #stance values
            ax2.plot(np.arange(nDays)[:11] + 1, stepDurationData[2][f][1][:11], 'o-', ms=2, label=None,
                     color=cmap(f / len(stepDurationData[0])), alpha=0.2)
            ax2.legend(loc="upper left", bbox_to_anchor=(10, 0.1))

        #limit average and SEM to 11 days
        swingDuration11days = stepDurationData[4][0][0:11]
        swingDurationdaysError = stepDurationData[4][1][0:11]
        stanceDuration11days = stepDurationData[5][0][0:11]
        stanceDuration11daysError = stepDurationData[5][1][0:11]


        #perform  and put stats as text
        (df,mdf, conf_int, pvalues, stars) = groupAnalysis.performMixedLinearModelRegression(stepDurationData[0], treatments=False,
                                                                          trialValues=False)
        (df,mdfStance, conf_intStance, pvaluesStance, starsStance) = groupAnalysis.performMixedLinearModelRegression(stepDurationData[2],
                                                                                      treatments=False,
                                                                                      trialValues=False)
        swingMxLM=[round(mdf.fe_params['recordingDay'], 4), round(mdf.pvalues['recordingDay'], 4), round(conf_int.iloc[1, 0], 3),
        round(conf_int.iloc[1, 1], 3)]
        stanceMxLM=[round(mdfStance.fe_params['recordingDay'], 4), round(mdfStance.pvalues['recordingDay'], 4),round(conf_intStance.iloc[1, 0], 3), round(conf_intStance.iloc[1, 1], 3)]
        (swingStars)=groupAnalysis.starMultiplier(swingMxLM[1])
        (stanceStars) = groupAnalysis.starMultiplier(stanceMxLM[1])


        ax0.text(0.0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (
        swingMxLM[0],  swingMxLM[1],  swingMxLM[2],  swingMxLM[3]),
                 ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=6, color='0.7')
        ax2.text(0, 0, 'Stance--MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (
        stanceMxLM[0],  stanceMxLM[1],  stanceMxLM[2],  stanceMxLM[3]),
                 ha='left', va='center', transform=ax2.transAxes, style='italic', fontsize=6, color='0.8')
        # ax0.text(0.40, 0.04, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResStance.anova_table.iloc[0, 2], 4), round(AnovaRMResStance.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7,color=cmap(0.1))
        
        #plot the average values
        #swing
        ax0.plot(np.arange(len(swingDuration11days)) + 1, swingDuration11days, '-', color='k',
                 linewidth=2)
        #stance
        ax2.plot(np.arange(len(stanceDuration11days)) + 1, stanceDuration11days, '-', color='0.5', linewidth=2)
        #SEM as transparent fill
        ax0.fill_between(np.arange(len(swingDuration11days)) + 1, swingDuration11days - swingDurationdaysError,
                         swingDuration11days + swingDurationdaysError, color='0.6', alpha=0.2)
        ax2.fill_between(np.arange(len(stanceDuration11days)) + 1, stanceDuration11days - stanceDuration11daysError,
                         stanceDuration11days + stanceDuration11daysError, color='0.7', alpha=0.2)
        #style and axes labels
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Mean swing duration (s)', Leg=[1, 9])
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='Mean stance duration (s)', Leg=[1, 9])
        ax0.set_title('Mean swing duration (all paws)')
        ax2.set_title('Mean stance duration (all paws)')
        ax0.text(0.9, 0.95, swingStars, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax2.text(0.9, 0.95, stanceStars, ha='left', va='center',transform=ax2.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(stepDurationData[0]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=10, color='k')
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        ax2.xaxis.set_major_locator(majorLocator_x)

        (PawSwingDuration, meanPawSwingDuration, semPawSwingDuration) = groupAnalysis.getAverageSingleGroup(
            stepDurationData[6])
        (swingDf, swingDurPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            stepDurationData[6],sessionValues=True, treatments=False)
        #plot paw specific values
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        for i in range(4):
            (swingStar)=groupAnalysis.starMultiplier(swingDurPValues[i])
            ax1 = plt.subplot(gs[1])
            ax1.plot(np.arange(10) + 1, meanPawSwingDuration[:, i][:11], '-', label='%s %s'%(pawId[i],swingStar), linewidth=2, c='C%s'%i)
            ax1.fill_between(np.arange(10) + 1, meanPawSwingDuration[:, i][:11] - semPawSwingDuration[:, i][:11],
                             meanPawSwingDuration[:, i][:11] + semPawSwingDuration[:, i][:11], color='C%s'%i, alpha=0.1)

            self.layoutOfPanel(ax1, xLabel='Days', yLabel=None, Leg=[1, 2])
            ax1.set_title('Mean swing duration (paw specific)')
            (PawStanceDuration, meanPawStanceDuration, semPawStanceDuration) = groupAnalysis.getAverageSingleGroup(
                stepDurationData[7])
            (stanceDf, stanceDurPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
                stepDurationData[7],sessionValues=True, treatments=False)
            (stanceStar)=groupAnalysis.starMultiplier(stanceDurPValues[i])

            ax4=plt.subplot(gs[3])
            ax4.plot(np.arange(10) + 1, meanPawStanceDuration[:, i][:11], '-', label='%s %s'%(pawId[i],stanceStar), linewidth=2, c='C%s'%i)
            ax4.fill_between(np.arange(10) + 1, meanPawStanceDuration[:, i][:11] - semPawStanceDuration[:, 0][:11],
                             meanPawStanceDuration[:, i][:11] + semPawStanceDuration[:, i][:11], color='C%s'%i, alpha=0.1)

            self.layoutOfPanel(ax4, xLabel='Days', yLabel=None, Leg=[1, 2])
            ax4.set_title('Mean stance duration (paw specific)')
        (swingDurStd_df, swingDurStd_mdf, swingDurStd_md_paw, swingDurStd_pawStars, swingDurStd_Stars_all, swingDur_tukey_paw) = groupAnalysis.pandaDataFrameAndMixedMLCompleteData(stepDurationData[11], treatments=False, varName='swing duration Std')
        swingDurStd_df['paw'].replace({0: 'FL', 1: 'FR', 2: 'HL', 3: 'HR'}, inplace=True)
        ax5=plt.subplot(gs[4])
        ax6=plt.subplot(gs[5])
        # sns.lineplot(data=swingDurStd_df, x='recordingDay', y='measuredValue', hue=None, errorbar='se',err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax5)
        sns.lineplot(data=swingDurStd_df, x='recordingDay', y='measuredValue', hue=None, err_style='band', err_kws={'alpha':0.07}, color='black', ax=ax5)
        sns.lineplot(data=swingDurStd_df, x='recordingDay', y='measuredValue', hue='mouse', errorbar=None, alpha=0.2, ax=ax5)
        sns.lineplot(data=swingDurStd_df, x='recordingDay', y='measuredValue', hue='paw',palette=sns.color_palette("tab10", n_colors=4), legend='auto', ax=ax6)
        self.layoutOfPanel(ax5, xLabel='Day', yLabel='Swing Duration Std', Leg=[1, 9])
        self.layoutOfPanel(ax6, xLabel='Day', yLabel='Swing Duration Std', Leg=[1, 9])
        star_day = groupAnalysis.starMultiplier(swingDurStd_mdf.pvalues['recordingDay'])
        star_trial = groupAnalysis.starMultiplier(swingDurStd_mdf.pvalues['trial'])
        ax5.text(0.9, 0.97, '%s' % (star_day), ha='center', va='center', transform=ax5.transAxes, style='italic',fontfamily='serif', fontsize=12, color='k')
        ax5.text(0.9, 0.92, '%s' % (star_trial.replace('*','#')), ha='center', va='center', transform=ax5.transAxes, style='italic',fontfamily='serif', fontsize=12, color='k')
        ax1.legend(loc="upper right", frameon=False, fontsize=8)

        ax4.legend(loc="upper right", frameon=False, fontsize=8)
        ax6.legend(frameon=False, fontsize=8)
        ax1.xaxis.set_major_locator(majorLocator_x)

        ax4.xaxis.set_major_locator(majorLocator_x)






        fname = '%s_swing_stance_duration_average'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')


    def createStepTimingFigure(self, stepDurationData,experiment):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')

        fig_width = 15  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 3,  # ,
                               width_ratios=[1, 1, 1],
                               height_ratios=[1, 1, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.6, hspace=0.25)
        plt.figtext(0.30, 0.95, "Limb duty factor, temporal symmetry", clip_on=False, color='black', size=14)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        # pdb.set_trace()
        (LDFArray,meanLDF, semLDF)= groupAnalysis.getAverageSingleGroup(
            stepDurationData[8])
        colors=['steelblue','darkorange','yellowgreen','salmon']
        labels = ['FL', 'FR', 'HL', 'HR']
        #pdb.set_trace()
        for p in range(4):
            ax0.plot(np.arange(11) + 1, meanLDF[:,p][:11], '-', label=labels[p], linewidth=2, c=colors[p])
            plt.fill_between(np.arange(11) + 1, meanLDF[:,p][:11] - semLDF[:,p][:11], meanLDF[:,p][:11] + semLDF[:,p][:11], color=colors[p], alpha=0.1)
        # ax0.text(0.75, 0.87, '(N=%s saline, %s muscimol)' % (nSaline, nMuscimol), ha='left', va='center',
        #          transform=ax0.transAxes, style='italic', fontsize=6, color='k')
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Limb duty factor', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        # ax1 = plt.subplot(gs[1])
        #
        # (meanTempHind,stdTempHind, semTemPHind)= groupAnalysis.getAverageSingleGroup(
        #     stepDurationData[9])
        # # pdb.set_trace()
        # ax1.plot(np.arange(10) + 1, meanTempHind, '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, meanTempHind - semTemPHind,
        #                  meanTempHind + semTemPHind, color='grey', alpha=0.1)
        #
        # majorLocator_x = MultipleLocator(1)
        # ax0.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax1, xLabel='Days', yLabel='Hind Limbs Temporal symmetry', Leg=[1, 9])
        #
        # ax2 = plt.subplot(gs[2])
        #
        # (meanTempFront,stdTempFront, semTempFront)= groupAnalysis.getAverageSingleGroup(
        #     stepDurationData[10])
        # # pdb.set_trace()
        # ax2.plot(np.arange(10) + 1, meanTempFront, '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, meanTempFront - semTempFront,
        #                  meanTempFront + semTempFront, color='grey', alpha=0.1)
        # majorLocator_x = MultipleLocator(1)
        # ax2.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax2, xLabel='Days', yLabel='Front Limbs Temporal symmetry', Leg=[1, 9])

        # ax3 = plt.subplot(gs[3])
        #
        # (salineAvgTempFR_HL, muscimolAvgTempFR_HL, saline, muscimol, nSaline,
        #  nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[11])
        # # pdb.set_trace()
        # ax3.plot(np.arange(10) + 1, salineAvgTempFR_HL[0], '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, salineAvgTempFR_HL[0] - salineAvgTempFR_HL[2],
        #                  salineAvgTempFR_HL[0] + salineAvgTempFR_HL[2], color='grey', alpha=0.1)
        # ax3.plot(np.arange(10) + 1, muscimolAvgTempFR_HL[0], '-', label='muscimol', linewidth=2, c='r')
        # plt.fill_between(np.arange(10) + 1, muscimolAvgTempFR_HL[0] - muscimolAvgTempFR_HL[2],
        #                  muscimolAvgTempFR_HL[0] + muscimolAvgTempFR_HL[2], color='r', alpha=0.1)
        # majorLocator_x = MultipleLocator(1)
        # ax3.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax3, xLabel='Days', yLabel='FR_HL Limbs Temporal symmetry', Leg=[1, 9])
        #
        # ax4 = plt.subplot(gs[4])
        #
        # (salineAvgTempFL_HR, muscimolAvgTempFL_HR, saline, muscimol, nSaline,
        #  nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[12])
        # # pdb.set_trace()
        # ax4.plot(np.arange(10) + 1, salineAvgTempFL_HR[0], '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, salineAvgTempFL_HR[0] - salineAvgTempFL_HR[2],
        #                  salineAvgTempFL_HR[0] + salineAvgTempFL_HR[2], color='grey', alpha=0.1)
        # ax4.plot(np.arange(10) + 1, muscimolAvgTempFL_HR[0], '-', label='muscimol', linewidth=2, c='r')
        # plt.fill_between(np.arange(10) + 1, muscimolAvgTempFL_HR[0] - muscimolAvgTempFL_HR[2],
        #                  muscimolAvgTempFL_HR[0] + muscimolAvgTempFL_HR[2], color='r', alpha=0.1)
        # majorLocator_x = MultipleLocator(1)
        # ax4.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax4, xLabel='Days', yLabel='FL_HR Limbs Temporal symmetry', Leg=[1, 9])
        #
        # ax5 = plt.subplot(gs[5])
        #
        # (salineAvgTempFR_HR, muscimolAvgTempFR_HR, saline, muscimol, nSaline,
        #  nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[13])
        # # pdb.set_trace()
        # ax5.plot(np.arange(10) + 1, salineAvgTempFR_HR[0], '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, salineAvgTempFR_HR[0] - salineAvgTempFR_HR[2],
        #                  salineAvgTempFR_HR[0] + salineAvgTempFR_HR[2], color='grey', alpha=0.1)
        # ax5.plot(np.arange(10) + 1, muscimolAvgTempFR_HR[0], '-', label='muscimol', linewidth=2, c='r')
        # plt.fill_between(np.arange(10) + 1, muscimolAvgTempFR_HR[0] - muscimolAvgTempFR_HR[2],
        #                  muscimolAvgTempFR_HR[0] + muscimolAvgTempFR_HR[2], color='r', alpha=0.1)
        # majorLocator_x = MultipleLocator(1)
        # ax5.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax5, xLabel='Days', yLabel='FR_HR Limbs Temporal symmetry', Leg=[1, 9])
        #
        # ax6 = plt.subplot(gs[6])
        #
        # (salineAvgTempFL_HL, muscimolAvgTempFL_HL, saline, muscimol, nSaline,
        #  nMuscimol) = groupAnalysis.getTreatmentAverages1D(stepDurationData[14])
        # # pdb.set_trace()
        # ax6.plot(np.arange(10) + 1, salineAvgTempFL_HL[0], '-', label='saline', linewidth=2, c='k')
        # plt.fill_between(np.arange(10) + 1, salineAvgTempFL_HL[0] - salineAvgTempFL_HL[2],
        #                  salineAvgTempFL_HL[0] + salineAvgTempFL_HL[2], color='grey', alpha=0.1)
        # ax6.plot(np.arange(10) + 1, muscimolAvgTempFL_HL[0], '-', label='muscimol', linewidth=2, c='r')
        # plt.fill_between(np.arange(10) + 1, muscimolAvgTempFL_HL[0] - muscimolAvgTempFL_HL[2],
        #                  muscimolAvgTempFL_HL[0] + muscimolAvgTempFL_HL[2], color='r', alpha=0.1)
        # majorLocator_x = MultipleLocator(1)
        # ax6.xaxis.set_major_locator(majorLocator_x)
        # self.layoutOfPanel(ax6, xLabel='Days', yLabel='FL_HL Limbs Temporal symmetry', Leg=[1, 9])

        fname = '%s_stride_timing_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')

    def createBehaviorFigureMaleFemaleAvg(self,  strideNumberData,stepDurationData,pawSpeedData,rungCrossedData,stepsQualityData):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        cmapM = cm.get_cmap('PuBu')
        cmapF = cm.get_cmap('PuRd')
        nDays = []
        xArray=[]
        animalNames = []
        for i in range(len(strideNumberData[1])):
            animalNames.append(strideNumberData[1][i][0])
        print(animalNames)
        fig_width = 18  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 3)
        #width_ratios=[1.2,1], height_ratios=[1, 2.5])
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        #plt.figtext(0.35, 0.95, "Average stride number", clip_on=False, color='black',size=14)

        (femaleStepsNb, maleStepsNb)=groupAnalysis.splitMaleFemale(strideNumberData[0])
        (df,pvalues)=groupAnalysis.ListToPandasDFAndStatsSex(strideNumberData[0], sessionValues=False, sex=True)

                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        #pdb.set_trace()
        for f in range(len(femaleStepsNb[0])):
            ax0.plot(np.arange(10)+1,femaleStepsNb[0][f][:10], 'o-',ms=2, label=None, color=cmapF(f/len(strideNumberData[1])),alpha=0.2)
        for f in range(len(maleStepsNb[0])):
            ax0.plot(np.arange(10)+1,maleStepsNb[0][f][:10], 'o-',ms=2, label=None, color=cmapM(f/len(strideNumberData[1])),alpha=0.2)
        ax0.legend(loc="upper left", bbox_to_anchor=(10,0.1))

        multiplier = 0
        if strideNumberData[2][2][1]<0.001:
            multiplier = 3
        elif strideNumberData[2][2][1]<0.01:
            multiplier = 2
        elif strideNumberData[2][2][1]<0.05:
            multiplier = 1
        ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        ax0.text(0.77, 0.9, '(N=%s males, %s females)' % (len(maleStepsNb[0]),len(femaleStepsNb[0])), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=8, color='k')
        ax0.text(0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' % (strideNumberData[2][2][0], strideNumberData[2][2][1],strideNumberData[2][2][2],strideNumberData[2][2][3]),ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=7,color='0.7')
        ax0.plot(np.arange(10)+1,maleStepsNb[1][:10],'-',label='maleStepsNb', linewidth=2,c='k')
        ax0.plot(np.arange(10) + 1, femaleStepsNb[1][:10], '-', label='femaleStepsNb', linewidth=2, c='salmon')
        #ax0.errorbar(np.arange(10),stepNb11days, yerr=stepNbSEM11days, color=cmap(0.01) )
        plt.fill_between(np.arange(10)+1, maleStepsNb[1][:10] - maleStepsNb[2][:10],maleStepsNb[1][:10] + maleStepsNb[2][:10], color='0.6', alpha=0.2)
        plt.fill_between(np.arange(10)+1, femaleStepsNb[1][:10] - femaleStepsNb[2][:10],femaleStepsNb[1][:10] + femaleStepsNb[2][:10], color='salmon', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride number (avg.)')#, Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
    #--------------------------------------second subplot swing Speed-------------------------------------------
        (femaleDurSwing, maleDurSwing)=groupAnalysis.splitMaleFemale(stepDurationData[0])
        (femaleDurStance, maleDurStance) = groupAnalysis.splitMaleFemale(stepDurationData[2])
        # pdb.set_trace()
        ax1 = plt.subplot(gs[1])
        for f in range(len(femaleDurSwing[0])):
            ax1.plot(np.arange(10)[:10]+1, femaleDurSwing[0][f][:10], 'o-', ms=2,label=None, color=cmapF(f/len(strideNumberData[1])),alpha=0.2)
            ax1.plot(np.arange(10)[:10] + 1, femaleDurStance[0][f][:10], 'o-', ms=2, label=None, color=cmapF(f/len(strideNumberData[1])),alpha=0.2)
        for f in range(len(maleDurSwing[0])):
            ax1.plot(np.arange(10)+1, maleDurSwing[0][f][:10], 'o-', ms=2,label=None, color=cmapM(f/len(strideNumberData[1])),alpha=0.2)
            ax1.plot(np.arange(10)+ 1, maleDurStance[0][f][:10], 'o-', ms=2, label=None, color=cmapM(f/len(strideNumberData[1])),alpha=0.2)

        ax1.legend(loc="upper left", bbox_to_anchor=(10, 0.1))
        multiplier = 0
        if stepDurationData[4][2][1]<0.001:
            multiplier = 3
        elif stepDurationData[4][2][1]<0.01:
            multiplier = 2
        elif stepDurationData[4][2][1]<0.05:
            multiplier = 1
        #ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=20, color='k')
        # ax1.text(0.80, 1, '(N=%s)' % len(stepDurationData[0]), ha='left', va='center', transform=ax1.transAxes, style='italic', fontsize=10, color='k')

        ax1.text(0.0, 0, 'Swing_MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[3][2][0], stepDurationData[3][2][1], stepDurationData[3][2][2],stepDurationData[3][2][3]), ha='left', va='center', transform=ax1.transAxes,style='italic', fontsize=7,color='0.7')
        ax1.text(0, 0.030, 'Stance_MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[4][2][0], stepDurationData[4][2][1], stepDurationData[4][2][2],stepDurationData[4][2][3]), ha='left', va='center', transform=ax1.transAxes, style='italic', fontsize=7,color='0.8')
        #ax0.text(0.40, 0.04, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResStance.anova_table.iloc[0, 2], 4), round(AnovaRMResStance.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7,color=cmap(0.1))


        ax1.plot(np.arange(10)+1, femaleDurSwing[1][:10], '--', label="swing  n.s.", color='salmon',linewidth=2)
        ax1.plot(np.arange(10) + 1, maleDurSwing[1][:10], '--', label="swing  n.s.", color='k', linewidth=2)
                 
        ax1.plot(np.arange(10)+1, femaleDurStance[1][:10], '-', label="stance  %s"%('*'*multiplier), color='salmon', linewidth=2)
        ax1.plot(np.arange(10)+1, maleDurStance[1][:10], '-', label="stance  %s"%('*'*multiplier), color='0.5', linewidth=2)
                 
        plt.fill_between(np.arange(10)+1, femaleDurSwing[1][:10] - femaleDurSwing[2][:10], femaleDurSwing[1][:10] + femaleDurSwing[2][:10],color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10)+1, maleDurSwing[1][:10] - maleDurSwing[2][:10],maleDurSwing[1][:10] + maleDurSwing[2][:10], color='0.7', alpha=0.2)
        plt.fill_between(np.arange(10)+1, femaleDurStance[1][:10] - femaleDurStance[2][:10], femaleDurStance[1][:10] + femaleDurStance[2][:10],color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10)+1, maleDurStance[1][:10] - maleDurStance[2][:10],maleDurStance[1][:10] + maleDurStance[2][:10], color='0.7', alpha=0.2)
        self.layoutOfPanel(ax1, xLabel='Days', yLabel='Mean swing/stance duration (s)', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax1.xaxis.set_major_locator(majorLocator_x)

    #---------------------third subplot-----Swing Speed-------------
        (femaleSwingSpeed, maleSwingSpeed)=groupAnalysis.splitMaleFemale(pawSpeedData[1])


        ax2 = plt.subplot(gs[2])
        for f in range (len(femaleSwingSpeed[0])):
            ax2.plot(np.arange(10)+1,femaleSwingSpeed[0][f][:10], 'o-', label=None, color=cmapF(f/len(pawSpeedData[1])),ms=2,alpha=0.2)
        for f in range (len(maleSwingSpeed[0])):
            ax2.plot(np.arange(10)+1,maleSwingSpeed[0][f][:10], 'o-', label=None, color=cmapM(f/len(pawSpeedData[1])),ms=2,alpha=0.2)
        ax2.legend(bbox_to_anchor=(1, 0.5),loc="upper left")
        plt.tight_layout()
        #ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        ax2.text(0.0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' %(pawSpeedData[2][2][0], pawSpeedData[2][2][1],pawSpeedData[2][2][2],pawSpeedData[2][2][3]),ha='left', va='center', transform=ax2.transAxes,style='italic', fontsize=7,color='0.7')
        multiplier = 0
        if pawSpeedData[2][2][1]<0.001:
            multiplier = 3
        elif pawSpeedData[2][2][1]<0.01:
            multiplier = 2
        elif pawSpeedData[2][2][1]<0.05:
            multiplier = 1
        ax2.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax2.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        # ax2.text(0.87, 0.9, '(N=%s)' % len(pawSpeedData[1]), ha='left', va='center', transform=ax2.transAxes, style='italic', fontsize=10, color='k')

        #ax0.errorbar(np.arange(10), speed11days, yerr=speed11daysError, color=cmap(0.01))

        ax2.plot(np.arange(10)+1,femaleSwingSpeed[1][:10],'-',label=None, color='salmon',linewidth=2)
        ax2.plot(np.arange(10)+1,maleSwingSpeed[1][:10],'-',label=None, color='k',linewidth=2)

        plt.fill_between(np.arange(10)+1,femaleSwingSpeed[1][:10]-femaleSwingSpeed[2][:10],femaleSwingSpeed[1][:10]+femaleSwingSpeed[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10)+1,maleSwingSpeed[1][:10]-maleSwingSpeed[2][:10],maleSwingSpeed[1][:10]+maleSwingSpeed[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='Average swing speed (cm/s)', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax2.xaxis.set_major_locator(majorLocator_x)
        ax2 = plt.subplot(gs[2])

    # ---------------------fourth subplot------------------rung crossed-----------------
        (femaleRungCrossed, maleRungCrossed)=groupAnalysis.splitMaleFemale(rungCrossedData[0])
        ax3 = plt.subplot(gs[3])
        nRec = 0

        for f in range (len(femaleRungCrossed[0])):
            ax3.plot(np.arange(10)+1,femaleRungCrossed[0][f][:10], 'o-', label=None, color=cmapF(f/len(rungCrossedData[0])),alpha=0.2,ms=2)
        for f in range(len(maleRungCrossed[0])):
            ax3.plot(np.arange(10)+1,maleRungCrossed[0][f][:10], 'o-', label=None, color=cmapM(f/len(rungCrossedData[0])),alpha=0.2,ms=2)

        ax3.legend(loc="upper left", bbox_to_anchor=(10,0.1))
        #ax0.text(0.4, 0.025, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResAll.anova_table.iloc[0, 2], 4), round(AnovaRMResAll.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7)
        ax3.text(0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (rungCrossedData[1][2][0], rungCrossedData[1][2][1],rungCrossedData[1][2][2], rungCrossedData[1][2][2]), ha='left', va='center', transform=ax3.transAxes,style='italic', fontsize=7,color='0.7')
        multiplier = 0
        if rungCrossedData[1][2][1]<0.001:
            multiplier = 3
        elif rungCrossedData[1][2][1]<0.01:
            multiplier = 2
        elif rungCrossedData[1][2][1]<0.05:
            multiplier = 1
        ax3.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax3.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        # ax3.text(0.87, 0.9, '(N=%s)' % len(rungCrossedData[0]), ha='left', va='center', transform=ax3.transAxes, style='italic', fontsize=10, color='k')


        ax3.plot(np.arange(10)+1,femaleRungCrossed[1][:10],'-',label=None, color='salmon',linewidth=2)
        ax3.plot(np.arange(10)+1,maleRungCrossed[1][:10],'-',label=None, color='k',linewidth=2)

        plt.fill_between(np.arange(10)+1,femaleRungCrossed[1][:10]-femaleRungCrossed[2][:10],femaleRungCrossed[1][:10]+femaleRungCrossed[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10)+1,maleRungCrossed[1][:10]-maleRungCrossed[2][:10],maleRungCrossed[1][:10]+maleRungCrossed[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax3, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax3.xaxis.set_major_locator(majorLocator_x)

    # ---------------------fourth subplot------------------------------------------------
        (femaleQuality, maleQuality)=groupAnalysis.splitMaleFemale(stepsQualityData[1])
        ax4 = plt.subplot(gs[4])
        for f in range(len(femaleQuality[0])):
            ax4.plot(np.arange(10) + 1, femaleQuality[0][f][:10], 'o-', label=None, color=cmapF(f / len(stepsQualityData[1][f][1])), ms=2, alpha=0.2)
        for f in range(len(maleQuality[0])):
            ax4.plot(np.arange(10) + 1, maleQuality[0][f][:10], 'o-', label=None, color=cmapM(f / len(stepsQualityData[1][f][1])), ms=2, alpha=0.2)

        ax4.legend(loc="upper left", bbox_to_anchor=(10, 0.1))
        inDstep11days = stepsQualityData[2][0][0:11]
        inDstepSEM11days = stepsQualityData[2][1][0:11]
        multiplier = 0
        if stepsQualityData[2][2][1] < 0.001:
            multiplier = 3
        elif stepsQualityData[2][2][1] < 0.01:
            multiplier = 2
        elif stepsQualityData[2][2][1] < 0.05:
            multiplier = 1
        ax4.text(0.9, 0.95, '*' * multiplier, ha='left', va='center', transform=ax4.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')
        # ax4.text(0.87, 0.9, '(N=%s)' % len(stepsQualityData[1]), ha='left', va='center', transform=ax4.transAxes,
        #          style='italic', fontsize=10, color='k')
        # ax0.errorbar(np.arange(10), inDstep11days, yerr=inDstepSEM11days, color=cmap(0.01))

        # ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        ax4.text(0.0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' % (
        stepsQualityData[2][2][0], stepsQualityData[2][2][1], stepsQualityData[2][2][2], stepsQualityData[2][2][3]),
                 ha='left', va='center', transform=ax4.transAxes, style='italic', fontsize=7, color='0.7')

        ax4.plot(np.arange(10) + 1, femaleQuality[1][:10], '-', label=None, color='salmon', linewidth=2)
        ax4.plot(np.arange(10) + 1, maleQuality[1][:10], '-', label=None, color='k', linewidth=2)

        plt.fill_between(np.arange(10) + 1, femaleQuality[1][:10] - femaleQuality[2][:10],femaleQuality[1][:10] + femaleQuality[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, maleQuality[1][:10] - maleQuality[2][:10],maleQuality[1][:10] + maleQuality[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax4, xLabel='Days', yLabel='Fraction of indecisive strides', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax4.xaxis.set_major_locator(majorLocator_x)

        fname = 'Behavior_summary_figure_male_Female'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')

    def createBehaviorFigureMuscimol(self, strideNumberData, stepDurationData, pawSpeedData, rungCrossedData,
                                          stepsQualityData, individualValues):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')
        cmapM = cm.get_cmap('PuBu')
        cmapF = cm.get_cmap('PuRd')
        nDays = []
        xArray = []
        animalNames = []
        for i in range(len(strideNumberData[1])):
            animalNames.append(strideNumberData[1][i][0])
        print(animalNames)
        fig_width = 30  # width in inches
        fig_height = 6  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 5)
        # width_ratios=[1.2,1], height_ratios=[1, 2.5])
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        plt.figtext(0.15, 0.01, "Mixed Linear Model Regression, treatment effect, *p<0.05, **p<0.01, ***p<0.001 ", clip_on=False, color='black',size=10)
        # first sub-plot #######################################################
        strideNumberDataFiltered=[]
        for a in range(len(strideNumberData[0])):
            if not ("220204_m50" or "201207_f42") in strideNumberData[0][a][0]  :
                strideNumberDataFiltered.append(strideNumberData[0][a])
        strideNumberData[0]=strideNumberDataFiltered

        (muscimolStepNbAvg, salineStepNbAvg) = groupAnalysis.splitTreatmentAvg(strideNumberData[0])
        (df, pvalues) = groupAnalysis.ListToPandasDFAndStatsSex(strideNumberData[0], sessionValues=False, sex=True)

        (mdf_stepNb, confInt_StepNb)=groupAnalysis.performMixedLinearModelRegression(strideNumberData[0],sessionValues=False, treatments=True)
        print("stats for stepNumber",mdf_stepNb.summary())

        pdb.set_trace()
        (anovaStepNb, posthocStepNb)=groupAnalysis.performMixedTwoWayANOVA(strideNumberData[1],sessionValues=True, treatments=True)
        # print('Treatment p-value=%s, Recording Day p-value=%s, Interaction p-value=%s' % (anovaStepNb.iloc[0, 6], anovaStepNb.iloc[1, 6], anovaStepNb.iloc[2, 6]))
        # print(anovaStepNb)


        ax0 = plt.subplot(gs[0])
        # treatmentStarsStepNb=groupAnalysis.starMultiplier(anovaStepNb.iloc[0, 6])
        treatmentStarsStepNb=groupAnalysis.starMultiplier(mdf_stepNb.pvalues['treatments[T.saline]'])
        ax0.text(0.5, 1.05, "Swing number", ha='center', va='center', transform=ax0.transAxes,fontfamily='sans-serif', fontsize=13, color='k')
        if individualValues==True:
            for f in range(len(muscimolStepNbAvg[0])):
                ax0.plot(np.arange(10) + 1, muscimolStepNbAvg[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapF(f / len(strideNumberData[1])), alpha=0.2)
            for f in range(len(salineStepNbAvg[0])):
                ax0.plot(np.arange(10) + 1, salineStepNbAvg[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapM(f / len(strideNumberData[1])), alpha=0.2)


        ax0.text(0.9, 0.95, treatmentStarsStepNb, ha='left', va='center', transform=ax0.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')


        ax0.plot(np.arange(10) + 1, salineStepNbAvg[1][:10], '-', label='saline (N=%s )'% (len(salineStepNbAvg[0])), linewidth=2, c='k')
        ax0.plot(np.arange(10) + 1, muscimolStepNbAvg[1][:10], '-', label='muscimol (N=%s )'% (len(muscimolStepNbAvg[0])), linewidth=2, c='salmon')
        #

        # ax0.errorbar(np.arange(10),stepNb11days, yerr=stepNbSEM11days, color=cmap(0.01) )
        plt.fill_between(np.arange(10) + 1, salineStepNbAvg[1][:10] - salineStepNbAvg[2][:10],
                         salineStepNbAvg[1][:10] + salineStepNbAvg[2][:10], color='0.6', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, muscimolStepNbAvg[1][:10] - muscimolStepNbAvg[2][:10],
                         muscimolStepNbAvg[1][:10] + muscimolStepNbAvg[2][:10], color='salmon', alpha=0.2)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride number (avg.)')  # , Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        ax0.legend(loc="upper left", bbox_to_anchor=(0.6,0.9), frameon=False)
        # ax0.legend()
     
        # --------------------------------------second subplot stride duration-------------------------------------------
        stepDurationDataFiltered=[]
        for a in range(len(stepDurationData[0])):
            if not ("220204_m50" or "201207_f42") in stepDurationData[0][a][0]  :
                stepDurationDataFiltered.append(stepDurationData[0][a])
        stepDurationData[0]=stepDurationDataFiltered
        stanceDurationDataFiltered=[]
        for a in range(len(stepDurationData[2])):
            if not ("220204_m50" or "201207_f42") in stepDurationData[2][a][0]  :
                stanceDurationDataFiltered.append(stepDurationData[2][a])
        stepDurationData[2]=stanceDurationDataFiltered

        (muscimolDurSwing, salineDurSwing) = groupAnalysis.splitTreatmentAvg(stepDurationData[0])
        (muscimolDurStance, salineDurStance) = groupAnalysis.splitTreatmentAvg(stepDurationData[2])
        # #pdb.set_trace()
        ax1 = plt.subplot(gs[1])
        ax1.text(0.5, 1.05, "Mean swing/stance duration", ha='center', va='center', transform=ax1.transAxes,
                 fontfamily='sans-serif', fontsize=13, color='k')
        #stats
        # (anovaSwing, posthocSwing) = groupAnalysis.performMixedTwoWayANOVA(stepDurationData[0], treatments=True,sessionValues=False)
        #
        # (anovaStance, posthocStance) = groupAnalysis.performMixedTwoWayANOVA(stepDurationData[2], treatments=True,sessionValues=False)
        # print("swing duration stats",anovaSwing)
        # print("stance duration stats",anovaStance)
        (mdf_SwingDur, confInt_SwingDur)=groupAnalysis.performMixedLinearModelRegression(stepDurationData[0],sessionValues=False, treatments=True )
        #pdb.set_trace()
        (mdf_StanceDur, confInt_StanceDur) = groupAnalysis.performMixedLinearModelRegression(stepDurationData[2], sessionValues=False,treatments=True )


        print("swing duration stats",mdf_SwingDur.summary())
        print("stance duration stats",mdf_StanceDur.summary())
        treatmentStarsSwingDur=groupAnalysis.starMultiplier(mdf_SwingDur.pvalues['treatments[T.saline]'])
        treatmentStarsStanceDur=groupAnalysis.starMultiplier(mdf_StanceDur.pvalues['treatments[T.saline]'])

        # treatmentStarsSwingDur=groupAnalysis.starMultiplier(anovaSwing.iloc[0, 6])
        # treatmentStarsStanceDur=groupAnalysis.starMultiplier(anovaStance.iloc[0, 6])
        ax1.text(0.9, 0.15, treatmentStarsSwingDur, ha='left', va='center', transform=ax1.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')
        ax1.text(0.9, 0.95, treatmentStarsStanceDur, ha='left', va='center', transform=ax1.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')
        if individualValues==True: 
            
            for f in range(len(muscimolDurSwing[0])):
                ax1.plot(np.arange(10)[:10] + 1, muscimolDurSwing[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapF(f / len(strideNumberData[1])), alpha=0.2)
                ax1.plot(np.arange(10)[:10] + 1, muscimolDurStance[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapF(f / len(strideNumberData[1])), alpha=0.2)
            for f in range(len(salineDurSwing[0])):
                ax1.plot(np.arange(10) + 1, salineDurSwing[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapM(f / len(strideNumberData[1])), alpha=0.2)
                ax1.plot(np.arange(10) + 1, salineDurStance[0][f][:10], 'o-', ms=2, label=None,
                         color=cmapM(f / len(strideNumberData[1])), alpha=0.2)


        ax1.plot(np.arange(10) + 1, muscimolDurStance[1][:10], '-', label="muscimol stance ",color='salmon', linewidth=2)
        ax1.plot(np.arange(10) + 1, salineDurStance[1][:10], '-', label="saline stance ", color='0.5',linewidth=2)
        ax1.plot(np.arange(10) + 1, muscimolDurSwing[1][:10], '--', label="muscimol swing", color='salmon', linewidth=2)
        ax1.plot(np.arange(10) + 1, salineDurSwing[1][:10], '--', label="saline swing", color='k', linewidth=2)

        plt.fill_between(np.arange(10) + 1, muscimolDurSwing[1][:10] - muscimolDurSwing[2][:10],
                         muscimolDurSwing[1][:10] + muscimolDurSwing[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, salineDurSwing[1][:10] - salineDurSwing[2][:10],
                         salineDurSwing[1][:10] + salineDurSwing[2][:10], color='0.7', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, muscimolDurStance[1][:10] - muscimolDurStance[2][:10],
                         muscimolDurStance[1][:10] + muscimolDurStance[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, salineDurStance[1][:10] - salineDurStance[2][:10],
                         salineDurStance[1][:10] + salineDurStance[2][:10], color='0.7', alpha=0.2)


        self.layoutOfPanel(ax1, xLabel='Days', yLabel='Mean swing/stance duration (s)', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax1.xaxis.set_major_locator(majorLocator_x)
        ax1.legend(loc='center left', bbox_to_anchor=(0.5,0.4),frameon=False)


        # ---------------------third subplot-----Swing Speed-------------
        stepSpeedDataFiltered=[]
        for a in range(len(pawSpeedData[1])):
            if not ("220204_m50" or "201207_f42") in pawSpeedData[1][a][0]  :
                stepSpeedDataFiltered.append(pawSpeedData[1][a])
        pawSpeedData[1]=stepSpeedDataFiltered
        (muscimolSwingSpeed, salineSwingSpeed) = groupAnalysis.splitTreatmentAvg(pawSpeedData[1])



        ax2 = plt.subplot(gs[2])

        # (anovaSpeed, posthocSpeed) = groupAnalysis.performMixedTwoWayANOVA(pawSpeedData[0], treatments=True,sessionValues=True)
        # print("swing speed stats",anovaSpeed)
        # treatmentStarsSpeed=groupAnalysis.starMultiplier(anovaSpeed.iloc[0, 6])
        (mdf_Speed, confInt_Speed)=groupAnalysis.performMixedLinearModelRegression(pawSpeedData[1],sessionValues=False, treatments=True)
        print("swing speed stats",mdf_Speed.summary())
        treatmentStarsSpeed=groupAnalysis.starMultiplier(mdf_Speed.pvalues['treatments[T.saline]'])

        #pdb.set_trace()
        ax2.text(0.9,0.95, treatmentStarsSpeed, ha='left', va='center', transform=ax2.transAxes, style='italic',
                 fontfamily='serif', fontsize=15, color='k')

        ax2.text(0.5, 1.05, "Average swing speed ", ha='center', va='center', transform=ax2.transAxes,
                 fontfamily='sans-serif', fontsize=13, color='k')
        if individualValues==True:
            for f in range(len(muscimolSwingSpeed[0])):
                ax2.plot(np.arange(10) + 1, muscimolSwingSpeed[0][f][:10], 'o-', label=None,
                         color=cmapF(f / len(pawSpeedData[1])), ms=2, alpha=0.2)
            for f in range(len(salineSwingSpeed[0])):
                ax2.plot(np.arange(10) + 1, salineSwingSpeed[0][f][:10], 'o-', label=None,
                         color=cmapM(f / len(pawSpeedData[1])), ms=2, alpha=0.2)
        ax2.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
        plt.tight_layout()


        ax2.plot(np.arange(10) + 1, muscimolSwingSpeed[1][:10], '-', label=None, color='salmon', linewidth=2)
        ax2.plot(np.arange(10) + 1, salineSwingSpeed[1][:10], '-', label=None, color='k', linewidth=2)

        plt.fill_between(np.arange(10) + 1, muscimolSwingSpeed[1][:10] - muscimolSwingSpeed[2][:10],
                         muscimolSwingSpeed[1][:10] + muscimolSwingSpeed[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, salineSwingSpeed[1][:10] - salineSwingSpeed[2][:10],
                         salineSwingSpeed[1][:10] + salineSwingSpeed[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='Average swing speed (cm/s)', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax2.xaxis.set_major_locator(majorLocator_x)
        ax2 = plt.subplot(gs[2])
        # plt.show()
        # #pdb.set_trace()

        # ---------------------fourth subplot------------------rung crossed-----------------
        rungCrossedDataDataFiltered=[]
        for a in range(len(rungCrossedData[0])):
            if not ("220204_m50" or "201207_f42") in rungCrossedData[0][a][0]  :
                rungCrossedDataDataFiltered.append(rungCrossedData[0][a])
        rungCrossedData[0]=rungCrossedDataDataFiltered


        (muscimolRungCrossed, salineRungCrossed) = groupAnalysis.splitTreatmentAvg(rungCrossedData[0])
        ax3 = plt.subplot(gs[3])
        # (anovaRungCrossed, posthocSpeed) = groupAnalysis.performMixedTwoWayANOVA(rungCrossedData[0], treatments=True,sessionValues=False)
        # print("swing rung crossed stats",anovaRungCrossed)
        (mdf_RungCrossed, confInt_RungCrossed)=groupAnalysis.performMixedLinearModelRegression(rungCrossedData[0],sessionValues=False, treatments=True)
        print("rung crossed stats",mdf_RungCrossed.summary())
        treatmentRungCrossedStars=groupAnalysis.starMultiplier(mdf_RungCrossed.pvalues['treatments[T.saline]'])


        # treatmentRungCrossed=groupAnalysis.starMultiplier(anovaRungCrossed.iloc[0, 6])
        ax3.text(0.9,0.95, treatmentRungCrossedStars, ha='left', va='center', transform=ax3.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')
        ax3.text(0.5, 1.05, 'Fraction of ' + r'$\geqq$' + '2 rungs crossed', ha='center', va='center',
                 transform=ax3.transAxes, fontfamily='sans-serif', fontsize=13, color='k')
        if individualValues==True:
            for f in range(len(muscimolRungCrossed[0])):
                ax3.plot(np.arange(10) + 1, muscimolRungCrossed[0][f][:10], 'o-', label=None,
                         color=cmapF(f / len(rungCrossedData[0])), alpha=0.2, ms=2)
            for f in range(len(salineRungCrossed[0])):
                ax3.plot(np.arange(10) + 1, salineRungCrossed[0][f][:10], 'o-', label=None,
                         color=cmapM(f / len(rungCrossedData[0])), alpha=0.2, ms=2)

        ax3.legend(loc="upper left", bbox_to_anchor=(10, 0.1))


        ax3.plot(np.arange(10) + 1, muscimolRungCrossed[1][:10], '-', label=None, color='salmon', linewidth=2)
        ax3.plot(np.arange(10) + 1, salineRungCrossed[1][:10], '-', label=None, color='k', linewidth=2)

        plt.fill_between(np.arange(10) + 1, muscimolRungCrossed[1][:10] - muscimolRungCrossed[2][:10],
                         muscimolRungCrossed[1][:10] + muscimolRungCrossed[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, salineRungCrossed[1][:10] - salineRungCrossed[2][:10],
                         salineRungCrossed[1][:10] + salineRungCrossed[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax3, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' + '2 rungs crossed', Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax3.xaxis.set_major_locator(majorLocator_x)

        # ---------------------fourth subplot------------------------------------------------
        stepsQualityDataDataFiltered=[]
        for a in range(len(stepsQualityData[1])):
            if not ("220204_m50" or "201207_f42") in stepsQualityData[1][a][0]  :
                stepsQualityDataDataFiltered.append(stepsQualityData[1][a])
        stepsQualityData[1]=stepsQualityDataDataFiltered

        (muscimolQuality, salineQuality) = groupAnalysis.splitTreatmentAvg(stepsQualityData[1])
        ax4 = plt.subplot(gs[4])

        # (anovaStepsQuality, posthocStepsQuality) = groupAnalysis.performMixedTwoWayANOVA(stepsQualityData[0], treatments=True,sessionValues=True)
        # print("swing quality stats",anovaStepsQuality)
        (mdf_stepQual, confInt_Step)=groupAnalysis.performMixedLinearModelRegression(stepsQualityData[1],sessionValues=False, treatments=True)
        print("swing quality stats",mdf_stepQual.summary())
        #pdb.set_trace()
        treatmentstepQualStars=groupAnalysis.starMultiplier(mdf_stepQual.pvalues['treatments[T.saline]'])
        ax4.text(0.9,0.95, treatmentstepQualStars, ha='left', va='center', transform=ax4.transAxes, style='italic',fontfamily='serif', fontsize=15, color='k')
        print("recording pvalue", round(mdf_stepNb.pvalues['treatments[T.saline]']))
        ax4.text(0.5, 1.05, "Fraction of indecisive strides", ha='center', va='center', transform=ax4.transAxes,
                 fontfamily='sans-serif', fontsize=13, color='k')
        if individualValues==True:
            for f in range(len(muscimolQuality[0])):
                ax4.plot(np.arange(10) + 1, muscimolQuality[0][f][:10], 'o-', label=None,
                         color=cmapF(f / len(stepsQualityData[1][f][1])), ms=2, alpha=0.2)
            for f in range(len(salineQuality[0])):
                ax4.plot(np.arange(10) + 1, salineQuality[0][f][:10], 'o-', label=None,
                         color=cmapM(f / len(stepsQualityData[1][f][1])), ms=2, alpha=0.2)

        ax4.legend(loc="upper left", bbox_to_anchor=(10, 0.1))

        ax4.plot(np.arange(10) + 1, muscimolQuality[1][:10], '-', label=None, color='salmon', linewidth=2)
        ax4.plot(np.arange(10) + 1, salineQuality[1][:10], '-', label=None, color='k', linewidth=2)

        plt.fill_between(np.arange(10) + 1, muscimolQuality[1][:10] - muscimolQuality[2][:10],
                         muscimolQuality[1][:10] + muscimolQuality[2][:10], color='salmon', alpha=0.2)
        plt.fill_between(np.arange(10) + 1, salineQuality[1][:10] - salineQuality[2][:10],
                         salineQuality[1][:10] + salineQuality[2][:10], color='0.6', alpha=0.2)
        self.layoutOfPanel(ax4, xLabel='Days', yLabel='Fraction of indecisive strides', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax4.xaxis.set_major_locator(majorLocator_x)

        fname = 'Behavior_summary_figure_Muscimol'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/Muscimol_experiment/' + fname + '.pdf')

    def createBehaviorSummaryFigureFENS(self,  swingNumber,strideDuration,swingSpeed,rungCrossed,indecisiveStrideFraction,strideLenght,pawCoordination):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        # create figure instance


        # define sub-panel grid and possibly width and height ratios

        if sessionValues==True:
            gs = gridspec.GridSpec(2, 5)
            fig_width = 25  # width in inches
            fig_height = 10  # height in inches
        else:
            fig_width = 23  # width in inches
            fig_height = 15  # height in inches
            gs = gridspec.GridSpec(2, 3)

        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 14,
                  'ytick.labelsize': 14,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        rcParams['font.sans-serif'] = 'Arial'

        fig = plt.figure()
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)


                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])

        for f in range (len(strideNumberData[1])):
            nDays = len(strideNumberData[1][f][1])


            ax0.plot(np.arange(nDays)[:11]+1,strideNumberData[0][f][1][:11], 'o-',ms=2, label=strideNumberData[1][f][0], color=cmap(f/len(strideNumberData[1])),alpha=0.2)

        ax0.legend(loc="upper left", bbox_to_anchor=(10,0.1))
        stepNb11days=strideNumberData[2][0][0:11]
        stepNbSEM11days=strideNumberData[2][1][0:11]
        #ax0.text(0, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        stepStars=groupAnalysis.starMultiplier(strideNumberData[2][2][1])

        ax0.text(0.5, 1, "Average stride number (all paws)", ha='center', va='center', transform=ax0.transAxes,fontfamily='sans-serif', fontsize=16, color='k')
        ax0.text(0.9, 0.95, '%s'%stepStars, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=16, color='k')
        ax0.text(0.87, 0.9, '(N=%s)' % len(strideNumberData[0]), ha='left', va='center', transform=ax0.transAxes, style='italic', fontsize=16, color='k')
        # ax0.text(0,0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' % (strideNumberData[2][2][0], strideNumberData[2][2][1],strideNumberData[2][2][2],strideNumberData[2][2][3]),ha='left', va='center', transform=ax0.transAxes,style='italic', fontsize=7,color='0.7')
        ax0.plot(np.arange(11)+1,stepNb11days,'-',label=None, linewidth=2,c='k')
        #ax0.errorbar(np.arange(11),stepNb11days, yerr=stepNbSEM11days, color=cmap(0.01) )
        plt.fill_between(np.arange(len(stepNb11days))+1, stepNb11days - stepNbSEM11days,stepNb11days + stepNbSEM11days, color='0.6', alpha=0.2)
        #plt.fill_between(np.arange(len(avg)),avg-error,avg+error, color='black', alpha=0.1)
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='Stride number (avg.)')#, Leg=[1, 9])

        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
#--------------------------------------second subplot-------------------------------------------
        ax1 = plt.subplot(gs[3])
        for f in range(len(stepDurationData[0])):
            nDays = len(stepDurationData[0][f][1])
            animalSteps = stepDurationData[0][f]
            # sns.relplot(x="recording days", y="average steps", kind="line", data=animalSteps);
            ax1.plot(np.arange(nDays)[:11]+1, stepDurationData[0][f][1][:11], 'o-', ms=2,label=None, color=cmap(f / len(stepDurationData[0])),alpha=0.2)
            ax1.plot(np.arange(nDays)[:11]+1, stepDurationData[2][f][1][:11], 'o-', ms=2,label=None,color=cmap(f / len(stepDurationData[0])), alpha=0.2)
        swingDuration11days = stepDurationData[3][0][0:11]
        swingDurationdaysError = stepDurationData[3][1][0:11]
        stanceDuration11days=stepDurationData[4][0][0:11]
        stanceDuration11daysError=stepDurationData[4][1][0:11]

        ax1.legend(loc="upper left", bbox_to_anchor=(10, 0.1))
        stepSwingDurStars = groupAnalysis.starMultiplier(stepDurationData[3][2][1])
        stepStanceDurStars=groupAnalysis.starMultiplier(stepDurationData[4][2][1])

        #ax0.text(0.9, 0.95, '*'*multiplier, ha='left', va='center',transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=20, color='k')
        # ax1.text(0.80, 1, '(N=%s)' % len(stepDurationData[0]), ha='left', va='center', transform=ax1.transAxes, style='italic', fontsize=16, color='k')
        ax1.text(0.5, 1, "Mean swing/stance duration (all paws)", ha='center', va='center', transform=ax1.transAxes,fontfamily='sans-serif', fontsize=16, color='k')
        # ax1.text(0.0, 0, 'Swing_MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[3][2][0], stepDurationData[3][2][1], stepDurationData[3][2][2],stepDurationData[3][2][3]), ha='left', va='center', transform=ax1.transAxes,style='italic', fontsize=7,color='0.7')
        # ax1.text(0, 0.030, 'Stance_MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (stepDurationData[4][2][0], stepDurationData[4][2][1], stepDurationData[4][2][2],stepDurationData[4][2][3]), ha='left', va='center', transform=ax1.transAxes, style='italic', fontsize=7,color='0.8')
        # #ax0.text(0.40, 0.04, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResStance.anova_table.iloc[0, 2], 4), round(AnovaRMResStance.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7,color=cmap(0.1))
        # ax1.text(0.5, 1, "Mean swing/stance duration", ha='center', va='center', transform=ax1.transAxes,fontfamily='sans-serif', fontsize=16, color='k')

        ax1.plot(np.arange(len(swingDuration11days))+1, swingDuration11days, '--', label="swing  %s"%(stepSwingDurStars), color='k',linewidth=2)

        ax1.plot(np.arange(len(stanceDuration11days))+1, stanceDuration11days, '-', label="stance  %s"%(stepStanceDurStars), color='0.5', linewidth=2)
        plt.fill_between(np.arange(len(swingDuration11days))+1, swingDuration11days - swingDurationdaysError, swingDuration11days + swingDurationdaysError,color='0.6', alpha=0.2)
        plt.fill_between(np.arange(len(stanceDuration11days))+1, stanceDuration11days - stanceDuration11daysError,stanceDuration11days + stanceDuration11daysError, color='0.7', alpha=0.2)
        self.layoutOfPanel(ax1, xLabel='Days', yLabel='Mean swing/stance duration (s)', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax1.xaxis.set_major_locator(majorLocator_x)
        ax1.legend(loc="upper right", frameon=False, fontsize=16)


# ---------------------fourth subplot------------------------------------------------
        ax3 = plt.subplot(gs[1])
        nRec = 0

        for f in range (len(rungCrossedData[0])):
            nDays = len(rungCrossedData[0][f][1])

            ax3.plot(np.arange(nDays)[:11]+1,rungCrossedData[0][f][1][:11], 'o-', label=None, color=cmap(f/len(rungCrossedData[0])),alpha=0.2,ms=2)

        ax3.legend(loc="upper left", bbox_to_anchor=(10,0.1))
        ax3.text(0.5, 1, 'Fraction of ' + r'$\geqq$' +'2 rungs crossed (all paws)', ha='center', va='center', transform=ax3.transAxes,fontfamily='sans-serif', fontsize=16, color='k')
        #ax0.text(0.4, 0.025, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMResAll.anova_table.iloc[0, 2], 4), round(AnovaRMResAll.anova_table.iloc[0, 3], 4)), ha='left',va='center', transform=ax0.transAxes, fontsize=7)
        # ax3.text(0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95conf-int=(%s ,%s)' % (rungCrossedData[1][2][0], rungCrossedData[1][2][1],rungCrossedData[1][2][2], rungCrossedData[1][2][2]), ha='left', va='center', transform=ax3.transAxes,style='italic', fontsize=7,color='0.7')

        rungCrossedStars = groupAnalysis.starMultiplier(rungCrossedData[1][2][1])
        ax3.text(0.9, 0.95, '%s'%rungCrossedStars, ha='left', va='center',transform=ax3.transAxes, style='italic',fontfamily='serif', fontsize=16, color='k')
        # ax3.text(0.87, 0.9, '(N=%s)' % len(rungCrossedData[0]), ha='left', va='center', transform=ax3.transAxes, style='italic', fontsize=16, color='k')
        rungCross11days = rungCrossedData[1][0][0:11]
        rungCross11daysError = rungCrossedData[1][1][0:11]

        #ax0.errorbar(np.arange(11), rungCross11daysError, yerr=rungCross11daysError, color=cmap(0.01))

        ax3.plot(np.arange(len(rungCross11days))+1,rungCross11days,'-',label=None, color='k',linewidth=2)
        plt.fill_between(np.arange(len(rungCross11days))+1,rungCross11days-rungCross11daysError,rungCross11days+rungCross11daysError, color='0.6', alpha=0.2)
        self.layoutOfPanel(ax3, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed', Leg=[1, 9])



        majorLocator_x = MultipleLocator(1)
        ax3.xaxis.set_major_locator(majorLocator_x)
        
        (pawRungCrossed, meanPawRungCrossed, semPawRungCrossed) = groupAnalysis.getAverageSingleGroup(
            rungCrossedData[2])
        (swingDf, pawPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            rungCrossedData[2], sessionValues=True, treatments=False)


        (FLStar) = groupAnalysis.starMultiplier(pawPValues[0])
        (FRStar) = groupAnalysis.starMultiplier(pawPValues[1])
        (HLStar) = groupAnalysis.starMultiplier(pawPValues[2])
        (HRStar) = groupAnalysis.starMultiplier(pawPValues[3])
        # pdb.set_trace()
        ax7 = plt.subplot(gs[2])
        ax7.plot(np.arange(11) + 1, meanPawRungCrossed[:, 0][:11], '-', label='FL %s' % FLStar, linewidth=2,
                 c='steelblue')
        ax7.plot(np.arange(11) + 1, meanPawRungCrossed[:, 1][:11], '-', label='FR %s' % FRStar, linewidth=2,
                 c='orange')
        plt.fill_between(np.arange(11) + 1, meanPawRungCrossed[:, 0][:11] - semPawRungCrossed[:, 0][:11],
                         meanPawRungCrossed[:, 0][:11] + semPawRungCrossed[:, 0][:11], color='steelblue', alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawRungCrossed[:, 1][:11] - semPawRungCrossed[:, 1][:11],
                         meanPawRungCrossed[:, 1][:11] + semPawRungCrossed[:, 1][:11], color='orange', alpha=0.1)
        self.layoutOfPanel(ax7, xLabel='Days', yLabel=None, Leg=[1, 2])


        ax7.plot(np.arange(11) + 1, meanPawRungCrossed[:, 2][:11], '-', label='HL %s' % HLStar, linewidth=2,
                 c='yellowgreen')
        ax7.plot(np.arange(11) + 1, meanPawRungCrossed[:, 3][:11], '-', label='HR %s' % HRStar, linewidth=2, c='red')
        plt.fill_between(np.arange(11) + 1, meanPawRungCrossed[:, 2][:11] - semPawRungCrossed[:, 2][:11],
                         meanPawRungCrossed[:, 2][:11] + semPawRungCrossed[:, 2][:11], color='yellowgreen', alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawRungCrossed[:, 3][:11] - semPawRungCrossed[:, 3][:11],
                         meanPawRungCrossed[:, 3][:11] + semPawRungCrossed[:, 3][:11], color='red', alpha=0.1)
        self.layoutOfPanel(ax7, xLabel='Days', yLabel='Fraction of ' + r'$\geqq$' +'2 rungs crossed', Leg=[1, 2])
        ax7.text(0.5, 1, 'Fraction of ' + r'$\geqq$' + '2 rungs crossed (single paw)', ha='center', va='center',
                 transform=ax7.transAxes, fontfamily='sans-serif', fontsize=16, color='k')
        ax7.legend(loc="upper right", frameon=False, fontsize=16)
        ax7.legend(loc="upper right", frameon=False, fontsize=16)
        ax7.xaxis.set_major_locator(majorLocator_x)
        ax7.xaxis.set_major_locator(majorLocator_x)
# ---------------------fourth subplot------------------------------------------------
        ax4 = plt.subplot(gs[4])
        for f in range(len(stepsQualityData[1])):
            nDays = len(stepsQualityData[1][f][1])

            ax4.plot(np.arange(nDays)[:11] + 1, stepsQualityData[1][f][1][:11], 'o-', label=None,
                     color=cmap(f / len(stepsQualityData[1][f][1])), ms=2, alpha=0.2)

        ax4.legend(loc="upper left", bbox_to_anchor=(10, 0.1))
        inDstep11days = stepsQualityData[2][0][0:11]
        inDstepSEM11days = stepsQualityData[2][1][0:11]
        indeciveStepStars = groupAnalysis.starMultiplier(stepsQualityData[2][2][1])
        ax4.text(0.9, 0.95, '%s'%indeciveStepStars, ha='left', va='center', transform=ax4.transAxes, style='italic',
                 fontfamily='serif', fontsize=16, color='k')
        # ax4.text(0.87, 0.9, '(N=%s)' % len(stepsQualityData[1]), ha='left', va='center', transform=ax4.transAxes,
        #          style='italic', fontsize=16, color='k')
        # ax0.errorbar(np.arange(11), inDstep11days, yerr=inDstepSEM11days, color=cmap(0.01))
        plt.fill_between(np.arange(len(inDstep11days)) + 1, inDstep11days - inDstepSEM11days,
                         inDstep11days + inDstepSEM11days, color='0.6', alpha=0.2)
        ax4.text(0.5, 1, "Fraction of indecisive strides (all paws)", ha='center', va='center', transform=ax4.transAxes,fontfamily='sans-serif', fontsize=16, color='k')
        (pawIndeciveStrides, meanPawIndeciveStrides, semPawIndeciveStrides) = groupAnalysis.getAverageSingleGroup(
            stepsQualityData[3])
        (pawSpeedDataDf, pawSpeedDataPValues) = groupAnalysis.PawListToPandasDFAndMxLM(
            stepsQualityData[3], sessionValues=True, treatments=False)
        # pdb.set_trace()
        (FLStar) = groupAnalysis.starMultiplier(pawSpeedDataPValues[0])
        (FRStar) = groupAnalysis.starMultiplier(pawSpeedDataPValues[1])
        (HLStar) = groupAnalysis.starMultiplier(pawSpeedDataPValues[2])
        (HRStar) = groupAnalysis.starMultiplier(pawSpeedDataPValues[3])
        # pdb.set_trace()
        ax8 = plt.subplot(gs[5])
        ax8.plot(np.arange(11) + 1, meanPawIndeciveStrides[:, 0][:11], '-', label='FL %s' % FLStar, linewidth=2,
                 c='steelblue')
        ax8.plot(np.arange(11) + 1, meanPawIndeciveStrides[:, 1][:11], '-', label='FR %s' % FRStar, linewidth=2,
                 c='orange')
        plt.fill_between(np.arange(11) + 1, meanPawIndeciveStrides[:, 0][:11] - semPawIndeciveStrides[:, 0][:11],
                         meanPawIndeciveStrides[:, 0][:11] + semPawIndeciveStrides[:, 0][:11], color='steelblue',
                         alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawIndeciveStrides[:, 1][:11] - semPawIndeciveStrides[:, 1][:11],
                         meanPawIndeciveStrides[:, 1][:11] + semPawIndeciveStrides[:, 1][:11], color='orange',
                         alpha=0.1)


        ax8 = plt.subplot(gs[5])
        ax8.plot(np.arange(11) + 1, meanPawIndeciveStrides[:, 2][:11], '-', label='HL %s' % HLStar, linewidth=2,
                 c='yellowgreen')
        ax8.plot(np.arange(11) + 1, meanPawIndeciveStrides[:, 3][:11], '-', label='HR %s' % HRStar, linewidth=2,
                 c='red')
        plt.fill_between(np.arange(11) + 1, meanPawIndeciveStrides[:, 2][:11] - semPawIndeciveStrides[:, 2][:11],
                         meanPawIndeciveStrides[:, 2][:11] + semPawIndeciveStrides[:, 2][:11], color='yellowgreen',
                         alpha=0.1)
        plt.fill_between(np.arange(11) + 1, meanPawIndeciveStrides[:, 3][:11] - semPawIndeciveStrides[:, 3][:11],
                         meanPawIndeciveStrides[:, 3][:11] + semPawIndeciveStrides[:, 3][:11], color='red',
                         alpha=0.1)
        self.layoutOfPanel(ax8, xLabel='Days', yLabel='indecisive steps fraction', Leg=[1, 2])
        ax8.text(0.5, 1, "Fraction of indecisive strides (single paws)", ha='center', va='center', transform=ax8.transAxes,
                 fontfamily='sans-serif', fontsize=16, color='k')
        ax8.legend(loc="upper right", frameon=False, fontsize=16)

        ax8.xaxis.set_major_locator(majorLocator_x)

        # ax0.text(0.40, 0.02, 'N=12 animals, AnovaRM: F value=%s, p value=%s' % (round(AnovaRMRes.anova_table.iloc[0, 2],4), round(AnovaRMRes.anova_table.iloc[0, 3],4)), ha='left', va='center',transform=ax0.transAxes, fontsize=7)
        # ax4.text(0.0, 0, 'MixedLM: Coef. recordings=%s, p value=%s, 95-conf-int=(%s ,%s)' % (
        # stepsQualityData[2][2][0], stepsQualityData[2][2][1], stepsQualityData[2][2][2], stepsQualityData[2][2][3]),
        #          ha='left', va='center', transform=ax4.transAxes, style='italic', fontsize=7, color='0.7')
        ax4.plot(np.arange(len(inDstep11days)) + 1, inDstep11days, '-', label=None, color='k', linewidth=2)
        plt.fill_between(np.arange(len(inDstep11days)) + 1, inDstep11days - inDstepSEM11days,
                         inDstep11days + inDstepSEM11days, color='0.6', alpha=0.2)
        self.layoutOfPanel(ax4, xLabel='Days', yLabel='Fraction of indecisive strides', Leg=[1, 9])
        majorLocator_x = MultipleLocator(1)
        ax4.xaxis.set_major_locator(majorLocator_x)
        if sessionValues:
            trialAllAvgStepArray=[]

            ax5 = plt.subplot(gs[1,0])
            for t in range (len(strideNumberData[1])):
                trialAvgStep=np.mean(strideNumberData[1][t][1],axis=0)
                trialAllAvgStepArray.append(trialAvgStep)

                # ax5.plot(np.arange(5)+1, trialAvgStep,color=cmap(t / len(strideNumberData[1])), ms=2, alpha=0.2)
            trialAllAvgStep =np.mean(trialAllAvgStepArray,axis=0)
            trialAllAvgSem=stats.sem(trialAllAvgStepArray, axis=0)
            # pdb.set_trace()
            ax5.plot(np.arange(5) + 1, trialAllAvgStep, color='k')
            ax5.fill_between(np.arange(5)+1,trialAllAvgStep-trialAllAvgSem, trialAllAvgStep+trialAllAvgSem, color='0.6',alpha=0.2)
            ax5.xaxis.set_major_locator(majorLocator_x)
            self.layoutOfPanel(ax5, xLabel='Trial', yLabel='Average Trial Step Number', Leg=[1, 9])
            trialAllAvgSwingDurArray=[]
            trialAllAvgStanceDurArray = []
            ax6= plt.subplot(gs[1,1])
            for t in range (len(stepDurationData[1])):
                trialAvgSwingDur=np.mean(stepDurationData[1][t][1],axis=0)
                trialAllAvgSwingDurArray.append(trialAvgSwingDur)
                # pdb.set_trace()
                trialAvgStanceDur=np.mean(stepDurationData[5][t][1],axis=0)
                trialAllAvgStanceDurArray.append(trialAvgStanceDur)

                # ax6.plot(np.arange(5)+1, trialAvgStanceDur,color=cmap(t / len(stepDurationData[1])), ms=2, alpha=0.2)
                # ax6.plot(np.arange(5)+1, trialAvgSwingDur,color=cmap(t / len(stepDurationData[1])), ms=2, alpha=0.2)
            trialAllAvgSwingDur =np.mean(trialAllAvgSwingDurArray,axis=0)
            trialAllAvgSwingDurSem=stats.sem(trialAllAvgSwingDurArray, axis=0)
            trialAllAvgStanceDur =np.mean(trialAllAvgStanceDurArray,axis=0)
            trialAllAvgStanceDurSem=stats.sem(trialAllAvgStanceDurArray, axis=0)
            ax6.plot(np.arange(5) + 1, trialAllAvgSwingDur, '--', color='k')
            ax6.fill_between(np.arange(5)+1,trialAllAvgSwingDur-trialAllAvgSwingDurSem, trialAllAvgSwingDur+trialAllAvgSwingDurSem, color='0.6',alpha=0.2)
            ax6.plot(np.arange(5) + 1, trialAllAvgStanceDur, '-', color='0.5')
            ax6.fill_between(np.arange(5)+1,trialAllAvgStanceDur-trialAllAvgStanceDurSem, trialAllAvgStanceDur+trialAllAvgStanceDurSem, color='0.6',alpha=0.2)
            ax6.xaxis.set_major_locator(majorLocator_x)
            self.layoutOfPanel(ax6, xLabel='Trial', yLabel='Mean trial swing/stance duration ', Leg=[1, 9])
        plt.figtext(0.15, 0.08, "Mixed Linear Model Regression, Time effect, *p<0.05, **p<0.01, ***p<0.001 ",
                    clip_on=False, color='black', size=16)
        fname = 'Behavior_summary_figure_FENS'
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')


    def createStrideAnalysisFigure(self, strideStats, experiment):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')
        col = ['C0', 'C1', 'C2', 'C3']
        pawList=['FL','FR','HL','HR']
        fig_width = 20  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               #width_ratios=[2,1]
                               height_ratios=[2,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)
        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], hspace=0.4, wspace=0.3)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1], hspace=0.4, wspace=0.6)
        var = ['swingLength', 'swingDuration']
        # var = ['swingLength', 'stanceDuration']
        for i in range(2):
            # pdb.set_trace()
            paw_strideStats = strideStats[(strideStats['paw'] == pawList[i])]
            avg_paw_strideStats=paw_strideStats.groupby(['trial', 'mouseId','day']).mean().reset_index()
            no_mask = paw_strideStats[var[0]] > paw_strideStats[var[0]].quantile(0)
            mask_inf = (avg_paw_strideStats[var[0]] < avg_paw_strideStats[var[0]].quantile(0.20)) & (avg_paw_strideStats[var[0]] > avg_paw_strideStats[var[0]].min())
            mask_sup = (avg_paw_strideStats[var[0]] > avg_paw_strideStats[var[0]].quantile(0.80)) & (avg_paw_strideStats[var[0]] < avg_paw_strideStats[var[0]].max())

            maskList = [no_mask,mask_inf, mask_sup]
            maskId=['all', 'low', 'high']
            # pdb.set_trace()
            for m in range(len(maskList)):
                ax0 = plt.subplot(gssub0[i,m])

                if m==0:
                    x0 = avg_paw_strideStats[var[0]]
                    y0 = avg_paw_strideStats[var[1]]
                else:
                    x0 = avg_paw_strideStats[var[0]][maskList[m]]
                    y0 = avg_paw_strideStats[var[1]][maskList[m]]
                slope, intercept, r_value, p_value, sterr = stats.linregress(x0, y0,
                                                                             alternative='two-sided')
                r2 = np.square(r_value)
                # sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                #             line_kws={'alpha': alpha})
                sns.regplot(x=x0, y=y0, ax=ax0, color=col[i])
                ax0.text(0.7, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",
                         transform=ax0.transAxes)
                ax0.set_title(f'{maskId[m]} {var[0]} vs {var[1]}')
                self.layoutOfPanel(ax0, xLabel=f'{var[0]}',
                                   yLabel=var[1])

            quantileRange = np.arange(0, 1.1, 0.2)


            #var = ['stanceDuration', 'strideLength']

            ax1=plt.subplot(gssub0[i,len(maskList)])
            rValueArray = []
            for q in range(len(quantileRange) - 1):
                maskQuant = (avg_paw_strideStats[var[0]] > avg_paw_strideStats[var[0]].quantile(
                    quantileRange[q])) & (avg_paw_strideStats[var[0]] < avg_paw_strideStats[var[0]].quantile(
                    quantileRange[q + 1]))

                # maskQ=((paw_strideStats[var[0]] > paw_strideStats[var[0]].quantile(
                #     quantileRange[q])) & (paw_strideStats[var[0]] < paw_strideStats[var[0]].quantile(
                #     quantileRange[q + 1])))&(paw_strideStats[var[1]] > paw_strideStats[var[1]].quantile(
                #     quantileRange[q])) & (paw_strideStats[var[1]] < paw_strideStats[var[1]].quantile(
                #     quantileRange[q + 1]))
                y1 = avg_paw_strideStats[var[1]][maskQuant]#[maskQuant]
                x1 = avg_paw_strideStats[var[0]][maskQuant]#[maskQuant]

                try:
                    slope1, intercept1, r_value1, p_value1, sterr1 = stats.linregress(x1, y1,
                                                                                      alternative='two-sided')


                except ValueError:
                    pdb.set_trace()
                if p_value1 > 0.05:
                    alpha = 0.2
                else:
                    alpha =1
                ax1.scatter(quantileRange[:len(quantileRange) - 1][q] * 100, r_value1, color=col[i], alpha=alpha)
                rValueArray.append(r_value1)
            # pdb.set_trace()
            ax1.plot(quantileRange[:len(quantileRange)-1]*100, rValueArray, color=col[i])

            ax1.set_xticklabels(['[0-20]','[20-40]','[40-60]','[60-80]','[80-100]'])
            ax1.set_xlim(0,100)
            self.layoutOfPanel(ax1, xLabel=f'{var[0]} range',
                               yLabel='correlation coefficient')

            # gssub1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], hspace=0.4, wspace=0.3)
            # # ax3 = plt.subplot(gssub1[0,i])
            # # sns.lineplot(data=avg_paw_strideStats, x='day', y='stepLength', ax=ax3, color=col[i])
            # # self.layoutOfPanel(ax3, xLabel='days',
            # #                    yLabel='stepLength')


            fname = f' {experiment}_swingLength_speed_stepLength'




        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')


    def pawCoordinationFigure(self, pawCoordination, experiment):
        from matplotlib import cm
        iqr_mice=pawCoordination[4]
        swingCountNorm=pawCoordination[3]
        fractionFRAhead=pawCoordination[1]
        fractionFLAhead=pawCoordination[0]

        pawList=['FL','FR','HL','HR']
        cmap = cm.get_cmap('tab20')
        cmap2 = cm.get_cmap('Reds')
        fig_width = 12  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 2,  # ,
                               width_ratios=[1, 1],
                               height_ratios=[1,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        # ########################IQR############################
        (iqr_df, iqr_md_paw, iqr_pawPvalues, iqr_stars_all)=groupAnalysis.pandaDataFrameAndMixedMLCompleteData(iqr_mice, treatments=False)
        iqr_df.to_csv(self.figureDirectory + '/' + 'iqr' + '.csv')
        ax0 = plt.subplot(gs[0])
        ax1=plt.subplot(gs[1])
        self.layoutOfPanel(ax0, xLabel='Days', yLabel='swing probability IQR (70-90%)', Leg=[1, 9])
        self.layoutOfPanel(ax1, xLabel='Days', yLabel='swing probability IQR (70-90%)', Leg=[1, 9])
        iqr_mice_meanArray=[[],[]]
        iqr_mice_meanArray=[[],[],[],[]]
        for m in range(len(iqr_mice)):
            for i in range(4):
                #ax0.plot(np.arange(10)+1, np.nanmean(iqr_mice[m][1][:,:,i], axis=1), '-o', label=None, color=cmap(m/len(iqr_mice[1])),ms=2,alpha=0.2)
                iqr_mice_meanArray[i].append(np.nanmean(iqr_mice[m][1][:,:,i], axis=1))

        #plot average
        iqr_mice_mean=np.nanmean(iqr_mice_meanArray,axis=1)
        iqr_mice_sem=stats.sem(iqr_mice_meanArray,axis=1, nan_policy='omit')

        for i in range(4):
            (iqr_pawStar_recording) = groupAnalysis.starMultiplier(iqr_pawPvalues[i][0])
            (iqr_pawStar_trials) = groupAnalysis.starMultiplier(iqr_pawPvalues[i][1])
            if i==0 or i==1:
                ax0.plot(np.arange(10)+1, iqr_mice_mean[i],'-o', color='C%s'%i, ms=2, label=pawList[i] + '%s %s'%(iqr_pawStar_recording, iqr_pawStar_trials.replace('*','#')))
                ax0.errorbar(np.arange(10)+1,iqr_mice_mean[i],iqr_mice_sem[i], capsize=3, color='C%s'%i)
                ax0.legend(loc="upper left", frameon=False, fontsize=10)
            else:
                ax1.plot(np.arange(10)+1, iqr_mice_mean[i],'-o', color='C%s'%i, ms=2, label=pawList[i] + '%s %s'%(iqr_pawStar_recording, iqr_pawStar_trials.replace('*','#')))
                ax1.errorbar(np.arange(10)+1,iqr_mice_mean[i],iqr_mice_sem[i], capsize=3, color='C%s'%i)
                ax1.legend(loc="upper left", frameon=False, fontsize=10)


        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)
        ax1.xaxis.set_major_locator(majorLocator_x)

        ###############################Paw ahead##################################
        ax2=plt.subplot(gs[2])
        meanFLAhead=[]
        meanFRAhead=[]
        for q in range(len(fractionFRAhead)):
            meanFLAhead.append(np.nanmean(fractionFLAhead[q][1],axis=1))
            meanFRAhead.append(np.nanmean(fractionFRAhead[q][1],axis=1))
            ax2.plot(np.arange(10) + 1, meanFRAhead[q], '-o', ms=2, alpha=0.2, label=fractionFRAhead[q][0])
            # ax2.scatter(meanFLAhead[0],meanFRAhead[0])
        mice_FLAhead_mean=np.nanmean(meanFLAhead, axis=0)
        mice_FLAhead_sem = stats.sem(meanFLAhead, axis=0, nan_policy='omit')
        mice_FRAhead_mean=np.nanmean(meanFRAhead, axis=0)
        mice_FRAhead_sem = stats.sem(meanFRAhead, axis=0, nan_policy='omit')
        ax2.plot(np.arange(10)+1,mice_FRAhead_mean,'-o', ms=2)
        ax2.errorbar(np.arange(10) + 1, mice_FRAhead_mean, mice_FRAhead_sem, capsize=3)
        ax2.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax2, xLabel='Days', yLabel='fraction of time FR ahead of FL', Leg=[1, 9])
        ax2.legend(loc="lower right", frameon=False, fontsize=8)

        swingCountNorm_sum=[]
        for m in range(len(swingCountNorm)):
            swingCountNorm_sum.append(np.nansum(swingCountNorm[m][1], axis=1))
        mice_swingCountNorm_sum=np.sum(swingCountNorm_sum, axis=0)
        ax3=plt.subplot(gs[3])
        for i in range(4):
            ax3.plot(np.linspace(0,100,51),mice_swingCountNorm_sum[:,i,:][0]/np.max(mice_swingCountNorm_sum[0]), color='C%s'%i)
            ax3.plot(np.linspace(0,100,51),mice_swingCountNorm_sum[:,i,:][9]/np.max(mice_swingCountNorm_sum[9]), '--', color='C%s'%i)
        self.layoutOfPanel(ax3, xLabel='% FL paw cycle', yLabel='Swing probability', Leg=[1, 9])
        # plt.show()
        fname = '%s_Hildebrand_figure'%experiment
        # plt.savefig(fname + '.png')
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
 ##########################################################################################
    def createUMAPClusteringFigure(self,  allVariablesDf, experiment):
        from matplotlib import cm
        cmap=cm.get_cmap('tab20')
        # create figure instance
        variableList = ['swingNumber', 'swingSpeed',  'acceleration','accelerationNumber', 'rungCrossed', 'strideLenght', 'swingDuration',
                        'stanceDuration', 'indecisiveStride', 'stanceOnsetMedian']
        YAxisList = ['Swing number (avg)', 'Swing Speed (cm/s)',  'Mean swing acceleration (cm/s²)', 'Swing acceleration phase number',  'Fraction of ' + r'$\geqq$' + '2 rungs crossed', 'Stride lenght (cm)', 'Swing duration (s)',
                        'Stance Duration (s)', 'Fraction of indecisive stride', 'Stance onset median (norm FL)']
        fig_width = 8  # width in inches
        fig_height = 5  # height in inches
        gs = gridspec.GridSpec(1, 2


                               )

        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        rcParams['font.sans-serif'] = 'Arial'

        fig = plt.figure()
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)


                    # first sub-plot #######################################################

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        # ax3 = plt.subplot(gs[3])


        useful=allVariablesDf.dropna(how='any')
        useful_matrix=useful[['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed','strideLenght','strideLenghtStd', 'swingDuration', 'SwingDurationStd','stanceDuration', 'indecisiveStride', 'stanceOnsetMedian']]
        useful_matrix_mouse=useful.groupby('mouse')
        useful_matrix=useful.groupby(['recordingDay', 'mouse'])
        mean_df_recordingDays=useful_matrix.mean().reset_index()
        # pdb.set_trace()
        mean_df_all=useful_matrix_mouse.mean().reset_index()
        mean_df_recordingDays.to_csv('recordingDayMeanMatrix.csv')
        mean_df=mean_df_all[['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian','stanceOnsetStd', 'wheelSpeed']]
        #standardize data
        scaler = StandardScaler()
        mean = mean_df.mean()
        std = mean_df.std()
        mean_df = scaler.fit_transform(mean_df)
        # Transform  data using UMAP
        umap_transformed = umap.UMAP(
        n_neighbors=5,
        # min_dist=0.3,
        n_components=2,
        random_state=42,
        ).fit_transform(mean_df)
        # # Use k-means to identify clusters in the low-dimensional data
        kmeans = KMeans(n_clusters=2)
        clusters = kmeans.fit_predict(umap_transformed)
        cluster_labels = kmeans.labels_
        # Plot the UMAP results with the cluster labels
        ax0.scatter(umap_transformed[:, 0], umap_transformed[:, 1], c=cluster_labels)
        nRCount = 0
        # pdb.set_trace()
        for n in range(11):

            ttt = mean_df_all['mouse'][n]
            # print(ttt)
            ax0.annotate(ttt, (umap_transformed[nRCount, 0], umap_transformed[nRCount, 1]), alpha=0.2,
                         size=6)
            nRCount += 1
        self.layoutOfPanel(ax0, xLabel='UMAP1', yLabel='UMAP 2', Leg=[1, 9])
        clustered_data = mean_df_all.groupby(cluster_labels)
        mean_df_all['cluster']=cluster_labels
        # mean_df_all.describe()
        cluster_stats = clustered_data.agg(['mean', 'std'])
        # cluster_stats.to_csv('clustering_results_statistics.csv')
        # def calculate_slope(group):
        #     # Split the data into x and y values
        #     # pdb.set_trace()
        #     x = np.array(group['recordingDay']).reshape(-1, 1)
        #     y = np.array(group['parameter'])
        #
        #     # Fit the linear regression model
        #     model = LinearRegression().fit(x, y)
        #
        #     # Calculate the slope of the fitted line
        #     slope = model.coef_[0]
        #
        #     return slope
        #
        # # Group the data by mouse
        # grouped_data = mean_df_recordingDays
        #
        # # Create a list to store the slopes for each parameter
        #
        # results_list_df=[]
        # # Loop through each parameter
        # for mouse in ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28','220716_f65','220716_f67']:
        #     slopes = []
        #     for parameter in ['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed', 'strideLenght',
        #                       'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
        #                       'stanceOnsetMedian']:
        #         # Select the data for the current parameter
        #         parameter_data = mean_df_recordingDays[['recordingDay', mouse, parameter]]
        #         pdb.set_trace()
        #
        #         # Rename the parameter column to 'parameter'
        #         parameter_data = parameter_data.rename(columns={parameter: 'parameter'})
        #
        #
        #     # # Apply the calculate_slope function to each group
        #         parameter_slopes = calculate_slope(parameter_data)
        #
        #         # Append the slopes to the list
        #         slopes.append(parameter_slopes)
        #
        #     # Create a dataframe with the slopes for each parameter
        #     results = pd.DataFrame(slopes, index=['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed',
        #                                           'strideLenght', 'strideLenghtStd', 'swingDuration', 'SwingDurationStd',
        #                                           'stanceDuration', 'indecisiveStride', 'stanceOnsetMedian']).transpose()
        #     results_list_df.append(results)
        # final_result= pd.concat(results_list_df, ignore_index=True)
        # pdb.set_trace()


        fname = '%s_clustering'%experiment
        if experiment=='muscimol':
            plt.savefig(self.figureDirectory + '/Muscimol_experiment/exploration/' + fname + '.pdf')
        else:
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def multivariateFigures(self, allVariablesDf, experiment):
        pawList=['FL','FR','HL','HR']
        treatment = ['saline', 'muscimol']
        # create figure instance
        if experiment=='ephy':
            gs = gridspec.GridSpec(1, 4)
            fig_width = 22  # width in inches
            fig_height = 5  # height in inches
            fig_size = [fig_width, fig_height]
        else:
            gs = gridspec.GridSpec(3, 4)
            fig_width = 22  # width in inches
            fig_height = 12  # height in inches
            fig_size = [fig_width, fig_height]

        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()


        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.25)

        useful = allVariablesDf.dropna(how='any')
        useful_matrix = useful[
            ['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed','strideLenght','strideLenghtStd', 'swingDuration', 'SwingDurationStd','stanceDuration', 'indecisiveStride', 'stanceOnsetMedian']]
        scaler = MinMaxScaler()
        matrix_scaled = pd.DataFrame(scaler.fit_transform(X=useful_matrix), columns=useful_matrix.columns)
        if experiment=='ephy':
            for p in range(4):
                ax0=plt.subplot(gs[0,p])
                pawDf=allVariablesDf[(allVariablesDf['paw'] == p)]
                sns.heatmap(pawDf.corr(), cmap=sns.diverging_palette(220, 20, n=200), ax=ax0)
                ax0.set_title('correlation map %s'%(pawList[p]))
                ax0.set_xticklabels(ax0.get_xticklabels(), rotation=90)
                ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)
        elif experiment == 'muscimol':
            for i in range(4):
                ax1 = plt.subplot(gs[2, i])
                for j in range(2):
                    ax2 = plt.subplot(gs[j, i])
                    paw_treatment = allVariablesDf[(allVariablesDf['paw'] == i) & (allVariablesDf['treatments'] == treatment[j])]
                    corr_mat_paw_treatment = paw_treatment.corr()
                    sns.heatmap(corr_mat_paw_treatment, cmap=sns.diverging_palette(220, 20, n=200), ax=ax2)
                    ax2.set_title('correlation map %s %s'%(pawList[i], treatment[j]))
                    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
                    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
                # for d in range(10):
                #     ax1 = plt.subplot(gs[2 + d, i])
                paw_saline = allVariablesDf[(allVariablesDf['paw'] == i) & (allVariablesDf['treatments'] == treatment[0]) ].corr()
                paw_muscimol = allVariablesDf[(allVariablesDf['paw'] == i) & (allVariablesDf['treatments'] == treatment[1]) ].corr()
                sns.heatmap(paw_muscimol - paw_saline, cmap=sns.diverging_palette(220, 20, n=200), ax=ax1)
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
                ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
                ax1.set_title('correlation map difference muscimol_saline %s'%pawList[i])

        fname = '%s_correlationMap' % experiment
        if experiment == 'muscimol':
            plt.savefig(self.figureDirectory + '/Muscimol_experiment/exploration/' + fname + '.pdf')
        elif experiment == 'ephy':
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def averagePlotsFigures (self, allVariablesDf, experiment):
        pawList = ['FL', 'FR', 'HL', 'HR']
        treatment = ['saline', 'muscimol']
        # create figure instance
        if experiment == 'ephy':
            gs = gridspec.GridSpec(15, 5)
            fig_width = 17  # width in inches
            fig_height = 27  # height in inches
            fig_size = [fig_width, fig_height]
        else:
            gs = gridspec.GridSpec(15, 6)
            fig_width = 22  # width in inches
            fig_height = 30  # height in inches
            fig_size = [fig_width, fig_height]

        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.5)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.10)

        # variableList = ['swingNumber', 'swingSpeed',  'acceleration','accelerationNumber', 'rungCrossed', 'strideLenght', 'swingDuration',
        #                 'stanceDuration', 'indecisiveStride', 'stanceOnsetMedian']
        variableList = ['swingNumber', 'swingSpeed',  'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian','stanceOnsetStd', 'wheelSpeed']
        # YAxisList = ['Swing number (avg)', 'Swing Speed (cm/s)',  'Mean swing acceleration (cm/s²)', 'Swing acceleration phase number',  'Fraction of ' + r'$\geqq$' + '2 rungs crossed', 'Stride lenght (cm)', 'Swing duration (s)',
        #                 'Stance Duration (s)', 'Fraction of indecisive stride', 'Stance onset median (norm FL)']
        YAxisList = ['Swing number (avg)', 'Swing Speed (cm/s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed', 'Mean stride lenght (cm)', 'Stride lenght Std (cm)', 'Swing duration (s)','Swing duration Std (s)',
                        'Stance Duration (s)', 'Fraction of indecisive stride', 'Stance onset median (norm FL)','Stance onset Std (norm FL)', 'Wheel speed (cm/s)']
        useful = allVariablesDf.dropna(how='any')
        useful_matrix = useful[
            ['swingNumber', 'swingSpeed', 'fractionRungCrossed', 'strideLenght',
             'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
             'stanceOnsetMedian','stanceOnsetStd', 'wheelSpeed']]
        scaler = MinMaxScaler()
        # matrix_scaled = pd.DataFrame(scaler.fit_transform(X=useful_matrix), columns=useful_matrix.columns)
        day1Df=allVariablesDf[(allVariablesDf['recordingDay'] == 1)][['swingNumber', 'swingSpeed', 'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian', 'wheelSpeed']]
        day10Df=allVariablesDf[(allVariablesDf['recordingDay'] == 10)][['swingNumber', 'swingSpeed' ,'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian', 'wheelSpeed']]
        allVariablesDf['paw'].replace({0:'FL', 1:'FR', 2:'HL', 3:'HR'}, inplace=True)

        allVariablesDf_stats=allVariablesDf.drop(allVariablesDf[allVariablesDf.recordingDay>9].index)
        # pvalues_all = {"day": mdf.pvalues['recordingDay'], "trial": mdf.pvalues['trial'], "paw":mdf.pvalues['paw']}
        # stars_all = {"day": starMultiplier(pvalues_all['day']), "trial": starMultiplier(pvalues_all['trial']), "paw":starMultiplier(pvalues_all['paw'])}
        if experiment == 'ephy':
            for v in range(len(variableList)):
                mdf = smf.mixedlm("%s ~ recordingDay*paw+trial*paw+trial*recordingDay"%variableList[v], allVariablesDf_stats, groups='mouse',missing='drop').fit()
                star_day = groupAnalysis.starMultiplier(mdf.pvalues['recordingDay'])
                star_trial = groupAnalysis.starMultiplier(mdf.pvalues['trial'])
                # print(variableList[v],mdf.summary(), mdf.pvalues['recordingDay'])
                # pdb.set_trace()
                ax=plt.subplot(gs[v,0])
                ax0=plt.subplot(gs[v,1])
                ax1 = plt.subplot(gs[v, 2])
                ax2 = plt.subplot(gs[v, 3])
                ax3=plt.subplot(gs[13:15,0:2])
                ax4 = plt.subplot(gs[13:15, 3:6])
                ax5 = plt.subplot(gs[v, 4])
                print(variableList[v])
                sns.lineplot(data=allVariablesDf, x='recordingDay', y=variableList[v], hue=None, errorbar=('se' if variableList[v]!='wheelSpeed' else 'sd'), err_style='bars', err_kws={'capsize':3, 'linewidth':1}, color='black', ax=ax)
                sns.lineplot(data=allVariablesDf, x='recordingDay', y=variableList[v], hue='mouse', errorbar=None, err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.1, legend=False, ax=ax)
                sns.lineplot(data=allVariablesDf.drop(allVariablesDf[allVariablesDf.trial>4].index), x='trial', y=variableList[v], errorbar=('se' if variableList[v]!='wheelSpeed' else 'sd'), err_style='bars', err_kws={'capsize':3, 'linewidth':1}, color='black', ax=ax0)
                sns.lineplot(data=allVariablesDf.drop(allVariablesDf[allVariablesDf.trial > 4].index), x='trial',y=variableList[v], hue='mouse', errorbar=None, err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, legend=False, alpha=0.1, ax=ax0)
                sns.lineplot(data=allVariablesDf, x='recordingDay', y=variableList[v], hue='paw',palette=sns.color_palette("tab10", n_colors=4), legend='auto',  ax=ax1)
                sns.lineplot(data=allVariablesDf.drop(allVariablesDf[allVariablesDf.trial>4].index), x='trial', y=variableList[v], hue='paw',palette=sns.color_palette("tab10", n_colors=4), legend=False, ax=ax2)
                sns.lineplot(data=allVariablesDf, x='recordingDay', y=variableList[v], hue='sex', errorbar='se',err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, palette=['pink', 'skyblue'], ax=ax5)
                self.layoutOfPanel(ax, xLabel='Day', yLabel='%s'%YAxisList[v], Leg=[1, 9])
                self.layoutOfPanel(ax0, xLabel='Trial', yLabel='', Leg=[1, 9])
                self.layoutOfPanel(ax1, xLabel='Day', yLabel='%s' % YAxisList[v], Leg=[1, 9])
                self.layoutOfPanel(ax2, xLabel='Trial', yLabel='', Leg=[1, 9])
                self.layoutOfPanel(ax5, xLabel='Day', yLabel='', Leg=[1, 9])
                ax1.legend( frameon=False, fontsize=8)
                ax.text(0.9, 0.97, '%s' % (star_day), ha='center', va='center', transform=ax.transAxes, style='italic',fontfamily='serif', fontsize=12, color='k')
                ax1.text(0.9, 0.97, '%s' % (star_trial), ha='center', va='center', transform=ax0.transAxes, style='italic',
                        fontfamily='serif', fontsize=12, color='k')

                if v!=0:
                    ax5.legend('',frameon=False)
                    ax1.legend('',frameon=False)

                # ax1.legend(bbox_to_anchor=(0.95,0.60), loc='center left', frameon=False, fontsize=6)#, labelcolor=['C0','C1', 'C2', 'C3'])
                # ax2.legend(bbox_to_anchor=(0.95, 0.60), loc='center left', frameon=False, fontsize=6)#, labelcolor=['C0','C1', 'C2', 'C3'])
            corr_pvalues_all, rvalues, annot, mask=groupAnalysis.calculate_correlation_pvalues(useful_matrix)
            mask_day10_d1=abs(day10Df.corr()-day1Df.corr())<0.20
            b=sns.heatmap(useful_matrix.corr(), annot=False, mask=mask,  fmt='',cmap=sns.diverging_palette(220, 20, n=100), ax=ax3)
            n=sns.heatmap(day10Df.corr()-day1Df.corr(), mask=mask_day10_d1, cmap=sns.diverging_palette(220, 20, n=100), ax=ax4)
            b.set_facecolor('whitesmoke')
            n.set_facecolor('whitesmoke')
            ax3.set_title('correlation map all' )
            ax4.set_title('correlation map day 10 - day 1' )
            fname = '%s_Behavior_summary' % (experiment)
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
        elif experiment == 'muscimol':
            paw_saline = allVariablesDf[(allVariablesDf['treatments'] == treatment[0])][['paw','swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian', "wheelSpeed"]]
            paw_muscimol = allVariablesDf[ (allVariablesDf['treatments'] == treatment[1])][['paw','swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian', 'wheelSpeed']]
            for v in range(len(variableList)):
                ax=plt.subplot(gs[v,0])
                ax0=plt.subplot(gs[v,1])


                ax3=plt.subplot(gs[13:15,0:2])
                ax4 = plt.subplot(gs[13:15, 2:4])
                ax5 = plt.subplot(gs[13:15, 4:6])
                sns.lineplot(data=allVariablesDf, x='recordingDay', y=variableList[v], style=('treatments' if variableList[v]!="wheelSpeed" else None), style_order=['saline', 'muscimol'], errorbar=('se' if variableList[v]!='wheelSpeed' else 'sd'), err_style='bars', err_kws={'capsize':3, 'linewidth':1}, color='black', legend=False, ax=ax)
                sns.lineplot(data=allVariablesDf, x='trial', y=variableList[v], style=('treatments' if variableList[v]!="wheelSpeed" else None), style_order=(['saline', 'muscimol']if variableList[v]!="wheelSpeed" else None), errorbar=('se' if variableList[v]!='wheelSpeed' else 'sd'), err_style='bars', err_kws={'capsize':3, 'linewidth':1}, color='black', legend=False, ax=ax0)
                for i in range(4):
                    sns.color_palette("tab10")
                    ax1=plt.subplot(gs[v, 2 + i])
                    # ax6=plt.subplot(gs[13,  i])
                    # ax7 = plt.subplot(gs[14, i])
                    sns.lineplot(data=allVariablesDf[ (allVariablesDf['paw'] == pawList[i])], x='recordingDay', style_order=(['saline', 'muscimol']if variableList[v]!="wheelSpeed" else None), y=variableList[v], hue=None, style=('treatments' if variableList[v]!="wheelSpeed" else None), errorbar=('se' if variableList[v]!='wheelSpeed' else 'sd'), err_style='bars', err_kws={'capsize':3, 'linewidth':1},color='C%s'%i, legend=False,  ax=ax1)
                    # pdb.set_trace()
                    # pawHeatSal = sns.heatmap(paw_saline[(paw_saline['paw']==i)].corr(), cmap=sns.diverging_palette(220, 20, n=200), ax=ax6)
                    # pawHeatMus = sns.heatmap(paw_muscimol[(paw_muscimol['paw']==i)].corr(),cmap=sns.diverging_palette(220, 20, n=200), ax=ax7)
                    if v==0:
                        ax1.set_title('%s'%pawList[i])
                    self.layoutOfPanel(ax1, xLabel='Trial', yLabel='%s' % YAxisList[v], Leg=[1, 9])



                self.layoutOfPanel(ax, xLabel='Day', yLabel='%s'%YAxisList[v], Leg=[1, 9])
                self.layoutOfPanel(ax0, xLabel='Trial', yLabel='%s' % YAxisList[v], Leg=[1, 9])

                salHeat=sns.heatmap(paw_saline.corr(), mask=abs(paw_saline.corr())<0.20, cmap=sns.diverging_palette(220, 20, n=200), ax=ax3)
                MusHeat=sns.heatmap(paw_muscimol.corr(),mask=abs(paw_muscimol.corr())<0.20, cmap=sns.diverging_palette(220, 20, n=200), ax=ax4)
                diffHeat=sns.heatmap(paw_muscimol.corr() - paw_saline.corr(), cmap=sns.diverging_palette(220, 20, n=200), mask=abs((paw_muscimol.corr() - paw_saline.corr()))<0.1, ax=ax5)
                salHeat.set_facecolor('whitesmoke')
                MusHeat.set_facecolor('whitesmoke')
                diffHeat.set_facecolor('whitesmoke')
                ax3.set_title('correlation map average saline')
                ax4.set_title('correlation map average muscimol')
                ax5.set_title('correlation map day muscimol - saline')
                fname = '%s_Behavior_summary' % (experiment)
                plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def wheelDistanceFigures (self, wheelDistanceDf_ephy, wheelDistanceDf_muscimol):
        gs = gridspec.GridSpec(1,1)
        fig_width = 7  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.5)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.10)



        ax1=plt.subplot(gs[0])

        sns.lineplot(data=wheelDistanceDf_ephy, x='recordingDay', y='measuredValue', hue=None, errorbar='se', err_style='bars', err_kws={'capsize':3, 'linewidth':1},  color='black', ax=ax1)
        sns.lineplot(data=wheelDistanceDf_ephy, x='recordingDay', y='measuredValue', hue='mouse', errorbar=None, err_style='bars', legend='auto', err_kws={'capsize':3, 'linewidth':1},  alpha=0.2, ax=ax1)
        sns.lineplot(data=wheelDistanceDf_muscimol, x='recordingDay', y='measuredValue', hue=None, errorbar='se', err_style='bars', err_kws={'capsize':3, 'linewidth':1},  color='red', ax=ax1)
        self.layoutOfPanel(ax1, xLabel='Day', yLabel='Wheel distance (cm)', Leg=[1, 9])
        # ax1.legend(['ephy','muscimol'], frameon=False)
        majorLocator_x = MultipleLocator(1)
        ax1.xaxis.set_major_locator(majorLocator_x)
        ax1.set_title('Wheel distance')
        fname = 'Wheel_distance_ephy_muscimol'
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def createPawTrajectoriesFigure(self, pawTrajectoriesData, experiment):
        from matplotlib import cm

        cmap = cm.get_cmap('tab20')

        fig_width = 17  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[10, 2.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure

        # create figure instance
        fig = plt.figure()
        plt.figtext(0.06, 0.96, 'N=%s Mice, 10 days of recording' % (len(pawTrajectoriesData[0])), clip_on=False, color='black',
                    size=14)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
        gssub0 = gridspec.GridSpecFromSubplotSpec(10, 4, subplot_spec=gs[0], hspace=0.2)
        # plot all swing phases ###############################################
        pawId=['FL','FR','HL','HR']
        pawTrajectoriesMean_Mice=[[],[],[],[]]
        pawTrajectoriesTimeMean_Mice=[[],[],[],[]]
        pawTrajectoriesMean=[[],[],[],[]]
        pawTrajectoriesTimeMean = [[], [], [], []]
        pawTrajectoriesStd= [[], [], [], []]
        for m in range(len(pawTrajectoriesData[0])):
            for p in range(4):
                nan_days=np.empty((100))
                nan_days.fill(np.nan)
                while len(pawTrajectoriesData[3][m][1][p])<10:
                    pawTrajectoriesData[3][m][1][p].append(nan_days)
                    pawTrajectoriesData[4][m][1][p].append(nan_days)
                    pawTrajectoriesData[5][m][1][p].append(nan_days)
                pawTrajectoriesMean[p].append(pawTrajectoriesData[3][m][1][p])
                pawTrajectoriesTimeMean[p].append(pawTrajectoriesData[4][m][1][p])
                pawTrajectoriesStd[p].append(pawTrajectoriesData[5][m][1][p])

        for i in range(4):
            # pawTrajectoriesMean_Mice[i].append(np.nanmean(pawTrajectoriesMean[i], axis=0))
            pawTrajectoriesMean_Mice[i].append(pawTrajectoriesMean[i])
            # pawTrajectoriesTimeMean_Mice[i].append( np.nanmean(pawTrajectoriesTimeMean[i], axis=0))
            pawTrajectoriesTimeMean_Mice[i].append(pawTrajectoriesTimeMean[i])
        pawTrajectoriesVar_Mouse = [[], [], [], []]
        meanTrajectVar =np.empty((4,len(pawTrajectoriesData[0]),10))
        meanTrajectVar.fill(np.nan)

        for m in range(len(pawTrajectoriesData[0])):
            nDays = len(pawTrajectoriesData[0][m][1])

            for n in range(nDays):


                for i in range(4):

                    ax = plt.subplot(gssub0[n,i])

                    meanTrajectVar[i][m][n]=np.nanmean(stats.variation(pawTrajectoriesData[0][m][1][n][i],axis=0, ddof=1, nan_policy='omit')[:30])
                    #meanTrajectVar[i][m][n] = np.nanmean(np.nanstd(pawTrajectoriesData[0][m][1][n][i], axis=0))
                    for k in range(len(pawTrajectoriesData[1][m][1][n][i])):
                        # pdb.set_trace()
                        # ax.plot(pawTrajectoriesData[1][m][1][n][i][k], pawTrajectoriesData[0][m][1][n][i][k],lw=0.2,alpha=0.1)

                        if experiment == 'muscimol':
                            if 'muscimol' in pawTrajectoriesData[1][m][0]:
                                color='red'
                            else:
                                color='k'
                            ax.plot(pawTrajectoriesData[1][m][1][n][i][k], pawTrajectoriesData[0][m][1][n][i][k],c=color, lw=0.2, alpha=0.05)
                    # pdb.set_trace()
                    # ax.plot(pawTrajectoriesTimeMean_Mice[i][0][m][n], pawTrajectoriesMean_Mice[i][0][m][n], alpha=0.7, lw=1.5, label=pawTrajectoriesData[0][m][0])
                    #
                    # if i==3:
                    #     ax.legend(frameon=False, fontsize=6, loc='upper right')
                    pawTrajectoriesAverageTime_AllMice=np.nanmean(pawTrajectoriesTimeMean_Mice[i][0], axis=0)
                    pawTrajectoriesAverage_AllMice=np.nanmean(pawTrajectoriesMean_Mice[i][0], axis=0)
                    medianAverageXpos=np.nanpercentile(pawTrajectoriesAverage_AllMice,50, axis=1)
                    maxAverageXpos = np.nanmax(pawTrajectoriesAverage_AllMice, axis=1)
                    medianAverageXposAgr=np.min(np.argwhere(pawTrajectoriesAverage_AllMice[n]>medianAverageXpos[n]))
                    # pdb.set_trace()
                    # maxTime=np.min(np.argwhere(pawTrajectoriesAverage_AllMice[n][~np.isnan(pawTrajectoriesAverage_AllMice[n])]>maxAverageXpos[n]))

                    ax.plot(pawTrajectoriesAverageTime_AllMice[n][:50], pawTrajectoriesAverage_AllMice[n][:50], c='k')
                    ax.axhline(medianAverageXpos[n], lw=0.5, ls=':', c='grey')


                    # ax.axvline(pawTrajectoriesAverageTime_AllMice[n][medianAverageXposAgr], lw=0.5, ls=':', c='grey')
                    # ax.axvline(pawTrajectoriesAverageTime_AllMice[n][maxTime], lw=0.5, ls=':', c='grey')


                    if i <2:
                        ax.set_ylim(-1,6)

                    else:
                        ax.set_ylim(-1,8)
                        ax.set_xlim(0, 0.3)
                    if n < (9):
                        # ax.plot(pawTrajectoriesData[4][m][1][i][n], pawTrajectoriesData[3][m][1][i][n], c='k')
                        if n == 0:
                            ax.set_title("%s" % pawId[i])
                        if (i == 0):

                            self.layoutOfPanel(ax, xLabel=None, yLabel='x-pos', xyInvisible=[True, False])
                            # if n == 0:
                            #     ax.text(0.9, 0.99, 'rung crossed', transform=ax.transAxes,ha='center', va='center')
                            #     for t in range(5):
                            #         ax.text(0.9, (0.9 - t * 0.1), '- %s' % t, color='C%s' % t, transform=ax.transAxes, ha='center', va='center')
                        elif (i == 2):
                            self.layoutOfPanel(ax, xLabel=None, yLabel=None, xyInvisible=[True, False])
                            # if n == 0:
                            #     ax.text(0.9, 0.99, 'rung crossed', transform=ax.transAxes,ha='center', va='center')
                            #     for t in range(5):
                            #         ax.text(0.9, (0.9 - t * 0.1), '- %s' % t, color='C%s' % t, transform=ax.transAxes, ha='center', va='center')
                        else:
                            self.layoutOfPanel(ax, xLabel=None, yLabel=None, xyInvisible=[True, True])
                    elif n==9:
                        if i == 0:
                            self.layoutOfPanel(ax, xLabel='time (s)', yLabel='x-pos', xyInvisible=[False, False])
                        elif (i == 2):
                            self.layoutOfPanel(ax, xLabel='time (s)', yLabel=None,xyInvisible=[False, False])
                        else:
                            self.layoutOfPanel(ax, xLabel='time (s)', yLabel=None, xyInvisible=[False, True])
        # pdb.set_trace()



        # mse_values = [[], [], [], []]
        # for p in range(4):
        #     for m in range((len(pawTrajectoriesData[0]))):
        #         for n in range(nDays):
        #             avg_paw_time_X = pawTrajectoriesMean_Mice[p][0][m][n][:50]
        #             avg_paw_traject_Y = pawTrajectoriesTimeMean_Mice[p][0][m][n][:50]
        #             avg_paw_time_X=avg_paw_time_X[~np.isnan(avg_paw_time_X)]
        #             avg_paw_traject_Y=avg_paw_traject_Y[~np.isnan(avg_paw_traject_Y)]
        #             # Fit linear regression model to the data
        #             if len(avg_paw_traject_Y)>2:
        #                 model = OLS(avg_paw_time_X, avg_paw_traject_Y, missing='drop')
        #                 results=model.fit()
        #
        #                 predictions = results.predict(avg_paw_time_X)# Calculate MSE for the current day
        #                 mse = mean_squared_error(avg_paw_traject_Y, predictions)  # Append MSE value to the list
        #                 mse_values[p].append(mse)
        # pdb.set_trace()
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.2,hspace=0.2)
        for i in range(4):
            ax1 = plt.subplot(gssub1[i])
            # pdb.set_trace()
            pawTrajectoryVar=np.nanmean(meanTrajectVar, axis=1)
            pawTrajectoryVar_sem = stats.sem(meanTrajectVar, axis=1, nan_policy='omit')
            # pdb.set_trace()
            ax1.plot(np.arange(10)+1,pawTrajectoryVar[i] )
            ax1.errorbar(np.arange(10) + 1, pawTrajectoryVar[i], pawTrajectoryVar_sem[i], capsize=3, linewidth=1, ms=5, c='k')
            for m in range(len(meanTrajectVar[i])):
                ax1.plot(np.arange(10) + 1, meanTrajectVar[i][m], alpha=0.2, label=pawTrajectoriesData[0][m][0])
            if i==0:
                self.layoutOfPanel(ax1, xLabel='days', yLabel='x-pos coefficient of variation', xyInvisible=[False, False])
            else:
                self.layoutOfPanel(ax1, xLabel='days', yLabel=None,xyInvisible=[False, False])
            if i==3:
                ax1.legend(frameon=False, fontsize=6, loc='upper left')
            majorLocator_x = MultipleLocator(1)
            ax1.xaxis.set_major_locator(majorLocator_x)


        fname = 'paw_Trajectories_%s'%experiment
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def createWheelSpeedFigure(self, strideLengthData, experiment):
        from matplotlib import cm

        cmap=cm.get_cmap('tab20')


        fig_width = 5  # width in inches
        fig_height = 5  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid':False   # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 1,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[1, 2.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure

        # create figure instance
        fig = plt.figure()
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.15)

        #plt.figtext(0.06, 0.96, animalNames, clip_on=False, color='black',size=10)


                    # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        wheelSpeedDf=groupAnalysis.convertListToPandasDF(strideLengthData[7], trialValues=False, treatments=False)
        (wheelSpeed_mean, wheelSpeed_std, wheelSpeed_sem)=groupAnalysis.getMeanStdNan(strideLengthData[6])
        sns.lineplot(data=wheelSpeedDf, x='recordingDay', y='measuredValue', errorbar='sd', hue=None, err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax0, legend=False)
        sns.lineplot(data=wheelSpeedDf, x='recordingDay', y='measuredValue',  hue='mouse', errorbar=None, alpha=0.2,
                     ax=ax0)
        self.layoutOfPanel(ax0, xLabel='Day', yLabel='Wheel Speed (cm/s)', Leg=[1, 9])
        ax0.legend(frameon=False, fontsize=8, loc='best')
        majorLocator_x = MultipleLocator(1)
        ax0.xaxis.set_major_locator(majorLocator_x)

        ax0.set_title('Wheel Speed')
        fname = 'wheel_speed_%s'%experiment
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def controlPlotsFigures(self, allVariablesDf, experiment):
        pawList = ['FL', 'FR', 'HL', 'HR']
        treatment = ['saline', 'muscimol']
        variableList = ['swingNumber', 'swingSpeed', 'swingAcceleration', 'fractionRungCrossed', 'strideLenght',
                        'strideLenghtStd', 'swingDuration', 'SwingDurationStd', 'stanceDuration', 'indecisiveStride',
                        'stanceOnsetMedian','stanceOnsetStd', 'wheelSpeed']
        fig_width = 22  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.5)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.10)
    ########################################################################################################################################
    def PSTHGroupFigure_before_after_event_modulation(self,recordings, df_psth, df_cells,condition,event):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0','C1','C2','C3']
        fig_width = 22  # width in inches
        fig_height = 38  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False, 'axes.spines.top':False, 'axes.spines.right':False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3,  # ,
                               width_ratios=[0.1,1,5]
                               #height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.03)
        gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        for i in range(4):
            # paw labels panel
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=25)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])
            gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub1[i], hspace=0.1, wspace=0.5, height_ratios=[1,2,1],width_ratios=[1])#, height_ratios=[1,1])
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            # paw_df = paw_df.dropna(subset=['before_swingOnset_z-score_AUC_0.1'])
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]

            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df=paw_psth_df.drop(index)
                        # pdb.set_trace()
            # events = ['stanceOnset', 'swingOnset']
            # for e in range(len(events)):
            #     event=events[e]
            ax1=plt.subplot(gssub2[1])

            #get the ids and df of all modulated cells
            cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df, paw_psth_df, event)

            #calculate correlation distribution
            modulated, non_modulated=groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(cells_id, modCells_df, event)
            #plot correlation distribution
            ax1.hist(modulated, bins=30, density=True, alpha=0.6, color='maroon', label=f'modulated {recordings}')
            ax1.hist(non_modulated, bins=30, density=True, alpha=0.6, color='steelblue', label=f'non modulated {recordings}')

            ax1.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
            ax1.legend(loc='best',frameon=False, fontsize=10)
            ax1.set_xlim(-0.5,1)
            ax1.set_ylim(0, 3)
            ax1.set_title(f'{event[:-5]} onset', fontsize=10)
            self.layoutOfPanel(ax1, xLabel='PSTH Correlation coefficient across trials', yLabel=('probability density' ),Leg=[1, 9])
            gssub3 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,2], hspace=0.22, height_ratios=[1,1,1,1])
            gssub4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub3[i], hspace=0.3)
            gssub5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub4[0], hspace=0.15, height_ratios=[1,12])
            times = ['before_', 'after_']
            for t in range(2):
                catList=['↓','↑','-']
                time = times[t]
                ax2 = plt.subplot(gssub5[0, t])
                # print('we are here!!!!!!',time, event, i)
                #get ids and counts of modulated cells
                modCells_Id, modCells_count, counts=groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList, time, event, condition=None)

                values = counts.values
                labels = counts.index
                percent = values / np.sum(values) * 100
                bottom_pos = 0
                label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
                label_colorsList=['cadetblue','indianred','lightsteelblue']
                for l, label in enumerate(labels):
                    # pdb.set_trace()
                    bar_width = percent[l]
                    bar_left = bottom_pos
                    ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],label=f'{label} {round(percent[l], 1)}% ({values[l]})')
                    # ax1.set_title(('modulation %s %s' % (time, event)) if i == 0 else '', loc='center')

                    # ax2.annotate(f'{label} {round(percent[l], 1)}%  ({values[l]}) ',(bar_left + bar_width / 2, -0.2), color='white', fontsize=12)
                    ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),bbox_to_anchor=(0.05, 0.7, 0.85, 0.1), frameon=False, fontsize=8)#, labelcolor=label_colors_List)
                    bottom_pos += bar_width
                    ax2.get_yaxis().set_visible(False)
                    # ax2.get_xaxis().set_visible(False)
                    ax2.spines[['left','right','bottom', 'top']].set_visible(False)
                ax2.text(0.5, 1.6, (f'modulation {time[:-1]} {event[:-5]} onset'), ha='center',va='center', transform=ax2.transAxes, fontsize=10)
                zscore = ['AUC', 'peak']
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[2]
                zscorePar = zscore[0]
                zscore_key = ['before_%s_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%s_z-score_%s_%s' % (event, zscorePar, interval)]
                for c in range(3):
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells_Id[catList[c]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial'] > 5].index)
                    modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index)
                    # if recordings=='MLI':
                    #     modulated_paw_df_visual = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_global_Id'] == 25].index)
                    # else:
                    modulated_paw_df_visual=modulated_paw_df

                    gssub6 = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gssub5[1, t], hspace=1.3, wspace=0.3,height_ratios=[2.5, 2,2,2,2,2,2])
                    ax3=plt.subplot(gssub6[0,c])
                    sns.lineplot(modulated_paw_df_visual, x='trial', y=zscore_key[t], hue='cell_global_Id', errorbar='se', ax=ax3, legend=False, alpha=0.06)
                    sns.lineplot(modulated_paw_df, x='trial', y=zscore_key[t], hue=None, errorbar='se', ax=ax3,legend=True, color=label_colorsList[c])
                    # ax3.plot(0,0, label='')
                    zscore_key1 = 'zScoreAUC_' + times[t] + event
                    modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    if len(modulated_paw_df) > 3:
                        mixedML_results, mixeML_trial_pvalue, tukey=groupAnalysis_psth.perform_mixedlm(modulated_paw_df,zscore_key1, 'trial', 'cell_global_Id')
                        # linReg_results, LinReg_trial_pvalue=groupAnalysis_psth.perform_linear_regression(modulated_paw_df,zscore_key1, 'trial')
                        # GLM_results, pvalue, tukey = groupAnalysis_psth.perform_GLM_group_and_Tukey(modulated_paw_df, zscore_key1, 'trial', 'cell_global_Id')
                        # results, p_value, tukey= groupAnalysis_psth.perform_GLM_and_Tukey(modulated_paw_df, zscore_key1, 'trial')
                        # modulated_paw_df = modulated_paw_df[modulated_paw_df['trial'].notna()]
                        # df_average_trial = modulated_paw_df.groupby(['trial', 'cell_global_Id'])[zscore_key1].agg(
                        #     ['mean']).reset_index()

                        # tukey_trial = pairwise_tukeyhsd(df_tukey_trial['mean'], df_tukey_trial['trial'])

                        # print('p_values',p_value, tukey)

                        # print(event, time, i, catList[c],tukey_trial.summary())
                        star_trial = groupAnalysis.starMultiplier(mixeML_trial_pvalue)

                        ax3.text(0.9, 0.97, '%s' % (star_trial), ha='center', va='center', transform=ax3.transAxes,style='italic', fontfamily='serif', fontsize=10, color='k')
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(f' {time[:-1]} {event[:-5]} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''), Leg=[1, 9])
                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)

                    behavior_par = ['swingDuration', 'swingLength', 'swingSpeed','stanceDuration', 'stanceOnsetStd','stanceOnsetMedian']
                    #behavior_par = ['acceleration', 'mean_acceleration', 'mean_deceleration', 'acc_duration','dec_duration']
                    # behavior_par = ['swingDuration', 'swingLength']
                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p, c])

                        modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df[zscore_key[t]]
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,alternative='two-sided')
                            r2 = np.square(r_value)
                        except ValueError:
                            pass
                        if (p_value > 0.05) or (r2 < 0.1) or (r2 == 1):
                            alpha = 0.1
                        else:
                            alpha = 0.8
                        sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                                    line_kws={'alpha': alpha})
                        ax4.text(0.7, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",transform=ax4.transAxes)
                        if p==0:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p!=0 else ''), yLabel=(f'swing {behavior_par[p][5:]} (s)' if (c == 0) else ''), Leg=[1, 9])
                        elif p==1:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p!=0 else ''), yLabel=(f'swing {behavior_par[p][5:]} (pixel)' if (c == 0 ) else ''), Leg=[1, 9])
                        elif p==2:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p!=0 else ''), yLabel=(f'swing {behavior_par[p][5:]} (pixel/s)' if (c == 0 ) else ''), Leg=[1, 9])
                        elif p == 3:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'stance {behavior_par[p][6:]} (s)' if (c == 0) else ''), Leg=[1, 9])
                        elif p == 4:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'stance {behavior_par[p][6:]} (s)' if (c == 0) else ''), Leg=[1, 9])
        fname = '%s_%s_ephys_psth_group_analysis_%s' % (event,recordings[3:],condition)

        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def PSTHGroupFigure_all_event_modulation(self, df, condition):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0','C1','C2','C3']
        fig_width = 20  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False, 'axes.spines.top':False, 'axes.spines.right':False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[0.008,1]
                               #height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.2)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.005, right=0.98, top=0.98, bottom=0.05)
        gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub0b = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1,0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1,subplot_spec=gs[0,1], hspace=0.2, wspace=0.13)

        for i in range(4):
            gssub3 = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=gssub1[i,0:2], hspace=0.8, wspace=0.13, height_ratios=[1,4,4,4,4])
            # paw labels
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=16)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])

            paw_df=df[(df['paw']==pawList[i])]

            time=['before_','after_']
            events=['swingOnset','stanceOnset']
            zscore=['AUC','peak']
            intervals=[0.1,0.2,0.3]
            interval=intervals[2]
            event = events[1]
            zscorePar=zscore[0]
            varKey='modulation_category_'
            zscore_key=['before_%s_z-score_%s_%s_ms'%(event,zscorePar,interval),'after_%s_z-score_%s_%s_ms'%(event,zscorePar,interval)]
            binCount_stance = ['modulation_count_before_stanceOnset', 'modulation_count_after_stanceOnset']
            behavior_par=['swingDuration', 'swingSpeed', 'swingLength', 'stanceDuration']
            catList=['↓','↑','-']
            if event=='stanceOnset':
                catList = ['↑↓', '↑-', '--']
            else:
                catList = ['↓↑', '-↑', '--']
            label_colors = {'↓↓': 'darkturquoise', '↓↑': 'turquoise', '↓-': 'paleturquoise', '↑↓': 'crimson', '↑↑': 'salmon', '↑-': 'sandybrown', '-↓': 'plum', '-↑': 'violet', '--': 'lightsteelblue'}
            catListColor = ['crimson','sandybrown','lightsteelblue']
            for t in range(2):

                gssub2 = gridspec.GridSpecFromSubplotSpec(5, 3, subplot_spec=gssub3[1:5, t], hspace=0.7, wspace=0.15)#, height_ratios=[2,4,4,4,4,4])
                key=varKey+time[t]+event
                key1=varKey+event
                for c in range(3):
                    modulated_cells = paw_df[(paw_df[key1] == catList[c])]
                    modulated_cells_list=np.unique(modulated_cells['cell_global_Id'])
                    # print('modulated',catList[c],modulated_cells_list)
                    modulated_mask = paw_df['cell_global_Id'].isin(modulated_cells_list)
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    modulated_paw_df=modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial']>5].index)
                    modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_trial_Nb']==1].index)
                    # modulated_paw_df.to_csv(groupAnalysisDir + '%s_modulated_psth.csv'%pawList[i])
                    # standardize data
                    # scaler = StandardScaler()
                    # columns_to_scale = ['swingDuration', 'swingSpeed', 'swingLength','stanceDuration','before_stanceOnset_z-score_AUC_%s_ms'%interval,	'after_stanceOnset_z-score_AUC_%s_ms'%interval,'modulation_count_before_stanceOnset', 'modulation_count_after_stanceOnset','before_swingOnset_z-score_AUC_%s_ms'%interval, 'after_swingOnset_z-score_AUC_%s_ms'%interval]
                    # if np.shape(modulated_paw_df[columns_to_scale])[0]!=0:
                    #     modulated_paw_df[columns_to_scale] = scaler.fit_transform(modulated_paw_df[columns_to_scale])
                    corrMatrix_stance=modulated_paw_df[['swingDuration', 'swingSpeed', 'swingLength','stanceDuration','before_stanceOnset_z-score_AUC_%s_ms'%interval,	'after_stanceOnset_z-score_AUC_%s_ms'%interval]]#,'modulation_count_before_stanceOnset', 'modulation_count_after_stanceOnset']]#,'before_swingOnset_z-score_AUC_%s_ms'%interval, 'after_swingOnset_z-score_AUC_%s_ms'%interval]]
                    # pdmodulated_cellsNb=[]b.set_trace()
                    # pdb.set_trace()
                    ax2 = plt.subplot(gssub2[0, c])

                    sns.lineplot(modulated_paw_df, x='trial', y=zscore_key[t], hue=None, errorbar='se', ax=ax2, legend='auto',color=label_colors[catList[c]])


                    # sns.lineplot(modulated_paw_df, x='trial', y=binCount_stance[t], hue=None, errorbar='se', ax=ax2,legend='auto', color=cat_colors[c])
                    # sns.lineplot(modulated_paw_df, x='trial', y='stanceDuration', hue=None, errorbar='se', ax=ax2, legend='auto', color='black')
                    # ax3.scatter(x=modulated_paw_df.groupby("trial").mean()['stanceDuration'], y=modulated_paw_df.groupby("trial").mean()[AUC_zscore_stance[t]])
                    #corr_pvalues_all, rvalues, annot, mask = groupAnalysis.calculate_correlation_pvalues(corrMatrix_stance)
                    # b = sns.heatmap(corrMatrix_stance.corr(), annot=False,  fmt='',
                    #                 cmap=sns.diverging_palette(220, 20, n=100), ax=ax3)
                    ax2.set_title(('Z-score %s %s %s'%(zscorePar ,time[t], event) if c==1 else ''))
                    self.layoutOfPanel(ax2, xLabel='Trial', yLabel=('Modulated MLI PSTH \n Z-score %s'%zscorePar if c == 0 else ''),Leg=[1, 9])
                    for p in range(4):
                        modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                        ax3 = plt.subplot(gssub2[p+1, c])
                        y = modulated_paw_df[behavior_par[p]]
                        # x=modulated_paw_df.groupby("trial").mean()[behavior_par[p]]
                        x = modulated_paw_df[zscore_key[t]]
                        sns.regplot(x=x, y=y, ax=ax3, color=label_colors[catList[c]])
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y, alternative='two-sided')
                            r2=np.square(r_value)
                        except ValueError:
                            pass


                        ax3.text(0.7, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}", transform=ax3.transAxes)
                        # try:
                        #     # y=modulated_paw_df.groupby("trial").mean()[AUC_zscore_stance[t]]
                        #     r_value, p_value = pearsonr(x, y)
                        #     ax3.text(0.7, 0.05, f"r = {r_value:.2f}\np = {p_value:.2f}", transform=ax3.transAxes)
                        # except ValueError:
                        #     pass
                        # sns.regplot(data=modulated_paw_df,x=behavior_par[p], y=AUC_zscore_stance[t],ax=ax3)


                    # sns.lineplot(modulated_paw_df, x='trial', y='swingLength', hue=None, errorbar='se', ax=ax2, legend='auto', color='black')
                    # sns.lineplot(modulated_paw_df, x='trial', y='swingSpeed', hue=None, errorbar='se', ax=ax2, legend='auto', color='black')
                    # sns.lineplot(modulated_paw_df, x='trial', y='stanceDuration', hue=None, errorbar='se', ax=ax2, legend='auto', color='black')

                        self.layoutOfPanel(ax3, xLabel='zscore_%s'%zscorePar, yLabel=(behavior_par[p] if c == 0 else ''),Leg=[1, 9], xyInvisible=[(True if p!=3 else False), False])
                    # ax2.legend(frameon=False)
                    majorLocator_x = MultipleLocator(1)
                    ax2.xaxis.set_major_locator(majorLocator_x)


                counts = paw_df[key1].value_counts()

                counts = counts.sort_index(ascending=False)
                # counts = (paw_df[key1].value_counts())
                values = counts.values
                labels = counts.index
                percent = values / np.sum(values) * 100
                # Create a list of starting positions for the bars
                bottom_pos = 0
                # print(labels)
                label_colors = {'↑': 'maroon', '↓': 'forestgreen', '-': 'steelblue'}
                label_colors = {'↓↓': 'darkgreen', '↓↑': 'green', '↓-': 'lightgreen', '↑↓': 'red', '↑↑': 'orange', '↑-': 'yellow', '-↓': 'purple', '-↑': 'lightblue', '--': 'grey'}
                label_colors = {'↓↓': 'darkturquoise', '↓↑': 'turquoise', '↓-': 'paleturquoise', '↑↓': 'crimson', '↑↑': 'salmon', '↑-': 'sandybrown', '-↓': 'plum', '-↑': 'violet', '--': 'lightsteelblue'}
            ax1 = plt.subplot(gssub3[0, 0:2])
            for l, label in enumerate(labels):

                bar_width = percent[l]
                bar_left = bottom_pos
                ax1.barh(0, bar_width, left=bar_left, height=1, label=f'{label} {round(percent[l], 1)}%',color=label_colors[label], alpha=1)
                # ax1.annotate(f'{label} {round(percent[l], 1)}%',
                #              (bar_left + bar_width / 2, 0))
                bottom_pos += bar_width
                # box = ax1.get_position()
                # ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                #                  box.width, box.height * 0.9])
                ax1.legend(loc='lower left', mode='expand', ncol=len(label_colors),bbox_to_anchor=(0,1.02,0.6, 0.2), frameon=False, fontsize=8)
                ax1.get_yaxis().set_visible(False)
                ax1.set_ylabel(('modulation %s %s' % (time[t], event)))

            # circle = plt.Circle((0, 0), 0.7, color='white')
            # plt.pie(mod_prop[key], labels=mod_prop['index'], autopct='%1.1f%%', startangle=90,
            #         labeldistance=1, textprops={'rotation':0})
            # plt.gca().add_artist(circle)
            # ax1.set_title(('modulation %s %s' % (time[t], event)) if i == 0 else '', y=-0.1)
                # self.layoutOfPanel(ax1, xLabel=None, yLabel=None, Leg=[1, 9], xyInvisible=[False,True])



        # plt.show()

        fname = '%s_%s_modulated_Z-score_%s_broadCategories'%(event, condition,zscorePar)
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def PSTH_correlation_figure (self, df, df_psth,condition, recordings):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0','C1','C2','C3']
        fig_width = 12  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False, 'axes.spines.top':False, 'axes.spines.right':False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[0.05,1]
                               #height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.05)
        gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub0b = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1,0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 3,subplot_spec=gs[0,1], hspace=0.3, wspace=0.3)


        events = ['stanceOnset', 'swingOnset']

        for i in range(4):
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=16)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])

            paw_df=df[(df['paw']==pawList[i])]
            paw_psth_df=df_psth[(df_psth['paw']==pawList[i])]
            #get the modulated and non modulated cells list
            for e in range(len(events)):
                event=events[e]
                #get the ids and df of all modulated cells
                cells_id, modCells_df=groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df, paw_psth_df, event)
                #calculate correlation distribution
                modulated, non_modulated=groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(cells_id, modCells_df, event)
                # pdb.set_trace()
                ax1=plt.subplot(gssub1[i,e])
                # a=sns.kdeplot(x=corr_coeff_all['modulated'], ax=ax1, legend=True)#, kwargs={'label':'modulated cells'} )
                # b=sns.kdeplot(x=corr_coeff_all['non_modulated'], ax=ax1, legend=True)
                # pdb.set_trace()
                ax1.hist(modulated, bins=30, density=True, alpha=0.6, color='C1', label='modulated cells')
                ax1.hist(non_modulated, bins=30, density=True, alpha=0.6, color='C0', label='non modulated cells')
                ax1.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
                # Perform  t-test
                _, p_value1 = stats.ttest_1samp(modulated, 0)
                _, p_value2 = stats.ttest_1samp(non_modulated, 0)
                star1 = groupAnalysis.starMultiplier(p_value1)
                star2 = groupAnalysis.starMultiplier(p_value2)
                hist1, edges1 = np.histogram(modulated, bins=30, density=True)
                peak1 = np.argmax(hist1)
                hist2, edges2 = np.histogram(non_modulated, bins=30, density=True)
                peak2 = np.argmax(hist2)
                # pdb.set_trace()
                # Add  p-value on the histograms
                ax1.annotate(f'{star1}', xy=(edges1[peak1], hist1[peak1]))#,xytext=(peak1, hist1[peak1] * 1.2),ha="center")
                ax1.annotate(f'{star2}', xy=(edges2[peak2], hist2[peak2]))#,xytext=(peak2, hist2[peak2] * 1.2),ha="center")
                # ax1.text(0.37, 0.98,f'{star}\nt={t_value:.2f}' , ha='center', va='center', transform=ax1.transAxes,style='italic', fontfamily='serif', fontsize=8, color='k')
                ax1.legend(frameon=False, fontsize=10)                # pdb.set_trace()
                # ax1.legend(frameon=False)

                self.layoutOfPanel(ax1, xLabel='PSTH Correlation coefficient', yLabel=('probability density %s' % events[e]),Leg=[1, 9])


            # ax2.legend(frameon=False)
            # for q in paw_df_swingLenPercent.values:
            #     ax2.axvline(q, ls='--', c='grey', alpha=0.2, lw=1)
            #     ax2.fill_between(0,q)
            # plt.show()
                # modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial'] > 5].index)
                # modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index)
        fname = '%s_%s_PSTH_correlation_modulated_non_modulated_cells' % (recordings, condition)
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def behavioralParameterDistribution_figure(self, df):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        fig_width = 4  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[0.1, 1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.6, hspace=0.2)
        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.05)
        gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 0], hspace=0.2)
        # gssub0b = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1,0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.5, wspace=0.3)
        for i in range(4):
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=16)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])

            paw_df=df[(df['paw']==pawList[i])]
            paw_df = paw_df.drop(paw_df[paw_df['trial'] > 5].index)

            # def percentile(n):
            #     def percentile_(x):
            #         return np.percentile(x, n)
            #     percentile_.__name__ = 'percentile_%s' % n
            #     return percentile_
            #"".agg([percentile(20),percentile(40),percentile(50),percentile(60),percentile(80)])#.reset_index()
            percentiles = paw_df['swingLength'].quantile([0,0.2, 0.4, 0.5, 0.6,0.8,1])
            # pdb.set_trace()
            ax2=plt.subplot(gssub1[i])
            plotOverallDist=True
            if plotOverallDist:
                sns.kdeplot(paw_df, x='swingLength', hue='trial', legend=True, ax=ax2)
                sns.kdeplot(paw_df, x='swingLength', legend=True, ax=ax2)
                ax2.fill_betweenx([0, 0.025], paw_df['swingLength'].min(), percentiles[0.2], color='red', alpha=0.05)
                ax2.fill_betweenx([0, 0.025], percentiles[0.2], percentiles[0.4], color='salmon', alpha=0.05)
                ax2.fill_betweenx([0, 0.025], percentiles[0.4], percentiles[0.6], color='green', alpha=0.05)
                ax2.fill_betweenx([0, 0.025], percentiles[0.6], percentiles[0.8], color='blue', alpha=0.05)
                ax2.fill_betweenx([0, 0.025], percentiles[0.8], paw_df['swingLength'].max(), color='yellow', alpha=0.05)
                ax2.text(percentiles[0.2], 0.025, '20th %', rotation=90, color='orange', fontsize=5)
                ax2.text(percentiles[0.4], 0.025, '40th %', rotation=90, color='green', fontsize=5)
                ax2.text(percentiles[0.6], 0.025, '60th %', rotation=90, color='blue', fontsize=5)
                ax2.text(percentiles[0.8], 0.025, '80th %', rotation=90, color='olive', fontsize=5)
                ax2.legend(title='trials', loc='upper right', frameon=False, labels=['5', '4', '3', '2', '1'])
                ax2.set_xlim(0,200)
            else:
                sns.kdeplot(paw_df, x='swingLength', hue='trial', legend=True, ax=ax2)

                ax2.legend(title='trials', loc='upper right', frameon=False, labels=['5', '4', '3', '2', '1'])
                ax2.fill_betweenx([0, 0.006], paw_df['swingLength'].min(), percentiles[0.2], color='red', alpha=0.05)
                ax2.fill_betweenx([0, 0.006], percentiles[0.2], percentiles[0.4], color='salmon', alpha=0.05)
                ax2.fill_betweenx([0, 0.006], percentiles[0.4], percentiles[0.6], color='green', alpha=0.05)
                ax2.fill_betweenx([0, 0.006], percentiles[0.6], percentiles[0.8], color='blue', alpha=0.05)
                ax2.fill_betweenx([0, 0.006], percentiles[0.8], paw_df['swingLength'].max(), color='yellow', alpha=0.05)


                # add labels
                ax2.text(percentiles[0.2], 0.006, '20th %', rotation=90, color='orange',fontsize=5)
                ax2.text(percentiles[0.4], 0.006, '40th %', rotation=90, color='green', fontsize=5)
                ax2.text(percentiles[0.6], 0.006, '60th %', rotation=90, color='blue',fontsize=5)
                ax2.text(percentiles[0.8], 0.006, '80th %', rotation=90, color='olive',fontsize=5)

        fname = 'ephy_swing_length_distribution'
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def PSTHGroupFigure_before_after_event_modulation_multiple_cond(self, recordings, df_psth, df_cells, conditionList, event,controlVar):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        fig_width = 22  # width in inches
        fig_height = 38  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[0.1, 1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.2)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.0025, right=0.98, top=0.98, bottom=0.03)


        for i in range(len(pawList)):
            gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 0], hspace=0.2)
            gssub1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 1], hspace=0.15)
            ax0 = plt.subplot(gssub0[0])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=25)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]
            # pdb.set_trace()
            # loop through the rows of the DataFrame and drop line when ['before_swingOnset_z-score_AUC_0.1'] column is empty to skip recordings with no strides that meet conditions
            for l in range(len(conditionList)):
                emptyColumnKey='before_swingOnset_z-score_AUC_0.1_%s'%conditionList[l]
                empty_cells = paw_df[emptyColumnKey].isna()
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                        # paw_psth_df=paw_psth_df.drop(index)
            gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1[0], hspace=0.06, height_ratios=[1,6])
            gssub3 = gridspec.GridSpecFromSubplotSpec(len(conditionList), 2, subplot_spec=gssub2[0], hspace=0.8, wspace=0.1)#, height_ratios=[1,1,1,1,1,1,12,12])
            times = ['before_', 'after_']
            modCellsList = {}
            modCellsList['before'] = []
            modCellsList['after'] = []
            for t in range(2):
                for l in range(len(conditionList)):
                    catList=['↓','↑','-']
                    time = times[t]
                    ax2 = plt.subplot(gssub3[l, t])
                    # print('we are here!!!!!!',time, event, i)
                    modCells_Id_allSteps, modCells_count_allSteps, counts_allSteps = groupAnalysis_psth.getModulatedcell_Id_count(
                        paw_df, catList, time, event, conditionList[0])
                    #get ids and counts of modulated cells
                    modCells_Id, modCells_count, counts=groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList, time, event, conditionList[l])
                    if conditionList[l]!='allSteps':
                        if t==0:
                            modCellsList['before'].append(modCells_Id)
                        if t==1:
                            modCellsList['after'].append(modCells_Id)


                    values = counts.values
                    labels = counts.index
                    percent = values / np.sum(values) * 100
                    bottom_pos = 0
                    label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
                    label_colorsList=['cadetblue','indianred','lightsteelblue']
                    for k, label in enumerate(labels):
                        # pdb.set_trace()
                        bar_width = percent[k]
                        bar_left = bottom_pos
                        ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],label=f'{label} {round(percent[k], 1)}% ({values[k]})')
                        # ax1.set_title(('modulation %s %s' % (time, event)) if i == 0 else '', loc='center')

                        # ax2.annotate(f'{label} {round(percent[l], 1)}%  ({values[l]}) ',(bar_left + bar_width / 2, -0.2), color='white', fontsize=12)
                        ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),bbox_to_anchor=(0.05, 0.75, 0.85, 0.1), frameon=False, fontsize=8)#, labelcolor=label_colors_List)
                        bottom_pos += bar_width
                        ax2.get_yaxis().set_visible(False)
                        ax2.get_xaxis().set_visible((True if l==5 else False))
                        ax2.spines[['left','right','bottom', 'top']].set_visible(False)
                        strideNbKey=f'stride_nb_{conditionList[l]}'
                        averageStrideNumber=paw_df[strideNbKey].mean()
                        # pdb.set_trace()
                    print(conditionList[l])
                    ax2.text(0.5, 1.4, (f'modulation {time[:-1]} {event[:-5]} onset {conditionList[l]} ({averageStrideNumber:.0f} strides)'), ha='center',va='center', transform=ax2.transAxes, fontsize=10)
                    zscore = ['AUC', 'peak']
                    intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                    interval = intervals[2]
                    zscorePar = zscore[0]
                    # zscore_key = ['before_%s_z-score_%s_%s_%s' % (event, zscorePar, interval,conditionList[l]),
                    #               'after_%s_z-score_%s_%s_%s' % (event, zscorePar, interval, conditionList[l])]
                    # print(zscore_key)
                    # pdb.set_trace()
                    for c in range(3):
                        modulated_mask = paw_df['cell_global_Id'].isin(modCells_Id[catList[c]])
                        modulated_paw_df = paw_df.loc[modulated_mask]
                        modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial'] >5].index)
                        modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index)
                        #get a df specificaly for allsteps
                        modulated_mask_allSteps = paw_df['cell_global_Id'].isin(modCells_Id_allSteps[catList[c]])
                        modulated_paw_df_allSteps= paw_df.loc[modulated_mask_allSteps]
                        modulated_paw_df_allSteps = modulated_paw_df_allSteps.drop(modulated_paw_df_allSteps[modulated_paw_df_allSteps['trial'] >5].index)
                        modulated_paw_df_allSteps = modulated_paw_df_allSteps.drop(modulated_paw_df_allSteps[modulated_paw_df_allSteps['cell_trial_Nb'] == 1].index)
                        # if recordings=='MLI':
                        #     modulated_paw_df_visual = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_global_Id'] == 25].index)
                        # else:
                        modulated_paw_df_visual=modulated_paw_df

                        gssub6 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], hspace=0.1, wspace=0.3, height_ratios=[1.1,1])
                        gssub7 = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gssub6[0,t],
                                                                  hspace=0.5, wspace=0.3)
                        gssub8 = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gssub6[1,t],
                                                                  hspace=0.5, wspace=0.3)

                        ax3=plt.subplot(gssub7[l,c])
                        ax3b=plt.subplot(gssub7[6,c])

                        #compute average across trial and cells
                        modulated_paw_df_allConditions_average_event=modulated_paw_df.groupby(by=['cell_global_Id', 'trial'])[
                                [f'{time}{event}_z-score_AUC_0.2_{conditionList[1]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[2]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[3]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[4]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[5]}']].mean().reset_index()
                        modulated_paw_df_allConditions_average_event['mean']=modulated_paw_df_allConditions_average_event[[f'{time}{event}_z-score_AUC_0.2_{conditionList[1]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[2]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[3]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[4]}',
                                 f'{time}{event}_z-score_AUC_0.2_{conditionList[5]}']].mean(axis=1)
                        # modulated_paw_df_allConditions_average_event.to_csv(
                        #     groupAnalysisDir + 'average_test_AUC.csv')
                        # beforeAvg.to_csv(
                        #     groupAnalysisDir + 'before_average_test_AUC.csv')

                        zscore_key = ['before_%s_z-score_%s_%s_%s' % (event, zscorePar, interval, conditionList[l]),
                                      'after_%s_z-score_%s_%s_%s' % (event, zscorePar, interval, conditionList[l])]
                        # modulated_paw_df_normalized= modulated_paw_df.assign(score_AUC_norm=modulated_paw_df.groupby(['cell_global_Id']).apply(lambda x: x.loc[x.trial == 1, zscore_key[t]].values[0]))
                        # pdb.set_trace()

                        zscore_key_avg=f'{time}{event}_z-score_AUC_0.2_{conditionList[1]}'
                        sns.lineplot(modulated_paw_df_visual, x='trial', y=zscore_key[t], hue='cell_global_Id', errorbar='se', ax=ax3, legend=False, alpha=0.06)
                        sns.lineplot(modulated_paw_df, x='trial', y=zscore_key[t], hue=None, errorbar='se', ax=ax3,legend=True, color=label_colorsList[c])
                        sns.lineplot(modulated_paw_df_allConditions_average_event, x='trial', y=zscore_key_avg, hue='cell_global_Id', errorbar=None, ax=ax3b,legend=False, alpha=0.06)
                        sns.lineplot(modulated_paw_df_allConditions_average_event, x='trial', y=zscore_key_avg, hue=None, errorbar='se', ax=ax3b,legend=False, color=label_colorsList[c])
                        zscore_key1 = 'zScoreAUC_' + times[t] + event+conditionList[l].replace('-','_')
                        modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                        zscore_key2 = 'zScoreAUC_' + times[t] + event + conditionList[l].replace('-', '_')
                        modulated_paw_df_allConditions_average_event[zscore_key2] = modulated_paw_df_allConditions_average_event.loc[:, zscore_key_avg]
                        if len(modulated_paw_df) > 3:
                            mixedML_results, mixeML_trial_pvalue, tukey = groupAnalysis_psth.perform_mixedlm(
                                modulated_paw_df, zscore_key1, 'trial', 'cell_global_Id')
                            #mixedML_results_avg, mixeML_trial_pvalue_avg, tukey = groupAnalysis_psth.perform_mixedlm(
                                #modulated_paw_df_allConditions_average_event, zscore_key2, 'trial', 'cell_global_Id')
                            star_trial = groupAnalysis.starMultiplier(mixeML_trial_pvalue)
                            #star_trial_avg=groupAnalysis.starMultiplier(mixeML_trial_pvalue_avg)
                            ax3.set_title((f'PSTH z-score AUC {conditionList[l]}'if c==1 else ''), fontsize=10)
                            ax3b.set_title((f'PSTH z-score AUC average across conditions' if c == 1 else ''), fontsize=10)
                            ax3.text(0.9, 0.97, '%s' % (star_trial), ha='center', va='center', transform=ax3.transAxes,
                                     style='italic', fontfamily='serif', fontsize=10, color='k')
                       # ax3b.text(0.9, 0.97, '%s' % (star_trial_avg), ha='center', va='center', transform=ax3b.transAxes,
                                 #style='italic', fontfamily='serif', fontsize=10, color='k')
                        self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(
                            f' {time[:-1]} {event[:-5]} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''),
                                           Leg=[1, 9])
                        self.layoutOfPanel(ax3b, xLabel='Trial', yLabel=(
                            f' {time[:-1]} {event[:-5]} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''),
                                           Leg=[1, 9])
                        majorLocator_x = MultipleLocator(1)
                        ax3.xaxis.set_major_locator(majorLocator_x)
                        allBehavior_par = ['swingDuration', 'swingLength', 'swingSpeed', 'stanceDuration',
                                        'stanceOnsetMedian']

                        #average the behavior across the intervals
                        modulated_paw_df_allConditions_average_event_behaviorpar_interval=modulated_paw_df.groupby(by=['cell_global_Id', 'trial'])[
                                [f'{controlVar}_{conditionList[1]}',
                                 f'{controlVar}_{conditionList[2]}',
                                 f'{controlVar}_{conditionList[3]}',
                                 f'{controlVar}_{conditionList[4]}',
                                 f'{controlVar}_{conditionList[5]}']].mean().reset_index()
                        modulated_paw_df_allConditions_average_event_behaviorpar_interval['mean']=modulated_paw_df_allConditions_average_event_behaviorpar_interval[[f'{controlVar}_{conditionList[1]}',
                                 f'{controlVar}_{conditionList[2]}',
                                 f'{controlVar}_{conditionList[3]}',
                                 f'{controlVar}_{conditionList[4]}',
                                 f'{controlVar}_{conditionList[5]}']].mean(axis=1)

                        # modulated_paw_df_allConditions_average_event_behaviorpar_interval.to_csv(groupAnalysisDir + 'average_test_behavior.csv')

                        behavior_par=[f'{controlVar}_{conditionList[l]}']
                        for p in range(len(behavior_par)):
                            ax4 = plt.subplot(gssub8[l, c])
                            ax4b = plt.subplot(gssub8[6, c])
                            modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                            y = modulated_paw_df[behavior_par[p]]
                            x = modulated_paw_df[zscore_key[t]]

                            y_avg=modulated_paw_df_allConditions_average_event_behaviorpar_interval['mean']
                            x_avg=modulated_paw_df_allConditions_average_event['mean']
                            try:
                                slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                             alternative='two-sided')
                                slope, intercept, r_value_avg, p_value_avg, sterr = stats.linregress(x_avg, y_avg,
                                                                                             alternative='two-sided')
                                r2 = np.square(r_value)
                                r2_avg = np.square(r_value_avg)
                            except ValueError:
                                pass
                            if (p_value > 0.05) or (r2 < 0.1) or (r2 == 1):
                                alpha = 0.1
                            else:
                                alpha = 0.8
                            if (p_value_avg > 0.05) or (r2_avg < 0.1) or (r2_avg == 1):
                                alpha_avg = 0.1
                            else:
                                alpha_avg = 0.8
                            sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                                        line_kws={'alpha': alpha})
                            sns.regplot(x=x_avg, y=y_avg, ax=ax4b, color=label_colorsList[c], scatter_kws={'alpha': alpha_avg},
                                        line_kws={'alpha': alpha_avg})
                            ax4.text(0.7, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",
                                     transform=ax4.transAxes)
                            ax4b.text(0.7, 0.05, f"r = {r_value_avg:.2f}\nr² = {r2_avg:.2f}\np = {p_value_avg:.2f}",
                                     transform=ax4b.transAxes)
                            ax4.set_title((f'swing {behavior_par[p][5:]} vs z-score AUC {conditionList[l]}'if c==1 else ''), fontsize=10)
                            ax4b.set_title((f'{controlVar} average across condition vs z-score AUC average across conditions' if c == 1 else ''),fontsize=10)
                            if p == 0:
                                self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if l== 5 else ''),
                                                   yLabel=(f'swing {controlVar[5:]}' if (c == 0) else ''),
                                                   Leg=[1, 9])
                            if p == 0:
                                self.layoutOfPanel(ax4b, xLabel=('zscore_%s' % zscorePar if l== 5 else ''),
                                                   yLabel=(f'swing {controlVar[5:]}' if (c == 0) else ''),
                                                   Leg=[1, 9])
            down_arrow_cells = np.concatenate([d['↓'] for d in modCellsList['before'] if '↓' in d])
            up_arrow_cells = np.concatenate([d['↑'] for d in modCellsList['before'] if '↑' in d])



            fname = f'{pawList[i]}_{event}_{recordings[3:]}_ephys_psth_group_analysis_{controlVar}_intervals_allTrials'

            # plt.savefig(fname + '.png')
            # plt.show()
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def fractionOfCellsModulated(self, recordings, df_psth, df_cells, condition):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        fig_width = 8  # width in inches
        fig_height = 4  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False,
                  'axes.spines.right': False  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               #width_ratios=[0.1, 1, 5]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.5)
        modulatedCellDic={}

        for i in range(4):

            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            # paw_df = paw_df.dropna(subset=['before_swingOnset_z-score_AUC_0.1'])
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]

            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)

            events=['swingOnset', 'stanceOnset']
            for e in range(2):
                event=events[e]
                modulatedCellDic[f'{pawList[i]}_{event}'] = []

            # get the ids and df of all modulated cells
                categoriesKey = f'modulation_category_{event}'
            # get the ids and df of all modulated cells
                cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df, paw_psth_df,
                                                                                 event)
                modulatedCellDic[f'{pawList[i]}_{event}'].append(cells_id)


        for e in  reversed(range(2)):
            ax1 = plt.subplot(gssub0[e])
            if e==0:
                alpha = 0.4
            else:
                alpha = 0.8
            for i  in reversed(range(4)):
                modulatedNb=len(modulatedCellDic[f'{pawList[i]}_{events[e]}'][0][0])
                total=len(modulatedCellDic[f'{pawList[i]}_{events[e]}'][0][0])+len(modulatedCellDic[f'{pawList[i]}_{events[e]}'][0][1])
                modulatedFrac=modulatedNb/total*100
                ax1.barh(pawList[i], [100,100,100,100], color='white', edgecolor='black', alpha=alpha)
                ax1.barh(pawList[i], modulatedFrac, color=f'C{i}', edgecolor='black', alpha=alpha)
                ax1.annotate(f'{modulatedNb}/{total}',(modulatedFrac+2,pawList[i]))
                ax1.spines['left'].set_visible(False)
                ax1.set_xlim(0,100)
                ax1.set_title(f'{events[e][:-5]}',fontsize=14)


        fname = 'ephys_psth_modulated_cells_fraction'


        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
        plt.savefig(self.figureDirectory + '/' + fname + '.png')
        # pdb.set_trace()
            # # calculate correlation distribution
            # times = ['before_', 'after_']
            # for t in range(2):
            #     catList = ['↓', '↑', '-']
            #     time = times[t]
            #     ax2 = plt.subplot(gssub5[0, t])
            #     # print('we are here!!!!!!',time, event, i)
            #     # get ids and counts of modulated cells
            #     modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df,
            #                                                                                        catList,
            #                                                                                        time, event,
            #                                                                                        condition=None)
            #
            #     values = counts.values
            #     labels = counts.index
            #     percent = values / np.sum(values) * 100
            #     bottom_pos = 0
            #     label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
            #     label_colorsList = ['cadetblue', 'indianred', 'lightsteelblue']
            #     for l, label in enumerate(labels):
            #         # pdb.set_trace()
            #         bar_width = percent[l]
            #         bar_left = bottom_pos
            #         ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],
            #                  label=f'{label} {round(percent[l], 1)}% ({values[l]})')
            #         # ax1.set_title(('modulation %s %s' % (time, event)) if i == 0 else '', loc='center')
            #
            #         # ax2.annotate(f'{label} {round(percent[l], 1)}%  ({values[l]}) ',(bar_left + bar_width / 2, -0.2), color='white', fontsize=12)
            #         ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
            #                    bbox_to_anchor=(0.05, 0.7, 0.85, 0.1), frameon=False,
            #                    fontsize=8)  # , labelcolor=label_colors_List)
            #         bottom_pos += bar_width
            #         ax2.get_yaxis().set_visible(False)
            #         # ax2.get_xaxis().set_visible(False)
            #         ax2.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
            #     ax2.text(0.5, 1.6, (f'modulation {time[:-1]} {event[:-5]} onset'), ha='center', va='center',
            #              transform=ax2.transAxes, fontsize=10)
    def PSTHGroupFigure_before_after_event_early_vs_late(self, recordings, df_psth, df_cells, condition, event):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        fig_width = 22  # width in inches
        fig_height = 50  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'font.size': 8, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3,  # ,
                               width_ratios=[0.1, 1, 5]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.03)
        gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        for i in range(4):
            # paw labels panel
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=25)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])
            gssub2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub1[i], hspace=0.1, wspace=0.5,
                                                      height_ratios=[1, 2, 1],
                                                      width_ratios=[1])  # , height_ratios=[1,1])
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            # paw_df = paw_df.dropna(subset=['before_swingOnset_z-score_AUC_0.1'])
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]

            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)
                    # pdb.set_trace()
            # events = ['stanceOnset', 'swingOnset']
            # for e in range(len(events)):
            #     event=events[e]
            ax1 = plt.subplot(gssub2[1])

            # get the ids and df of all modulated cells
            cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df,
                                                                                                  paw_psth_df,
                                                                                                  event)

            # calculate correlation distribution
            modulated, non_modulated = groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(cells_id,
                                                                                                  modCells_df,
                                                                                                  event)
            # plot correlation distribution
            ax1.hist(modulated, bins=30, density=True, alpha=0.6, color='maroon', label=f'modulated {recordings}')
            ax1.hist(non_modulated, bins=30, density=True, alpha=0.6, color='steelblue',
                     label=f'non modulated {recordings}')

            ax1.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
            ax1.legend(loc='best', frameon=False, fontsize=10)
            ax1.set_xlim(-0.5, 1)
            ax1.set_ylim(0, 3)
            ax1.set_title(f'{event[:-5]} onset', fontsize=10)
            self.layoutOfPanel(ax1, xLabel='PSTH Correlation coefficient across trials',
                               yLabel=('probability density'), Leg=[1, 9])
            gssub3 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 2], hspace=0.22,
                                                      height_ratios=[1, 1, 1, 1])
            gssub4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub3[i], hspace=0.3)
            gssub5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub4[0], hspace=0.15,
                                                      height_ratios=[1, 12])
            times = ['before_', 'after_']
            for t in range(2):
                catList = ['↓', '↑', '-']
                time = times[t]
                ax2 = plt.subplot(gssub5[0, t])
                # print('we are here!!!!!!',time, event, i)
                # get ids and counts of modulated cells
                modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
                                                                                                   time, event,
                                                                                                   condition=None)

                values = counts.values
                labels = counts.index
                percent = values / np.sum(values) * 100
                bottom_pos = 0
                label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
                label_colorsList = ['cadetblue', 'indianred', 'lightsteelblue']
                for l, label in enumerate(labels):
                    # pdb.set_trace()
                    bar_width = percent[l]
                    bar_left = bottom_pos
                    ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],
                             label=f'{label} {round(percent[l], 1)}% ({values[l]})')
                    # ax1.set_title(('modulation %s %s' % (time, event)) if i == 0 else '', loc='center')

                    # ax2.annotate(f'{label} {round(percent[l], 1)}%  ({values[l]}) ',(bar_left + bar_width / 2, -0.2), color='white', fontsize=12)
                    ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                               bbox_to_anchor=(0.05, 0.7, 0.85, 0.1), frameon=False,
                               fontsize=8)  # , labelcolor=label_colors_List)
                    bottom_pos += bar_width
                    ax2.get_yaxis().set_visible(False)
                    # ax2.get_xaxis().set_visible(False)
                    ax2.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
                ax2.text(0.5, 1.6, (f'modulation {time[:-1]} {event[:-5]} onset'), ha='center', va='center',
                         transform=ax2.transAxes, fontsize=10)
                zscore = ['AUC', 'peak']
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[2]
                zscorePar = zscore[0]
                zscore_key = ['before_%s_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%s_z-score_%s_%s' % (event, zscorePar, interval)]
                for c in range(3):
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells_Id[catList[c]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    # modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial'] > 5].index).reset_index()
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                    # if recordings=='MLI':
                    #     modulated_paw_df_visual = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_global_Id'] == 25].index)
                    # else:
                    modulated_paw_df_visual = modulated_paw_df

                    gssub6 = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gssub5[1, t], hspace=1.3,
                                                              wspace=0.3, height_ratios=[6, 2, 2, 2, 2, 2, 2])
                    ax3 = plt.subplot(gssub6[0, c])
                    # sns.barplot(modulated_paw_df_visual, x='day_category', y=zscore_key[t], hue='cell_global_Id',
                    #              errorbar='se', ax=ax3,  alpha=0.06)
                    modulated_paw_df_early_late=modulated_paw_df.groupby(['cell_global_Id','trial_category'])[zscore_key[t]].mean().reset_index()
                    # pdb.set_trace()
                    # sns.boxplot(modulated_paw_df_early_late, x='trial_category', y=zscore_key[t], hue=None,  ax=ax3,
                    #            color=label_colorsList[c])
                    # sns.barplot(modulated_paw_df_early_late, x='trial_category', hue=None, y=zscore_key[t],  ax=ax3,
                    #             color=label_colorsList[c], alpha=0.001, errorbar=None)
                    sns.lineplot(modulated_paw_df_early_late, x='trial_category', hue='cell_global_Id', y=zscore_key[t],  ax=ax3,
                                color=label_colorsList[c], legend=False, alpha=0.3)
                    # sns.lineplot(modulated_paw_df_early_late, x='trial_category', errorbar='se',err_style='bars',hue=None, y=zscore_key[t],  ax=ax3,
                    #             color=label_colorsList[c], legend=False, alpha=0.8)
                    late_days=modulated_paw_df_early_late[modulated_paw_df_early_late['trial_category']=='first']
                    early_days = modulated_paw_df_early_late[modulated_paw_df_early_late['trial_category'] == 'last']

                    t_value, t_test_p_value=stats.ttest_rel(late_days[zscore_key[t]], early_days[zscore_key[t]], axis=0)

                    # ax3.plot(0,0, label='')
                    zscore_key1 = 'zScoreAUC_' + times[t] + event
                    modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    # if len(modulated_paw_df) > 3:
                    #     try:
                    #         mixedML_results, mixeML_trial_pvalue, tukey = groupAnalysis_psth.perform_mixedlm(
                    #             modulated_paw_df, zscore_key1, 'day_category_bin', 'cell_global_Id')
                    #     except:
                    #         mixeML_trial_pvalue=1
                        # linReg_results, LinReg_trial_pvalue=groupAnalysis_psth.perform_linear_regression(modulated_paw_df,zscore_key1, 'trial')
                        # GLM_results, pvalue, tukey = groupAnalysis_psth.perform_GLM_group_and_Tukey(modulated_paw_df, zscore_key1, 'trial', 'cell_global_Id')
                        # results, p_value, tukey= groupAnalysis_psth.perform_GLM_and_Tukey(modulated_paw_df, zscore_key1, 'trial')
                        # modulated_paw_df = modulated_paw_df[modulated_paw_df['trial'].notna()]
                        # df_average_trial = modulated_paw_df.groupby(['trial', 'cell_global_Id'])[zscore_key1].agg(
                        #     ['mean']).reset_index()

                        # tukey_trial = pairwise_tukeyhsd(df_tukey_trial['mean'], df_tukey_trial['trial'])

                        # print('p_values',p_value, tukey)

                        # print(event, time, i, catList[c],tukey_trial.summary())
                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    #
                    ax3.text(0.5, 0.99, f'{star_trial} \np={t_test_p_value:.2f}', ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=8, color=label_colorsList[c])
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(
                        f' {time[:-1]} {event[:-5]} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''),
                                       Leg=[1, 9])
                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)

                    behavior_par = ['swingDuration', 'swingLength', 'swingSpeed', 'stanceDuration',
                                    'stanceOnsetStd', 'stanceOnsetMedian']
                    # behavior_par = ['acceleration', 'mean_acceleration', 'mean_deceleration', 'acc_duration','dec_duration']
                    # behavior_par = ['swingDuration', 'swingLength']
                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p, c])

                        modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df[zscore_key[t]]
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            r2 = np.square(r_value)
                        except ValueError:
                            pass
                        if (p_value > 0.05) or (r2 < 0.1) or (r2 == 1):
                            alpha = 0.1
                        else:
                            alpha = 0.8
                        sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                                    line_kws={'alpha': alpha})
                        ax4.text(0.7, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",
                                 transform=ax4.transAxes)
                        if p == 0:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''),
                                               yLabel=(f'swing {behavior_par[p][5:]} (s)' if (c == 0) else ''),
                                               Leg=[1, 9])
                        elif p == 1:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''),
                                               yLabel=(f'swing {behavior_par[p][5:]} (pixel)' if (c == 0) else ''),
                                               Leg=[1, 9])
                        elif p == 2:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'swing {behavior_par[p][5:]} (pixel/s)' if (c == 0) else ''), Leg=[1, 9])
                        elif p == 3:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'stance {behavior_par[p][6:]} (s)' if (c == 0) else ''), Leg=[1, 9])
                        elif p == 4:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'stance {behavior_par[p][6:]} (s)' if (c == 0) else ''), Leg=[1, 9])
        fname = '%s_%s_ephys_psth_group_analysis_early_late%s' % (event, recordings[3:], condition)

        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def psthGroupFigure_early_vs_late_short(self, recordings, df_psth, df_cells, condition, event):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        pawNb = 2
        fig_width = 24  # width in inches
        fig_height = 15*pawNb  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 12, 'xtick.labelsize': 12,
                  'ytick.labelsize': 12,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,  # ,
                               width_ratios=[0.1, 1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.005, right=0.98, top=0.92, bottom=0.07)
        gssub0 = gridspec.GridSpecFromSubplotSpec(pawNb, 1, subplot_spec=gs[0, 0], hspace=0.2)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(pawNb, 1, subplot_spec=gs[0, 1], hspace=0.15)
        for i in range(pawNb):
            # paw labels panel
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=25)
            self.layoutOfPanel(ax0, xyInvisible=[True, True])
            # gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1[i], hspace=0.1, wspace=0.5,
            #                                           height_ratios=[1, 1],
            #                                           width_ratios=[1])  # , height_ratios=[1,1])
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            # paw_df = paw_df.dropna(subset=['before_swingOnset_z-score_AUC_0.1'])
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]

            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)
                    # pdb.set_trace()
            # events = ['stanceOnset', 'swingOnset']
            # for e in range(len(events)):
            #     event=events[e]
            # ax1 = plt.subplot(gssub2[1])
            #
            # # get the ids and df of all modulated cells
            # cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df,
            #                                                                                       paw_psth_df,
            #                                                                                       event)
            #
            # # calculate correlation distribution
            # modulated, non_modulated = groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(cells_id,
            #                                                                                       modCells_df,
            #                                                                                       event)
            # # plot correlation distribution
            # ax1.hist(modulated, bins=30, density=True, alpha=0.6, color='maroon', label=f'modulated {recordings}')
            # ax1.hist(non_modulated, bins=30, density=True, alpha=0.6, color='steelblue',
            #          label=f'non modulated {recordings}')
            #
            # ax1.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
            # ax1.legend(loc='best', frameon=False, fontsize=10)
            # ax1.set_xlim(-0.5, 1)
            # ax1.set_ylim(0, 3)
            # ax1.set_title(f'{event[:-5]} onset', fontsize=10)
            # self.layoutOfPanel(ax1, xLabel='PSTH Correlation coefficient across trials',
            #                    yLabel=('probability density'), Leg=[1, 9])
            gssub3 = gridspec.GridSpecFromSubplotSpec(pawNb, 1, subplot_spec=gs[0, 1], hspace=0.15)
            gssub4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub3[i], hspace=0.1)
            gssub5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub4[0], hspace=0.1,
                                                      height_ratios=[1, 20])
            times = ['before_', 'after_']
            for t in range(2):
                catList = ['↓', '↑', '-']
                time = times[t]
                ax2 = plt.subplot(gssub5[0, t])
                # print('we are here!!!!!!',time, event, i)
                # get ids and counts of modulated cells
                modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
                                                                                                   time, event,
                                                                                                   condition=None)

                values = counts.values
                labels = counts.index
                percent = values / np.sum(values) * 100
                bottom_pos = 0
                label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
                label_colorsList = ['cadetblue', 'indianred', 'lightsteelblue']
                for l, label in enumerate(labels):
                    # pdb.set_trace()
                    bar_width = percent[l]
                    bar_left = bottom_pos
                    ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],
                             label=f'{label} {round(percent[l], 1)}% ({values[l]})')
                    # ax1.set_title(('modulation %s %s' % (time, event)) if i == 0 else '', loc='center')

                    # ax2.annotate(f'{label} {round(percent[l], 1)}%  ({values[l]}) ',(bar_left + bar_width / 2, -0.2), color='white', fontsize=12)
                    ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                               bbox_to_anchor=(0.05, 0.8, 0.9, 0.1), frameon=False,
                               fontsize=14)  # , labelcolor=label_colors_List)
                    bottom_pos += bar_width
                    ax2.get_yaxis().set_visible(False)
                    # ax2.get_xaxis().set_visible(False)
                    ax2.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
                ax2.text(0.5, 2, (f'modulation {time[:-1]} {event} onset'), ha='center', va='center',
                         transform=ax2.transAxes, fontsize=16)
                zscore = ['AUC', 'peak']
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[0]
                zscorePar = zscore[0]
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                for c in range(3):
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells_Id[catList[c]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    # modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['trial'] > 5].index).reset_index()
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()

                    behavior_par = ['swingDuration', 'swingLength', 'swingSpeed', 'stanceDuration',
                                    'stanceOnsetStd', 'stanceOnsetMedian']
                    behavior_par = ['swingDuration', 'swingLength',  'stepLength']# 'stanceDuration',
                                   # 'stanceOnsetStd', 'stanceOnsetMedian']
                    hr = np.full(len(behavior_par) + 1, 2)
                    hr[0] = 2.8
                    gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) + 1, 3, subplot_spec=gssub5[1, t],
                                                              hspace=0.5,
                                                              wspace=0.3, height_ratios=hr)
                    ax3 = plt.subplot(gssub6[0, c])
                    # sns.barplot(modulated_paw_df_visual, x='day_category', y=zscore_key[t], hue='cell_global_Id',
                    #              errorbar='se', ax=ax3,  alpha=0.06)
                    modulated_paw_df_early_late=modulated_paw_df.groupby(['cell_global_Id','trial_category'])[zscore_key[t]].mean().reset_index()
                    # pdb.set_trace()
                    # sns.boxplot(modulated_paw_df_early_late, x='trial_category', y=zscore_key[t], hue=None,  ax=ax3,
                    #            color=label_colorsList[c])

                    # sns.barplot(modulated_paw_df_early_late, x='trial_category', hue=None, y=zscore_key[t],  ax=ax3,
                    #             color=label_colorsList[c], alpha=0.001, errorbar=None)
                    sns.lineplot(modulated_paw_df_early_late, x='trial_category', hue='cell_global_Id', y=zscore_key[t],  ax=ax3,
                                color=label_colorsList[c], legend=False, alpha=0.5)
                    # sns.lineplot(modulated_paw_df_early_late, x='trial_category', errorbar='se',err_style='bars',hue=None, y=zscore_key[t],  ax=ax3,
                    #             color=label_colorsList[c], legend=False, alpha=0.8)
                    early_days=modulated_paw_df_early_late[modulated_paw_df_early_late['trial_category']=='first']
                    late_days = modulated_paw_df_early_late[modulated_paw_df_early_late['trial_category'] == 'last']
                    try:
                        t_value, t_test_p_value=stats.ttest_rel(late_days[zscore_key[t]], early_days[zscore_key[t]], axis=0)
                    except ValueError:
                        pass

                    ax3.set_title((f'PSTH Z-score AUC first vs last trial ' if c==1 else ''))
                    zscore_key1 = 'zScoreAUC_' + times[t] + event
                    modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]

                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    #
                    ax3.text(0.5, 0.96, f'{star_trial} \np={t_test_p_value:.2f}', ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=11, color=label_colorsList[c])
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(
                        f' {time[:-1]} {event[:-5]} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''),
                                       Leg=[1, 9])

                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)

                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p, c])

                        modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df[zscore_key[t]]



                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            r2 = np.square(r_value)
                        except ValueError:
                            pass
                        if (p_value > 0.05) or (r2 < 0.05) or (r2 == 1):
                            alpha = 0.1
                        else:
                            alpha = 0.8
                        sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                                    line_kws={'alpha': alpha})
                        ax4.text(0.50, 0.05, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",
                                 transform=ax4.transAxes, fontsize=10)
                        if p!=2:
                            ax4.set_title((f'swing {behavior_par[p][5:]}  vs PSTH Z-score AUC' if c==1  else ''))
                        else:
                            ax4.set_title((f'{behavior_par[p]}  vs PSTH Z-score AUC' if c == 1 else ''))
                        if behavior_par[p]=='swingLengthLinear':
                            ax4.set_title((f'Swing Length  vs PSTH Z-score AUC' if c == 1 else ''))
                        if p == 0:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''),
                                               yLabel=(f'swing {behavior_par[p][5:]} (s)' if (c == 0) else ''),
                                               Leg=[1, 9])
                        elif p == 1:
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''),
                                               yLabel=(f'swing {behavior_par[p][5:]} (cm)' if (c == 0) else ''),
                                               Leg=[1, 9])
                        else :
                            self.layoutOfPanel(ax4, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                                f'{behavior_par[p]}' if (c == 0) else ''), Leg=[1, 9])
                    # trialCat=['early', 'late']
                    # for l in range(2):
                    #     y0 = modulated_paw_df[modulated_paw_df['trial_type'] == trialCat[l]]['swingLength']
                    #     x0 = modulated_paw_df[modulated_paw_df['trial_type'] == trialCat[l]][zscore_key[t]]
                    #     ax5 = plt.subplot(gssub6[1 + len(behavior_par) + l, c])
                    #     try:
                    #         slope, intercept, r_value0, p_value0, sterr = stats.linregress(x0, y0,
                    #                                                                      alternative='two-sided')
                    #         r20 = np.square(r_value)
                    #     except ValueError:
                    #         pass
                    #     if (p_value > 0.05) or (r2 < 0.05) or (r2 == 1):
                    #         alpha = 0.1
                    #     else:
                    #         alpha = 0.8
                    #     sns.regplot(x=x0, y=y0, ax=ax5, color=label_colorsList[c], scatter_kws={'alpha': alpha},
                    #                 line_kws={'alpha': alpha})
                    #     ax5.text(0.50, 0.05, f"r = {r_value0:.2f}\nr² = {r20:.2f}\np = {p_value0:.2f}",
                    #              transform=ax5.transAxes, fontsize=10)
                    #     self.layoutOfPanel(ax5, xLabel=('zscore_%s' % zscorePar if p != 0 else ''), yLabel=(
                    #         f'{behavior_par[p]}' if (c == 0) else ''), Leg=[1, 9])

        fname = f'{event}_{recordings[3:]}_ephys_psth_group_analysis_zscore_{zscorePar}_early_late_{condition}'

        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
        # plt.savefig(self.figureDirectory + '/' + fname + '.png')
    def ephyBehavior(self, swingStanceD, ephys, pawPos, pawSpeed, ephysPSTHDict, pawNb,ephys2,pawPos2):

        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 14, 'xtick.labelsize': 12,
                  'ytick.labelsize': 12, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[1.4, 4, 3.5]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.45, hspace=0.4)


        # gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.2, wspace=0.1)

        #paw Ids
        pawList = ['FL', 'FR', 'HL', 'HR']
        #paw colors
        col = ['C0', 'C1', 'C2', 'C3']

        #swing and stance colors for PSTH
        twoCols = {'swing':'blueviolet', 'stance':'0.4'}
        firingRate = []

        #length of example trace
        startx = [20]
        xLength = 8
        #first subplot
        ax0 = plt.subplot(gs[1])
        # ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        # isis = np.diff(ephys[0][(ephys[0] > 10.) & (ephys[0] <= 52)])
        # firingRate.append(1. / np.mean(isis))
        stepParameters = dataAnalysis_psth.calculateStepPar(pawPos, swingStanceD)
        # stepParameters = dataAnalysis_psth.calculateStepParameters(pawPos, swingStanceD)
        # plot paw position

        for i in [0]:
            linearPawPos = (swingStanceD['forFit'][i][5])
            ax0.plot(pawPos[i][:, 0], pawPos[i][:, 1], c=col[i], lw=2, label=pawList[i])
            # ax0.plot(linearPawPos[:, 0], linearPawPos[:, 1], c=col[i], lw=2, label=pawList[i])
            # ax0.set_ylim(np.min(linearPawPos[:, 1]), 150)
            # speed_smoothed = gaussian_filter1d(pawSpeed[i][:, 1], 2)
            # ax1.plot(pawSpeed[i][:, 0], speed_smoothed, c=col[i], lw=2, label=pawList[i])
            # accelerationDic=groupAnalysis.calc_acceleration(pawSpeed[i][:, 1],pawSpeed[i][:, 0])
            # acceleration_smoothed = gaussian_filter1d(accelerationDic['acceleration'], 2)
            # ax2.plot(pawSpeed[i][:, 0][1:], acceleration_smoothed, c=col[i], lw=2, label=pawList[i])
            ax0.legend(frameon=False)


            ax2 = plt.subplot(gs[0])
            ax2.plot(pawPos2[i][:, 0], pawPos2[i][:, 1], c=col[i], lw=2, label=pawList[i])
            ax2.legend(frameon=False)


        #plot spiking activity
        alpha=0.7

        ax0.eventplot(ephys, lineoffsets=300, linelengths=700, linewidths=0.1, color='0.4', alpha=alpha)
        # ax1.eventplot(ephys, lineoffsets=2000, linelengths=8000, linewidths=0.1, color='0.4', alpha=alpha)
        ax2.eventplot(ephys2, lineoffsets=300, linelengths=700, linewidths=0.1, color='0.4', alpha=alpha)
        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        self.layoutOfPanel(ax2, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[True, False], Leg=[1, 9])
        ax0.set_xlim(startx[0], startx[0] + xLength)
        # ax1.set_xlim(startx[0], startx[0] + xLength)
        ax2.set_xlim(startx[0], startx[0] + xLength)
        ax0.set_ylim(np.min(pawPos[0][:, 1]), 680)
        ax2.set_ylim(np.min(pawPos[1][:, 1]), 680)
        ax0.set_title("last trial", fontsize=20)
        ax2.set_title("first trial", fontsize=20)
        fname = f'ephy_behavior_trial_1_trial_5'

        plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def ephyBehavior_PC_SS_CS(self, ephysPSTHData, ephysPSTHData_complex, conditions):

        # figure #################################
        fig_width = 40  # width in inches
        fig_height = 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 14, 'xtick.labelsize': 12,
                  'ytick.labelsize': 12, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 1,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[1.4, 4, 3.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.45, hspace=0.4)


        # gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.2, wspace=0.1)

        #paw Ids
        pawList = ['FL', 'FR', 'HL', 'HR']
        #paw colors
        col = ['C0', 'C1', 'C2', 'C3']

        #swing and stance colors for PSTH
        twoCols = {'swing':'blueviolet', 'stance':'0.4'}
        firingRate = []
        for l, condition in enumerate(conditions):
            for nDay in range(len(ephysPSTHData)):
                for nRec in  range(len(ephysPSTHData[nDay][2])):
                    ephysPSTHDict = ephysPSTHData[nDay][3][nRec][condition][events[e]]
                    ephysPSTHDict_day = ephysPSTHData[nDay][3]
                    ephysPSTHDict_complex = ephysPSTHData_complex[nDay][3][nRec][condition][events[e]]
                    swingStanceD = ephysPSTHData[nDay][2][nRec][4]
                    ephys = ephysPSTHData[nDay][2][nRec][3]
                    ephys_complex = ephysPSTHData_complex[nDay][2][nRec][3]
                    pawPos = ephysPSTHData[nDay][2][nRec][2]
                    pawSpeed = ephysPSTHData[nDay][2][nRec][5]
        #length of example trace
        startx = [30]
        xLength = 15
        #first subplot
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        isis = np.diff(ephys[0][(ephys[0] > 10.) & (ephys[0] <= 52)])
        firingRate.append(1. / np.mean(isis))
        binWidth = 1.E-3  # in sec
        spikecountwindow = 0.05
        nspikecountwindow = spikecountwindow / binWidth
        #tbins = np.linspace(0., len(eData) * dt, int(len(eData) * dt / binWidth) + 1)
        tbins = np.linspace(0.,60.,int(60/binWidth)+1,endpoint=True)

        binnedspikes, _ = np.histogram(ephys, tbins)
        spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
        # convert the convolved spike trains to units of spikes/sec
        spikesconv *= 1. / binWidth
        # stepParameters = dataAnalysis_psth.calculateStepParameters(pawPos, swingStanceD)
        # plot paw position
        for i in [0,1]:
            ax0.plot(pawPos[i][:, 0], pawPos[i][:, 1], c=col[i], lw=2, label=pawList[i])

            speed_smoothed = gaussian_filter1d(pawSpeed[i][:, 1], 3)
            ax1.plot(pawSpeed[i][:, 0], speed_smoothed, c=col[i], lw=2, label=pawList[i])
            # accelerationDic=groupAnalysis.calc_acceleration(pawSpeed[i][:, 1],pawSpeed[i][:, 0])
            # acceleration_smoothed = gaussian_filter1d(accelerationDic['acceleration'], 2)
            # ax2.plot(pawSpeed[i][:, 0][1:], acceleration_smoothed, c=col[i], lw=2, label=pawList[i])
            ax0.legend(frameon=False)

        # for i  in [2,3]:
        #     ax2 = plt.subplot(gs[2])
        #     ax2.plot(pawPos[i][:, 0], pawPos[i][:, 1], c=col[i], lw=2, label=pawList[i])
        #     ax2.legend(frameon=False)

        #plot spiking activity
        alpha=0.7
        mmax = np.max(spikesconv)
        mmin = np.min(spikesconv)
        ax0.eventplot(ephys, lineoffsets=300, linelengths=700, linewidths=0.1, color='0.4', alpha=alpha)

        # ax0.plot((tbins[1:]+tbins[:-1])/2,spikesconv,color='0.5')
        ax0.eventplot(ephys_complex, lineoffsets=300, linelengths=700, linewidths=2, color='red', alpha=alpha)
        ax1.eventplot(ephys, lineoffsets=2000, linelengths=8000, linewidths=0.1, color='0.4', alpha=alpha)
        ax1.eventplot(ephys_complex, lineoffsets=2000, linelengths=8000, linewidths=2, color='red', alpha=alpha)

        # ax2.eventplot(ephys, lineoffsets=100, linelengths=800, linewidths=0.1, color='0.4', alpha=alpha)
        # ax2.eventplot(ephys_complex, lineoffsets=100, linelengths=800, linewidths=2, color='red', alpha=alpha)
        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        self.layoutOfPanel(ax1, xLabel='time (s)', yLabel= 'x-speed (pixel/s)',
                           xyInvisible=[False, False], Leg=[1, 9])
        # self.layoutOfPanel(ax2, xLabel='time (s)', yLabel= 'x (pixel)',
        #                    xyInvisible=[False, False], Leg=[1, 9])
        ax0.set_xlim(startx[0], startx[0] + xLength)
        ax1.set_xlim(startx[0], startx[0] + xLength)
        # ax2.set_xlim(startx[0], startx[0] + xLength)
        ax0.set_ylim(np.min(pawPos[1][:, 1]), 680)
        # ax2.set_ylim(np.min(pawPos[3][:, 1]), 680)
        ax1.set_ylim(-100, 3200)
        # ax2.set_ylim(-1.4e5, 1.4e5)
        fname = f'ephy_behavior_PC_SS_CS'

        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def muscimol_group_analysis_averages(self, stridePar, strideTraj):
        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13,
                  'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios

        gs = gridspec.GridSpec(1,1)
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        #regroup data per trial, day, paw and mouse
        stridePar = stridePar.drop(stridePar[(stridePar['day'] > 9)].index).reset_index()
        # print(stridePar['keys'])
        # pdb.set_trace()
        times='day','trial'
        for time in times:
            stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
            stridePar_Recordings=stridePar_Recordings.reset_index()
            #average data per day for visualization
            stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
            stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

            # stridePar_Recordings_day_normal_speed_mice=stridePar_Recordings_day[(stridePar_Recordings_day['LinWheelSpeed_avg'] > 7) & (stridePar_Recordings_day['day']==1) ]['mouseId']
            # stridePar_Recordings_day_batch_2=stridePar_Recordings_day[~stridePar_Recordings_day['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_batch_2=stridePar_Recordings[~stridePar_Recordings['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_batch_1 = stridePar_Recordings[stridePar_Recordings['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_day_batch_1 = stridePar_Recordings_day[stridePar_Recordings_day['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]


            parameters = ['swingNumber', 'swingSpeed', 'swingSpeed_Max','acceleration', 'swingLength','swingLengthLinear',
                          'swingDuration', 'stanceDuration', 'strideDuration',  'dutyFactor','twoRungsFraction', 'rungsCrossed_std', 'rungCrossed',
                          'indecisiveFraction', 'wheelDistance', 'LinWheelSpeed_avg', 'frequency','iqr_25_75_ref_FL', 'iqr_70_90_ref_FL', 'stanceOnStd_ref_FL', 'swingOnMedian_FL', 'stanceOnMedian_ref_FL', 'stanceOn_iqr_25_75_ref_FL', 'stanceOn_iqr_70_90_ref_FL']

            # parameters_Y = ['swing number (avg.)','swing duration (s)', 'swing speed (cm/s)', 'indecisive strides fraction','stance duration (s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed']
            # 		iqr_25_75_ref_FL	iqr_70_90_ref_FL	stanceOnStd_ref_FL	swingOnMedian_FL	stanceOnMedian_ref_FL	stanceOn_iqr_25_75_ref_FL	stanceOn_iqr_70_90_ref_FL	iqr_25_75_ref_FR	iqr_70_90_ref_FR	stanceOnStd_ref_FR	swingOnMedian_FR	stanceOnMedian_ref_FR	stanceOn_iqr_25_75_ref_FR	stanceOn_iqr_70_90_ref_FR


            for par in range(len(parameters)):

                gssub1 = gridspec.GridSpecFromSubplotSpec(int(len(parameters)/3), 4, subplot_spec=gs[0], hspace=0.4, wspace=0.35)
                ax1 = plt.subplot(gssub1[par])
                pars_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par],treatments=True)
                sns.lineplot(data=stridePar_Recordings_day, x=time, y=parameters[par], hue='treatment', hue_order=['tdTomato', 'opsin'], style='treatment', style_order=['tdTomato', 'opsin'],
                             errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                             err_kws={'capsize': 3, 'linewidth': 1}, palette=['black','red'], ax=(ax1 ), marker='o')
                sns.lineplot(data=stridePar_Recordings_day, x=time, y=parameters[par], hue='mouseId', style='treatment', style_order=['tdTomato', 'opsin'],
                             errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)], alpha=0.1,
                             err_kws={'capsize': 3, 'linewidth': 1}, palette=['red','black'], ax=(ax1))
                interactionTerm=f'{time}:treatment[T.tdTomato]'
                ax1.text(1, 0.31, f'{pars_summary["stars"]["all"][interactionTerm].replace("*", "°")}',
                         ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                         fontsize=15, color='k')
                ax1.legend([f'tdTomato {pars_summary["stars"]["tdTomato"]["day"]} {pars_summary["stars"]["tdTomato"]["trial"].replace("*","#")}', f'opsin {pars_summary["stars"]["opsin"]["day"]} {pars_summary["stars"]["opsin"]["trial"].replace("*","#")}'],loc='best',frameon=False)
                ax1.xaxis.set_major_locator(MultipleLocator(1))

                self.layoutOfPanel(ax1, xLabel=('session' if time=='day' else 'trials'))
            fname = f'fig_opto_averages_plots_all_{time}'
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def optoPawTiming(self, stridePar, strideTraj, exp):
        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13,
                  'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios

        gs = gridspec.GridSpec(1, 1)
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        timingVars=['FL_HL_time','FR_HR_time','FL_FR_time','FL_HR_time','FR_HL_time']
        strideTraj = strideTraj.drop(strideTraj[(strideTraj['day'] > 4)].index).reset_index()
        pawDf = strideTraj[(strideTraj['paw'] == pawId[0])]
        if exp=='muscimol':
            treat='muscimol'
            cont='saline'
        else:
            treat='opsin'
            cont='tdTomato'

        for p in range(len(timingVars)):
            gssub1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], hspace=0.25,
                                                      wspace=0.35)
            opsin=pawDf[(pawDf['treatment']==treat)][timingVars[p]].values
            tdTomato = pawDf[(pawDf['treatment'] == cont)][timingVars[p]].values

            def remove_duplicate_arrays(arrays):
                unique_arrays = [arrays[0]]
                for array in arrays[1:]:
                    if not any(np.array_equal(array, unique_array) for unique_array in unique_arrays):
                        unique_arrays.append(array)
                return np.array(unique_arrays)

            unique_opsin = remove_duplicate_arrays(opsin)
            unique_tdTomato = remove_duplicate_arrays(tdTomato)
            opsin_unique = np.concatenate(unique_opsin)
            tdTomato_unique = np.concatenate(unique_tdTomato)

            ax0 = plt.subplot(gssub1[p])
            sns.kdeplot(opsin_unique, label=treat, shade=True, ax=ax0, alpha=0.5, color='C3')
            sns.kdeplot(tdTomato_unique, label=cont, shade=True, ax=ax0, alpha=0.5, color='k')
            ax0.axvline(np.median(opsin_unique), ls='--', color='C3', alpha=0.5)
            ax0.axvline(np.median(tdTomato_unique), ls='--', color='k', alpha=0.5)
            # Performing t-test
            t_stat, p_value = stats.ttest_ind(opsin_unique, tdTomato_unique)

            # Display p-value on the plot
            ax0.text(0.5,1,f'p={p_value:.4f}',ha='center', va='center', transform=ax0.transAxes, style='italic', fontfamily='serif',
                                 fontsize=15, color='k')
            self.layoutOfPanel(ax0, xLabel=timingVars[p], yLabel='density', xyInvisible=([False, False]))
            ax0.legend(frameon=False, loc='upper left')
        fname = f'fig_timing_distribution_plots_{exp}'
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def muscimol_group_analysis_averages_single_paw(self, stridePar, strideTraj):
        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13,
                  'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios

        gs = gridspec.GridSpec(1, 1)
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        treatmentA='tdTomato'
        treatmentB='opsin'
        times='day','trial'
        for time in times:
            # modulated_paw_df = modulated_paw_df.drop(
            #     modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
            stridePar=stridePar.drop(stridePar[(stridePar['day']>9)].index).reset_index()
            # regroup data per trial, day, paw and mouse
            stridePar_Recordings = stridePar.groupby(['trial', 'day', 'paw', 'mouseId', 'treatment']).mean()
            stridePar_Recordings = stridePar_Recordings.reset_index()
            stridePar_Recordings_median = stridePar.groupby(['trial', 'day', 'paw', 'mouseId', 'treatment']).median()
            stridePar_Recordings_median=stridePar_Recordings_median.reset_index()
            # average data per day for visualization
            stridePar_Recordings_day = stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
            stridePar_Recordings_day = stridePar_Recordings_day.reset_index()

            # stridePar_Recordings_day_normal_speed_mice = stridePar_Recordings_day[
            #     (stridePar_Recordings_day['LinWheelSpeed_avg'] > 7) & (stridePar_Recordings_day['day'] == 1)]['mouseId']
            # stridePar_Recordings_day_batch_2 = stridePar_Recordings_day[
            #     ~stridePar_Recordings_day['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_batch_2 = stridePar_Recordings[
            #     ~stridePar_Recordings['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_batch_1 = stridePar_Recordings[
            #     stridePar_Recordings['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]
            # stridePar_Recordings_day_batch_1 = stridePar_Recordings_day[
            #     stridePar_Recordings_day['mouseId'].isin(stridePar_Recordings_day_normal_speed_mice)]

            parameters = ['swingNumber', 'swingSpeed', 'swingSpeed_Max','acceleration', 'swingLength','swingLengthLinear',
                          'swingDuration', 'stanceDuration', 'strideDuration',  'dutyFactor','twoRungsFraction', 'rungsCrossed_std', 'rungCrossed',
                          'indecisiveFraction', 'wheelDistance', 'LinWheelSpeed_avg', 'frequency','iqr_25_75_ref_FL', 'iqr_70_90_ref_FL', 'stanceOnStd_ref_FL', 'swingOnMedian_FL', 'stanceOnMedian_ref_FL', 'stanceOn_iqr_25_75_ref_FL', 'stanceOn_iqr_70_90_ref_FL']
            # parameters_Y = ['swing number (avg.)','swing duration (s)', 'swing speed (cm/s)', 'indecisive strides fraction','stance duration (s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed']
            # 		iqr_25_75_ref_FL	iqr_70_90_ref_FL	stanceOnStd_ref_FL	swingOnMedian_FL	stanceOnMedian_ref_FL	stanceOn_iqr_25_75_ref_FL	stanceOn_iqr_70_90_ref_FL	iqr_25_75_ref_FR	iqr_70_90_ref_FR	stanceOnStd_ref_FR	swingOnMedian_FR	stanceOnMedian_ref_FR	stanceOn_iqr_25_75_ref_FR	stanceOn_iqr_70_90_ref_FR

            for par in range(len(parameters)):
                gssub1 = gridspec.GridSpecFromSubplotSpec(int(len(parameters) / 3), 4, subplot_spec=gs[0], hspace=0.4,
                                                          wspace=0.35)
                gssub1a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub1[par], hspace=0.4, wspace=0.35)
                for i in range(4):
                    ax2 = plt.subplot(gssub1a[i])
                    pawDf = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[i])]
                    pars_summary = groupAnalysis.perform_mixedlm_treatment_single_paw(pawDf, parameters[par])
                    # if i==1 and parameters[par]=='indecisiveFraction':
                    #     print(pars_summary['table']['all'].summary())
                    # if pars_summary['pvalues']['all']['treatment[T.saline]:trial']<0.05:
                    #     print('trial effect !!!!',parameters[par], pars_summary['table']['all'].summary())
                    # if pars_summary['pvalues']['all']['day:treatment[T.saline]']<0.05:
                    #     print('effect !!!!', pawId[i],parameters[par], pars_summary['table']['all'].summary())
                    sns.lineplot(data=pawDf, x=time, y=parameters[par], hue=None,
                                 hue_order=[treatmentA, treatmentB], style='treatment', style_order=[treatmentA, treatmentB],
                                 errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
                                 err_kws={'capsize': 3, 'linewidth': 1}, color=f'C{i}', ax=(ax2), marker='o')
                    interactionTerm = f'{time}:treatment[T.tdTomato]'
                    if pars_summary["pvalues"]["all"][interactionTerm]<0.05:
                        ax2.text(1, 0.33, f'{pars_summary["stars"]["all"][interactionTerm].replace("*", "°")}',
                                 ha='center', va='center', transform=ax2.transAxes, style='italic', fontfamily='serif',
                                 fontsize=15, color='k')
                    else:
                        ax2.text(1, 0.33, f'{pars_summary["pvalues"]["all"][interactionTerm]:.2f}',
                                 ha='center', va='center', transform=ax2.transAxes, style='italic', fontfamily='serif',
                                 fontsize=10, color='k')
                    ax2.legend([f'tdTomato {pars_summary["stars"]["tdTomato"]["day"]} {pars_summary["stars"]["tdTomato"]["trial"].replace("*", "#")}',f'opsin {pars_summary["stars"]["opsin"]["day"]} {pars_summary["stars"]["tdTomato"]["trial"].replace("*", "#")}'], bbox_to_anchor=(0.35,0.7,0.5,0.5),loc='upper left', frameon=False, fontsize=10)
                    if i == 2:
                        self.layoutOfPanel(ax2, xLabel=('session' if time=='day' else 'trial'), yLabel=parameters[par],
                                           xyInvisible=([False, False]))

                    elif i == 1:
                        self.layoutOfPanel(ax2, xLabel=('session' if time=='day' else 'trial'), yLabel='', xyInvisible=([True, True]))

                    elif i == 0:
                        self.layoutOfPanel(ax2, xLabel=('session' if time=='day' else 'trial'), yLabel='', xyInvisible=([True, False]))

                    elif i == 3:
                        self.layoutOfPanel(ax2, xLabel=('session' if time=='day' else 'trial'), yLabel='', xyInvisible=([False, True]))

                    ax2.xaxis.set_major_locator(MultipleLocator(2))


            fname =f'fig_opto_averages_plots_single paw_{time}'
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def muscimol_group_analysis_correlations(self, stridePar, strideTraj):
        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13,
                  'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios

        gs = gridspec.GridSpec(1, 2)
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        # regroup data per trial, day, paw and mouse
        stridePar_Recordings = stridePar.groupby(['trial', 'day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings = stridePar_Recordings.reset_index()
        # average data per day for visualization
        stridePar_Recordings_day = stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day = stridePar_Recordings_day.reset_index()
        ax0=plt.subplot(gs[0])
        treatmentList = ['muscimol', 'saline']
        colors_t=['red', 'black']
        for t in range(len(treatmentList)):
            xdata_day = stridePar_Recordings[(stridePar_Recordings['treatment'] == treatmentList[t])]['wheelDistance']
            ydata_day = stridePar_Recordings[(stridePar_Recordings['treatment'] == treatmentList[t])]['stanceDuration']
            sns.regplot(x=xdata_day, y=ydata_day, ax=ax0,color=colors_t[t])
            # ax1.xaxis.set_major_locator(MultipleLocator(1))

            self.layoutOfPanel(ax0)
        # parameters = ['swingNumber', 'swingSpeed', 'swingSpeed_Max', 'swingLength', 'stanceDuration',
        #               'swingDuration', 'stanceDuration', 'strideDuration', 'twoRungsFraction', 'rungsCrossed_std',
        #               'indecisiveFraction', 'indecisiveStrides', 'wheelDistance', 'LinWheelSpeed_avg', 'frequency',
        #               'dutyFactor', 'max_acceleration', 'mean_acceleration', 'acceleration_phases',
        #               'max_deceleration', 'mean_deceleration', 'deceleration_phases', 'acc_duration',
        #               'dec_duration', 'acceleration', 'rungCrossed']
        # parameters_Y = ['swing number (avg.)','swing duration (s)', 'swing speed (cm/s)', 'indecisive strides fraction','stance duration (s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed']
        # 		iqr_25_75_ref_FL	iqr_70_90_ref_FL	stanceOnStd_ref_FL	swingOnMedian_FL	stanceOnMedian_ref_FL	stanceOn_iqr_25_75_ref_FL	stanceOn_iqr_70_90_ref_FL	iqr_25_75_ref_FR	iqr_70_90_ref_FR	stanceOnStd_ref_FR	swingOnMedian_FR	stanceOnMedian_ref_FR	stanceOn_iqr_25_75_ref_FR	stanceOn_iqr_70_90_ref_FR

        # for par in range(len(parameters)):
        #     gssub1 = gridspec.GridSpecFromSubplotSpec(int(len(parameters) / 3), 4, subplot_spec=gs[0], hspace=0.4,
        #                                               wspace=0.35)
        #     ax1 = plt.subplot(gssub1[par])
        #     pars_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par], treatments=True)
        #     sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='treatment',
        #                  hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
        #                  errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
        #                  err_kws={'capsize': 3, 'linewidth': 1}, palette=['black', 'red'], ax=(ax1), marker='o')
        #     sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId',
        #                  style='treatment', style_order=['saline', 'muscimol'],
        #                  errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)], alpha=0.1,
        #                  err_kws={'capsize': 3, 'linewidth': 1}, palette=['red', 'black'], ax=(ax1))
        #     ax1.text(1, 0.31, f'{pars_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
        #              ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
        #              fontsize=15, color='k')
        #     ax1.legend([
        #                    f'saline {pars_summary["stars"]["saline"]["day"]} {pars_summary["stars"]["saline"]["trial"].replace("*", "#")}',
        #                    f'muscimol {pars_summary["stars"]["muscimol"]["day"]} {pars_summary["stars"]["muscimol"]["trial"].replace("*", "#")}'],
        #                loc='best', frameon=False)

        fname = 'fig_muscimol_correlation_plots'
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def psthGroupFigure_missteps(self,  cellType, dfs, spikeType):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 22  # width in inches

        fig_height = 20
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False, 'ytick.direction':'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)
        modCells = {}
        events=['swing', 'stance']
        swingType = ['indecisive', 'not_indecisive', 'allSteps']
        pawNb=0
        modCells['swing']= {}
        modCells['stance']={}
        for ev in range(len(events)):
            event = events[ev]
            for e in range(3):
                modCells['swing'][swingType[e]]=[]
                modCells['stance'][swingType[e]]=[]
            for e in range(3):
                for i in [pawNb]:
                    dayCat=['all','early','late']
                    # looking at specific paw values in the df that contains single values
                    paw_df = dfs[swingType[e]][(dfs[swingType[e]]['paw'] == pawList[i])]
                    paw_df_early=dfs[swingType[e]][(dfs[swingType[e]]['paw'] == pawList[i]) & (dfs[swingType[e]]['day_category']=='early')]
                    paw_df_late= dfs[swingType[e]][(dfs[swingType[e]]['paw'] == pawList[i]) & (dfs[swingType[e]]['day_category']=='late')]
                    psth_df_K=f'psth_{swingType[e]}'
                    # looking at specific paw values in the df that contains psth and zscore
                    paw_psth_df = dfs[psth_df_K][(dfs[psth_df_K]['paw'] == pawList[i])]
                    # detect and remove empty cells, happens when there's no stride in a condition
                    empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                    # loop through the rows of the DataFrame
                    for index, value in empty_cells.iteritems():
                        if value:
                            print(f'Empty cell detected in line {index}')
                            paw_df = paw_df.drop(index)
                            paw_psth_df = paw_psth_df.drop(index)
                    # big panels for each paw


                    gssub4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], hspace=0.1,wspace=0.3)

                        # sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                    gssub5 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub4[0,e], hspace=0.2,
                                                              height_ratios=[7, 5, 24])

                    # time around event for keys
                    times = ['before_', 'after_']
                    # iterate through time (before/after)
                    timePoints = [paw_df, paw_df_early, paw_df_late]
                    for z in range(len(timePoints)):
                        modCells[pawList[i]] = {}
                        modCells[pawList[i]]['all'] = np.empty(0)
                        for t in range(2):
                            # modCells[pawList[i]]['before']=np.empty(0)
                            # modCells[pawList[i]]['after']=np.empty(0)
                            # define modulation catgories
                            catList = ['↓', '↑', '-']
                            # time to look at
                            time = times[t]
                            # change position of plot if looking at more than one paw
                            gssub5z = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub5[1], hspace=0.005)


                            bottom_pos=0

                            ax2 = plt.subplot(gssub5z[z])


                            # get Id and counts of modulated cells
                            modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(timePoints[z], catList,
                                                                                                               time, event,
                                                                                                               condition=None)
                            if z==1:
                                print(modCells_Id)
                                # pdb.set_trace()
                            if t == 0:
                                modCells[pawList[i]]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                                modCells[pawList[i]]['not_before']=modCells_Id['-']
                            else:
                                modCells[pawList[i]]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                                modCells[pawList[i]]['not_after'] = modCells_Id['-']
                        # pdb.set_trace()
                        modCells[pawList[i]]['all_mod'] = np.unique(np.concatenate((modCells[pawList[i]]['before'], modCells[pawList[i]]['after'])))
                        # modCells[pawList[i]]['all_non']=np.unique(np.concatenate((modCells[pawList[i]]['not_before'], modCells[pawList[i]]['not_after'])))
                        # modCells[pawList[i]]['all_non']=np.setdiff1d(modCells[pawList[i]]['all_non'],np.intersect1d(modCells[pawList[i]]['all_mod'],modCells[pawList[i]]['all_non']))
                        # all_cells=np.arange(64)+1
                        all_cells=np.unique(timePoints[z]['cell_global_Id'])
                        modCells[pawList[i]]['all_non']=np.setdiff1d(all_cells,modCells[pawList[i]]['all_mod'])
                        # print(len(modCells[pawList[i]]['all_mod'] )+len(modCells[pawList[i]]['all_non'] ))
                        # pdb.set_trace()

                        twoCol=['0.2','0.7']
                        alpha_e=[0.4,0.8,1]
                        for i in [pawNb]:

                            bar_width = len(modCells[pawList[i]]['all_mod'])/len(all_cells)*100
                            bar_left = bottom_pos

                            ax2.barh([0], [100], color='0.9',  alpha=0.8)
                            ax2.barh(0, bar_width, left=bar_left, color=f'C{i}', alpha=alpha_e[e])
                            ax2.annotate(f'{bar_width:.1f}% ({ len(modCells[pawList[i]]["all_mod"])}/{len(all_cells)})', (bar_width/ 2, 0),
                                          fontsize=12, ha='center', va='center', c=('white' if e==1 else 'k'))

                            ax2.xaxis.set_major_locator(MultipleLocator(25))
                            ax2.set_ylim(1, -1)
                            # the text above with percent and counts bbox (left, up, spacing, right)
                            # ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                            #            bbox_to_anchor=(0.05, 0.7, 0.9, 0.1), frameon=False,
                            #            fontsize=14, labelcolor='k')  # , labelcolor=label_colors_List)
                            bottom_pos += bar_width

                            # axis style
                            # ax2.get_yaxis().set_visible(False)
                            ax2.set_xlim(0, 100)
                            ax2.set_yticklabels(['',dayCat[z],''])
                            ax2.tick_params(axis='y', which='both', length=0)
                            # ax2.yaxis.set_major_locator(MultipleLocator(1))
                            if z!=2:
                                ax2.spines[['left', 'right', 'top','bottom']].set_visible(False)
                                ax2.get_xaxis().set_visible(False)
                            else:
                                ax2.spines[['left', 'right', 'top']].set_visible(False)
                                ax2.text(0.5, -0.6, (f'fraction of  {pawList[i]} {swingType[e]} {event} onset modulated {cellType}  (%)'),ha='center', va='center', transform=ax2.transAxes)
                        # plot title


                        zscore = ['AUC', 'peak']
                        zscorePar = zscore[0]
                        # choose interval
                        intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                        interval = intervals[0]
                        # key to access the variable (AUC or Peak)
                        zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                                      'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                        modList=['all_mod', 'all_non']
                        for m in range(2):
                                # only look at cell modulated showing one category of modulation
                            modulated_mask = paw_df['cell_global_Id'].isin(modCells[pawList[i]][modList[m]])
                            modulated_paw_df = paw_df.loc[modulated_mask]


                            modulated_paw_psth_df = paw_psth_df.loc[modulated_mask]
                            # drop cells with only one rec
                            modulated_paw_df = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                            modulated_paw_df['z_score_abs']=abs(modulated_paw_df[zscore_key[0]])+abs(modulated_paw_df[zscore_key[1]])
                            # key for z_score and z_score time arrays
                            z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                            z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
                            # ids of modulated cells
                            cellsId = np.unique(modulated_paw_psth_df['cell_global_Id'])
                            modCells[events[ev]][swingType[e]].append(cellsId)
                            print('cellsID:', cellsId)
                            if spikeType=='simple':
                                if e==2:
                                    exampleCell = 0
                                else:
                                    exampleCell = 1
                                # # choose an example cell for each condition
                                # if condition == 'allSteps':
                                #     if pawNb == 0 :
                                #         if (e == 0 and m==0 ):
                                #             exampleCell = 21 #19 #25 #31 #41 #2 is perfect
                                #         elif (e == 0 and m==1) :
                                #             exampleCell = 8 #7
                                #         elif (e==1 and m==0):
                                #             exampleCell=2 #19 #13 #3 #21 #25 #36
                                #         elif (e == 1 and m == 1):
                                #             exampleCell=7 #2
                                #     elif pawNb ==1:
                                #         exampleCell=18  #16 maybe[ 1  2  4  5  6  9 11 12 13 17 19 21 26 27 29 30 32 34 35 36 37 38 40 41 42 44 45 46 47 50 52 56 58 59 61 63]
                                # elif condition == 'swingLengthLinear_lastRec_20_80':
                                #     if pawNb == 0 :
                                #         if (e == 0 and m == 0):
                                #             exampleCell = 1  # 19 #25 #31 #41 #2 is perfect
                                #         elif (e == 0 and m == 1):
                                #             exampleCell = 1  # 7
                                #         elif (e == 1 and m == 0):
                                #             exampleCell = 1  # 19 #13 #3 #21 #25 #36
                                #         elif (e == 1 and m == 1):
                                #             exampleCell = 1  # 2
                                #     else:
                                #         exampleCell = 1
                                # else:
                                #     exampleCell=2
                                if spikeType == 'simple':
                                            # get the zscore of and zscore time of example cells for first and last trials
                                    earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                                    zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                                    lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (
                                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                                # subplot before and after for z-score examples
                                    # subplot before and after for z-score examples

                            if z == 0:
                                gssub5b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub5[0], hspace=0.2)
                                if spikeType == 'simple':
                                    if e == 0:
                                        # for time before we need two examples for up and down modulated (c==0 and c==1)
                                        gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], wspace=0.3)
                                        # we don't care about non modulated (c==2)

                                        if m < 2:
                                            ax1 = plt.subplot(gssub5c[m])
                                            # vertical fill concerned time window (before -100ms to 0)
                                            # plot the example z-score for first and last trial

                                            ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))
                                            ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                                     alpha=1, label='last trial')

                                            ax1.set_ylim(-3.5,4.5)

                                            # timeAfterMask=zscoreTime[0]>0
                                            ax1.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                                            ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                                            ax1.legend(frameon=False, loc='upper right')
                                    if e == 1:
                                        gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], hspace=0.2, wspace=0.3)
                                        if m < 2:
                                            ax1 = plt.subplot(gssub5c[m])
                                            # vertical fill concerned time window (after 0ms to 100)
                                            # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                                            # plot the example z-score for first and last trial for after event
                                            if pawNb==0:
                                                ax1.set_ylim(-5,10)
                                            else:
                                                ax1.set_ylim(-3.5, 6)

                                            ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                                            ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                                     alpha=1,
                                                     label='last trial')
                                            ax1.legend(frameon=False)

                                            ax1.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                                            ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))

                                            ax1.legend(frameon=False, loc='upper right')
                                    if e == 2:
                                        gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], hspace=0.2, wspace=0.3)
                                        if m < 2:
                                            ax1 = plt.subplot(gssub5c[m])
                                            # vertical fill concerned time window (after 0ms to 100)
                                            # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                                            # plot the example z-score for first and last trial for after event
                                            if pawNb==0:
                                                ax1.set_ylim(-5,10)
                                            else:
                                                ax1.set_ylim(-3.5, 6)

                                            ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                                            ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                                     alpha=1,
                                                     label='last trial')
                                            ax1.legend(frameon=False)

                                            ax1.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                                            ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))

                                            ax1.legend(frameon=False, loc='upper right')
                                    #draw 0 line
                                    ax1.axvline(0, ls='--', color='grey', lw=1,alpha=0.1)
                                    ax1.axvline(0, ls='--', color='grey', alpha=0.1)
                                    ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                                    ax1.axhline(0, ls='--', color='grey', alpha=0.1)
                                    self.layoutOfPanel(ax1, xLabel=f'{event} onset time (s)', yLabel=('PSTH Z-score' if m==0 else ''), xyInvisible=[(False), False])
                                    ax1.yaxis.set_major_locator(MultipleLocator(2))

                                # for FL look only at those behavior parameters

                                behavior_par = ['swingDuration', 'swingLengthLinear']
                                behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                                hr = np.full(len(behavior_par) + 2, 2)
                                hr[0] = 3.2
                                gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) + 2, 2, subplot_spec=gssub5[2],
                                                                          hspace=0.5,
                                                                          wspace=0.4, height_ratios=hr)

                                # comparison panel for each categories

                                # group the data per cell and trial categories
                                modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                                    'z_score_abs'].mean().reset_index()

                                modulated_paw_df_early_late_days = modulated_paw_df.groupby(['cell_global_Id', 'day_category'])[
                                    'z_score_abs'].mean().reset_index()

                                categories=[modulated_paw_df_first_late,modulated_paw_df_early_late_days]
                                categoriesNames=['trial_category', 'day_category']
                                for cate in range(2):
                                    ax3 = plt.subplot(gssub6[cate, m])
                                    # define colors for seaborn
                                    palette = sns.color_palette([f'C{i}'], 1)
                                    palette_non = sns.color_palette(['0.8'], 1)
                                    grey = sns.color_palette(['grey'], 1)


                                    # the invisible bar plot is to tighten things
                                    sns.barplot(categories[cate], x=categoriesNames[cate], hue=None, y='z_score_abs', ax=ax3,
                                                color='k', alpha=0.001, errorbar=None)
                                    # lineplot for the line between cells
                                    sns.lineplot(categories[cate], x=categoriesNames[cate], hue='cell_global_Id', y='z_score_abs',
                                                 ax=ax3,
                                                 palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                                    # scatterplot for the AUC values
                                    sns.scatterplot(categories[cate], x=categoriesNames[cate], hue='cell_global_Id',
                                                    y='z_score_abs',
                                                    ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                                                    lw=0.1, s=80)

                                    print()
                                    if cate==0:
                                        ax3.set_xlim(-0.9,1.9)
                                        ax3.plot([0,1],[np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']),np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs'])],ls='-',c='k',lw=2)
                                        # extract first and last trial z-score AUC arrays
                                        first_trial = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']
                                        last_trial = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']
                                        # perform t-test
                                        t_value, t_test_p_value = stats.ttest_rel(first_trial['z_score_abs'], last_trial['z_score_abs'],
                                                                                  axis=0)
                                    elif cate==1:
                                        first_day = modulated_paw_df_early_late_days[modulated_paw_df_early_late_days['day_category'] == 'early']
                                        last_day = modulated_paw_df_early_late_days[modulated_paw_df_early_late_days['day_category'] == 'late']
                                        # perform t-test
                                        t_value, t_test_p_value = stats.ttest_ind(first_day['z_score_abs'], last_day['z_score_abs'],
                                                                                  axis=0)

                                    print('paired t-test (t-value, p): ',event, t_value, t_test_p_value)
                                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                                    if t_test_p_value<0.05:
                                        ax3.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                                                 transform=ax3.transAxes,
                                                 style='italic', fontfamily='serif', fontsize=18, color='k')
                                    else:
                                        ax3.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',
                                                 transform=ax3.transAxes,
                                                 style='italic', fontfamily='serif', fontsize=12, color='k')
                                    self.layoutOfPanel(ax3, xLabel=('Trial' if cate==0 else 'Day'), yLabel=(f' {event} onset PSTH \n  Z-score {zscorePar} (abs)' if m==0 else ''),
                                                       Leg=[1, 9])

                                majorLocator_x = MultipleLocator(1)
                                ax3.xaxis.set_major_locator(majorLocator_x)
                                # allSteps y-axis limits
                                #if event == 'swing' :
                                #    ax3.set_ylim(0,0.815)
                                #elif event == 'stance':
                                #    ax3.set_ylim(0,1.255)
                                # 20-80 swing length y-axis limits
                                # if  pawNb==0:
                                #     if event == 'swing':
                                #         ax3.set_ylim(0, 0.85)
                                #     elif event == 'stance':
                                #         ax3.set_ylim(0, 1.25)
                                # else:
                                #     if event == 'swing':
                                #         ax3.set_ylim(0, 0.45)
                                #     elif event == 'stance':
                                #         ax3.set_ylim(0, 0.5)

                                # create a new column with a name without '-' to put the z-score value (necessary because statsmodel and scipy dont like '-' in the key names
                                # zscore_key1 = 'zScoreAUC_' + times[t] + event
                                # modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                                # for each behavior parameter perform linear regression
                                for p in range(len(behavior_par)):
                                    ax4 = plt.subplot(gssub6[2 + p,m])

                                    modulated_paw_df.dropna(subset=['z_score_abs'], inplace=True)
                                    y = modulated_paw_df[behavior_par[p]]
                                    x = modulated_paw_df['z_score_abs']
                                    # we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                                    try:
                                        slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                                     alternative='two-sided')
                                    except ValueError:
                                        pass
                                    # set a treshold to reduce alpha and highlight only pvalues<0.05 and r2>0.05
                                    if p_value>0.05:
                                        alpha=0.1
                                    else:
                                        alpha=alpha_e[e]


                                    # show linear reg plots
                                    sns.regplot(x=x, y=y, ax=ax4, color=(f'C{i}'if m==0 else '0.7'),
                                                scatter_kws={'alpha': alpha, 'edgecolor': 'k', 'lw': 0.1},
                                                line_kws={'alpha': alpha, 'lw': 1.5})
                                    # annotate with values of r, r², p_values$
                                    corr_star=groupAnalysis.starMultiplier(p_value)
                                    if p_value>0.05:
                                        ax4.text(0.73, 0.80, f"r = {r_value:.2f}\np = {p_value:.2f}",
                                                 transform=ax4.transAxes, fontsize=10, color='dimgrey')

                                    else:
                                        ax4.text(0.73, 0.80, f"r = {r_value:.2f}",
                                                 transform=ax4.transAxes, fontsize=10, color='dimgrey')
                                        ax4.text(0.5, 0.98, f"{corr_star}",
                                                 transform=ax4.transAxes, fontsize=18, color='k')
                                    self.layoutOfPanel(ax4, xLabel=('Z-score %s (abs)' % zscorePar if p == len(behavior_par) - 1 else ''), yLabel=(f'{behavior_par_Name[p]} ' if m==0 else ''),Leg=[1, 9])
                                    print(event,p)
                                    # allSteps AND 20-80 percentile y-axis limits
                                    if behavior_par[p] == 'swingDuration':
                                        ax4.set_ylim(0.065,0.36)
                                    elif behavior_par[p] == 'swingLengthLinear':
                                        ax4.set_ylim(1.1,4.7)

                paw=pawList[pawNb]
            if spikeType=='simple':
                fname = f'fig_ephys_psth_Z-score_{event}_missSteps_{zscorePar}_cell_based_{cellType}_{paw}'
            else:
                fname = f'fig_ephys_psth_Z-score_{event}_missSteps_{zscorePar}_cell_based_{cellType}_{paw}_complex'
            # plt.savefig(fname + '.png')
            # plt.show()
            plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def psthGroupFigure_cell_based_summary(self,  cellType, df_cells, condition,  pawNb, spikeType):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 20  # width in inches

        fig_height = 14
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False, 'ytick.direction':'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.2)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.1)
        modCells = {}
        events=['swing', 'stance']
        for e in reversed(range(2)):
            event=events[e]
            for i in range(4):
                paw_df = df_cells[(df_cells['paw'] == pawList[i])]
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.1,wspace=0.3)
                gssub5 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub4[0,e], hspace=0.5,
                                                          height_ratios=[1.5, 0.4,4])
                # time around event for keys
                times = ['before_', 'after_']
                # iterate through time (before/after)

                modCells[pawList[i]] = {}
                modCells[pawList[i]]['all'] = np.empty(0)
                for t in range(2):
                    catList = ['↓', '↑', '-']
                    # time to look at
                    time = times[t]
                    # change position of plot if looking at more than one paw
                    # get Id and counts of modulated cells
                    modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
                                                                                                       time, event,
                                                                                                       condition=None)
                    if t == 0:
                        modCells[pawList[i]]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_before']=modCells_Id['-']
                    else:
                        modCells[pawList[i]]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_after'] = modCells_Id['-']
                # pdb.set_trace()
                modCells[pawList[i]]['all_mod'] = np.unique(np.concatenate((modCells[pawList[i]]['before'], modCells[pawList[i]]['after'])))
                # modCells[pawList[i]]['all_non']=np.unique(np.concatenate((modCells[pawList[i]]['not_before'], modCells[pawList[i]]['not_after'])))
                # modCells[pawList[i]]['all_non']=np.setdiff1d(modCells[pawList[i]]['all_non'],np.intersect1d(modCells[pawList[i]]['all_mod'],modCells[pawList[i]]['all_non']))
                all_cells=np.arange(64)+1
                modCells[pawList[i]]['all_non']=np.setdiff1d(all_cells,modCells[pawList[i]]['all_mod'])
                # print(len(modCells[pawList[i]]['all_mod'] )+len(modCells[pawList[i]]['all_non'] ))
            # pdb.set_trace()
            for i in reversed(range(4)):
                twoCol=['0.2','0.7']
                alpha_e=[0.4,0.8]
                ax2 = plt.subplot(gssub5[0])


                bottom_pos = 0
                # gssub5a = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gssub5[0], hspace=0.2)

                bar_width = len(modCells[pawList[i]]["all_mod"])/len(all_cells)*100
                bar_left = bottom_pos
                ax2.barh(pawList[i], [100,100,100,100], color='white', edgecolor='k', alpha=0.8)
                ax2.barh(pawList[i], bar_width, color=f'C{i}', edgecolor='black', alpha=alpha_e[e])
                ax2.annotate(f'{bar_width:.1f}%', (bar_width/ 2, pawList[i]),
                              fontsize=12, ha='center', va='center', c=('white' if e==1 else 'k'))
                ax2.annotate(f'{len(modCells[pawList[i]]["all_mod"])}/{len(all_cells)}', (bar_width+5, pawList[i]),
                              fontsize=16, ha='center', va='center', c=('k' if e==1 else 'k'))
                ax2.xaxis.set_major_locator(MultipleLocator(25))

                bottom_pos += bar_width
                ax2.spines[['left', 'right', 'top']].set_visible(False)
                # # axis style
                ax2.set_xlim(0, 100)

                first_last_stats_df = pd.DataFrame(columns=['paw', 't_test_p_value', 'tvalue', 'stars'])
                first_last_stats_df['paw'] = pawList

                first_last_stats_list=[]
            ax2.text(0.5, -0.2, (f'fraction of  {event} onset modulated {cellType}  (%)'),ha='center', va='center',transform=ax2.transAxes)
            ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=24)
            for i in range(4):
                paw_df = df_cells[(df_cells['paw'] == pawList[i])]
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)

                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[0]
                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                modList=['all_mod', 'all_non']
                for m in range(2):
                    first_last_stats = {}
                    first_last_stats['paw'] = pawList[i]

                    # only look at cell modulated showing one category of modulation
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells[pawList[i]][modList[m]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    # modulated_paw_psth_df = paw_psth_df.loc[modulated_mask]
                    # drop cells with only one rec
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                    modulated_paw_df['z_score_abs']=abs(modulated_paw_df[zscore_key[0]])+abs(modulated_paw_df[zscore_key[1]])

                    modulated_paw_df_first_last = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()


                    last_trial = modulated_paw_df_first_last[modulated_paw_df_first_last['trial_category'] == 'first']
                    first_trial = modulated_paw_df_first_last[modulated_paw_df_first_last['trial_category'] == 'last']
                    # perform t-test
                    t_value, t_test_p_value = stats.ttest_rel(first_trial['z_score_abs'],last_trial['z_score_abs'], axis=0)
                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    if m == 0:
                        first_last_stats['modulation'] = 'modulated'
                    elif m == 1:
                        first_last_stats['modulation'] = 'non_modulated'
                    behavior_par = ['swingDuration',	'stanceDuration',	'swingLength',	'stepLength',	'rungCrossed'	,'stepDuration',	'stepMeanSpeed',	'swingLengthLinear',	'swingSpeed',	'stanceOnsetStd'	,'stanceOnsetMedian']
                    behavior_par = ['swingDuration',	'stanceDuration',	'swingLength','swingLengthLinear',	'swingSpeed',	'stanceOnsetStd'	,'stanceOnsetMedian']
                    for p in range(len(behavior_par)):
                        key_pvalue=f'{behavior_par[p]}_pvalue'
                        key_rvalue=f'{behavior_par[p]}_rvalue'
                        key_stars=f'{behavior_par[p]}_stars'
                        modulated_paw_df.dropna(subset=['z_score_abs'], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df['z_score_abs']
                        # we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            corr_star = groupAnalysis.starMultiplier(p_value)
                            first_last_stats[key_pvalue]=p_value
                            first_last_stats[key_rvalue] = r_value
                            first_last_stats[key_stars] = corr_star
                        except ValueError:
                            first_last_stats[key_pvalue]=np.nan
                            first_last_stats[key_rvalue] = np.nan
                            pass


                    first_last_stats['nCells'] = len(last_trial)
                    first_last_stats_df.loc[i, 't_value']=t_value
                    first_last_stats_df.loc[i,'t_test_p_value']=t_test_p_value
                    first_last_stats_df.loc[i, 'stars'] = star_trial
                    first_last_stats['t_value']=t_value
                    first_last_stats['t_test_p_value'] = t_test_p_value
                    first_last_stats['stars']=star_trial
                    first_last_stats_list.append(first_last_stats)

                    behavior_par = ['swingDuration', 'swingLengthLinear']
                    behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
            df_cells_stats = pd.DataFrame(first_last_stats_list)
            gssub5a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5[1], hspace=0.2)
            modulation=['modulated','non_modulated']
            for m in range(2):
                ax2a = plt.subplot(gssub5a[0,m])
                modulation_stats_df=df_cells_stats[df_cells_stats['modulation'] == modulation[m]]
                # pdb.set_trace()
                # ax2a.imshow(np.array(modulation_stats_df['t_test_p_value']).reshape(1,4),cmap='Reds_r')
                # print(modulation_stats_df['t_value'])
                # heatmap=ax2a.pcolor(np.array(modulation_stats_df['t_value']).reshape(1,4), cmap='PiYG_r')
                if m==1:
                    heatmap=sns.heatmap(np.array(modulation_stats_df['t_value']).reshape(1,4), ax=ax2a, fmt='',cmap=sns.color_palette("PiYG_r", 5), vmin=-3, vmax=3, cbar_kws={'label': 't-value'})


                    cbar = ax2a.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=10)
                    cbar.set_label(label='t-value', size=10, rotation=90)
                else:
                    heatmap = sns.heatmap(np.array(modulation_stats_df['t_value']).reshape(1, 4), ax=ax2a, fmt='',
                                          cmap=sns.color_palette("PiYG_r", 5), vmin=-3, vmax=3,
                                          cbar=False)

                if m==1:
                    # plt.colorbar(heatmap, ax=ax2a, location='right')
                    ax2a.text(0.2, -1.8, ('first vs last trial PSTH z-score AUC'), ha='center', va='center')

                self.layoutOfPanel(ax2a, xLabel=f' {modulation[m]} MLI', yLabel=None, xyInvisible=[False, True])
                ax2a.set_xticks(ticks=[0.5,1.5,2.5,3.5],labels=['FL','FR','HL','HR'])

                my_colors=['C0','C1','C2','C3']
                for ticklabel, tickcolor in zip(ax2a.get_xticklabels(), my_colors):
                    ticklabel.set_color(tickcolor)
                ax2a.set_ylim(0,1)

                for l in range(4):
                    ax2a.annotate((np.array(modulation_stats_df['stars'])[l] ) , (0.5+l,0.35), fontsize=15, ha='center', va='center')

                behavior_par = ['swingDuration', 'stanceDuration', 'swingLength', 'swingLengthLinear', 'swingSpeed',
                                'stanceOnsetStd', 'stanceOnsetMedian']
                gssub5b = gridspec.GridSpecFromSubplotSpec(len(behavior_par), 2, subplot_spec=gssub5[2], hspace=0.1)

                for p in range(len(behavior_par)):
                    key_pvalue = f'{behavior_par[p]}_pvalue'
                    key_rvalue = f'{behavior_par[p]}_rvalue'
                    key_stars = f'{behavior_par[p]}_stars'
                    ax2b=plt.subplot(gssub5b[p,m])
                    print(modulation_stats_df[key_rvalue])

                    if p!=len(behavior_par)-1:
                        heatmap_2=sns.heatmap(np.array(modulation_stats_df[key_rvalue]).reshape(1,4), yticklabels=[behavior_par[p]], ax=ax2b, fmt='',cmap=sns.color_palette("PiYG_r", 10), vmin=-0.5, vmax=0.5, cbar=False)
                        if m==0:
                            self.layoutOfPanel(ax2b, xLabel=None, yLabel=None, xyInvisible=[True, False])
                        else:
                            self.layoutOfPanel(ax2b, xLabel=None, yLabel=None, xyInvisible=[True, True])
                        ax2b.set_yticklabels(ax2b.get_yticklabels(), rotation=0, fontsize=10)
                    else:

                        heatmap_2=sns.heatmap(np.array(modulation_stats_df[key_rvalue]).reshape(1,4), yticklabels=[behavior_par[p]],ax=ax2b, fmt='',cmap=sns.color_palette("PiYG_r", 10), vmin=-0.5, vmax=0.5, cbar=False)
                        if m==0:
                            self.layoutOfPanel(ax2b, xLabel=f' {modulation[m]} MLI', yLabel=None, xyInvisible=[False, False])
                        else:
                            self.layoutOfPanel(ax2b, xLabel=f' {modulation[m]} MLI', yLabel=None, xyInvisible=[False, True])
                            ax2a.text(0.2, -1.8, ('first vs last trial PSTH z-score AUC'), ha='center', va='center')
                        ax2b.set_xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['FL', 'FR', 'HL', 'HR'])
                        my_colors = ['C0', 'C1', 'C2', 'C3']
                        ax2b.set_yticklabels(ax2b.get_yticklabels(), rotation=0, fontsize=10)
                        for ticklabel, tickcolor in zip(ax2b.get_xticklabels(), my_colors):
                            ticklabel.set_color(tickcolor)
                    for l in range(4):
                        ax2b.annotate(f"{(np.array(modulation_stats_df[key_stars])[l])}\n r={(np.array(modulation_stats_df[key_rvalue])[l]):.2f}", (0.5 + l, 0.35), fontsize=11,
                                      ha='center', va='center')
                    # ax2b.set_yticks(ax2b.get_xticklabels(), labels=[behavior_par[p]], rotation=0, fontsize=10)


            # plt.show()
            # pdb.set_trace()

        paw=pawList[pawNb]
        if spikeType=='simple':
            fname = f'fig_ephys_psth_cell_based_summary_{cellType}_{paw}_{condition}'
        else:
            fname = f'fig_ephys_psth_cell_based_summary_{cellType}_{paw}_{condition}_complex'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')


    def psthGroupFigure_cell_based_single_paw  (self,  cellType, df_psth, df_cells, condition,  pawNb):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 25  # width in inches

        fig_height = 17
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False, 'ytick.direction':'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)
        modCells = {}
        events=['swing', 'stance']
        for e in reversed(range(2)):
            event=events[e]
            for i in [pawNb]:
                # looking at specific paw values in the df that contains single values
                paw_df = df_cells[(df_cells['paw'] == pawList[i])]

                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                        paw_psth_df = paw_psth_df.drop(index)
                # big panels for each paw


                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.1,wspace=0.3)

                    # sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub4[0,e], hspace=0.2,
                                                          height_ratios=[7, 2, 24])

                # time around event for keys
                times = ['before_', 'after_']
                # iterate through time (before/after)

                modCells[pawList[i]] = {}
                modCells[pawList[i]]['all'] = np.empty(0)
                for t in range(2):
                    # modCells[pawList[i]]['before']=np.empty(0)
                    # modCells[pawList[i]]['after']=np.empty(0)
                    # define modulation catgories
                    catList = ['↓', '↑', '-']
                    # time to look at
                    time = times[t]
                    # change position of plot if looking at more than one paw

                    ax2 = plt.subplot(gssub5[1])


                    # get Id and counts of modulated cells
                    modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
                                                                                                       time, event,
                                                                                                       condition=None)
                    if t == 0:
                        modCells[pawList[i]]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_before']=modCells_Id['-']
                    else:
                        modCells[pawList[i]]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_after'] = modCells_Id['-']
                # pdb.set_trace()
                modCells[pawList[i]]['all_mod'] = np.unique(np.concatenate((modCells[pawList[i]]['before'], modCells[pawList[i]]['after'])))
                # modCells[pawList[i]]['all_non']=np.unique(np.concatenate((modCells[pawList[i]]['not_before'], modCells[pawList[i]]['not_after'])))
                # modCells[pawList[i]]['all_non']=np.setdiff1d(modCells[pawList[i]]['all_non'],np.intersect1d(modCells[pawList[i]]['all_mod'],modCells[pawList[i]]['all_non']))
                all_cells=np.arange(64)+1
                modCells[pawList[i]]['all_non']=np.setdiff1d(all_cells,modCells[pawList[i]]['all_mod'])
                # print(len(modCells[pawList[i]]['all_mod'] )+len(modCells[pawList[i]]['all_non'] ))
            # pdb.set_trace()
            bottom_pos=0
            twoCol=['0.2','0.7']
            alpha_e=[0.4,0.8]
            for i in [pawNb]:
                bar_width = len(modCells[pawList[i]]['all_mod'])/len(all_cells)*100
                bar_left = bottom_pos

                ax2.barh([0], [100], color='0.9',  alpha=0.8)
                ax2.barh(0, bar_width, left=bar_left, color=f'C{i}', alpha=alpha_e[e])
                ax2.annotate(f'{bar_width:.1f}%', (bar_width/ 2, 0),
                              fontsize=12, ha='center', va='center', c=('white' if e==1 else 'k'))

                ax2.xaxis.set_major_locator(MultipleLocator(25))
                ax2.set_ylim(1, -1)
                # the text above with percent and counts bbox (left, up, spacing, right)
                # ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                #            bbox_to_anchor=(0.05, 0.7, 0.9, 0.1), frameon=False,
                #            fontsize=14, labelcolor='k')  # , labelcolor=label_colors_List)
                bottom_pos += bar_width

                # axis style
                ax2.get_yaxis().set_visible(False)
                ax2.set_xlim(0, 100)
                ax2.spines[['left', 'right', 'top']].set_visible(False)
            # plot title
                ax2.text(0.5, -0.6, (f'fraction of  {pawList[i]} {event} onset modulated {cellType}  (%)'),
                         ha='center', va='center',
                         transform=ax2.transAxes)

                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                if cellType=='MLI':
                    interval = intervals[0]
                else:
                    interval = intervals[2]
                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                modList=['all_mod', 'all_non']
                for m in range(2):
                        # only look at cell modulated showing one category of modulation
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells[pawList[i]][modList[m]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    modulated_paw_psth_df = paw_psth_df.loc[modulated_mask]
                    # drop cells with only one rec
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                    modulated_paw_df['z_score_abs']=abs(modulated_paw_df[zscore_key[0]])+abs(modulated_paw_df[zscore_key[1]])
                    # key for z_score and z_score time arrays
                    z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                    z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
                    # ids of modulated cells
                    cellsId = np.unique(modulated_paw_psth_df['cell_global_Id'])
                    print('cellsID:', cellsId)
                    # exampleCell = 1
                    # choose an example cell for each condition
                    if cellType=='MLI':
                        if condition == 'allSteps':
                            if pawNb == 0 :
                                if (e == 0 and m==0 ):
                                    exampleCell = 21 #19 #25 #31 #41 #2 is perfect
                                elif (e == 0 and m==1) :
                                    exampleCell = 8 #7
                                elif (e==1 and m==0):
                                    exampleCell=2 #19 #13 #3 #21 #25 #36
                                elif (e == 1 and m == 1):
                                    exampleCell=7 #2
                            elif pawNb ==1:
                                exampleCell=18  #16 maybe[ 1  2  4  5  6  9 11 12 13 17 19 21 26 27 29 30 32 34 35 36 37 38 40 41 42 44 45 46 47 50 52 56 58 59 61 63]
                        elif condition == 'swingLengthLinear_lastRec_20_80':
                            if pawNb == 0 :
                                if (e == 0 and m == 0):
                                    exampleCell = 1  # 19 #25 #31 #41 #2 is perfect
                                elif (e == 0 and m == 1):
                                    exampleCell = 1  # 7
                                elif (e == 1 and m == 0):
                                    exampleCell = 1  # 19 #13 #3 #21 #25 #36
                                elif (e == 1 and m == 1):
                                    exampleCell = 1  # 2
                            else:
                                exampleCell = 1
                        else:
                            exampleCell=2
                    else:
                        exampleCell = 5


                                # get the zscore of and zscore time of example cells for first and last trials
                    earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                    lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (
                            modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    # subplot before and after for z-score examples
                        # subplot before and after for z-score examples
                    gssub5b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub5[0], hspace=0.2)
                    if e == 0:
                        # for time before we need two examples for up and down modulated (c==0 and c==1)
                        gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], wspace=0.3)
                        # we don't care about non modulated (c==2)
                        if m < 2:
                            ax1 = plt.subplot(gssub5c[m])
                            # vertical fill concerned time window (before -100ms to 0)
                            # plot the example z-score for first and last trial
                            ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))
                            ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                     alpha=1, label='last trial')

                            ax1.set_ylim(-3.5,4.5)

                            # timeAfterMask=zscoreTime[0]>0
                            ax1.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                            ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                            ax1.legend(frameon=False, loc='upper right')
                    if e == 1:
                        gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], hspace=0.2, wspace=0.3)
                        if m < 2:
                            ax1 = plt.subplot(gssub5c[m])
                            # vertical fill concerned time window (after 0ms to 100)
                            # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                            # plot the example z-score for first and last trial for after event
                            if condition == 'allSteps' and pawNb==0:
                                ax1.set_ylim(-5,10)
                            else:
                                ax1.set_ylim(-3.5, 6)

                            ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                            ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                     alpha=1,
                                     label='last trial')
                            ax1.legend(frameon=False)

                            ax1.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                            ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))

                            ax1.legend(frameon=False, loc='upper right')
                    #draw 0 line
                    ax1.axvline(0, ls='--', color='grey', lw=1,alpha=0.1)
                    ax1.axvline(0, ls='--', color='grey', alpha=0.1)
                    ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    ax1.axhline(0, ls='--', color='grey', alpha=0.1)
                    self.layoutOfPanel(ax1, xLabel=f'time centered on {event} onset (s)', yLabel=('PSTH Z-score' if m==0 else ''), xyInvisible=[(False), False])
                    ax1.yaxis.set_major_locator(MultipleLocator(2))

                    # for FL look only at those behavior parameters

                    behavior_par = ['swingDuration', 'swingLengthLinear']
                    behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                    hr = np.full(len(behavior_par) + 1, 2)
                    hr[0] = 3.2
                    gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) + 1, 2, subplot_spec=gssub5[2],
                                                              hspace=0.5,
                                                              wspace=0.2, height_ratios=hr)

                    # comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0,m])
                    # group the data per cell and trial categories
                    modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()
                    # define colors for seaborn
                    palette = sns.color_palette([f'C{i}'], 1)
                    palette_non = sns.color_palette(['0.8'], 1)
                    grey = sns.color_palette(['grey'], 1)


                    # the invisible bar plot is to tighten things
                    sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y='z_score_abs', ax=ax3,
                                color='k', alpha=0.001, errorbar=None)
                    # lineplot for the line between cells
                    sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y='z_score_abs',
                                 ax=ax3,
                                 palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                    # scatterplot for the AUC values
                    sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id',
                                    y='z_score_abs',
                                    ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                                    lw=0.1, s=80)
                    ax3.set_xlim(-0.9,1.9)
                    #ax3.plot([0,1],[np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']),np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs'])],ls='-',c='k',lw=2)
                    # extract first and last trial z-score AUC arrays
                    late_days = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']
                    first_days = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']
                    # perform t-test
                    t_value, t_test_p_value = stats.ttest_rel(late_days['z_score_abs'], first_days['z_score_abs'],
                                                              axis=0)
                    print('paired t-test (t-value, p): ',event, t_value, t_test_p_value)
                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    if t_test_p_value<0.05:
                        ax3.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                                 transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=18, color='k')
                    else:
                        ax3.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',
                                 transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=12, color='k')
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(
                        f' {event} onset PSTH \n  Z-score {zscorePar} (abs)' if m==0 else ''),
                                       Leg=[1, 9])

                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)
                    # allSteps y-axis limits
                    #if event == 'swing' :
                    #    ax3.set_ylim(0,0.815)
                    #elif event == 'stance':
                    #    ax3.set_ylim(0,1.255)
                    # 20-80 swing length y-axis limits
                    if condition == 'allSteps' and pawNb==0:
                        if event == 'swing':
                            ax3.set_ylim(0, 0.85)
                        elif event == 'stance':
                            ax3.set_ylim(0, 1.25)
                    else:
                        if event == 'swing':
                            ax3.set_ylim(0, 0.45)
                        elif event == 'stance':
                            ax3.set_ylim(0, 0.5)

                    # create a new column with a name without '-' to put the z-score value (necessary because statsmodel and scipy dont like '-' in the key names
                    # zscore_key1 = 'zScoreAUC_' + times[t] + event
                    # modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    # for each behavior parameter perform linear regression
                    for p in range(len(behavior_par)):

                        gssub6a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub6[1+p,m],
                                                                  hspace=0.15,
                                                                  wspace=0.3)
                        trials=['first', 'last']
                        for t in range(2):

                            ax4 = plt.subplot(gssub6a[t])

                            modulated_paw_df.dropna(subset=['z_score_abs'], inplace=True)
                            y=modulated_paw_df[modulated_paw_df['trial_category'] == trials[t]][behavior_par[p]]

                            x = modulated_paw_df[modulated_paw_df['trial_category'] == trials[t]]['z_score_abs']
                            # we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                            try:
                                slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                             alternative='two-sided')
                            except ValueError:
                                pass
                            # set a treshold to reduce alpha and highlight only pvalues<0.05 and r2>0.05
                            if p_value>0.05:
                                alpha=0.1
                            else:
                                alpha=alpha_e[e]


                            # show linear reg plots
                            sns.regplot(x=x, y=y, ax=ax4, color=(f'C{i}'if m==0 else '0.7'),
                                        scatter_kws={'alpha': alpha, 'edgecolor': 'k', 'lw': 0.1},
                                        line_kws={'alpha': alpha, 'lw': 1.5})
                            # annotate with values of r, r², p_values$
                            corr_star=groupAnalysis.starMultiplier(p_value)
                            if p_value>0.05:
                                ax4.text(0.73, 0.80, f"r = {r_value:.2f}\np = {p_value:.2f}",
                                         transform=ax4.transAxes, fontsize=10, color='dimgrey')

                            else:
                                ax4.text(0.73, 0.80, f"r = {r_value:.2f}",
                                         transform=ax4.transAxes, fontsize=10, color='dimgrey')
                                ax4.text(0.5, 0.98, f"{corr_star}",
                                         transform=ax4.transAxes, fontsize=18, color='k')
                            self.layoutOfPanel(ax4, xLabel=('Z-score %s (abs)' % zscorePar if p == len(behavior_par) - 1 else ''),
                                               yLabel=(f'{behavior_par_Name[p]} ' if t==0 else ''),
                                               Leg=[1, 9])
                            print(event,p)
                            # allSteps AND 20-80 percentile y-axis limits
                            if behavior_par[p] == 'swingDuration':
                                ax4.set_ylim(0.065,0.36)
                            elif behavior_par[p] == 'swingLengthLinear':
                                ax4.set_ylim(1.1,4.7)

        paw=pawList[pawNb]
        fname = f'fig_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paw}_{condition}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def psthGroupFigure_cell_based_single_paw__all_modulated_zscore (self, cellType, df_psth, df_cells, condition, pawNb):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 60  # width in inches
        if cellType=='PC':
            fig_height = 45
        else:
            fig_height = 70
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.02)
        modCells = {}
        events = ['swing', 'stance']
        for e in reversed(range(2)):
            event = events[e]
            for i in [pawNb]:
                # looking at specific paw values in the df that contains single values
                paw_df = df_cells[(df_cells['paw'] == pawList[i])]

                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                        paw_psth_df = paw_psth_df.drop(index)
                # big panels for each paw

                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.1, wspace=0.1)

                # sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub4[0, e], hspace=0.1, height_ratios=[2,1.5])

                # time around event for keys
                times = ['before_', 'after_']
                # iterate through time (before/after)

                modCells[pawList[i]] = {}
                modCells[pawList[i]]['all'] = np.empty(0)
                for t in range(2):
                    # modCells[pawList[i]]['before']=np.empty(0)
                    # modCells[pawList[i]]['after']=np.empty(0)
                    # define modulation catgories
                    catList = ['↓', '↑', '-']
                    # time to look at
                    time = times[t]
                    # change position of plot if looking at more than one paw

                    # ax2 = plt.subplot(gssub5[1])

                    # get Id and counts of modulated cells
                    modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df,
                                                                                                       catList,
                                                                                                       time, event,
                                                                                                       condition=None)
                    if t == 0:
                        modCells[pawList[i]]['before'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_before'] = modCells_Id['-']
                    else:
                        modCells[pawList[i]]['after'] = np.concatenate((modCells_Id['↓'], modCells_Id['↑']))
                        modCells[pawList[i]]['not_after'] = modCells_Id['-']
                # pdb.set_trace()
                modCells[pawList[i]]['all_mod'] = np.unique(
                    np.concatenate((modCells[pawList[i]]['before'], modCells[pawList[i]]['after'])))
                # modCells[pawList[i]]['all_non']=np.unique(np.concatenate((modCells[pawList[i]]['not_before'], modCells[pawList[i]]['not_after'])))
                # modCells[pawList[i]]['all_non']=np.setdiff1d(modCells[pawList[i]]['all_non'],np.intersect1d(modCells[pawList[i]]['all_mod'],modCells[pawList[i]]['all_non']))
                all_cells = np.arange(64) + 1
                modCells[pawList[i]]['all_non'] = np.setdiff1d(all_cells, modCells[pawList[i]]['all_mod'])
                # print(len(modCells[pawList[i]]['all_mod'] )+len(modCells[pawList[i]]['all_non'] ))
            # pdb.set_trace()
            bottom_pos = 0
            twoCol = ['0.2', '0.7']
            alpha_e = [0.4, 0.8]
            for i in [pawNb]:
                # bar_width = len(modCells[pawList[i]]['all_mod']) / len(all_cells) * 100
                # bar_left = bottom_pos
                #
                # ax2.barh([0], [100], color='0.9', alpha=0.8)
                # ax2.barh(0, bar_width, left=bar_left, color=f'C{i}', alpha=alpha_e[e])
                # ax2.annotate(f'{bar_width:.1f}%', (bar_width / 2, 0),
                #              fontsize=12, ha='center', va='center', c=('white' if e == 1 else 'k'))
                #
                # ax2.xaxis.set_major_locator(MultipleLocator(25))
                # ax2.set_ylim(1, -1)
                # # the text above with percent and counts bbox (left, up, spacing, right)
                # # ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                # #            bbox_to_anchor=(0.05, 0.7, 0.9, 0.1), frameon=False,
                # #            fontsize=14, labelcolor='k')  # , labelcolor=label_colors_List)
                # bottom_pos += bar_width
                #
                # # axis style
                # ax2.get_yaxis().set_visible(False)
                # ax2.set_xlim(0, 100)
                # ax2.spines[['left', 'right', 'top']].set_visible(False)
                # # plot title
                # ax2.text(0.5, -0.6, (f'fraction of  {pawList[i]} {event} onset modulated {cellType}  (%)'),
                #          ha='center', va='center',
                #          transform=ax2.transAxes)

                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                if cellType == 'MLI':
                    interval = intervals[0]
                else:
                    interval = intervals[0]
                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                modList = ['all_mod', 'all_non']
                modId=['modulated', 'non-modulated']
                for m in range(2):
                    # only look at cell modulated showing one category of modulation
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells[pawList[i]][modList[m]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    modulated_paw_psth_df = paw_psth_df.loc[modulated_mask]
                    # drop cells with only one rec
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                    modulated_paw_df['z_score_abs'] = abs(modulated_paw_df[zscore_key[0]]) + abs(
                        modulated_paw_df[zscore_key[1]])
                    # key for z_score and z_score time arrays
                    z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                    z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
                    # ids of modulated cells
                    cells=modulated_paw_psth_df['cell_global_Id']
                    cells=np.array(cells)
                    dayCat=modulated_paw_psth_df['day_category']
                    dayCatMask=np.argsort(dayCat)
                    orderedCells=cells[dayCatMask]
                    cellsId, cellsIdx = np.unique(orderedCells,return_index=True)
                    cellsId=cellsId[np.argsort(cellsIdx)]
                    # cellsId = np.unique(modulated_paw_psth_df['cell_global_Id'])
                    cellsDayId = np.unique(modulated_paw_psth_df['dayNb'])
                    daySortMask=np.argsort(cellsDayId)
                    cellsDayCat= np.unique(modulated_paw_psth_df['day_category'])
                    print('cellsID:', cellsId)
                    # pdb.set_trace()
                    cell=0
                    if m == 0:
                        if cellType == 'PC':
                            line = 6
                            column = 5
                        else:
                            line = 9
                            column = 5
                    else:
                        if cellType == 'PC':
                            line = 6
                            column = 5
                        else:
                            line = 8
                            column = 5
                    gssub5b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub5[m], hspace=0.2)
                    gssub5c = gridspec.GridSpecFromSubplotSpec(line, column, subplot_spec=gssub5b[0], wspace=0.3)
                    for exampleCell in range(len(cellsId)):

                            # get the zscore of and zscore time of example cells for first and last trials
                        earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                        zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                        lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (
                                modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                        zscoreSingle = modulated_paw_psth_df[(modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                        zscoreSingleTime = modulated_paw_psth_df[(modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][
                                z_scoreTimeKey].values
                        category=np.unique(modulated_paw_psth_df[(modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])]['day_category'])[0]

                        # subplot before and after for z-score examples
                        # subplot before and after for z-score examples
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler(feature_range=(-2, 4))
                        if e == 0:
                            # for time before we need two examples for up and down modulated (c==0 and c==1)


                            # we don't care about non modulated (c==2)
                            if cell==2:
                                ax1.text(0.5, 1.2, f'{event} {modId[m]} {cellType}',
                                         fontsize=50,
                                         transform=ax1.transAxes)
                            if m < 2:
                                ax1 = plt.subplot(gssub5c[cell+1])
                                ax1a = plt.subplot(gssub5c[0])
                                ax1.text(0.02, 0.8, f'Id {cellsId[exampleCell]}\nIdx {exampleCell}\n{category}', fontsize=20,
                                         transform=ax1.transAxes)
                                try:
                                    # ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid',
                                    #          color=(f'C{i}' if m == 0 else '0.6'),
                                    #          lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                                    ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid',
                                             color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                                             alpha=1, label='last trial')
                                    # scaled_values = scaler.fit_transform(lateZscoreA[0][1].reshape(-1, 1))
                                    #
                                    # scaled_values = scaled_values.flatten()
                                    if np.max(lateZscoreA[0][1])<10:
                                        smoothed = gaussian_filter1d(lateZscoreA[0][1], 0.8)
                                        ax1a.plot(zscoreTime[0], smoothed, lw=0.5,
                                                 alpha=0.4, label='last trial')
                                    # ax1.set_ylim(-3.5, 4.5)

                                    # timeAfterMask=zscoreTime[0]>0
                                    # ax1.fill_between(zscoreTime[0], lateZscoreA[0][1], 0,
                                    #                  where=((zscoreTime[0] >= (-0.1)) & (
                                    #                          zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0,
                                    #                  color=(f'C{i}' if m == 0 else '0.6'))
                                    #
                                    # ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1], 0,
                                    #                  where=((zscoreTime[0] >= (-0.1)) & (
                                    #                          zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0,
                                    #                  color=(f'C{i}' if m == 0 else '0.6'))
                                except IndexError:
                                    ax1.step(zscoreSingleTime[0], zscoreSingle[0][1], where='mid',
                                             color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                                             alpha=1, label='last trial')
                            if exampleCell==1:
                                ax1.legend(frameon=False, loc='upper right')
                        if e == 1:
                            # gssub5c = gridspec.GridSpecFromSubplotSpec(line, column, subplot_spec=gssub5b[0], hspace=0.2,
                            #                                            wspace=0.3)
                            if cell==2:
                                ax1.text(0.5, 1.2, f'{event} {modId[m]} {cellType}',
                                         fontsize=50,
                                         transform=ax1.transAxes)
                            if m < 2:
                                ax1 = plt.subplot(gssub5c[cell+1])
                                ax1a = plt.subplot(gssub5c[0])
                                ax1.text(0.02, 0.8, f'Id {cellsId[exampleCell]}\nIdx {exampleCell}\n{category}', fontsize=20, transform=ax1.transAxes)

                                # vertical fill concerned time window (after 0ms to 100)
                                # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                                # plot the example z-score for first and last trial for after event
                                # if condition == 'allSteps' and pawNb == 0:
                                #     ax1.set_ylim(-5, 10)
                                # else:
                                #     ax1.set_ylim(-3.5, 6)
                                try:
                                    # ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid',
                                    #          color=(f'C{i}' if m == 0 else '0.6'),
                                    #          lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                                    ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid',
                                             color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                                             alpha=1,
                                             label='last trial')

                                    # scaled_values = scaler.fit_transform(lateZscoreA[0][1].reshape(-1, 1))
                                    #
                                    # scaled_values = scaled_values.flatten()
                                    if np.max(lateZscoreA[0][1])<8:
                                        smoothed = gaussian_filter1d(lateZscoreA[0][1], 0.8)
                                        ax1a.plot(zscoreTime[0], smoothed, lw=0.5,
                                                 alpha=0.4, label='last trial')
                                    if exampleCell == 1:
                                        ax1.legend(frameon=False, loc='upper right')

                                    # ax1.fill_between(zscoreTime[0], lateZscoreA[0][1], 0,
                                    #                  where=((zscoreTime[0] >= (-0.1)) & (
                                    #                          zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0,
                                    #                  color=(f'C{i}' if m == 0 else '0.6'))
                                    #
                                    # ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1], 0,
                                    #                  where=((zscoreTime[0] >= (-0.1)) & (
                                    #                          zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0,
                                    #                  color=(f'C{i}' if m == 0 else '0.6'))
                                except IndexError:
                                    print('single trial cell', cellsId[exampleCell])
                                    ax1.step(zscoreSingleTime[0], zscoreSingle[0][1], where='mid',
                                             color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                                             alpha=1, label='last trial')

                                if exampleCell == 1:
                                    ax1.legend(frameon=False, loc='upper right')
                        # draw 0 line
                        ax1.axvline(0, ls='--', color='grey', lw=1, alpha=0.3)
                        ax1.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
                        ax1.axhline(0, ls='--', color='grey', alpha=0.3)
                        self.layoutOfPanel(ax1, xLabel=f'', yLabel=('' if m == 0 else ''), xyInvisible=[(False), False])
                        ax1.yaxis.set_major_locator(MultipleLocator(2))
                        ax1a.axvline(0, ls='--', color='grey', lw=1, alpha=0.3)
                        ax1a.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax1a.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
                        ax1a.axhline(0, ls='--', color='grey', alpha=0.3)
                        self.layoutOfPanel(ax1a, xLabel=f'', yLabel=('' if m == 0 else ''),
                                           xyInvisible=[(False), False])
                        ax1a.yaxis.set_major_locator(MultipleLocator(2))
                        cell += 1

        paw=pawList[pawNb]
        fname = f'fig_ephys_psth_Z-score_{zscorePar}_allModCells_{cellType}_{paw}_{condition}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def averageZscoreMLI_PC(self, df_psth_MLI, df_MLI, df_psth_PC, df_PC):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 15  # width in inches

        fig_height = 14
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        modCells = {}
        events = ['swing', 'stance']
        pawNb = [0,1,2,3]
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.2,
                                                  wspace=0.3)
        for e in range(2):
            event = events[e]
            for i in pawNb:
                # looking at specific paw values in the df that contains single values
                paw_df_MLI = df_MLI[(df_MLI['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_MLI = df_psth_MLI[(df_psth_MLI['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                paw_df_PC = df_PC[(df_PC['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_PC = df_psth_PC[(df_psth_PC['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_PC['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df_MLI = paw_df_MLI.drop(index)
                        paw_psth_df_MLI = paw_psth_df_MLI.drop(index)
                        paw_df_PC = paw_df_PC.drop(index)
                        paw_psth_df_PC = paw_psth_df_PC.drop(index)
                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

                interval = intervals[0]

                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]

                modCells_MLI = groupAnalysis_psth.getModCellsDic(paw_df_MLI, event, pawList[i])
                modCells_PC = groupAnalysis_psth.getModCellsDic(paw_df_PC, event, pawList[i])

                modulated_mask_MLI = paw_df_MLI['cell_global_Id'].isin(modCells_MLI[pawList[i]]['all_mod'])
                modulated_paw_df_MLI = paw_df_MLI.loc[modulated_mask_MLI]
                modulated_paw_psth_df_MLI = paw_psth_df_MLI.loc[modulated_mask_MLI]

                modulated_mask_PC = paw_df_PC['cell_global_Id'].isin(modCells_PC[pawList[i]]['all_mod'])
                modulated_paw_df_PC = paw_df_PC.loc[modulated_mask_PC]
                modulated_paw_psth_df_PC = paw_psth_df_PC.loc[modulated_mask_PC]
                modulatedList = [modulated_paw_df_MLI, modulated_paw_df_PC]
                zscoreArrayList = [modulated_paw_psth_df_MLI, modulated_paw_psth_df_PC]
                z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                z_scoreTimeKey = f'psth_{event}OnsetAligned_time'

                times = ['before', 'after']
                cells = ['MLI', 'PC']
                for t in range(2):
                    gssub1a = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gssub1[e], hspace=0.2,
                                                               wspace=0.2)
                    for m in [0,1]:
                        cellsId = np.unique(modulatedList[m]['cell_global_Id'])
                        upCells, downCells, zscoreArray,zscoreArrayUp, zscoreArrayDown,zscoreSingleTime = groupAnalysis_psth.classify_cells_with_tsne(cellsId, zscoreArrayList, m, z_scoreKey, z_scoreTimeKey, interval)

                        # pdb.set_trace()
                        # zscoreArray = []
                        # zscoreArrayUp = []
                        # zscoreArrayDown = []
                        # upCells = []
                        # downCells = []
                        # for c in range(len(cellsId)):
                        #     zscoreSingle = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][
                        #         z_scoreKey].values
                        #
                        #     zscoreSingleTime = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][
                        #         z_scoreTimeKey].values
                        #     zscore = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[c])][
                        #         z_scoreKey].values
                        #     zscoreArray.append(zscore[0][1])
                        #     # if e == 1:
                        #     # timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                        #     #  else:
                        #     # timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                        #     timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                        #     if np.mean(zscore[0][1][timeMask]) > 0:
                        #         zscoreArrayUp.append(zscore[0][1])
                        #         upCells.append(cellsId[c])
                        #     else:
                        #         zscoreArrayDown.append(zscore[0][1])
                        #         downCells.append(cellsId[c])
                        meanZscoreUp = gaussian_filter1d(np.mean(zscoreArrayUp, axis=0), 0.8)
                        meanZscoreDown = gaussian_filter1d(np.mean(zscoreArrayDown, axis=0), 0.8)
                        semZscoreUp = gaussian_filter1d(stats.sem(zscoreArrayUp, axis=0), 0.8)
                        semZscoreDown = gaussian_filter1d(stats.sem(zscoreArrayDown, axis=0), 0.8)
                        meanZscore = np.mean(zscoreArray, axis=0)
                        meanZscore = gaussian_filter1d(meanZscore, 0.8)
                        semZscore = stats.sem(zscoreArray, axis=0)
                        semZscore = gaussian_filter1d(semZscore, 0.8)
                        varList = [meanZscore]
                        semList = [semZscore]
                        PCINames=['PC']
                        if m == 0:
                            varList = [meanZscore]
                            semList = [semZscore]
                        else:
                            varList = [meanZscoreUp, meanZscoreDown]
                            semList = [semZscoreUp, semZscoreDown]
                            PCINames = ['PC 1', 'PC 2']
                            upDownPCNb = [len(upCells), len(downCells)]
                        ax0 = plt.subplot(gssub1a[i])
                        if e == 1:
                            timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                        else:
                            timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                        maxTimeMask = (meanZscore[timeMask] == np.max(meanZscore[timeMask]))
                        # ax0.axvline(zscoreSingleTime[0][timeMask][maxTimeMask], ls='--', color=(f'C{i}' if m == 0 else 'C4'), alpha=0.6)
                        for v in range(len(varList)):
                            ax0.plot(zscoreSingleTime[0], varList[v],
                                     color=(f'C{i}' if m != 1 else f'C{v + 4}'), lw=2,
                                     alpha=(1 if v==0 else 0.6),
                                     label=(f' MLI ({len(cellsId)})' if m != 1 else f'{PCINames[v]} ({upDownPCNb[v]})'))
                            ax0.fill_between(zscoreSingleTime[0], varList[v] - semList[v], varList[v] + semList[v],
                                             color=(f'C{i}' if m != 1 else f'C{v + 4}'), alpha=(0.1 if v==0 else 0.06),)
                            if m==0:
                                ax0.set_ylim(-2,2.5)
                            else:
                                ax0.set_ylim(-2, 2.5)
                            self.layoutOfPanel(ax0, yLabel=f' {cells[m]} average z-score ', xLabel=f'{event} onset',
                                               xyInvisible=[(False), (True if i!=0  else False)])
                            # if t == 0:
                        ax0.legend(frameon=False, fontsize=10)
                        ax0.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax0.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
        # paw = pawList[i]
        fname = f'fig_ephys_avg_Z-score_AUC_ModCells_MLI_PC'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def classifyAverageZscoreMLI_PC(self, df_psth_MLI, df_MLI, df_psth_PC, df_PC):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 22  # width in inches

        fig_height = 12
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.25)
        modCells = {}
        events = ['swing', 'stance']
        pawNb = [0,1]
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.1,
                                                  wspace=0.3)
        for e in range(2):
            event = events[e]
            for i in pawNb:
                # looking at specific paw values in the df that contains single values
                paw_df_MLI = df_MLI[(df_MLI['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_MLI = df_psth_MLI[(df_psth_MLI['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                paw_df_PC = df_PC[(df_PC['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_PC = df_psth_PC[(df_psth_PC['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_PC['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame


                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df_MLI = paw_df_MLI.drop(index)
                        paw_psth_df_MLI = paw_psth_df_MLI.drop(index)
                        paw_df_PC = paw_df_PC.drop(index)
                        paw_psth_df_PC = paw_psth_df_PC.drop(index)
                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

                interval = intervals[0]

                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                timeP = ['early', 'late']


                    
                modCells_MLI = groupAnalysis_psth.getModCellsDic(paw_df_MLI, event, pawList[i])
                modCells_PC = groupAnalysis_psth.getModCellsDic(paw_df_PC, event, pawList[i])

                modulated_mask_MLI = paw_df_MLI['cell_global_Id'].isin(modCells_MLI[pawList[i]]['all_mod'])
                modulated_paw_df_MLI = paw_df_MLI.loc[modulated_mask_MLI]
                modulated_paw_psth_df_MLI = paw_psth_df_MLI.loc[modulated_mask_MLI]

                modulated_mask_PC = paw_df_PC['cell_global_Id'].isin(modCells_PC[pawList[i]]['all_mod'])
                modulated_paw_df_PC = paw_df_PC.loc[modulated_mask_PC]
                modulated_paw_psth_df_PC = paw_psth_df_PC.loc[modulated_mask_PC]

                timeDfMLI_early=modulated_paw_psth_df_MLI[(modulated_paw_psth_df_MLI['day_category'] == timeP[0])]
                timeDfPC_early=modulated_paw_psth_df_PC[(modulated_paw_psth_df_PC['day_category'] == timeP[0])]
                timeDfMLI_late=modulated_paw_psth_df_MLI[(modulated_paw_psth_df_MLI['day_category'] == timeP[1])]
                timeDfPC_late=modulated_paw_psth_df_PC[(modulated_paw_psth_df_PC['day_category'] == timeP[1])]

                modulatedList = [modulated_paw_df_MLI, modulated_paw_df_PC]
                zscoreArrayList = [modulated_paw_psth_df_MLI, modulated_paw_psth_df_PC]

                zscoreArrayList_all= [modulated_paw_psth_df_MLI, timeDfMLI_early,timeDfMLI_late,modulated_paw_psth_df_PC,timeDfPC_early,timeDfPC_late]
                zscoreArrayList_all = [modulated_paw_psth_df_MLI,
                                       modulated_paw_psth_df_PC]
                zscoreArrayList_all_Ids = ['all MLI', 'early MLI',' late MLI', 'all PC','early PC', 'late PC']
                zscoreArrayList_all_Ids = ['all MLI', 'all PC']
                z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                z_scoreTimeKey = f'psth_{event}OnsetAligned_time'

                # times = ['before', 'after']
                cells = ['MLI', 'PC']


                gssub1a = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gssub1[e], hspace=0.4,
                                                           wspace=0.4)
                for m in range(len(zscoreArrayList_all)):

                    gssub1b = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gssub1a[0,m], hspace=0.4,
                                                               wspace=0.2)

                    gssub1c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub1a[1:3, m],
                                                               hspace=0.4,
                                                               wspace=0.2)
                    ax0 = plt.subplot(gssub1b[i])
                    #else:


                    cellsId = np.unique(zscoreArrayList_all[m]['cell_global_Id'])
                    earlyMod = np.unique(zscoreArrayList_all[m][(zscoreArrayList_all[m]['day_category'] == timeP[0])]['cell_global_Id'])
                    lateMod = np.unique(zscoreArrayList_all[m][(zscoreArrayList_all[m]['day_category'] == timeP[1])]['cell_global_Id'])
                    timeList=[earlyMod,lateMod]
                    method='Kmeans'
                    if len(cellsId)>2:
                        # best_cluster_num,cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime= groupAnalysis_psth.find_cell_clusters(cellsId, zscoreArrayList_all, m, z_scoreKey, z_scoreTimeKey, interval, zscoreArrayList_all_Ids[m], max_clusters=5, showFig=False)
                        best_cluster_num, cluster_ids, cluster_zscores, zscoreArray, zscoreSingleTime = groupAnalysis_psth.find_cell_clusters(
                            cellsId, zscoreArrayList_all, m, z_scoreKey, z_scoreTimeKey, interval,
                            zscoreArrayList_all_Ids[m], method,max_clusters=10, showFig=False, )
                        # pdb.set_trace()
                        # try:
                        print(best_cluster_num)
                        if  best_cluster_num> 1:
                            max_cells = max(len(cluster) for cluster in cluster_zscores)
                            alpha_start = 0.1  # Initial alpha value for the first line
                            alpha_end = 1.0  # Maximum alpha value for the last line
                            alpha_range = alpha_end - alpha_start

                            for cl in range(len(cluster_zscores)):
                                meanClusterZscore=np.nanmean(cluster_zscores[cl], axis=0)
                                semClusterZscore = stats.sem(cluster_zscores[cl], axis=0, nan_policy='omit')
                                nCells=len(cluster_zscores[cl])


                                try:
                                    meanClusterZscore = gaussian_filter1d(meanClusterZscore, 0.8)
                                    semClusterZscore = gaussian_filter1d(semClusterZscore, 0.8)

                                except:
                                    pass
                                try:
                                    alpha = alpha_start + alpha_range * (nCells - 3) / (max_cells - 3)
                                except:
                                    alpha=1
                                if alpha>1:
                                    alpha=1
                                if nCells>3:
                                    ax0.plot(zscoreSingleTime[0], meanClusterZscore, color=(f'C{i}'), lw=2, alpha=alpha,label=(f' {zscoreArrayList_all_Ids[m]} ({nCells})'))
                                    ax0.fill_between(zscoreSingleTime[0], meanClusterZscore- semClusterZscore, meanClusterZscore + semClusterZscore,color=(f'C{i}'),alpha=alpha/5)
                                timeIds=['early','late']
                                if cl!=90:

                                    for tm in range(2):

                                        ax0b = plt.subplot(gssub1c[tm,i])
                                        timeIdx, timeVal=groupAnalysis_psth.get_matching_indices_and_elements(cluster_ids[cl],timeList[tm])
                                        timeIdx=np.array(timeIdx)
                                        clusterArray=np.array(cluster_zscores[cl])
                                        if timeIdx.size>0:
                                            meanClusterZscore_time=np.nanmean(clusterArray[timeIdx], axis=0)
                                            semClusterZscore_time= stats.sem(clusterArray[timeIdx], axis=0, nan_policy='omit')
                                            try:
                                                meanClusterZscore_time = gaussian_filter1d(meanClusterZscore_time, 0.8)
                                                semClusterZscore_time = gaussian_filter1d(semClusterZscore_time, 0.8)
                                            except:
                                                pass
                                            max_cells = max(len(cluster) for cluster in cluster_zscores)
                                            nCellsTime=len(timeVal)
                                            alpha1 = alpha_start + alpha_range * (nCellsTime ) / (max_cells - 3)
                                            if alpha1 > 1:
                                                alpha1 = 1
                                            if nCellsTime>2:
                                                ax0b.plot(zscoreSingleTime[0], meanClusterZscore_time, color=(f'C{i}'), lw=2, alpha=alpha1,
                                                     label=(f' {timeIds[tm]} {cells[m]} ({nCellsTime})'))
                                                ax0b.fill_between(zscoreSingleTime[0], meanClusterZscore_time - semClusterZscore_time,
                                                         meanClusterZscore_time + semClusterZscore_time, color=(f'C{i}'), alpha=alpha1 / 3)
                                        self.layoutOfPanel(ax0b, yLabel=f' {timeIds[tm]} {cells[m]} average z-score ', xLabel=f'{event} onset',
                                               xyInvisible=[(False), (True if i != 0 else False)])
                                        ax0b.legend(frameon=False, fontsize=10)
                                        ax0b.set_ylim(-2, 4)
                        elif best_cluster_num==1:
                            meanClusterZscore = np.nanmean(cluster_zscores, axis=0)

                            semClusterZscore = stats.sem(cluster_zscores, axis=0, nan_policy='omit')
                            nCells = len(cluster_zscores)
                            pdb.set_trace()
                            if nCells>3:
                                ax0.plot(zscoreSingleTime[0], meanClusterZscore, color=(f'C{i}'), lw=2, alpha=1,
                                         label=(f' {zscoreArrayList_all_Ids[m]} ({nCells})'))
                                ax0.fill_between(zscoreSingleTime[0], meanClusterZscore - semClusterZscore,
                                                 meanClusterZscore + semClusterZscore, color=(f'C{i}'), alpha=(0.1))

                            for tm in range(2):
                                gssub1c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1a[1 + tm, m],
                                                                           hspace=0.4,
                                                                           wspace=0.2)
                                ax0b = plt.subplot(gssub1c[i])
                                timeIdx, timeVal = groupAnalysis_psth.get_matching_indices_and_elements(cluster_ids,
                                                                                                        timeList[tm])
                                timeIdx = np.array(timeIdx)
                                clusterArray = np.array(cluster_zscores)
                                pdb.set_trace()
                                meanClusterZscore_time = np.nanmean(clusterArray[timeIdx], axis=0)
                                semClusterZscore_time = stats.sem(clusterArray[timeIdx], axis=0, nan_policy='omit')
                                try:
                                    meanClusterZscore_time = gaussian_filter1d(meanClusterZscore_time, 0.8)
                                    semClusterZscore_time = gaussian_filter1d(semClusterZscore_time, 0.8)
                                except:
                                    pass
                                max_cells = max(len(cluster) for cluster in cluster_zscores)
                                nCellsTime = len(timeVal)
                                alpha1 = alpha_start + alpha_range * (nCellsTime) / (max_cells - 4)
                                ax0b.plot(zscoreSingleTime[0], meanClusterZscore_time, color=(f'C{i}'), lw=2, alpha=alpha1,
                                          label=(f' {timeIds[tm]} {cells[m]} ({nCellsTime})'))
                                ax0b.fill_between(zscoreSingleTime[0], meanClusterZscore_time - semClusterZscore_time,
                                                  meanClusterZscore_time + semClusterZscore_time, color=(f'C{i}'),
                                                  alpha=alpha1 / 3)
                                self.layoutOfPanel(ax0b, yLabel=f' {timeIds[tm]} {cells[m]} average z-score ',
                                                   xLabel=f'{event} onset',
                                                   xyInvisible=[(False), (True if i != 0 else False)])
                                ax0b.legend(frameon=False, fontsize=10)
                        # if m == 0:
                        #     ax0.set_ylim(-2, 2.5)
                        # else:

                        self.layoutOfPanel(ax0, yLabel=f' {zscoreArrayList_all_Ids[m]} average z-score ', xLabel=f'{event} onset',
                                           xyInvisible=[(False), (True if i != 0 else False)])
                        # if t == 0:
                        ax0.legend(frameon=False, fontsize=10)
                        ax0.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax0.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
                        ax0.set_ylim(-2, 4)

                        # if t == 0:

                        ax0b.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax0b.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)


        fname = f'fig_ephys_avg_Z-score_AUC_ModCells_MLI_PC_overTime_front_paws-{method}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def compareMLI_PC (self, df_psth_MLI,df_MLI,df_psth_PC,df_PC):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 15  # width in inches

        fig_height = 15
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)
        modCells = {}
        events = ['swing', 'stance']
        pawNb=0
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2,
                                                  wspace=0.3)
        for e in range(2):
            event = events[e]
            for i in [pawNb]:
                # looking at specific paw values in the df that contains single values
                paw_df_MLI = df_MLI[(df_MLI['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_MLI = df_psth_MLI[(df_psth_MLI['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_MLI['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                # looking at specific paw values in the df that contains single values
                paw_df_PC = df_PC[(df_PC['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_PC = df_psth_PC[(df_psth_PC['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_PC['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df_MLI = paw_df_MLI.drop(index)
                        paw_psth_df_MLI = paw_psth_df_MLI.drop(index)
                        paw_df_PC = paw_df_PC.drop(index)
                        paw_psth_df_PC = paw_psth_df_PC.drop(index)
                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

                interval = intervals[0]

                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                #t_value, t_test_p_value = stats.ttest_ind(paw_df_MLI[zscore_key[0]], paw_df_PC[zscore_key[0]],axis=0)
                #t_value1, t_test_p_value1 = stats.ttest_ind(paw_df_MLI[zscore_key[1]], paw_df_PC[zscore_key[1]], axis=0)
                #print('before', event, 'p=', t_test_p_value, 't=', t_value1, 'average MLI', np.nanmean(paw_df_MLI[zscore_key[0]]), 'average PC', np.nanmean(paw_df_PC[zscore_key[0]]))
                #print('after',event, 'p=',t_test_p_value1, 't=', t_value1,  'average MLI', np.nanmean(paw_df_MLI[zscore_key[1]]), 'average PC', np.nanmean(paw_df_PC[zscore_key[1]]))

                gssub1a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub1[e], hspace=0.2,
                                                           wspace=0.45, height_ratios=[2,1])

                modCells_MLI=groupAnalysis_psth.getModCellsDic(paw_df_MLI,event, pawList[i])
                modCells_PC = groupAnalysis_psth.getModCellsDic(paw_df_PC, event, pawList[i])
                
                modulated_mask_MLI= paw_df_MLI['cell_global_Id'].isin(modCells_MLI[pawList[i]]['all_mod'])
                modulated_paw_df_MLI = paw_df_MLI.loc[modulated_mask_MLI]
                modulated_paw_psth_df_MLI = paw_psth_df_MLI.loc[modulated_mask_MLI]
                
                modulated_mask_PC= paw_df_PC['cell_global_Id'].isin(modCells_PC[pawList[i]]['all_mod'])
                modulated_paw_df_PC = paw_df_PC.loc[modulated_mask_PC]
                modulated_paw_psth_df_PC = paw_psth_df_PC.loc[modulated_mask_PC]
                modulatedList=[modulated_paw_df_MLI, modulated_paw_df_PC]
                zscoreArrayList=[modulated_paw_psth_df_MLI,modulated_paw_psth_df_PC]
                z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
               # t_value, t_test_p_value = stats.ttest_ind(modulated_paw_df_MLI[zscore_key[0]], modulated_paw_df_PC[zscore_key[0]],
                                                          #axis=0)

                times=['before','after']
                cells=['MLI','PC']
                for t in range(2):

                    for m in range(1):
                        if m==0 and e==1:
                            exampleCell=15
                        elif m==0 and e==0:
                            exampleCell = 34
                        elif m == 1 and e == 0:
                            exampleCell=11
                        elif m == 1 and e == 1:
                            exampleCell = 10

                        cellsId = np.unique(modulatedList[m]['cell_global_Id'])
                        zscoreArray=[]
                        zscoreArrayUp=[]
                        zscoreArrayDown = []
                        upCells=[]
                        downCells = []
                        for c in range(len(cellsId)):
                            zscoreSingle =  zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreKey].values

                            zscoreSingleTime =  zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][z_scoreTimeKey].values
                            zscore = zscoreArrayList[m][ (zscoreArrayList[m]['cell_global_Id'] == cellsId[c])][z_scoreKey].values
                            zscoreArray.append(zscore[0][1])
                           # if e == 1:
                               # timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                          #  else:
                                #timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                            timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                            if np.mean(zscore[0][1][timeMask])>0 :
                                zscoreArrayUp.append(zscore[0][1])
                                upCells.append(cellsId[c])
                            else:
                                zscoreArrayDown.append(zscore[0][1])
                                downCells.append(cellsId[c])
                        meanZscoreUp=gaussian_filter1d(np.mean(zscoreArrayUp, axis=0), 0.8)
                        meanZscoreDown = gaussian_filter1d(np.mean(zscoreArrayDown, axis=0), 0.8)
                        semZscoreUp = gaussian_filter1d(stats.sem(zscoreArrayUp, axis=0), 0.8)
                        semZscoreDown = gaussian_filter1d(stats.sem(zscoreArrayDown, axis=0), 0.8)
                        meanZscore=np.mean(zscoreArray, axis=0)
                        meanZscore=gaussian_filter1d(meanZscore, 0.8)
                        semZscore = stats.sem(zscoreArray, axis=0)
                        semZscore = gaussian_filter1d(semZscore, 0.8)
                        varList = [meanZscore]
                        semList = [semZscore]
                        if m==0:
                           varList = [meanZscore]
                           semList = [semZscore]
                        else:
                            varList=[meanZscoreUp,meanZscoreDown]
                            semList=[semZscoreUp,semZscoreDown]
                            PCINames=['PC 1', 'PC 2']
                            upDownPCNb=[len(upCells),len(downCells)]
                        ax0=plt.subplot(gssub1a[0,0:2])
                        if e==1:
                            timeMask=(zscoreSingleTime[0]<0)&(zscoreSingleTime[0]>-interval)
                        else:
                            timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                        maxTimeMask = (meanZscore[timeMask] == np.max(meanZscore[timeMask]))
                        #ax0.axvline(zscoreSingleTime[0][timeMask][maxTimeMask], ls='--', color=(f'C{i}' if m == 0 else 'C4'), alpha=0.6)
                        for v in range(len(varList)):
                            ax0.plot(zscoreSingleTime[0], varList[v],
                                     color=(f'C{i}' if m == 0 else f'C{v+4}'), lw=2,
                                     alpha=1, label=(f'MLI ({len(cellsId)})' if m==0 else f'{PCINames[v]} ({upDownPCNb[v]})'))
                            ax0.fill_between(zscoreSingleTime[0], varList[v]-semList[v], varList[v]+semList[v],
                                     color=(f'C{i}' if m == 0 else f'C{v+4}'), alpha=0.1)

                            self.layoutOfPanel(ax0, yLabel=f'z-score' ,xLabel=f'{event} onset',
                                               xyInvisible=[(False), False])
                            if t==0:
                                ax0.legend(frameon=False)
                        ax0.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax0.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)

                        modulatedList = [modulated_paw_df_MLI, modulated_paw_df_PC]
                        zscoreMLI=modulated_paw_df_MLI.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean = np.mean(zscoreMLI[zscore_key[t]])
                        zscore_sem = stats.sem(zscoreMLI[zscore_key[t]])
                        zscore=zscoreMLI[zscore_key[t]]

                        modulated_up_mask = modulatedList[m]['cell_global_Id'].isin(upCells)
                        modulated_up = modulatedList[m].loc[modulated_up_mask]
                        modulated_up = modulated_up.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean_up = np.mean(modulated_up[zscore_key[t]])
                        zscore_sem_up = stats.sem(modulated_up[zscore_key[t]])
                        zscore_up=modulated_up[zscore_key[t]]
                        
                        modulated_down_mask = modulatedList[m]['cell_global_Id'].isin(downCells)
                        modulated_down = modulatedList[m].loc[modulated_down_mask]
                        modulated_down = modulated_down.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean_down = np.mean(modulated_down[zscore_key[t]])
                        zscore_sem_down = stats.sem(modulated_down[zscore_key[t]])
                        zscore_down=modulated_down[zscore_key[t]]



                        ax1 = plt.subplot(gssub1a[1,t])
                        if m==0:
                            ax1.bar(0,zscore_mean, yerr=zscore_sem, color=(f'C{i}' if m == 0 else 'C4'))
                            ax1.scatter(np.zeros(len(zscore)), zscore, edgecolor=(f'C{i}' if m == 0 else 'C4'), facecolor='white', marker='o')
                        else:
                            ax1.bar(1, zscore_mean_up,  yerr=zscore_sem_up,  color=(f'C{i}' if m == 0 else 'C4'))
                            ax1.scatter(np.ones(len(zscore_up)), zscore_up, edgecolor=(f'C{i}' if m == 0 else 'C4'), facecolor='white', marker='o')
                            ax1.bar(2, zscore_mean_down,  yerr=zscore_sem_down,  color=(f'C{i}' if m == 0 else 'C5'))
                            ax1.scatter(np.repeat(2,len(zscore_down)), zscore_down, edgecolor=(f'C{i}' if m == 0 else 'C5'), facecolor='white', marker='o')
                        ax1.xaxis.set_major_locator(MultipleLocator(1))
                        ax1.set_xticklabels(['','MLI','PC \nup', 'PC \ndown'])
                        plt.setp(ax1.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                        self.layoutOfPanel(ax1, yLabel=f'mean z-score AUC {times[t]} {event}', xyInvisible=[(False), False])
                    onewayStats, OneWay_p_value = stats.f_oneway(zscore, zscore_up, zscore_down,
                                                          axis=0)
                    if OneWay_p_value < 0.05:
                        t_value1, t_test_p_value1 = stats.ttest_ind(zscore, zscore_up, axis=0)
                        t_value2, t_test_p_value2 = stats.ttest_ind(zscore, zscore_down, axis=0)
                        t_value3, t_test_p_value3 = stats.ttest_ind(zscore_up, zscore_down, axis=0)

                        pvalues = [t_test_p_value1, t_test_p_value2, t_test_p_value3]
                        xpos = [0.28, 0.5, 0.72]
                        for p in range(3):
                            star_trial = groupAnalysis.starMultiplier(pvalues[p])
                            if pvalues[p] < 0.05:
                                ax1.text(xpos[p], (0.90 if p != 1 else 0.99), (f'{star_trial} '), ha='center',
                                         va='center',
                                         transform=ax1.transAxes,
                                         style='italic', fontfamily='serif', fontsize=18, color='k')
                            else:
                                ax1.text(xpos[p], (0.90 if p != 1 else 0.99), (f'p={pvalues[p]:.2f}'), ha='center',
                                         va='center',
                                         transform=ax1.transAxes,
                                         style='italic', fontfamily='serif', fontsize=12, color='k')
                    else:
                        ax1.text(0.5,0.99, 'n.s', ha='center',
                                 va='center',
                                 transform=ax1.transAxes,
                                 style='italic', fontfamily='serif', fontsize=12, color='k')




            paw = pawList[pawNb]
            fname = f'fig_ephys_psth_Z-score_{zscorePar}_avgModCells_{paw}'
            # plt.savefig(fname + '.png')
            # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def compareMLI_PC(self, df_psth_MLI, df_MLI, df_psth_PC, df_PC):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 15  # width in inches

        fig_height = 15
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 16, 'font.size': 16, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)
        modCells = {}
        events = ['swing', 'stance']
        pawNb = 0
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2,
                                                  wspace=0.3)
        for e in range(2):
            event = events[e]
            for i in [pawNb]:
                # looking at specific paw values in the df that contains single values
                paw_df_MLI = df_MLI[(df_MLI['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_MLI = df_psth_MLI[(df_psth_MLI['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_MLI['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                # looking at specific paw values in the df that contains single values
                paw_df_PC = df_PC[(df_PC['paw'] == pawList[i])]
                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df_PC = df_psth_PC[(df_psth_PC['paw'] == pawList[i])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df_PC['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df_MLI = paw_df_MLI.drop(index)
                        paw_psth_df_MLI = paw_psth_df_MLI.drop(index)
                        paw_df_PC = paw_df_PC.drop(index)
                        paw_psth_df_PC = paw_psth_df_PC.drop(index)
                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

                interval = intervals[0]

                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                # t_value, t_test_p_value = stats.ttest_ind(paw_df_MLI[zscore_key[0]], paw_df_PC[zscore_key[0]],axis=0)
                # t_value1, t_test_p_value1 = stats.ttest_ind(paw_df_MLI[zscore_key[1]], paw_df_PC[zscore_key[1]], axis=0)
                # print('before', event, 'p=', t_test_p_value, 't=', t_value1, 'average MLI', np.nanmean(paw_df_MLI[zscore_key[0]]), 'average PC', np.nanmean(paw_df_PC[zscore_key[0]]))
                # print('after',event, 'p=',t_test_p_value1, 't=', t_value1,  'average MLI', np.nanmean(paw_df_MLI[zscore_key[1]]), 'average PC', np.nanmean(paw_df_PC[zscore_key[1]]))

                gssub1a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub1[e], hspace=0.2,
                                                           wspace=0.45, height_ratios=[2, 1])

                modCells_MLI = groupAnalysis_psth.getModCellsDic(paw_df_MLI, event, pawList[i])
                modCells_PC = groupAnalysis_psth.getModCellsDic(paw_df_PC, event, pawList[i])

                modulated_mask_MLI = paw_df_MLI['cell_global_Id'].isin(modCells_MLI[pawList[i]]['all_mod'])
                modulated_paw_df_MLI = paw_df_MLI.loc[modulated_mask_MLI]
                modulated_paw_psth_df_MLI = paw_psth_df_MLI.loc[modulated_mask_MLI]

                modulated_mask_PC = paw_df_PC['cell_global_Id'].isin(modCells_PC[pawList[i]]['all_mod'])
                modulated_paw_df_PC = paw_df_PC.loc[modulated_mask_PC]
                modulated_paw_psth_df_PC = paw_psth_df_PC.loc[modulated_mask_PC]
                modulatedList = [modulated_paw_df_MLI, modulated_paw_df_PC]
                zscoreArrayList = [modulated_paw_psth_df_MLI, modulated_paw_psth_df_PC]
                z_scoreKey = f'psth_{event}OnsetAligned_zscore'
                z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
                # t_value, t_test_p_value = stats.ttest_ind(modulated_paw_df_MLI[zscore_key[0]], modulated_paw_df_PC[zscore_key[0]],
                # axis=0)

                times = ['before', 'after']
                cells = ['MLI', 'PC']
                for t in range(2):

                    for m in range(1):
                        if m == 0 and e == 1:
                            exampleCell = 15
                        elif m == 0 and e == 0:
                            exampleCell = 34
                        elif m == 1 and e == 0:
                            exampleCell = 11
                        elif m == 1 and e == 1:
                            exampleCell = 10

                        cellsId = np.unique(modulatedList[m]['cell_global_Id'])
                        zscoreArray = []
                        zscoreArrayUp = []
                        zscoreArrayDown = []
                        upCells = []
                        downCells = []
                        for c in range(len(cellsId)):
                            zscoreSingle = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][
                                z_scoreKey].values

                            zscoreSingleTime = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[0])][
                                z_scoreTimeKey].values
                            zscore = zscoreArrayList[m][(zscoreArrayList[m]['cell_global_Id'] == cellsId[c])][
                                z_scoreKey].values
                            zscoreArray.append(zscore[0][1])
                            # if e == 1:
                            # timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                            #  else:
                            # timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                            timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                            if np.mean(zscore[0][1][timeMask]) > 0:
                                zscoreArrayUp.append(zscore[0][1])
                                upCells.append(cellsId[c])
                            else:
                                zscoreArrayDown.append(zscore[0][1])
                                downCells.append(cellsId[c])
                        meanZscoreUp = gaussian_filter1d(np.mean(zscoreArrayUp, axis=0), 0.8)
                        meanZscoreDown = gaussian_filter1d(np.mean(zscoreArrayDown, axis=0), 0.8)
                        semZscoreUp = gaussian_filter1d(stats.sem(zscoreArrayUp, axis=0), 0.8)
                        semZscoreDown = gaussian_filter1d(stats.sem(zscoreArrayDown, axis=0), 0.8)
                        meanZscore = np.mean(zscoreArray, axis=0)
                        meanZscore = gaussian_filter1d(meanZscore, 0.8)
                        semZscore = stats.sem(zscoreArray, axis=0)
                        semZscore = gaussian_filter1d(semZscore, 0.8)
                        varList = [meanZscore]
                        semList = [semZscore]
                        if m == 0:
                            varList = [meanZscore]
                            semList = [semZscore]
                        else:
                            varList = [meanZscoreUp, meanZscoreDown]
                            semList = [semZscoreUp, semZscoreDown]
                            PCINames = ['PC 1', 'PC 2']
                            upDownPCNb = [len(upCells), len(downCells)]
                        ax0 = plt.subplot(gssub1a[0, 0:2])
                        if e == 1:
                            timeMask = (zscoreSingleTime[0] < 0) & (zscoreSingleTime[0] > -interval)
                        else:
                            timeMask = (zscoreSingleTime[0] > 0) & (zscoreSingleTime[0] < interval)
                        maxTimeMask = (meanZscore[timeMask] == np.max(meanZscore[timeMask]))
                        # ax0.axvline(zscoreSingleTime[0][timeMask][maxTimeMask], ls='--', color=(f'C{i}' if m == 0 else 'C4'), alpha=0.6)
                        for v in range(len(varList)):
                            ax0.plot(zscoreSingleTime[0], varList[v],
                                     color=(f'C{i}' if m == 0 else f'C{v + 4}'), lw=2,
                                     alpha=1,
                                     label=(f'MLI ({len(cellsId)})' if m == 0 else f'{PCINames[v]} ({upDownPCNb[v]})'))
                            ax0.fill_between(zscoreSingleTime[0], varList[v] - semList[v], varList[v] + semList[v],
                                             color=(f'C{i}' if m == 0 else f'C{v + 4}'), alpha=0.1)

                            self.layoutOfPanel(ax0, yLabel=f'z-score', xLabel=f'{event} onset',
                                               xyInvisible=[(False), False])
                            if t == 0:
                                ax0.legend(frameon=False)
                        ax0.axvline(0, ls='--', color='grey', alpha=0.3)
                        ax0.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)

                        modulatedList = [modulated_paw_df_MLI, modulated_paw_df_PC]
                        zscoreMLI = modulated_paw_df_MLI.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean = np.mean(zscoreMLI[zscore_key[t]])
                        zscore_sem = stats.sem(zscoreMLI[zscore_key[t]])
                        zscore = zscoreMLI[zscore_key[t]]

                        modulated_up_mask = modulatedList[m]['cell_global_Id'].isin(upCells)
                        modulated_up = modulatedList[m].loc[modulated_up_mask]
                        modulated_up = modulated_up.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean_up = np.mean(modulated_up[zscore_key[t]])
                        zscore_sem_up = stats.sem(modulated_up[zscore_key[t]])
                        zscore_up = modulated_up[zscore_key[t]]

                        modulated_down_mask = modulatedList[m]['cell_global_Id'].isin(downCells)
                        modulated_down = modulatedList[m].loc[modulated_down_mask]
                        modulated_down = modulated_down.groupby(['cell_global_Id', zscore_key[t]]).mean().reset_index()
                        zscore_mean_down = np.mean(modulated_down[zscore_key[t]])
                        zscore_sem_down = stats.sem(modulated_down[zscore_key[t]])
                        zscore_down = modulated_down[zscore_key[t]]

                        ax1 = plt.subplot(gssub1a[1, t])
                        if m == 0:
                            ax1.bar(0, zscore_mean, yerr=zscore_sem, color=(f'C{i}' if m == 0 else 'C4'))
                            ax1.scatter(np.zeros(len(zscore)), zscore, edgecolor=(f'C{i}' if m == 0 else 'C4'),
                                        facecolor='white', marker='o')
                        else:
                            ax1.bar(1, zscore_mean_up, yerr=zscore_sem_up, color=(f'C{i}' if m == 0 else 'C4'))
                            ax1.scatter(np.ones(len(zscore_up)), zscore_up, edgecolor=(f'C{i}' if m == 0 else 'C4'),
                                        facecolor='white', marker='o')
                            ax1.bar(2, zscore_mean_down, yerr=zscore_sem_down, color=(f'C{i}' if m == 0 else 'C5'))
                            ax1.scatter(np.repeat(2, len(zscore_down)), zscore_down,
                                        edgecolor=(f'C{i}' if m == 0 else 'C5'), facecolor='white', marker='o')
                        ax1.xaxis.set_major_locator(MultipleLocator(1))
                        ax1.set_xticklabels(['', 'MLI', 'PC \nup', 'PC \ndown'])
                        plt.setp(ax1.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                        self.layoutOfPanel(ax1, yLabel=f'mean z-score AUC {times[t]} {event}',
                                           xyInvisible=[(False), False])
                    onewayStats, OneWay_p_value = stats.f_oneway(zscore, zscore_up, zscore_down,
                                                                 axis=0)
                    if OneWay_p_value < 0.05:
                        t_value1, t_test_p_value1 = stats.ttest_ind(zscore, zscore_up, axis=0)
                        t_value2, t_test_p_value2 = stats.ttest_ind(zscore, zscore_down, axis=0)
                        t_value3, t_test_p_value3 = stats.ttest_ind(zscore_up, zscore_down, axis=0)

                        pvalues = [t_test_p_value1, t_test_p_value2, t_test_p_value3]
                        xpos = [0.28, 0.5, 0.72]
                        for p in range(3):
                            star_trial = groupAnalysis.starMultiplier(pvalues[p])
                            if pvalues[p] < 0.05:
                                ax1.text(xpos[p], (0.90 if p != 1 else 0.99), (f'{star_trial} '), ha='center',
                                         va='center',
                                         transform=ax1.transAxes,
                                         style='italic', fontfamily='serif', fontsize=18, color='k')
                            else:
                                ax1.text(xpos[p], (0.90 if p != 1 else 0.99), (f'p={pvalues[p]:.2f}'), ha='center',
                                         va='center',
                                         transform=ax1.transAxes,
                                         style='italic', fontfamily='serif', fontsize=12, color='k')
                    else:
                        ax1.text(0.5, 0.99, 'n.s', ha='center',
                                 va='center',
                                 transform=ax1.transAxes,
                                 style='italic', fontfamily='serif', fontsize=12, color='k')

            paw = pawList[pawNb]
            fname = f'fig_ephys_psth_Z-score_{zscorePar}_avgModCells_{paw}'
            # plt.savefig(fname + '.png')
            # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def createObstacleStrategiesFigure(self, df_strat):


        cc = ['C0', 'C1', 'C2', 'C3']


        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 9,
                  'ytick.labelsize': 9, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,
                                #width_ratios=[3,1],
                               height_ratios=[1, 3]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.90, top=0.85, bottom=0.1)
        dates=np.unique(df_strat['date'])
        # pdb.set_trace()
        sessionNb=len(dates)
        mouseNb=len(np.unique(df_strat['mouse']))
        plt.figtext(0.06, 0.92, f'{dates}', size=7)
        paws = ['left','right']
        pawsLetter=['L','R','both']
        behavior = ['touch_count','success', 'rungMiss', 'obsMiss']
        # behavior=['success', 'rungMiss', 'obsMiss']
        behaviorYaxis= ['obstacle number', 'after cross success (%)', ' after cross misstep (%)', 'obstacle miss (%)']
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.4, wspace=0.4)
        timePoint = ['trial','session']




        ax = plt.subplot(gssub0[0, 0])
        axa = plt.subplot(gssub0[0, 1])
        if mouseNb>1:
            df_strat_mean_trial = df_strat.groupby(['trial'])['fraction_both_count', 'fraction_Rtouch_count', 'fraction_Ltouch_count'].mean().reset_index()
            df_strat_mean_session = df_strat.groupby(['session'])['fraction_both_count', 'fraction_Rtouch_count', 'fraction_Ltouch_count'].mean().reset_index()
            meanArray_stack=[df_strat_mean_trial['trial'],df_strat_mean_session['session']]
            # pdb.set_trace()
        # else:
        #     df_strat_mean = df_strat.groupby([timePoint[s]])['fraction_both_count', 'fraction_Rtouch_count', 'fraction_Ltouch_count'].mean().reset_index()

            # noTouch=100-(df_strat_mean['fraction_both_count']  +df_strat_mean['fraction_Rtouch_count']+df_strat_mean['fraction_Ltouch_count'])
            ax.stackplot(df_strat_mean_trial['trial'],df_strat_mean_trial['fraction_both_count'],df_strat_mean_trial['fraction_Rtouch_count'],df_strat_mean_trial['fraction_Ltouch_count'], labels=['both','FR','FL','none'],colors=['C6','C1','C0','C7'])
            axa.stackplot(df_strat_mean_session['session'],df_strat_mean_session['fraction_both_count'],df_strat_mean_session['fraction_Rtouch_count'],df_strat_mean_session['fraction_Ltouch_count'], labels=['both','FR','FL','none'],colors=['C6','C1','C0','C7'])

            ax.xaxis.set_major_locator(MultipleLocator(1))
            self.layoutOfPanel(ax, xLabel=('trials' ), yLabel='touch fraction (%)', Leg=[0, 9])
            ax.legend(frameon=False, bbox_to_anchor=(0, 1.05, 0.9, 0.5), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10)
            axa.xaxis.set_major_locator(MultipleLocator(1))
            self.layoutOfPanel(axa, xLabel=( 'sessions'), yLabel='touch fraction (%)', Leg=[0, 9])
            axa.legend(frameon=False, bbox_to_anchor=(0, 1.05, 0.9, 0.5), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10)
        gssub0b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.4, wspace=0.4, width_ratios=[1,3])
        gssub1a = gridspec.GridSpecFromSubplotSpec(len(behavior), 1, subplot_spec=gssub0b[0], hspace=0.4, wspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(len(behavior), 2, subplot_spec=gssub0b[1], hspace=0.4, wspace=0.1, width_ratios=[4,1])
        #gssub2 = gridspec.GridSpecFromSubplotSpec(len(behavior), 1, subplot_spec=gs[1,1], hspace=0.4, wspace=0.4)
        df_strat['fraction_success_mean']=df_strat[['fraction_left_success','fraction_right_success']].mean(axis=1)
        df_strat['fraction_rungMiss_mean']=df_strat[['fraction_LrungMiss','fraction_RrungMiss']].mean(axis=1)
        df_strat['fraction_obsMiss_mean'] = df_strat[['fraction_LobsMiss','fraction_RobsMiss']].mean(axis=1)

        meanList=['fraction_success_mean','fraction_rungMiss_mean','fraction_obsMiss_mean']
        for p in range(2):
            for b in range(len(behavior)):
                if b==1:
                    key = f'fraction_{paws[p]}_{behavior[b]}'
                elif b!=1:
                    key=f'fraction_{pawsLetter[p]}{behavior[b]}'


                # tukey_success = pairwise_tukeyhsd(endog=df_strat['fraction_success_mean'], groups=df_strat['session'])
                # tukey_successDf=pd.DataFrame(tukey_success)
                # tukey_successDf.to_csv('/home/andry/Documents/stats/tukey_success.csv')
                # print(tukey_success)
                # anova_success = pg.rm_anova(data=df_strat, dv='fraction_success_mean', subject='mouse',
                #                          within=['session','whikers'], effsize='n2').round(3)
                # anova_success = AnovaRM(data=df_strat, depvar='fraction_success_mean', subject='mouse', within=['session', 'whikers'], aggregate_func='mean').fit()
                # print(anova_success)


                # anova_success.to_csv('/home/andry/Documents/stats/anova_success.csv')
                # print(anova_success)
                # print(tukey_success)
                # posthoc = pairwise_tukeyhsd(endog=df['measuredValue'], groups=df['treatments'], alpha=0.05)
                # pdb.set_trace()

                import scikit_posthocs as sp

                ax0 = plt.subplot(gssub1a[b])
                # sns.lineplot(data=df_strat, x='trial', y=key, color=cc[p], ax=ax0,errorbar=('se'), err_style='bars', alpha=0.3,marker='o')
                # paw_data = df_strat.drop(df_strat[df_strat['session'] > 12].index)
                df_strat_train=df_strat[df_strat['session']<13].reset_index()
                df_strat_half_intact=df_strat[(df_strat['whikers'] == 'half_trimmed') |(df_strat['whikers'] == 'intact')]
                strategy_summary = groupAnalysis.perform_mixedlm_obstacle(df_strat_train, meanList[b - 1])
                posthocTukey=sp.posthoc_ttest(df_strat_half_intact, val_col=meanList[b-1], group_col='session', p_adjust='fdr_bh')
                posthocTukey.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/{meanList[b-1]}_trainPeriod_scikit__Benjamini_and_Hochberg.csv')
                # tukey = pg.pairwise_tukey(data=df_strat_train, dv=meanList[b-1], between='session')
                # tukey.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/{meanList[b-1]}_trainPeriod.csv')
                # tukey = pg.pairwise_tukey(data=df_strat_train, dv=meanList[b-1], between='session')
                # tukey.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/{meanList[b-1]}_trainPeriod.csv')
                # tukey_trial = pg.pairwise_tukey(data=df_strat_train, dv=meanList[b-1], between='trial')
                # tukey_trial.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/_trial_{meanList[b-1]}_trainPeriod.csv')

                df_strat_whis = df_strat[df_strat['session'] > 12].reset_index()

                strategy_summary_whis = groupAnalysis.perform_mixedlm_obstacle(df_strat_whis, meanList[b - 1], treatment=True)
                # tukeyWhis = pg.pairwise_tukey(data=df_strat_whis, dv=meanList[b-1], between='session')
                # tukeyWhis.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/{meanList[b-1]}_whisPeriod.csv')
                # tukeyWhis_trial = pg.pairwise_tukey(data=df_strat_whis, dv=meanList[b-1], between='trial')
                # tukeyWhis_trial.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/_trial_{meanList[b-1]}_WhisPeriod.csv')
                #
                # tukeyWhis_effect = pg.pairwise_tukey(data=df_strat_whis, dv=meanList[b-1], between='whikers')
                # tukeyWhis_effect.to_csv(f'/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/stats_obstacle_strategies/_whikers_{meanList[b-1]}_WhisPeriod.csv')
                trialDf=df_strat.drop(df_strat[df_strat['session'] > 12].index)
                if b==0:
                    sns.lineplot(data=df_strat_train, x='trial', y='obsNb', color='k', ax=ax0, errorbar=('se'),   err_style='bars',marker='o')

                if b>0:
                    sns.lineplot(data=df_strat_train, x='trial', y=meanList[b-1], color='k', ax=ax0, errorbar=('se'),marker='o',
                                 err_style='bars')

                    # sns.lineplot(data=trialDf, x='trial', y=key, color=f'C{p}', ax=ax0, errorbar=('se'),marker='o',
                    #              err_style='bars')
                    # ax0.text(0.5, 0.9, '%s' %(strategy_summary['stars']['trial']), ha='center', va='center', transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=13, color='k')
                self.layoutOfPanel(ax0, xLabel=('trials' if b==3 else ''), yLabel=behaviorYaxis[b], Leg=[0, 9])
                ax0.xaxis.set_major_locator(MultipleLocator(1))

                ax1 = plt.subplot(gssub1[b, 0])

                # sns.lineplot(data=df_strat, x='session', y=key, color=cc[p], ax=ax1, errorbar=('se'), err_style='bars',alpha=0.3,marker='o')

                if b > 0:
                    ax2 = plt.subplot(gssub1[b, 1])
                    paletteW = sns.color_palette(['0.4','0.8'], 2)
                    spalette = sns.color_palette(['cyan', 'hotpink'], 2)
                    # sns.lineplot(data=df_strat_train, x='session', y=meanList[b-1], color='k', ax=ax1, errorbar=('se'),marker='o',
                    #              err_style='bars')
                    sns.lineplot(data=df_strat_train, x='session', y=key, color=f'C{p}', ax=ax1, errorbar=('se'),marker='o',
                                 err_style='bars')
                    # sns.lineplot(data=df_strat_whis[df_strat_whis['whikers']=='half_trimmed'], x='session', y=meanList[b - 1], hue='whikers', style='sex', legend=False,ax=ax2,
                    #              errorbar=('se'), marker='o',   err_style='bars', palette=paletteW)

                    sns.lineplot(data=df_strat_whis[df_strat_whis['whikers']=='half_trimmed'], x='session', y=key, color=f'C{p}', ax=ax1, errorbar=('se'),marker='o',
                                 err_style='bars')
                    # sns.lineplot(data=df_strat_train, x='session', y=meanList[b-1], palette=spalette, ax=ax1, errorbar=('se'),marker='o',
                    #              err_style='bars', hue='sex', alpha=0.5, legend=False)
                    # if b==1:
                    #     ax1.set_ylim(60,100)
                    #     ax2.set_ylim(60,100)
                    # if b == 2:
                    #     ax1.set_ylim(0, 40)
                    #     ax2.set_ylim(0, 40)
                    # if b == 3:
                    #     ax1.set_ylim(15, 60)
                    #     ax2.set_ylim(15, 60)
                    self.layoutOfPanel(ax2, xLabel=('' if b == 3 else ''), yLabel='', Leg=[0, 9],
                                       xyInvisible=[True, True])
                    ax2.xaxis.set_major_locator(MultipleLocator(1))
                    # sns.lineplot(data=df_strat, x='session', y=key, color=f'C{p}', ax=ax1, style='animal', hue='animal', errorbar=('se'),marker='o',
                    #              err_style='bars', legend=False)
                    # ax1.text(0.5, 1, '%s' % (strategy_summary['stars']['session']), ha='center', va='center',
                    #          transform=ax1.transAxes, style='italic', fontfamily='serif', fontsize=13, color='k')


                # if p==0 and b !=0:
                #     ax1.fill_betweenx(y=[np.mean(df_strat_train[key]) - 10, np.mean(df_strat_train[key]) + 10], x1=12.7,
                #                       x2=14.3, where=None, step=None, interpolate=False, data=None, alpha=0.1,
                #                       color='indianred', lw=0)
                #     ax1.fill_betweenx(y=[np.mean(df_strat_train[key]) - 10, np.mean(df_strat_train[key]) + 10], x1=14.7,
                #                       x2=16.3, where=None, step=None, interpolate=False, data=None, alpha=0.1,
                #                       color='olive', lw=0)
                # ax1.set_xlim(0.5,16.5)
                # ax1.set_xlim(0.5,12.5)
                # if b==2:
                #     ax1.set_ylim(-1, 12)
                if b==0:
                    sns.lineplot(data=df_strat, x='session', y='obsNb', marker='o', color='k', ax=ax1, errorbar=('se'),   err_style='bars')

                self.layoutOfPanel(ax1, xLabel=('sessions' if b==3 else ''), yLabel=behaviorYaxis[b], Leg=[0, 9], xyInvisible=[False,False])
                ax1.xaxis.set_major_locator(MultipleLocator(1))

        fname = f'fig_obstacle_strategies_new_differentScale_paws'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def createGroupWhiskerObsFigure(self, whiskerTouch): #creating group plot


        cc = ['C0', 'C1', 'C2', 'C3']


        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 15, 'font.size': 11, 'xtick.labelsize': 14,
                  'ytick.labelsize': 14, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  # major tick size in points
                  # 'edgecolor' : 'white'
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,
                                #width_ratios=[3,1],
                               height_ratios=[1, 1]
                               )


        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.8) # To change spacing

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.90, top=0.85, bottom=0.1)

        sessionNumber = len(whiskerTouch) - 6
        touch_time = []
        allWhiskerTouchTimes = []
        allmedianWhiskerTouchTime = []
        for i in range(sessionNumber):
            whiskerTouchSessions = []
            for j in range(len(whiskerTouch[i])):
                obsTimeDict = whiskerTouch[i][j]
                if len(obsTimeDict) > 0:
                    for value in obsTimeDict['obsNumber']:
                        # if len(obsTimeDict) == len(obsTimeDict['obsNumber']):
                        touch_time.append(obsTimeDict[value]['normalized_touch_time'] * 1000)
                        whiskerTouchSessions.append(obsTimeDict[value]['normalized_touch_time'] * 1000)
                else:
                    pass
            allWhiskerTouchTimes.append(whiskerTouchSessions)
            medianSessionWhiskerTouchTime = np.median(whiskerTouchSessions)
            allmedianWhiskerTouchTime.append(medianSessionWhiskerTouchTime)


        ax = plt.subplot(gs[0])
        ax.hist(touch_time, bins=40, color = 'slateblue', alpha = 0.8)
        cm = plt.cm.get_cmap('Blues')
        n, bins, patches = ax.hist(touch_time, bins=40, color='slateblue', alpha=0.6)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for patch, color in zip(patches, cm(bin_centers / max(bin_centers))):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        touch_time_iqr = scipy.stats.iqr(touch_time)
        touch_time_25th = np.percentile(touch_time, 25)
        touch_time_75th = np.percentile(touch_time, 75)
        ax.text(0.9, 0.81, f'N = {len(touch_time)}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')

        ax.text(0.9, 0.75, f'IQR = {touch_time_iqr:.2f}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')

        median = np.median(touch_time)
        ax.axvline(x=median, color='r', linestyle='--', label='Median')
        # ax.axvline(x=touch_time_25th, color='magenta', linestyle='--', label='25th')
        # ax.axvline(x=touch_time_75th, color='magenta', linestyle='--', label='75th')

        ax.set_title('Histogram of Whisker Touch Time Over\n All Sessions and Trials of a Single Mouse',
                     fontsize=20, pad=20)
        ax.set_xlabel('Whisker Touch Time (ms)', labelpad=10)
        ax.set_ylabel('Frequency', labelpad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.legend(fontsize=15, frameon=False)

        ax1 = plt.subplot(gs[1])
        x_axisArray = np.arange(1, (sessionNumber + 1))
        allmedianWhiskerTouchTime = np.array(allmedianWhiskerTouchTime)
        ax1.plot(x_axisArray, allmedianWhiskerTouchTime)
        ax1.fill_between(x_axisArray, allmedianWhiskerTouchTime - touch_time_iqr/2,
                         allmedianWhiskerTouchTime + touch_time_iqr/2,
                         color='C2', alpha=0.2)

        ax1.set_title('Median Whisker Touch Time Over Each\nSession of a Single Mouse', fontsize=20, pad=20)
        ax1.xaxis.set_major_locator(MultipleLocator(1))

        ax1.set_ylabel('Median Whisker Touch Time (ms)', labelpad=10)
        ax1.set_xlabel('Sessions', labelpad=10)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        fname = f'fig_whiskerTouch_Time_histogram'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def createHistogramObsXPosAtTouchFigure(self, whiskerTouch):
        cc = ['C0', 'C1', 'C2', 'C3']

        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 15, 'font.size': 11, 'xtick.labelsize': 14,
                  'ytick.labelsize': 14, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  # major tick size in points
                  # 'edgecolor' : 'white'
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,
                               # width_ratios=[3,1],
                               height_ratios=[1, 1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.5)  # To change spacing

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.90, top=0.85, bottom=0.1)

        sessionNumber = len(whiskerTouch) - 6
        obsPosAtTouch = []
        allObsXPosAtTouch = []
        allObsYPosAtTouch = []
        obsYPosAtTouch=[]
        allmedianObsXPosAtTouch = []
        allmedianObsYPosAtTouch = []
        allObsIds=[]
        for i in range(sessionNumber):
            obsXPosAtTouchSessions = []
            obsYPosAtTouchSessions=[]
            for j in range(len(whiskerTouch[i])):
                obsTimeDict = whiskerTouch[i][j]
                if len(obsTimeDict) > 0:
                    for value in obsTimeDict['obsNumber']:
                        # if len(obsTimeDict) == len(obsTimeDict['obsNumber']):
                        allObsIds.append(obsTimeDict[value]['obsID'])
                        obsPosAtTouch.append(obsTimeDict[value]['obsXPosAtTouch'])
                        obsYPosAtTouch.append(obsTimeDict[value]['obsYPosAtTouch'])
                        obsXPosAtTouchSessions.append(obsTimeDict[value]['obsXPosAtTouch'])
                        obsYPosAtTouchSessions.append(obsTimeDict[value]['obsYPosAtTouch'])
                else:
                    pass

            allObsXPosAtTouch.append(obsXPosAtTouchSessions)
            allObsYPosAtTouch.append(obsYPosAtTouchSessions)
            medianSessionObsXPosAtTouch = np.median(obsXPosAtTouchSessions)
            medianSessionObsYPosAtTouch = np.median(obsYPosAtTouchSessions)
            allmedianObsXPosAtTouch.append(medianSessionObsXPosAtTouch)
            allmedianObsYPosAtTouch.append(medianSessionObsYPosAtTouch)

        ax = plt.subplot(gs[0])
        allObsIds=np.asarray(allObsIds)
        obsPosAtTouch = np.asarray(obsPosAtTouch)

        maskObs1=allObsIds==1
        maskObs2 = allObsIds == 2


        obsPosAtTouch1=obsPosAtTouch[maskObs1]
        obsPosAtTouch2 = obsPosAtTouch[maskObs2]
        # pdb.set_trace()
        median1=np.median(obsPosAtTouch1)
        median2=np.median(obsPosAtTouch2)
        ax.hist(obsPosAtTouch1, bins=40, color='slateblue', alpha=0.8, label='obstacle 1')
        ax.hist(obsPosAtTouch2, bins=40, color='powderblue', label='obstacle 2')
        cm = plt.cm.get_cmap('Blues')
        # n, bins, patches = ax.hist(obsPosAtTouch, bins=40, color='slateblue', alpha=0.6)
        # median = np.median(obsPosAtTouch)
        ax.axvline(x=median1, color='slateblue', linestyle='--', label='Median obs 1')
        ax.axvline(x=median2, color='cadetblue', linestyle='--', label='Median obs 2')
        # bin_centers = 0.5 * (bins[:-1] + bins[1:])
        #
        # for patch, color in zip(patches, cm(bin_centers / max(bin_centers))):
        #     patch.set_facecolor(color)
        #     patch.set_alpha(0.8)

        obsXPos_iqr1 = scipy.stats.iqr(obsPosAtTouch1)
        obsXPos_iqr2 = scipy.stats.iqr(obsPosAtTouch2)
        ax.text(0.17, 0.62, f'N = {len(obsPosAtTouch2)}, IQR obs2 {obsXPos_iqr2:.2f}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=13, color='powderblue')
        ax.text(0.17, 0.68, f'N = {len(obsPosAtTouch1)}, IQR obs1 {obsXPos_iqr1:.2f}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=13, color='slateblue')
        # ax.text(0.9, 0.75, f'IQR = ', ha='center', va='center',
        #         transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')

        # median = np.median(obsPosAtTouch)
        # ax.axvline(x=median, color='r', linestyle='--', label='Median')

        ax.set_title('Histogram of Obstacle X Position At Whisker Touch Over\n All Sessions and Trials of a Single Mouse',
                     fontsize=20, pad=20)
        ax.set_xlabel('Obstacle X Position At Whisker Touch (pixels)', labelpad=10)
        ax.set_ylabel('Frequency', labelpad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.legend(fontsize=15, frameon=False)

        ax1 = plt.subplot(gs[1])
        obsYPosAtTouch = np.asarray(obsYPosAtTouch)

        maskObs1=allObsIds==1
        maskObs2 = allObsIds == 2


        obsYPosAtTouch1=obsYPosAtTouch[maskObs1]
        obsYPosAtTouch2 = obsYPosAtTouch[maskObs2]
        # pdb.set_trace()
        medianY1=np.median(obsYPosAtTouch1)
        medianY2=np.median(obsYPosAtTouch2)
        ax1.hist(obsYPosAtTouch1, bins=40, color='slateblue', alpha=0.8, label='obstacle 1')
        ax1.hist(obsYPosAtTouch2, bins=40, color='powderblue', label='obstacle 2')
        cm = plt.cm.get_cmap('Blues')
        # n, bins, patches = ax1.hist(obsPosAtTouch, bins=40, color='slateblue', alpha=0.6)
        # median = np.median(obsPosAtTouch)
        ax1.axvline(x=medianY1, color='slateblue', linestyle='--', label='Median obs 1')
        ax1.axvline(x=medianY2, color='cadetblue', linestyle='--', label='Median obs 2')
        # bin_centers = 0.5 * (bins[:-1] + bins[1:])
        #
        # for patch, color in zip(patches, cm(bin_centers / max1(bin_centers))):
        #     patch.set_facecolor(color)
        #     patch.set_alpha(0.8)

        obsYPos_iqr1 = scipy.stats.iqr(obsYPosAtTouch1)
        obsYPos_iqr2 = scipy.stats.iqr(obsYPosAtTouch2)
        ax1.text(0.17, 0.62, f'N = {len(obsYPosAtTouch2)}, IQR obs1 {obsYPos_iqr2:.2f}', ha='center', va='center',
                transform=ax1.transAxes, style='italic', fontfamily='serif', fontsize=13, color='powderblue')
        ax1.text(0.17, 0.68, f'N = {len(obsYPosAtTouch1)}, IQR obs1 {obsYPos_iqr1:.2f}', ha='center', va='center',
                transform=ax1.transAxes, style='italic', fontfamily='serif', fontsize=13, color='slateblue')

        # median = np.median(obsPosAtTouch)
        # ax1.ax1vline(x=median, color='r', linestyle='--', label='Median')

        ax1.set_title('Histogram of Obstacle Y Position At Whisker Touch Over\n All Sessions and Trials of a Single Mouse',
                     fontsize=20, pad=20)
        ax1.set_xlabel('Obstacle Y Position At Whisker Touch (pixels)', labelpad=10)
        ax1.set_ylabel('Frequency', labelpad=10)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        ax1.legend(fontsize=15, frameon=False, loc='upper left')
        # x_axisArray = np.arange(1, (sessionNumber + 1))
        # allmedianObsXPosAtTouch = np.array(allmedianObsXPosAtTouch)
        # ax1.plot(x_axisArray, allmedianObsXPosAtTouch)
        # ax1.fill_between(x_axisArray, allmedianObsXPosAtTouch - obsXPos_iqr,
        #                  allmedianObsXPosAtTouch + obsXPos_iqr,
        #                  color='palevioletred', alpha=0.2)
        # 
        # ax1.set_title('Median Obstacle X Position At Whisker Touch Over Each\nSession of a Single Mouse', fontsize=20, pad=20)
        # ax1.xaxis.set_major_locator(MultipleLocator(1))
        # 
        # ax1.set_ylabel('Median Obstacle X Position at Whisker Touch (pixel)', labelpad=10)
        # ax1.set_xlabel('Sessions', labelpad=10)
        # 
        # ax1.spines['top'].set_visible(False)
        # ax1.spines['right'].set_visible(False)
        # 
        # ax1.grid(axis='y', linestyle='--', alpha=0.5)

        fname = f'fig_whiskerTouch_ObsPosition_X_histogram'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')


    def createHistogramObsYPosAtTouchFigure(self, whiskerTouch):
        cc = ['C0', 'C1', 'C2', 'C3']

        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 15, 'font.size': 11, 'xtick.labelsize': 14,
                  'ytick.labelsize': 14, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  # major tick size in points
                  # 'edgecolor' : 'white'
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,
                               # width_ratios=[3,1],
                               height_ratios=[1, 1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.4, hspace=0.8)  # To change spacing

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.90, top=0.85, bottom=0.1)

        sessionNumber = len(whiskerTouch) - 6
        obsPosAtTouch = []
        allObsYPosAtTouch = []
        allmedianObsYPosAtTouch = []
        for i in range(sessionNumber):
            obsYPosAtTouchSessions = []
            for j in range(len(whiskerTouch[i])):
                obsTimeDict = whiskerTouch[i][j]
                if len(obsTimeDict) > 0:
                    for value in obsTimeDict['obsNumber']:
                        # if len(obsTimeDict) == len(obsTimeDict['obsNumber']):
                        obsPosAtTouch.append(obsTimeDict[value]['obsYPosAtTouch'])
                        obsYPosAtTouchSessions.append(obsTimeDict[value]['obsYPosAtTouch'])
                else:
                    pass
            allObsYPosAtTouch.append(obsYPosAtTouchSessions)
            medianSessionObsYPosAtTouch = np.median(obsYPosAtTouchSessions)
            allmedianObsYPosAtTouch.append(medianSessionObsYPosAtTouch)

        ax = plt.subplot(gs[0])
        ax.hist(obsPosAtTouch, bins=40, color='mediumaquamarine', alpha=0.8)
        cm = plt.cm.get_cmap('Blues')
        n, bins, patches = ax.hist(obsPosAtTouch, bins=40, color='mediumaquamarine', alpha=0.6)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for patch, color in zip(patches, cm(bin_centers / max(bin_centers))):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        obsYPos_iqr = scipy.stats.iqr(obsPosAtTouch)
        ax.text(0.9, 0.81, f'N = {len(obsPosAtTouch)}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')

        ax.text(0.9, 0.75, f'IQR = {obsYPos_iqr:.2f}', ha='center', va='center',
                transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')

        median = np.median(obsPosAtTouch)
        ax.axvline(x=median, color='r', linestyle='--', label='Median')

        ax.set_title(
            'Histogram of Obstacle Y Position At Whisker Touch Over\n All Sessions and Trials of a Single Mouse',
            fontsize=20, pad=20)
        ax.set_xlabel('Obstacle Y Position At Whisker Touch (pixel)', labelpad=10)
        ax.set_ylabel('Frequency', labelpad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.legend(fontsize=15)

        ax1 = plt.subplot(gs[1])
        x_axisArray = np.arange(1, (sessionNumber + 1))
        allmedianObsYPosAtTouch = np.array(allmedianObsYPosAtTouch)
        ax1.plot(x_axisArray, allmedianObsYPosAtTouch)
        ax1.fill_between(x_axisArray, allmedianObsYPosAtTouch - obsYPos_iqr,
                         allmedianObsYPosAtTouch + obsYPos_iqr,
                         color='lightcoral', alpha=0.2)

        ax1.set_title('Median Obstacle Y Position At Whisker Touch Over Each\nSession of a Single Mouse',
                      fontsize=20, pad=20)
        ax1.xaxis.set_major_locator(MultipleLocator(1))

        ax1.set_ylabel('Median Obstacle Y Position at Whisker Touch (pixel)', labelpad=10)
        ax1.set_xlabel('Sessions', labelpad=10)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.grid(axis='y', linestyle='--', alpha=0.5)


        fname = f'fig_whiskerTouch_ObsPosition_Y_histogram'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    # ax.text(0.9, 0.81, f'N = {len(touch_time)}', ha='center', va='center',
    #                      transform=ax.transAxes, style='italic', fontfamily='serif', fontsize=15, color='k')
    # touch_timeIQR = scipy.stats.iqr(touch_time) #Inter quartile range
    #
    # ax.set_title('Histogram of Whisker Touch Time Over\n All Sessions and Trials of a Single Mouse', fontsize = 20, pad = 20)
    #
    # self.layoutOfPanel(ax, xLabel=('Whisker Touch Time (ms)'), yLabel='Frequency', Leg=[0, 9])
    # ax.set_ylabel('Frequency', labelpad=10)
    # ax.set_xlabel('Whisker Touch Time (ms)', labelpad=10)
    # median = np.median(touch_time)
    # ax.axvline(x=median, color='r', linestyle='--', label='Median')
    # ax.legend(fontsize=15)
    #
    # ax1 = plt.subplot(gs[1])
    # x_axisArray = np.arange(1, (sessionNumber + 1))
    # allmedianWhiskerTouchTime = np.array(allmedianWhiskerTouchTime)
    # ax1.plot(x_axisArray, allmedianWhiskerTouchTime)
    # ax1.fill_between(x_axisArray, allmedianWhiskerTouchTime - touch_timeIQR, allmedianWhiskerTouchTime + touch_timeIQR, color = 'C2', alpha = 0.2)
    #
    # ax1.set_title('Median Whisker Touch Time Over Each\nSession of a Single Mouse', fontsize=20, pad = 20)
    # ax1.xaxis.set_major_locator(MultipleLocator(1))
    #
    # self.layoutOfPanel(ax1, xLabel=('Sessions'), yLabel='Median Whisker Touch Time (ms)', Leg=[0, 9])
    # ax1.set_ylabel('Median Whisker Touch Time (ms)', labelpad=10)
    # ax1.set_xlabel('Sessions', labelpad=10)
    # plt.show()
    # pdb.set_trace()


    def plotMLIswingLengthResponse(self, cellType,allModTraces, variable):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 20  # width in inches

        fig_height = 12
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 20, 'axes.titlesize': 16, 'font.size': 20, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        plt.figtext(0.02, 0.97, 'A', clip_on=False, color='black', size=26)
        plt.figtext(0.35, 0.97, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.65, 0.97, 'C', clip_on=False, color='black', size=26)
        plt.figtext(0.02, 0.5, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.35, 0.5, 'E', clip_on=False, color='black', size=26)
        plt.figtext(0.65, 0.5, 'F', clip_on=False, color='black', size=26)
        modCells = {}
        events = ['swing', 'stance']
        pawNb = [0,1]
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], hspace=0.3,
                                                  wspace=0.4, width_ratios=[1,1.2,1])

        possibleConditions = [f'{variable}_allRecs_percentiles_0_20',
                              f'{variable}_allRecs_percentiles_20_40',
                              f'{variable}_allRecs_percentiles_40_60',
                              f'{variable}_allRecs_percentiles_60_80',
                              f'{variable}_allRecs_percentiles_80_100'
                              ]
        percentiles = ['0-20', '20-40', '40-60', '60-80', '80-100']
        percentilesc = ['[0-20]', '[20-40]', '[40-60]', '[60-80]', '[80-100]']
        numberArray=[[],[]]
        traceAverageArray=[[],[]]
        traceSemArray = [[],[]]
        traceAUC = [[],[]]
        var = 'peak'
        if cellType=='PC':
            timeInterval=0.25
            totCells=34
        else:
            timeInterval=0.15
            totCells=64

        cellDicList=[]

        for l, condition in enumerate(possibleConditions):
            for e in range(2):
                nCells=len(allModTraces[condition][events[e]]['traces'])
                traceAUC[e].append(np.empty((nCells)))
                # AUCdic[percentiles[l]][events[e]]=[]
        for l, condition in enumerate(possibleConditions):
            for e in range(2):
                time=allModTraces[condition][events[e]]['time']
                numberArray[e].append(allModTraces[condition][events[e]]['number'])
                eventOnsetMask=(time>-timeInterval) & (time<timeInterval)
                dt=np.diff(time)[0]
                for c in range(len(allModTraces[condition][events[e]]['traces'])):
                    trace = allModTraces[condition][events[e]]['traces'][c]
                    cellDic={}
                    AUC= np.trapz(allModTraces[condition][events[e]]['traces'][c][eventOnsetMask],dx=dt)
                    cellDic['AUC']=abs(AUC)
                    cellDic['condition']=f'{percentilesc[l]}' # {allModTraces[condition][events[e]]["number"]}'
                    cellDic['event']=events[e]
                    cellDic['id']=c
                    # Find positive peak
                    pos_peak = trace[eventOnsetMask].max()
                    pos_idx = np.argmax(trace[eventOnsetMask])
                    pos_latency = time[eventOnsetMask][pos_idx]

                    # Find negative peak
                    neg_peak = trace[eventOnsetMask].min()
                    neg_idx = np.argmin(trace[eventOnsetMask])
                    neg_latency = time[eventOnsetMask][neg_idx]

                    # Add to dictionary
                    cellDic['PosPeak'] = pos_peak
                    cellDic['PosLatency'] = pos_latency

                    cellDic['NegPeak'] = neg_peak
                    cellDic['NegLatency'] = neg_latency
                    # Get onset time
                    onset = time[eventOnsetMask][0]

                    # Find peak
                    peak = np.max(trace[eventOnsetMask])
                    peak_time = time[np.argmax(trace[eventOnsetMask])]

                    # Time to peak
                    time_to_peak = peak_time - onset

                    # Find trough
                    trough = np.min(trace[eventOnsetMask])
                    trough_time = time[np.argmin(trace[eventOnsetMask])]

                    # Time to trough
                    time_to_trough = trough_time - onset

                    # Store in dictionary
                    cellDic['TimeToPeak'] = time_to_peak
                    cellDic['TimeToTrough'] = time_to_trough

                    # Also store peak and trough values
                    cellDic['Peak'] = peak
                    cellDic['Trough'] = trough

                    cellDicList.append(cellDic)
                meanZscore = np.mean(allModTraces[condition][events[e]]['traces'], axis=0)
                meanZscore = gaussian_filter1d(meanZscore, 0.8)
                semZscore = stats.sem(allModTraces[condition][events[e]]['traces'], axis=0)
                traceAverageArray[e].append(meanZscore)
        dfCells = pd.DataFrame(cellDicList)
        for e in range(2):
            ax0=plt.subplot(gssub1[e,0])
            if e==0:
                alpha=0.6
            else:
                alpha=1


            numberArrayEv=np.array(numberArray[e])

            MLIfrac=(numberArrayEv/totCells)*100
            x=np.arange(5)
            y=MLIfrac
            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                         alternative='two-sided')
            ax0.plot(percentilesc,MLIfrac, '-o', lw=2, c='C0', alpha=alpha)
            sns.regplot(x=np.arange(5),y=MLIfrac, scatter_kws={'alpha': 0.8, 'edgecolor': 'k', 'lw': 0.1},
                                            line_kws={'alpha': alpha, 'lw': 1.5}, color='0.8')
            corr_star = groupAnalysis.starMultiplier(p_value)
            ax0.set_ylim(0,70)
            if p_value > 0.05:
                ax0.text(0.73, 0.80, f"r = {r_value:.2f}\np = {p_value:.2f}",
                         transform=ax0.transAxes, fontsize=10, color='dimgrey')

            else:
                ax0.text(0.75, 0.4, f"r = {r_value:.2f}",
                         transform=ax0.transAxes, fontsize=18, color='dimgrey')
                ax0.text(0.5, 0.98, f"{corr_star}",
                         transform=ax0.transAxes, fontsize=18, color='k')
            if variable!='swingLengthLinear':
                self.layoutOfPanel(ax0, xLabel=f'swing {variable[5:].lower()} percentile (%)', yLabel=f'fraction of {events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                   xyInvisible=[False, False])
            else:
                self.layoutOfPanel(ax0, xLabel=f'swing length percentile (%)', yLabel=f'fraction of {events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                   xyInvisible=[False, False])
            ax1=plt.subplot(gssub1[e,1])
            import matplotlib.colors as mcolors
            from matplotlib.colors import ColorConverter

            cc = ColorConverter()
            blue = mcolors.to_rgba('C0')

            colorList0=['#d2e3e4', '#b9d5e4', '#9bc7e4', '#7db9e4', '#5cace4']
            colorList=['#8ca5bf', '#6b8dbf', '#4c74bf', '#2d5abf', '#0d40bf']
            colorList=['#a3cef0', '#80bce5', '#5daeda', '#3a9dcf', '#167fc4']

            for a in range(5):
                intensity = (a + 1) / 5
                color = blue[:3] + (intensity,)  # adjust alpha channel\
                averageZscore=gaussian_filter1d(traceAverageArray[e][a], 0.8)
                line=ax1.plot(time, averageZscore, color=color, lw=2,
                         label=f'[{percentiles[a]}]')

                meanAUC=np.mean(traceAUC[e][a])
                # pdb.set_trace()

                ax1.text(0.42, 0.97, f"{events[e]} onset",
                         transform=ax1.transAxes, fontsize=12, color='grey')
                ax1.axvline(0, ls='--', color='grey', alpha=0.3)
                ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
                # self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                #                    xyInvisible=[False, False])
                self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                                   xyInvisible=[False, False])
            variables=['AUC','PosPeak', 'PosLatency', 'NegPeak', 'NegLatency' , 'TimeToPeak', 'TimeToTrough']
            variables=['AUC']
            varNames=['AUC (abs)', 'positive peak', 'positive peak latency', 'negative peak', 'negative peak latency', 'Time To Peak', 'Time To Trough']
            for v, var in enumerate(variables):
                ax2=plt.subplot(gssub1[e,2+v])
                cellDfEvent=dfCells[(dfCells['event']==events[e])]
                cellDfEventHigh=dfCells[(dfCells['event']==events[e]) & (dfCells['condition']=='[80-100]')][var]
                cellDfEventLow = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[0-20]')][var]

                sns.pointplot(cellDfEvent, x='condition', y=var, hue='condition', ax=ax2, errorbar='se', palette=colorList)
                sns.lineplot(cellDfEvent, x='condition', y=var,  ax=ax2, errorbar=None, lw=2)

                t_value, t_test_p_value = stats.ttest_ind(cellDfEventLow, cellDfEventHigh)

                star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                if t_test_p_value < 0.05:
                    ax2.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                             transform=ax2.transAxes,
                             style='italic', fontfamily='serif', fontsize=18, color='k')
                else:
                    ax2.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',
                             transform=ax2.transAxes,
                             style='italic', fontfamily='serif', fontsize=12, color='k')


                if variable!='swingLengthLinear':
                    self.layoutOfPanel(ax2, xLabel=f'swing {variable[5:].lower()} percentile (%)', yLabel=(f'{events[e]} onset PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                       xyInvisible=[False, False])
                else:
                    self.layoutOfPanel(ax2, xLabel=f'swing length percentile (%)', yLabel=(f'{events[e]} onset PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                       xyInvisible=[False, False])
                ax2.legend([],[], frameon=False)

            # ax0.set_ylim(0,65)
            # ax1.set_ylim(-1,1.5)
            # ax2.set_ylim(0.05,0.2)
        fname = f'fig_fractionOfModulated{cellType}{variable}Perc_allVariables_{timeInterval}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    def plotMLIindecisiveResponse(self, cellType, allModTraces, variable):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 40  # width in inches

        fig_height = 12
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 20, 'axes.titlesize': 16, 'font.size': 20, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4, 'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
                  'ytick.direction': 'in',
                  'xtick.direction': 'in',
                  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.2, hspace=0.3)

        plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1)
        modCells = {}
        events = ['swing', 'stance']
        pawNb = [0, 1]
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 9, subplot_spec=gs[0], hspace=0.3,
                                                  wspace=0.4, width_ratios=[1,1.2,1,1,1,1,1,1,1])

        possibleConditions = [ 'certainSteps', 'indecisiveSteps']

        percentilesc =['regular','miss']
        numberArray=[[],[]]
        traceAverageArray=[[],[]]
        traceSemArray = [[],[]]
        traceAUC = [[],[]]
        if cellType=='PC':
            timeInterval=0.10
            totCells=34
        else:
            timeInterval=0.10
            totCells=64
        cellDicList=[]
        colors = ['k', 'C6']
        for l, condition in enumerate(possibleConditions):
            for e in range(2):
                numberArray[e].append(allModTraces[condition][events[e]]['number'])

                meanZscore = np.mean(allModTraces[condition][events[e]]['traces'], axis=0)
                meanZscore = gaussian_filter1d(meanZscore, 0.8)
                semZscore = stats.sem(allModTraces[condition][events[e]]['traces'], axis=0)
                traceAverageArray[e].append(meanZscore)
                time = allModTraces[condition][events[e]]['time']

                eventOnsetMask=(time>-timeInterval) & (time<timeInterval)
                dt=np.diff(time)[0]
                for c in range(len(allModTraces[condition][events[e]]['traces'])):
                    trace = allModTraces[condition][events[e]]['traces'][c]
                    cellDic={}
                    AUC= np.trapz(allModTraces[condition][events[e]]['traces'][c][eventOnsetMask],dx=dt)
                    cellDic['AUC']=abs(AUC)
                    cellDic['condition']=f'{percentilesc[l]}' # {allModTraces[condition][events[e]]["number"]}'
                    cellDic['event']=events[e]
                    cellDic['id']=c
                    # Find positive peak
                    pos_peak = trace[eventOnsetMask].max()
                    pos_idx = np.argmax(trace[eventOnsetMask])
                    pos_latency = time[eventOnsetMask][pos_idx]

                    # Find negative peak
                    neg_peak = trace[eventOnsetMask].min()
                    neg_idx = np.argmin(trace[eventOnsetMask])
                    neg_latency = time[eventOnsetMask][neg_idx]

                    # Add to dictionary
                    cellDic['PosPeak'] = pos_peak
                    cellDic['PosLatency'] = pos_latency

                    cellDic['NegPeak'] = neg_peak
                    cellDic['NegLatency'] = neg_latency
                    # Get onset time
                    onset = time[eventOnsetMask][0]

                    # Find peak
                    peak = np.max(trace[eventOnsetMask])
                    peak_time = time[np.argmax(trace[eventOnsetMask])]

                    # Time to peak
                    time_to_peak = peak_time - onset

                    # Find trough
                    trough = np.min(trace[eventOnsetMask])
                    trough_time = time[np.argmin(trace[eventOnsetMask])]

                    # Time to trough
                    time_to_trough = trough_time - onset

                    # Store in dictionary
                    cellDic['TimeToPeak'] = time_to_peak
                    cellDic['TimeToTrough'] = time_to_trough

                    # Also store peak and trough values
                    cellDic['Peak'] = peak
                    cellDic['Trough'] = trough
                    cellDicList.append(cellDic)
        dfCells = pd.DataFrame(cellDicList)
        for e in range(2):
            ax0 = plt.subplot(gssub1[e, 0])
            if e == 0:
                alpha = 0.6
            else:
                alpha = 1
            percentiles = [ 'regular', 'miss']
            if cellType=='MLI':
                totalCell=64
            else:
                totalCell=34
            numberArrayEv = np.array(numberArray[e])
            MLIfrac = (numberArrayEv / totalCell) * 100
            x = np.arange(5)
            y = MLIfrac
            # slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
            #                                                              alternative='two-sided')
            ax0.bar(percentiles, MLIfrac, color=colors, alpha=alpha)

            if variable != 'swingLengthLinear':
                self.layoutOfPanel(ax0, xLabel=f'swing type',
                                   yLabel=f'fraction of {events[e]} modulated MLI (%)', Leg=[0, 9],
                                   xyInvisible=[False, False])
            ax1 = plt.subplot(gssub1[e, 1])

            import matplotlib.colors as mcolors

            blue = mcolors.to_rgba('C0')
            var = 'AUC'
            variables=['AUC','PosPeak', 'PosLatency', 'NegPeak', 'NegLatency' , 'TimeToPeak', 'TimeToTrough']
            varNames=['AUC (abs)', 'positive peak', 'positive peak latency', 'negative peak', 'negative peak latency', 'Time To Peak', 'Time To Trough']

            for a in range(2):
                intensity = (a + 1) / 2
                color = blue[:3] + (intensity,)  # adjust alpha channel\

                averageZscore = gaussian_filter1d(traceAverageArray[e][a], 0.8)
                ax1.plot(time, averageZscore, color=colors[a], lw=2,
                         label=f'[{percentiles[a]}]')
            ax1.text(0.42, 0.97, f"{events[e]} onset",
                     transform=ax1.transAxes, fontsize=12, color='grey')
            ax1.axvline(0, ls='--', color='grey', alpha=0.3)
            ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
            self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=' average Z-score', Leg=[0, 9],
                               xyInvisible=[False, False])
            ax1.text(0.42, 0.97, f"{events[e]} onset",
                     transform=ax1.transAxes, fontsize=12, color='grey')
            ax1.axvline(0, ls='--', color='grey', alpha=0.3)
            ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
            self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=' average Z-score', Leg=[0, 9],
                               xyInvisible=[False, False])
            self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=' average Z-score', Leg=[0, 9],
                               xyInvisible=[False, False])

            for v, var in enumerate(variables):
                ax2 = plt.subplot(gssub1[e, 2+v])
                cellDfEvent=dfCells[(dfCells['event']==events[e])]
                cellDfEventMiss=dfCells[(dfCells['event']==events[e]) & (dfCells['condition']=='miss')][var]
                cellDfEventReg = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == 'regular')][var]

                sns.pointplot(cellDfEvent, x='condition', y=var, hue='condition', ax=ax2, errorbar='se', palette=['k', 'C6'])
                # sns.lineplot(cellDfEvent, x='condition', y=var,  ax=ax2, errorbar=None, lw=2)
                # sns.scatterplot(cellDfEvent, x='condition', y='AUC', hue='condition', ax=ax2,
                #               palette=colorList)



                t_value, t_test_p_value = stats.ttest_ind(cellDfEventMiss, cellDfEventReg)

                star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                if t_test_p_value < 0.05:
                    ax2.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                             transform=ax2.transAxes,
                             style='italic', fontfamily='serif', fontsize=18, color='k')
                else:
                    ax2.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',
                             transform=ax2.transAxes,
                             style='italic', fontfamily='serif', fontsize=12, color='k')
                self.layoutOfPanel(ax2, xLabel=f'steps type', yLabel=f'z-score {varNames[v]}', Leg=[0, 9],
                                   xyInvisible=[False, False])
                ax2.legend([], [], frameon=False)
        fname = f'fig_fractionOfModulated{cellType}Miss_{timeInterval}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    def createComplexSpikeAnalysisFigure(self, df_CS,compCS_df):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')
        col = ['C0', 'C1', 'C2', 'C3']
        pawList=['FL','FR','HL','HR']
        fig_width = 15  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 10, 'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 1,  # ,
                               #width_ratios=[2,1]
                               height_ratios=[1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)
        gssub0 = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=gs[0], hspace=0.5, wspace=0.55)


        ax0=plt.subplot(gssub0[0,2])
        ax1= plt.subplot(gssub0[0,0])
        ax2=plt.subplot(gssub0[0,1])

        compCS_trial_df_mean = compCS_df.groupby(['Id', 'trialCat']).mean().reset_index()
        compCS_trial_df_mean_paw = compCS_df.groupby(['Id', 'trialCat', 'paw']).mean().reset_index()
        compCS_trial_df_mean['ratio']=compCS_trial_df_mean['locoCSFreq']/compCS_trial_df_mean['restCSFreq']
        # ratiosFirst=compCS_trial_df_mean[(compCS_trial_df_mean['trialCat']=='first')]['ratio'].values
        # ratiosLast=compCS_trial_df_mean[(compCS_trial_df_mean['trialCat']=='last')]['ratio'].values
        # pdb.set_trace()
        # sns.lineplot(compCS_trial_df_mean, x='trialCat', y='ratio',  ax=ax2, hue='Id', legend=False, lw=0.2, alpha=0.5, palette=['grey'])
        # sns.scatterplot(compCS_trial_df_mean, x='trialCat', y='ratio', ax=ax2, palette=['#a3cef0', '#167fc4'], edgecolor='grey',lw=0.1, s=80, alpha=0.5)
        cellsId=np.unique(compCS_trial_df_mean['Id'])
        ratiosLastAll=[]
        ratiosFirstAll=[]
        for cell in cellsId:
            try:
                last=compCS_trial_df_mean[(compCS_trial_df_mean['Id'] ==cell)&(compCS_trial_df_mean['trialCat'] == 'last')]
                first=compCS_trial_df_mean[(compCS_trial_df_mean['Id'] ==cell)&(compCS_trial_df_mean['trialCat'] == 'first')]
                ratiosLast = last['locoCSFreq'].values[0]/ last['restCSFreq'].values[0]
                ratiosFirst = first['locoCSFreq'].values[0]/ first['restCSFreq'].values[0]
            except:
                pass


            ratiosLastAll.append(ratiosLast)
            ratiosFirstAll.append(ratiosFirst)
            ax2.plot(['first','last'],[ratiosFirst/ratiosFirst, ratiosLast/ratiosFirst],lw=0.2,alpha=0.5, color='grey')
            ax2.scatter(['first', 'last'], [ratiosFirst / ratiosFirst,ratiosLast / ratiosFirst],edgecolor='k', facecolor=(f'C0'),lw=0.1, s=80)
        ax2.set_xlim(-0.9, 1.9)
        t_value, t_test_p_value2 = stats.ttest_rel(ratiosLastAll, ratiosFirstAll, axis=0)
        star_trial = groupAnalysis.starMultiplier(t_test_p_value2)
        if t_test_p_value2 < 0.05:
            ax2.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                     transform=ax2.transAxes,
                     style='italic', fontfamily='serif', fontsize=18, color='k')
        else:
            ax2.text(0.5, 0.99, (f'p={t_test_p_value2:.2f}'), ha='center', va='center',
                     transform=ax2.transAxes,
                     style='italic', fontfamily='serif', fontsize=12, color='k')
        compCS_df_day=compCS_df.groupby(['Id', 'day']).mean().reset_index()


        compCS_df_mean = compCS_df.groupby(['Id', 'trial']).mean().reset_index()
        compCS_df_mean_day_trial = compCS_df.groupby(['Id', 'trial','day']).mean().reset_index()
        cellsCs = np.unique(compCS_df_mean['Id'])
        ratios=[]
        for n in cellsCs:
            CS_loco=np.array(compCS_df_mean[compCS_df_mean['Id']==n]['locoCSFreq'])
            CS_rest =np.array( compCS_df_mean[compCS_df_mean['Id'] == n]['restCSFreq'])
            for r in range(len(CS_loco)):
                ratios.append(CS_loco[r]/CS_rest[r])
                if CS_loco[r]>0:
                    alpha = r / 5
                    alpha = max(alpha, 0.2)  # set minimum
                    ax1.scatter(['rest','loco'], [CS_rest[r]/CS_rest[r],CS_loco[r]/CS_rest[r]],edgecolor='k', facecolor=(f'C0'),lw=0.1, s=80, alpha=alpha)
                    ax1.plot(['rest','loco'], [CS_rest[r]/CS_rest[r],CS_loco[r]/CS_rest[r]], lw=0.2,alpha=0.5, color='grey')

                ax1.set_xlim(-0.9, 1.9)


        allCSxPos=[[],[]]

        for i in [0,1]:
            ax3 = plt.subplot(gssub0[1+i, 0])
            ax4 = plt.subplot(gssub0[1+i, 1:3])
            ax5 = plt.subplot(gssub0[1 + i, 3])
            probability = []
            missSwing=[]
            missStance=[]
            all_CS_swingLength=[]
            all_nonCS_swingLength = []
            all_CS_nextSwingLength=[]
            ct = 0
            for n in cellsCs:
                CS_sw=np.array(compCS_trial_df_mean_paw[(compCS_trial_df_mean_paw['Id']==n)&(compCS_trial_df_mean_paw['paw']==i)]['swingCSprob'])
                CS_stance =np.array(compCS_trial_df_mean_paw[(compCS_trial_df_mean_paw['Id']==n)&(compCS_trial_df_mean_paw['paw']==i)]['stanceCSProb'])


                CS_stanceX =np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSstancePos'])
                CS_stanceT = np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSstanceTimes'])
                CS_swingX =np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSswingPos'])
                CS_swingT = np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSswingTimes'])
                

                CS_stanceMiss =np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSstanceMiss'])


                CS_swingMiss = np.array(compCS_df[(compCS_df['Id'] == n)&(compCS_df['paw'] == i)]['CSswingMiss'])

                CS_swingLength= np.concatenate(np.array(compCS_df[(compCS_df['Id'] == n) & (compCS_df['paw'] == i)]['swingLengthStanceCS']))
                nonCS_swingLength = np.concatenate(np.array(compCS_df[(compCS_df['Id'] == n) & (compCS_df['paw'] == i)]['swingLengthStanceNoCS']))
                CS_nextSwingLength = np.concatenate(np.array(compCS_df[(compCS_df['Id'] == n) & (compCS_df['paw'] == i)]['swingLengthPostCSSwing']))
                all_CS_swingLength.append(CS_swingLength)
                all_nonCS_swingLength.append(nonCS_swingLength)
                all_CS_nextSwingLength.append(CS_nextSwingLength)
                CS_swingLength= np.concatenate(np.array(compCS_df[(compCS_df['Id'] == n) & (compCS_df['paw'] == i)]['swingLengthStanceCS']))
                nonCS_swingLength = np.concatenate(np.array(compCS_df[(compCS_df['Id'] == n) & (compCS_df['paw'] == i)]['swingLengthStanceNoCS']))

                for r in range(len(CS_sw)):
                    CS_stanceMissFrac=sum(CS_stanceMiss[r])/len(CS_stanceMiss[r])
                    CS_swingMissFrac = sum(CS_swingMiss[r]) / len(CS_swingMiss[r])
                    missSwing.append(CS_swingMissFrac)
                    missStance.append(CS_stanceMissFrac)
                    probability.append(CS_stance[r]/CS_sw[r])
                    if CS_sw[r]>0:
                        alpha = r / 5
                        alpha = max(alpha, 0.1)  # set minimum
                        ax3.scatter(['swing','stance'], [CS_sw[r]/CS_sw[r],CS_stance[r]/CS_sw[r]],edgecolor='k', facecolor=(f'C{i}'),lw=0.1, s=80, alpha=alpha)
                        ax3.plot(['swing','stance'], [CS_sw[r]/CS_sw[r],CS_stance[r]/CS_sw[r]], lw=0.2,alpha=0.5, color='grey')
                        ax5.scatter(['swing','stance'], [CS_swingMissFrac,CS_stanceMissFrac],edgecolor='k', facecolor=(f'C{i}'),lw=0.1, s=80, alpha=alpha)
                        ax5.plot(['swing','stance'], [CS_swingMissFrac,CS_stanceMissFrac], lw=0.2,alpha=0.5, color='grey')
                        ax5.axhline(0.5, ls='--', color='grey', lw=1, alpha=0.3)
                        max_len = max([len(traj) for traj in CS_stanceX[r]])
                        max_lenSw = max([len(trajSw) for trajSw in CS_swingT[r]])
                        # Pad all trajectories to max length
                        padded_time=[np.pad(traj, (0, max_len - len(traj))) for traj in CS_stanceT[r]]
                        padded_trajs = [np.pad(traj, (0, max_len - len(traj))) for traj in CS_stanceX[r]]
                        padded_timeSw=[np.pad(traj, (0, max_lenSw - len(traj))) for traj in CS_swingT[r]]
                        padded_trajsSw = [np.pad(traj, (0, max_lenSw - len(traj))) for traj in CS_swingX[r]]
                        # Take mean of padded trajectories
                        avg_traj = np.mean(padded_trajs, axis=0)
                        avg_time = np.mean(padded_time, axis=0)
                        avg_trajSw = np.mean(padded_trajsSw, axis=0)
                        avg_timeSw = np.mean(padded_timeSw, axis=0)
                        # num_points = len(padded_trajs[0])
                        # times = np.linspace(-0.3, 0.3, num=num_points)

                        ax4.plot(avg_time, avg_traj, lw=1, c='k', alpha=alpha, label=('stance CS' if ct==50 else ''))
                        ax4.plot(avg_timeSw, avg_trajSw, lw=1, c='purple', alpha=alpha, label=('swing CS'if ct==50 else ''))
                        ct+=1

                        for cs in range(len(CS_stanceX[r])):
                            # ax4.plot(np.linspace(-0.3, 0.3, num=len(CS_stanceX[r][cs])), CS_stanceX[r][cs], lw=0.1, alpha=0.2)
                            allCSxPos[i].append(CS_stanceX[r][cs])
                    ax5.set_xlim(-0.9, 1.9)
                    ax3.set_xlim(-0.9, 1.9)
                    ax4.set_xlim(-0.4, 0.4)
                    if i==0:
                        ax4.set_ylim(480, 620)
                    else:
                        ax4.set_ylim(460, 600)

                    ax3.axhline(1, ls='--', color='grey', lw=1, alpha=0.3)
                    ax4.axvline(0, ls='--', color='grey', alpha=0.3)

                    self.layoutOfPanel(ax4, xLabel=' time centered on CS onset (s)', yLabel=f' trial average  {pawList[i]} x-pos',                               Leg=[0, 9], xyInvisible=[False, False])
                    self.layoutOfPanel(ax3, xLabel=' ', yLabel='CS probability (fold)', Leg=[0, 9],                                       xyInvisible=[False, False])
                    self.layoutOfPanel(ax5, xLabel=' ', yLabel='CS during miss steps fraction', Leg=[0, 9],                                       xyInvisible=[False, False])

            ax6 = plt.subplot(gssub0[1 + i, 4])
            sns.kdeplot(x=np.concatenate(all_CS_swingLength), ax=ax6, color='C6', alpha=0.8)

            
            sns.kdeplot(x=np.concatenate(all_nonCS_swingLength), ax=ax6, color='C7', alpha=0.8)
            # sns.kdeplot(x=np.concatenate(all_CS_nextSwingLength), ax=ax6, ls='--', color='C6', alpha=0.8)

            # ax6.axvline(np.median(np.concatenate(all_CS_nextSwingLength)), ls='--', color='C6', alpha=0.3)
            ax6.axvline(np.median(np.concatenate(all_CS_swingLength)), ls='-', color='C6', alpha=0.3)
            ax6.axvline(np.median(np.concatenate(all_nonCS_swingLength)), ls='-', color='C7', alpha=0.3)
            self.layoutOfPanel(ax6, xLabel=' swing length (cm)', yLabel='probability density', Leg=[0, 9],
                               xyInvisible=[False, False])
            ax6.legend(['CS stride', 'no CS stride'], frameon=False, loc='upper left', fontsize=8)
            t_value, t_test_p_value_swingLen = stats.ttest_ind(np.concatenate(all_CS_swingLength), np.concatenate(all_nonCS_swingLength))
            star_swinlen = groupAnalysis.starMultiplier(t_test_p_value_swingLen)
            if t_test_p_value_swingLen < 0.05:
                ax6.text(0.5, 1.1, (f'{star_swinlen} '), ha='center', va='center', transform=ax6.transAxes,                         style='italic', fontfamily='serif', fontsize=18, color='k')
            else:
                ax6.text(0.5, 1.1, (f'p={t_test_p_value_swingLen:.2f}'), ha='center', va='center', transform=ax6.transAxes,                         style='italic', fontfamily='serif', fontsize=12, color='k')
            ax6.set_xlim(-1, 7)
            # ax6.set_xlim(0, 200)
            _, p_value1 = stats.ttest_1samp(probability, 1)
            star_trial2 = groupAnalysis.starMultiplier(p_value1)

            _, p_value2 = stats.ttest_1samp(missSwing, 0.5)
            star_trial3 = groupAnalysis.starMultiplier(p_value2)
            _, p_value3 = stats.ttest_1samp(missStance, 0.5)
            star_trial4 = groupAnalysis.starMultiplier(p_value3)

            if p_value1 < 0.05:
                ax3.text(0.5, 1.1, (f'{star_trial2} '), ha='center', va='center', transform=ax3.transAxes,                         style='italic', fontfamily='serif', fontsize=18, color='k')
            else:
                ax3.text(0.5, 1.1, (f'p={p_value1:.2f}'), ha='center', va='center', transform=ax3.transAxes,                         style='italic', fontfamily='serif', fontsize=12, color='k')

            if p_value2 < 0.05:
                ax5.text(0.35, 1.1, (f'{star_trial3} '), ha='center', va='center', transform=ax5.transAxes,                         style='italic', fontfamily='serif', fontsize=8, color='k')
            else:
                ax5.text(0.35, 1.1, (f'p={p_value2:.2f}'), ha='center', va='center', transform=ax5.transAxes,                         style='italic', fontfamily='serif', fontsize=8, color='k')

            if p_value3 < 0.05:
                ax5.text(0.65, 1.1, (f'{star_trial4} '), ha='center', va='center', transform=ax5.transAxes,                         style='italic', fontfamily='serif', fontsize=8, color='k')
            else:
                ax5.text(0.65, 1.1, (f'p={p_value3:.2f}'), ha='center', va='center', transform=ax5.transAxes,                         style='italic', fontfamily='serif', fontsize=8, color='k')

        compCS_df_mean_day_trial['ratio']=compCS_df_mean_day_trial['locoCSFreq']/compCS_df_mean_day_trial['restCSFreq']
        ratiosEarly=compCS_df_mean_day_trial[(compCS_df_mean_day_trial['day']=='early')]['ratio'].values
        ratiosLate=compCS_df_mean_day_trial[(compCS_df_mean_day_trial['day']=='late')]['ratio'].values


        ax0.bar(['early','late'], [np.mean(ratiosEarly),np.mean(ratiosLate)], yerr=[stats.sem(ratiosEarly),stats.sem(ratiosLate)], color=['C8', 'C4'])
        sns.scatterplot(compCS_df, x='day', y='CS_ratio', ax=ax0, palette=['C8', 'C4'], edgecolor='grey',
                        lw=0.1, s=80, alpha=0.3)
        ax0.set_xlim(-0.9, 1.9)
        t_value, t_test_p_value = stats.ttest_ind(ratiosEarly, ratiosLate)
        _, p_value = stats.ttest_1samp(ratios, 1)

        star_trial = groupAnalysis.starMultiplier(t_test_p_value)
        star_trial1 = groupAnalysis.starMultiplier(p_value)

        if t_test_p_value < 0.05:
            ax0.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',                     transform=ax0.transAxes,                     style='italic', fontfamily='serif', fontsize=18, color='k')
        else:
            ax0.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',                     transform=ax0.transAxes,                     style='italic', fontfamily='serif', fontsize=12, color='k')
        if p_value < 0.05:
            ax1.text(0.5, 0.99, (f'{star_trial1} '), ha='center', va='center',                     transform=ax1.transAxes,                     style='italic', fontfamily='serif', fontsize=18, color='k')
        else:
            ax1.text(0.5, 0.99, (f'p={p_value:.2f}'), ha='center', va='center',                     transform=ax1.transAxes,                     style='italic', fontfamily='serif', fontsize=12, color='k')


        self.layoutOfPanel(ax0, xLabel='sessions', yLabel=' CS frequency (spikes/s)', Leg=[0, 9],                           xyInvisible=[False, False])
        self.layoutOfPanel(ax1, xLabel='', yLabel='CS frequency (fold)', Leg=[0, 9],                           xyInvisible=[False, False])
        self.layoutOfPanel(ax2, xLabel=' trial', yLabel='CS frequency (fold)', Leg=[0, 9],                           xyInvisible=[False, False])



        ax1.axhline(1, ls='--', color='grey', lw=1, alpha=0.3)
        ax2.axhline(1, ls='--', color='grey', lw=1, alpha=0.3)

        # sns.scatterplot(compCS_trial_df_mean, x='trialCat', y='stanceCSProb', ax=ax3, palette=['#a3cef0', '#167fc4'], edgecolor='grey',lw=0.1, s=80, alpha=0.5)
        # sns.lineplot(compCS_trial_df_mean, x='trialCat', y='swingCSprob',  ax=ax3, hue='Id', legend=False, lw=0.2, alpha=0.2, palette=['grey'])
        # sns.scatterplot(compCS_trial_df_mean, x='trialCat', y='swingCSprob', ax=ax3, palette=['#a3cef0', '#167fc4'], edgecolor='grey',lw=0.1, s=80, alpha=0.2)


        fname = f'fig_CS_analysis'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ###########################################################
    def speedPSTHGroupFigure_rescaledTime(self,date, recordings, spikeType, PSTHSummaryAllAnimals, rescaled = False, alignment='Stance'):

        if rescaled:
            specification = 'psth_speed%sOnsetAlignedRescaled' % alignment
            xlim = [0,2]
            xvLine = 1
        else:
            specification = 'psth_speed%sOnsetAligned' % alignment
            xlim = [-0.3,0.4]
            xvLine = 0

        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0','C1','C2','C3']
        #tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals['allMLI'])

        fig_width = 12  # width in inches
        fig_height = 26  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 10, 'axes.titlesize': 10, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False, 'axes.spines.top':False, 'axes.spines.right':False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(5, 2,  # ,
                               #width_ratios=[0.1,1,5]
                               height_ratios=[2, 2, 1.5,2.2,2.2]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.3)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        #gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        #gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)

        psthRescaledAllMLIs = []
        nMice = len(PSTHSummaryAllAnimals['allMLI'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allMLI'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allMLI'][n]['PSTHdata']
            #pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            #gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):
                nRecs = len(PSTHData[m][3]) # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][1]
                        resTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][1]
                psthRescaledAllMLIs.append((tempPSTH/nRecs))

        psthRescaledAllPCs = []
        nMice = len(PSTHSummaryAllAnimals['allPC'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allPC'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allPC'][n]['PSTHdata']
            #pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            #gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):
                nRecs = len(PSTHData[m][3]) # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][1]
                        resTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][1]
                psthRescaledAllPCs.append((tempPSTH/nRecs))

        #pdb.set_trace()
        psthRescaledAllMLIs = np.asarray(psthRescaledAllMLIs)
        psthRescaledAllPCs = np.asarray(psthRescaledAllPCs)
        peak_indicesMLI = np.argmax(psthRescaledAllMLIs, axis=1)
        # Sort rows by the position of the positive peak
        sorted_indicesMLI = np.argsort(peak_indicesMLI)
        sorted_zs_stance_MLI = psthRescaledAllMLIs[sorted_indicesMLI]
        ax0= plt.subplot(gs[0])
        ax0.set_title('all MLI')
        factor = 0.7
        heatmapSwing = ax0.imshow(sorted_zs_stance_MLI, aspect='auto', cmap='RdBu_r', interpolation='none', extent=[resTime[0], resTime[-1], 0, sorted_zs_stance_MLI.shape[0]],
                                   vmin=-factor * np.max(np.abs(sorted_zs_stance_MLI)),  # Ensure 0 is white
                                   vmax=factor * np.max(np.abs(sorted_zs_stance_MLI)))
        # Add colorbar
        ax0.axvline(xvLine, ls='--', color='k')
        cbar = plt.colorbar(heatmapSwing)
        cbar.set_label('Z-score')
        ax0.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax0, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel='cell (N)')

        ######################
        peak_indicesPC = np.argmax(psthRescaledAllPCs, axis=1)
        # Sort rows by the position of the positive peak
        sorted_indicesPC = np.argsort(peak_indicesPC)
        sorted_zs_stance_PC = psthRescaledAllPCs[sorted_indicesPC]
        ax1= plt.subplot(gs[1])
        factor = 0.6
        ax1.set_title('all PC')
        heatmapSwing = ax1.imshow(sorted_zs_stance_PC, aspect='auto', cmap='RdBu_r', interpolation='none', extent=[resTime[0], resTime[-1], 0, sorted_zs_stance_PC.shape[0]],
                                   vmin=-factor * np.max(np.abs(sorted_zs_stance_PC)),  # Ensure 0 is white
                                   vmax=factor * np.max(np.abs(sorted_zs_stance_PC)))
        # Add colorbar
        ax1.axvline(xvLine, ls='--', color='k')
        cbar = plt.colorbar(heatmapSwing)
        cbar.set_label('Z-score')
        ax1.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax1, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel='cell (N)')
        ##############################
        ax2 = plt.subplot(gs[2])
        mliPeaks = []
        cmap = plt.get_cmap("nipy_spectral")

        # Normalize indices to [0,1] to map to colormap
        colors = cmap(np.linspace(0, 1, len(sorted_zs_stance_MLI)))
        for i in range(len(sorted_zs_stance_MLI)):
            maxIdx = np.argmax(sorted_zs_stance_MLI[i])
            minIdx = np.argmin(sorted_zs_stance_MLI[i])
            #ax2.plot(resTime[maxIdx],sorted_zs_stance_MLI[i][maxIdx],'+',c='C6')
            #ax2.plot(resTime[minIdx], sorted_zs_stance_MLI[i][minIdx], '+', c='C7')
            if resTime[maxIdx]<resTime[minIdx] or (not rescaled):
                ax2.plot([resTime[maxIdx], resTime[minIdx]], [sorted_zs_stance_MLI[i][maxIdx], sorted_zs_stance_MLI[i][minIdx]], 'o-',color=colors[i],alpha=0.3)
            else:
                x0 = resTime[maxIdx]
                y0 = sorted_zs_stance_MLI[i][maxIdx]
                x1 = resTime[minIdx]
                y1 = sorted_zs_stance_MLI[i][minIdx]
                m = (y0-y1)/((x0-2.)-x1) # slope
                b = y0 - m*x0 # intercept
                y2 = m*2. + b
                print('point at 2, slope, intercept :', y2, m, b)
                ax2.plot([x0], [y0], 'o',color=colors[i], alpha=0.3)
                ax2.plot([x0,2], [y0,y2], '-',color=colors[i], alpha=0.3)
                ax2.plot([0, x1], [y2, y1], '-', color=colors[i],alpha=0.3)
                ax2.plot([x1], [y1], 'o',color=colors[i], alpha=0.3)
            mliPeaks.append([i,resTime[maxIdx],resTime[minIdx],sorted_zs_stance_MLI[i][maxIdx],sorted_zs_stance_MLI[i][minIdx]])
        mliPeaks = np.asarray(mliPeaks)
        ax2.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax2.axvline(xvLine, ls='--', color='k')
        ax2.axhline(0, ls='--', color='gray')
        ax2.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax2, xLabel=(f'resacled time of z-score peak :\n FL %s onset' % alignment if rescaled else 'time of z-score peak :\n FL %s onset' % alignment), yLabel=f'peak amplitude (z-score)')
        ##############################
        ax3 = plt.subplot(gs[3])
        pcPeaks = []
        cmap = plt.get_cmap("nipy_spectral")

        # Normalize indices to [0,1] to map to colormap
        colors = cmap(np.linspace(0, 1, len(sorted_zs_stance_PC)))
        for i in range(len(sorted_zs_stance_PC)):
            maxIdx = np.argmax(sorted_zs_stance_PC[i])
            minIdx = np.argmin(sorted_zs_stance_PC[i])
            #ax3.plot(resTime[maxIdx], sorted_zs_stance_PC[i][maxIdx], '+', c='C6')
            #ax3.plot(resTime[minIdx], sorted_zs_stance_PC[i][minIdx], '+', c='C7')
            if resTime[maxIdx]<resTime[minIdx] or (not rescaled):
                ax3.plot([resTime[maxIdx],resTime[minIdx]],[sorted_zs_stance_PC[i][maxIdx],sorted_zs_stance_PC[i][minIdx]], 'o-',color=colors[i],alpha=0.3)
            else:
                x0 = resTime[maxIdx]
                y0 = sorted_zs_stance_PC[i][maxIdx]
                x1 = resTime[minIdx]
                y1 = sorted_zs_stance_PC[i][minIdx]
                m = (y0 - y1) / ((x0 - 2.) - x1)  # slope
                b = y0 - m * x0  # intercept
                y2 = m * 2. + b
                #print('point at 2, slope, intercept :', y2, m, b)
                ax3.plot([x0], [y0], 'o',color=colors[i], alpha=0.3)
                ax3.plot([x0, 2], [y0, y2], '-',color=colors[i], alpha=0.3)
                ax3.plot([0, x1], [y2, y1], '-',color=colors[i], alpha=0.3)
                ax3.plot([x1], [y1], 'o',color=colors[i], alpha=0.3)
            pcPeaks.append([i,resTime[maxIdx],resTime[minIdx],sorted_zs_stance_PC[i][maxIdx],sorted_zs_stance_PC[i][minIdx]])
        pcPeaks = np.asarray(pcPeaks)
        ax3.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax3.axvline(xvLine, ls='--', color='k')
        ax3.axhline(0, ls='--', color='gray')
        ax3.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax3, xLabel=(f'resacled time of z-score peak :\n FL %s onset' % alignment if rescaled else 'time of z-score peak :\n FL %s onset' % alignment), yLabel=f'peak amplitude (z-score)')
        ##############################
        ax4 = plt.subplot(gs[4])
        ax4.plot(resTime,np.mean(sorted_zs_stance_MLI,axis=0))

        ax4.axvline(xvLine, ls='--', color='k')
        ax4.axhline(0, ls='--', color='gray')
        ax4.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax4, xLabel=('rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'mean z-score across all cells')
        ##############################
        ax5 = plt.subplot(gs[5])
        ax5.plot(resTime, np.mean(sorted_zs_stance_PC,axis=0))
        ax5.axvline(xvLine, ls='--', color='k')
        ax5.axhline(0, ls='--', color='gray')
        ax5.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax5, xLabel=('rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'mean z-score across all cells')
        ##############################
        gssub10 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[6], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax6 = plt.subplot(gssub10[1])
        maxTimesMLI = []
        for i in range(len(sorted_zs_stance_MLI)):
            smoothed = self.moving_average(sorted_zs_stance_MLI[i],5)
            dd = np.diff(smoothed)
            idxMax = np.argmax(np.abs(dd))
            timeMax = resTime[idxMax]
            maxTimesMLI.append(timeMax)
            #dd = np.concatenate(([0],dd))
            #maxIdx = min(np.argmax(dd),len(resTime)-1)
            #minIdx = max(np.argmin(dd),0)
            #ax6.plot(resTime[maxIdx], dd[maxIdx], '+', c='C6')
            #ax6.plot(resTime[minIdx], dd[minIdx], '+', c='C7')
            ax6.plot(resTime[1:], dd,alpha=0.3)
        #ax6.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax6.axvline(xvLine, ls='--', color='k')
        ax6.axhline(0, ls='--', color='gray')
        ax6.set_xlim(xlim[0], xlim[1])
        #ax6.set_ylim(-3, 3)
        self.layoutOfPanel(ax6, xLabel=('rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'derivative of the PSTH')
        ##############
        ax_histx1 = plt.subplot(gssub10[0], sharex=ax6)
        ax_histx1.hist(maxTimesMLI, bins=10, color='gray', alpha=0.7)

        # Remove tick labels for cleaner look
        ax_histx1.tick_params(axis="x", labelbottom=False)

        ##############################
        gssub11 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[7], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax7 = plt.subplot(gssub11[1])
        maxTimesPC  = []
        for i in range(len(sorted_zs_stance_PC)):
            smoothed = self.moving_average(sorted_zs_stance_PC[i],5)
            dd = np.diff(smoothed)
            idxMax = np.argmax(np.abs(dd))
            timeMax = resTime[idxMax]
            maxTimesPC.append(timeMax)
            #dd = np.concatenate(([0],dd))
            #maxIdx = np.argmax(sorted_zs_stance_PC[i])
            #minIdx = np.argmin(sorted_zs_stance_PC[i])
            #maxIdx = min(np.argmax(dd),len(resTime)-1)
            #minIdx = max(np.argmin(dd),0)
            #ax7.plot(resTime[maxIdx], dd[maxIdx], '+', c='C6')
            #ax7.plot(resTime[minIdx], dd[minIdx], '+', c='C7')
            ax7.plot(resTime[1:],dd,alpha=0.3)
        #ax7.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax7.axvline(xvLine, ls='--', color='k')
        ax7.axhline(0, ls='--', color='gray')
        ax7.set_xlim(xlim[0], xlim[1])
        #ax7.set_ylim(-3, 3)
        ######################
        self.layoutOfPanel(ax7, xLabel=('rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'derivative of the PSTH')

        ax_histx1 = plt.subplot(gssub11[0], sharex=ax7)
        ax_histx1.hist(maxTimesPC, bins=10, color='gray', alpha=0.7)

        # Remove tick labels for cleaner look
        ax_histx1.tick_params(axis="x", labelbottom=False)

        #########################################################
        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[8], width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Main scatter plot
        ax_scatter = plt.subplot(gssub0[2])
        ax_scatter.scatter(mliPeaks[:, 1], mliPeaks[:, 2], alpha=0.6)
        ax_scatter.axvline(xvLine, ls='--', color='k')
        ax_scatter.axhline(xvLine, ls='--', color='gray')
        ax_scatter.set_xlim(xlim[0], xlim[1])
        ax_scatter.set_ylim(xlim[0], xlim[1])
        # Top histogram
        ax_histx = plt.subplot(gssub0[0], sharex=ax_scatter)
        ax_histx.hist(mliPeaks[:, 1], bins=10, color='gray', alpha=0.7)

        # Right histogram
        ax_histy = plt.subplot(gssub0[3], sharey=ax_scatter)
        ax_histy.hist(mliPeaks[:, 2], bins=10, color='gray', alpha=0.7, orientation='horizontal')

        # Remove tick labels for cleaner look
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax_scatter.set_xlabel('time of positive peak')
        ax_scatter.set_ylabel('time of negative peak')

        # ax8 = plt.subplot(gs[8])
        # for i in range(len(sorted_zs_stance_MLI)):
        #     ax8.plot(mliPeaks[i,1], mliPeaks[i,2], 'o-')
        # # ax6.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        # ax8.axvline(1, ls='--', color='k')
        # ax8.axhline(1, ls='--', color='gray')
        # ax8.set_xlim(0, 2)
        # ax8.set_ylim(0, 2)
        # self.layoutOfPanel(ax8, xLabel=f'time of positive peak', yLabel=f'time of negative peak')

        ##############################
        #gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[9], width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Main scatter plot
        ax_scatter = plt.subplot(gssub1[2])
        ax_scatter.scatter(pcPeaks[:,1], pcPeaks[:,2], alpha=0.6)
        ax_scatter.axvline(xvLine, ls='--', color='k')
        ax_scatter.axhline(xvLine, ls='--', color='gray')
        ax_scatter.set_xlim(xlim[0], xlim[1])
        ax_scatter.set_ylim(xlim[0], xlim[1])
        # Top histogram
        ax_histx = plt.subplot(gssub1[0], sharex=ax_scatter)
        ax_histx.hist(pcPeaks[:,1], bins=10, color='gray', alpha=0.7)

        # Right histogram
        ax_histy = plt.subplot(gssub1[3], sharey=ax_scatter)
        ax_histy.hist(pcPeaks[:,2], bins=10, color='gray', alpha=0.7, orientation='horizontal')

        # Remove tick labels for cleaner look
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax_scatter.set_xlabel('time of positive peak')
        ax_scatter.set_ylabel('time of negative peak')

        #ax9 = plt.subplot(gs[9])
        #for i in range(len(sorted_zs_stance_PC)):
        #    ax9.plot(pcPeaks[i,1], pcPeaks[i,2], 'o-')
        ## ax7.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        #ax9.axvline(1, ls='--', color='k')
        #ax9.axhline(1, ls='--', color='gray')
        #ax9.set_xlim(0, 2)
        #ax9.set_ylim(0, 2)
        #self.layoutOfPanel(ax9,  xLabel=f'time of positive peak', yLabel=f'time of negative peak')

        ##############################
        if rescaled :
            fname = 'speed_rescaled-psth_group_analysis_%s_%s' % ('allSteps', alignment)
        else:
            fname = 'speed_psth_group_analysis_%s_%s' % ('allSteps', alignment)

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ###############################################################################################################################
    def PSTHGroupFigure_rescaledTime(self,date, recordings, spikeType, PSTHSummaryAllAnimals, rescaled = False, alignment='stance'):

        if rescaled:
            specification = 'psth_%sOnsetAlignedRescaled' % alignment
            xlim = [0,2]
            xvLine = 1
        else:
            specification = 'psth_%sOnsetAligned' % alignment
            xlim = [-0.3,0.4]
            xvLine = 0

        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0','C1','C2','C3']
        #tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals['allMLI'])

        fig_width = 12  # width in inches
        fig_height = 26  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 10, 'axes.titlesize': 10, 'font.size': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4,  'axes.grid': False, 'axes.spines.top':False, 'axes.spines.right':False # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        fig = plt.figure()
        gs = gridspec.GridSpec(5, 2,  # ,
                               #width_ratios=[0.1,1,5]
                               height_ratios=[2, 2, 1.5,2.2,2.2]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.3)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        #gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        #gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)

        psthRescaledAllMLIs = []
        nMice = len(PSTHSummaryAllAnimals['allMLI'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allMLI'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allMLI'][n]['PSTHdata']
            #pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            #gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):
                nRecs = len(PSTHData[m][3]) # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s_z-scored' % specification][1]
                        resTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s_z-scored' % specification][1]
                psthRescaledAllMLIs.append((tempPSTH/nRecs))

        psthRescaledAllPCs = []
        nMice = len(PSTHSummaryAllAnimals['allPC'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allPC'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allPC'][n]['PSTHdata']
            #pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            #gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):
                nRecs = len(PSTHData[m][3]) # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s_z-scored' % specification][1]
                        resTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s' % specification][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['%s_z-scored' % specification][1]
                psthRescaledAllPCs.append((tempPSTH/nRecs))

        #pdb.set_trace()
        psthRescaledAllMLIs = np.asarray(psthRescaledAllMLIs)
        psthRescaledAllPCs = np.asarray(psthRescaledAllPCs)
        peak_indicesMLI = np.argmax(psthRescaledAllMLIs, axis=1)
        # Sort rows by the position of the positive peak
        sorted_indicesMLI = np.argsort(peak_indicesMLI)
        sorted_zs_stance_MLI = psthRescaledAllMLIs[sorted_indicesMLI]
        ax0= plt.subplot(gs[0])
        ax0.set_title('all MLI')
        factor = 0.4
        heatmapSwing = ax0.imshow(sorted_zs_stance_MLI, aspect='auto', cmap='RdBu_r', interpolation='none', extent=[resTime[0], resTime[-1], 0, sorted_zs_stance_MLI.shape[0]],
                                   vmin=-factor * np.max(np.abs(sorted_zs_stance_MLI)),  # Ensure 0 is white
                                   vmax=factor * np.max(np.abs(sorted_zs_stance_MLI)))
        # Add colorbar
        ax0.axvline(xvLine, ls='--', color='k')
        cbar = plt.colorbar(heatmapSwing)
        cbar.set_label('Z-score')
        ax0.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax0, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel='cell (N)')

        ######################
        peak_indicesPC = np.argmax(psthRescaledAllPCs, axis=1)
        # Sort rows by the position of the positive peak
        sorted_indicesPC = np.argsort(peak_indicesPC)
        sorted_zs_stance_PC = psthRescaledAllPCs[sorted_indicesPC]
        ax1= plt.subplot(gs[1])
        factor = 0.4
        ax1.set_title('all PC')
        heatmapSwing = ax1.imshow(sorted_zs_stance_PC, aspect='auto', cmap='RdBu_r', interpolation='none', extent=[resTime[0], resTime[-1], 0, sorted_zs_stance_PC.shape[0]],
                                   vmin=-factor * np.max(np.abs(sorted_zs_stance_PC)),  # Ensure 0 is white
                                   vmax=factor * np.max(np.abs(sorted_zs_stance_PC)))
        # Add colorbar
        ax1.axvline(xvLine, ls='--', color='k')
        cbar = plt.colorbar(heatmapSwing)
        cbar.set_label('Z-score')
        ax1.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax1, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel='cell (N)')
        ##############################
        ax2 = plt.subplot(gs[2])
        mliPeaks = []
        cmap = plt.get_cmap("nipy_spectral")

        # Normalize indices to [0,1] to map to colormap
        colors = cmap(np.linspace(0, 1, len(sorted_zs_stance_MLI)))
        for i in range(len(sorted_zs_stance_MLI)):
            maxIdx = np.argmax(sorted_zs_stance_MLI[i])
            minIdx = np.argmin(sorted_zs_stance_MLI[i])
            #ax2.plot(resTime[maxIdx],sorted_zs_stance_MLI[i][maxIdx],'+',c='C6')
            #ax2.plot(resTime[minIdx], sorted_zs_stance_MLI[i][minIdx], '+', c='C7')
            if resTime[maxIdx]<resTime[minIdx] or (not rescaled):
                ax2.plot([resTime[maxIdx], resTime[minIdx]], [sorted_zs_stance_MLI[i][maxIdx], sorted_zs_stance_MLI[i][minIdx]], 'o-',color=colors[i],alpha=0.3)
            else:
                x0 = resTime[maxIdx]
                y0 = sorted_zs_stance_MLI[i][maxIdx]
                x1 = resTime[minIdx]
                y1 = sorted_zs_stance_MLI[i][minIdx]
                m = (y0-y1)/((x0-2.)-x1) # slope
                b = y0 - m*x0 # intercept
                y2 = m*2. + b
                print('point at 2, slope, intercept :', y2, m, b)
                ax2.plot([x0], [y0], 'o',color=colors[i], alpha=0.3)
                ax2.plot([x0,2], [y0,y2], '-',color=colors[i], alpha=0.3)
                ax2.plot([0, x1], [y2, y1], '-', color=colors[i],alpha=0.3)
                ax2.plot([x1], [y1], 'o',color=colors[i], alpha=0.3)
            mliPeaks.append([i,resTime[maxIdx],resTime[minIdx],sorted_zs_stance_MLI[i][maxIdx],sorted_zs_stance_MLI[i][minIdx]])
        mliPeaks = np.asarray(mliPeaks)
        ax2.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax2.axvline(xvLine, ls='--', color='k')
        ax2.axhline(0, ls='--', color='gray')
        ax2.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax2, xLabel=(f'resacled time of z-score peak :\n FL %s onset' % alignment if rescaled else f'time of z-score peak :\n FL %s onset' % alignment), yLabel=f'peak amplitude (z-score)')
        ##############################
        ax3 = plt.subplot(gs[3])
        pcPeaks = []
        cmap = plt.get_cmap("nipy_spectral")

        # Normalize indices to [0,1] to map to colormap
        colors = cmap(np.linspace(0, 1, len(sorted_zs_stance_PC)))
        for i in range(len(sorted_zs_stance_PC)):
            maxIdx = np.argmax(sorted_zs_stance_PC[i])
            minIdx = np.argmin(sorted_zs_stance_PC[i])
            #ax3.plot(resTime[maxIdx], sorted_zs_stance_PC[i][maxIdx], '+', c='C6')
            #ax3.plot(resTime[minIdx], sorted_zs_stance_PC[i][minIdx], '+', c='C7')
            if resTime[maxIdx]<resTime[minIdx] or (not rescaled):
                ax3.plot([resTime[maxIdx],resTime[minIdx]],[sorted_zs_stance_PC[i][maxIdx],sorted_zs_stance_PC[i][minIdx]], 'o-',color=colors[i],alpha=0.3)
            else:
                x0 = resTime[maxIdx]
                y0 = sorted_zs_stance_PC[i][maxIdx]
                x1 = resTime[minIdx]
                y1 = sorted_zs_stance_PC[i][minIdx]
                m = (y0 - y1) / ((x0 - 2.) - x1)  # slope
                b = y0 - m * x0  # intercept
                y2 = m * 2. + b
                #print('point at 2, slope, intercept :', y2, m, b)
                ax3.plot([x0], [y0], 'o',color=colors[i], alpha=0.3)
                ax3.plot([x0, 2], [y0, y2], '-',color=colors[i], alpha=0.3)
                ax3.plot([0, x1], [y2, y1], '-',color=colors[i], alpha=0.3)
                ax3.plot([x1], [y1], 'o',color=colors[i], alpha=0.3)
            pcPeaks.append([i,resTime[maxIdx],resTime[minIdx],sorted_zs_stance_PC[i][maxIdx],sorted_zs_stance_PC[i][minIdx]])
        pcPeaks = np.asarray(pcPeaks)
        ax3.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax3.axvline(xvLine, ls='--', color='k')
        ax3.axhline(0, ls='--', color='gray')
        ax3.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax3, xLabel=(f'resacled time of z-score peak :\n FL %s onset' % alignment if rescaled else f'time of z-score peak :\n FL %s onset' % alignment), yLabel=f'peak amplitude (z-score)')
        ##############################
        ax4 = plt.subplot(gs[4])
        ax4.plot(resTime,np.mean(sorted_zs_stance_MLI,axis=0))

        ax4.axvline(xvLine, ls='--', color='k')
        ax4.axhline(0, ls='--', color='gray')
        ax4.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax4, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'mean z-score across all cells')
        ##############################
        ax5 = plt.subplot(gs[5])
        ax5.plot(resTime, np.mean(sorted_zs_stance_PC,axis=0))
        ax5.axvline(xvLine, ls='--', color='k')
        ax5.axhline(0, ls='--', color='gray')
        ax5.set_xlim(xlim[0], xlim[1])
        self.layoutOfPanel(ax5, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'mean z-score across all cells')
        ##############################
        gssub10 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[6], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax6 = plt.subplot(gssub10[1])
        maxTimesMLI = []
        for i in range(len(sorted_zs_stance_MLI)):
            smoothed = self.moving_average(sorted_zs_stance_MLI[i],5)
            dd = np.diff(smoothed)
            idxMax = np.argmax(np.abs(dd))
            timeMax = resTime[idxMax]
            maxTimesMLI.append(timeMax)
            #dd = np.concatenate(([0],dd))
            #maxIdx = min(np.argmax(dd),len(resTime)-1)
            #minIdx = max(np.argmin(dd),0)
            #ax6.plot(resTime[maxIdx], dd[maxIdx], '+', c='C6')
            #ax6.plot(resTime[minIdx], dd[minIdx], '+', c='C7')
            ax6.plot(resTime[1:], dd,alpha=0.3)
        #ax6.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax6.axvline(xvLine, ls='--', color='k')
        ax6.axhline(0, ls='--', color='gray')
        ax6.set_xlim(xlim[0], xlim[1])
        #ax6.set_ylim(-3, 3)
        self.layoutOfPanel(ax6, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'derivative of the PSTH')
        ##############
        ax_histx1 = plt.subplot(gssub10[0], sharex=ax6)
        ax_histx1.hist(maxTimesMLI, bins=10, color='gray', alpha=0.7)

        # Remove tick labels for cleaner look
        ax_histx1.tick_params(axis="x", labelbottom=False)

        ##############################
        gssub11 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[7], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax7 = plt.subplot(gssub11[1])
        maxTimesPC  = []
        for i in range(len(sorted_zs_stance_PC)):
            smoothed = self.moving_average(sorted_zs_stance_PC[i],5)
            dd = np.diff(smoothed)
            idxMax = np.argmax(np.abs(dd))
            timeMax = resTime[idxMax]
            maxTimesPC.append(timeMax)
            #dd = np.concatenate(([0],dd))
            #maxIdx = np.argmax(sorted_zs_stance_PC[i])
            #minIdx = np.argmin(sorted_zs_stance_PC[i])
            #maxIdx = min(np.argmax(dd),len(resTime)-1)
            #minIdx = max(np.argmin(dd),0)
            #ax7.plot(resTime[maxIdx], dd[maxIdx], '+', c='C6')
            #ax7.plot(resTime[minIdx], dd[minIdx], '+', c='C7')
            ax7.plot(resTime[1:],dd,alpha=0.3)
        #ax7.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        ax7.axvline(xvLine, ls='--', color='k')
        ax7.axhline(0, ls='--', color='gray')
        ax7.set_xlim(xlim[0], xlim[1])
        #ax7.set_ylim(-3, 3)
        ######################
        self.layoutOfPanel(ax7, xLabel=(f'rescaled time centered on %s onset' % alignment if rescaled else 'time centered on %s onset' % alignment), yLabel=f'derivative of the PSTH')

        ax_histx1 = plt.subplot(gssub11[0], sharex=ax7)
        ax_histx1.hist(maxTimesPC, bins=10, color='gray', alpha=0.7)

        # Remove tick labels for cleaner look
        ax_histx1.tick_params(axis="x", labelbottom=False)

        #########################################################
        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[8], width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Main scatter plot
        ax_scatter = plt.subplot(gssub0[2])
        ax_scatter.scatter(mliPeaks[:, 1], mliPeaks[:, 2], alpha=0.6)
        ax_scatter.axvline(xvLine, ls='--', color='k')
        ax_scatter.axhline(xvLine, ls='--', color='gray')
        ax_scatter.set_xlim(xlim[0], xlim[1])
        ax_scatter.set_ylim(xlim[0], xlim[1])
        # Top histogram
        ax_histx = plt.subplot(gssub0[0], sharex=ax_scatter)
        ax_histx.hist(mliPeaks[:, 1], bins=10, color='gray', alpha=0.7)

        # Right histogram
        ax_histy = plt.subplot(gssub0[3], sharey=ax_scatter)
        ax_histy.hist(mliPeaks[:, 2], bins=10, color='gray', alpha=0.7, orientation='horizontal')

        # Remove tick labels for cleaner look
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax_scatter.set_xlabel('time of positive peak')
        ax_scatter.set_ylabel('time of negative peak')

        # ax8 = plt.subplot(gs[8])
        # for i in range(len(sorted_zs_stance_MLI)):
        #     ax8.plot(mliPeaks[i,1], mliPeaks[i,2], 'o-')
        # # ax6.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        # ax8.axvline(1, ls='--', color='k')
        # ax8.axhline(1, ls='--', color='gray')
        # ax8.set_xlim(0, 2)
        # ax8.set_ylim(0, 2)
        # self.layoutOfPanel(ax8, xLabel=f'time of positive peak', yLabel=f'time of negative peak')

        ##############################
        #gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[9], width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Main scatter plot
        ax_scatter = plt.subplot(gssub1[2])
        ax_scatter.scatter(pcPeaks[:,1], pcPeaks[:,2], alpha=0.6)
        ax_scatter.axvline(xvLine, ls='--', color='k')
        ax_scatter.axhline(xvLine, ls='--', color='gray')
        ax_scatter.set_xlim(xlim[0], xlim[1])
        ax_scatter.set_ylim(xlim[0], xlim[1])
        # Top histogram
        ax_histx = plt.subplot(gssub1[0], sharex=ax_scatter)
        ax_histx.hist(pcPeaks[:,1], bins=10, color='gray', alpha=0.7)

        # Right histogram
        ax_histy = plt.subplot(gssub1[3], sharey=ax_scatter)
        ax_histy.hist(pcPeaks[:,2], bins=10, color='gray', alpha=0.7, orientation='horizontal')

        # Remove tick labels for cleaner look
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax_scatter.set_xlabel('time of positive peak')
        ax_scatter.set_ylabel('time of negative peak')

        #ax9 = plt.subplot(gs[9])
        #for i in range(len(sorted_zs_stance_PC)):
        #    ax9.plot(pcPeaks[i,1], pcPeaks[i,2], 'o-')
        ## ax7.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
        #ax9.axvline(1, ls='--', color='k')
        #ax9.axhline(1, ls='--', color='gray')
        #ax9.set_xlim(0, 2)
        #ax9.set_ylim(0, 2)
        #self.layoutOfPanel(ax9,  xLabel=f'time of positive peak', yLabel=f'time of negative peak')

        ##############################
        if rescaled :
            fname = 'psth_rescaled-psth_group_analysis_%s_%s' % ('allSteps',alignment)
        else:
            fname = 'psth_psth_group_analysis_%s_%s' % ('allSteps',alignment)

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ################################################################################################################
    def PSTHGroupFigure_missStep(self, date, recordings, spikeType, PSTHSummaryAllAnimals,cellType='allMLI'):
        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0', 'C1', 'C2', 'C3']
        # tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals[cellType])
        if cellType == 'allMLI':
            nRowsCols = 8
        elif cellType == 'allPC':
            nRowsCols = 6

        fig_width = 20  # width in inches
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
        gs.update(wspace=0.25, hspace=0.25)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        # gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        nFig = 0
        nMice = len(PSTHSummaryAllAnimals[cellType])
        PSTHquantifyDict = {}
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals[cellType][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals[cellType][n]['PSTHdata']
            # pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            PSTHquantifyDict[mouse] = {}
            # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):  # cells per animal
                gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[nFig], hspace=0.1, wspace=0.5)  # , height_ratios=[1,2,1],width_ratios=[1])
                ax0 = plt.subplot(gssub2[0])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                PSTHquantifyDict[mouse][m] = {}
                for k in range(nRecs):
                    if k == 0:
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedMissstep_z-scored'][1]
                        tempExpStanceMis = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_expectedStanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful_z-scored'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful'][0]
                    else:
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedMissstep_z-scored'][1]
                        tempExpStanceMis += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_expectedStanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful_z-scored'][1]
                ax0.plot(tTime, tempPSTHSucc/nRecs, c='C0', label='successful step')
                ax0.plot(tTime, tempPSTHMiss/nRecs,c='C1', label='miss step, actual stance onset')
                ax0.plot(tTime, tempExpStanceMis / nRecs, c='C2', label='miss step, expected stance onset')
                #psths.append([tTime,tempPSTHSucc/nRecs,tempPSTHMiss/nRecs,tempExpStanceMis / nRecs])
                PSTHquantifyDict[mouse][m]['timePSTH'] = tTime
                PSTHquantifyDict[mouse][m]['psthSucc'] = tempPSTHSucc/nRecs
                PSTHquantifyDict[mouse][m]['psthMiss'] = tempPSTHMiss/nRecs
                PSTHquantifyDict[mouse][m]['psthExp'] = tempExpStanceMis / nRecs
                PSTHquantifyDict[mouse][m]['peakTimeSuccessfulStep'] = tTime[np.argmax(tempPSTHSucc/nRecs)]
                PSTHquantifyDict[mouse][m]['peakTimeMissStep'] = tTime[np.argmax(tempPSTHMiss/nRecs)]
                PSTHquantifyDict[mouse][m]['peakTimeExpectedStanceMissStep'] = tTime[np.argmax(tempExpStanceMis/nRecs)]
                PSTHquantifyDict[mouse][m]['psthStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['stanceStart']
                ax0.axvline(0, ls='--', color='gray',alpha=0.5)
                ax0.axhline(0, ls=':', color='gray',alpha=0.5)
                ax0.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax0, xLabel=None, yLabel=(r'activity \n(z-score)' if not (nFig%nRowsCols) else None),xyInvisible=[True,False],Leg=(([0.1,1.1],9) if nFig==0 else None))
                #
                ax1 = plt.subplot(gssub2[1])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempExpStanceMis = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][0]
                    else:
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempExpStanceMis += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                ax1.plot(tTime, tempPSTHMiss / nRecs, c='C1', label='miss step, actual stance onset')
                ax1.plot(tTime, tempPSTHSucc / nRecs, c='C0', label='successful step')
                ax1.plot(tTime, tempExpStanceMis / nRecs, c='C2', label='miss step, expected stance onset')
                PSTHquantifyDict[mouse][m]['timeSpeed'] = tTime
                PSTHquantifyDict[mouse][m]['speedSucc'] = tempPSTHSucc / nRecs
                PSTHquantifyDict[mouse][m]['speedMiss'] = tempPSTHMiss / nRecs
                PSTHquantifyDict[mouse][m]['speedExp'] = tempExpStanceMis / nRecs
                PSTHquantifyDict[mouse][m]['speedStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['speedCenteredStanceStart']
                ax1.axvline(0, ls='--', color='gray', alpha=0.5)
                ax1.axhline(0, ls=':', color='gray', alpha=0.5)
                ax1.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax1, xLabel=(f'time centered on actual/expected \n stance onset (s)' if nFig >= nRowsCols*(nRowsCols-1) else None), yLabel=('speed' if not (nFig % nRowsCols) else None))
                #
                nFig +=1

        ###### save pickle file
        pickle.dump(PSTHquantifyDict, open('psth-speed-profiles_%s.p' % cellType, 'wb'))

        ##############################
        fname = 'ephys_missstep-success_psth_group_analysis_stanceOnset_%s_%s' % ('allSteps',cellType)

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ################################################################################################################
    def SpeedGroupFigure_missStep(self, date, recordings, spikeType, PSTHSummaryAllAnimals):
        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0', 'C1', 'C2', 'C3']
        # tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals['allMLI'])

        fig_width = 20  # width in inches
        fig_height = 20  # height in inches
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
        gs = gridspec.GridSpec(8, 8,  # ,
                               # width_ratios=[0.1,1,5]
                               # height_ratios=[2, 1.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.25, hspace=0.25)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        # gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        nFig = 0
        nMice = len(PSTHSummaryAllAnimals['allMLI'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allMLI'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allMLI'][n]['PSTHdata']
            # pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):  # cells per animal
                ax0 = plt.subplot(gs[nFig])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempExpStanceMis = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][0]
                    else:
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempExpStanceMis += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                ax0.plot(tTime, tempPSTHMiss/nRecs,c='C1', label='miss step, actual stance onset')
                ax0.plot(tTime, tempPSTHSucc/nRecs, c='C0', label='successful step')
                ax0.plot(tTime, tempExpStanceMis / nRecs, c='C2', label='miss step, expected stance onset')
                ax0.axvline(0, ls='--', color='gray',alpha=0.5)
                ax0.axhline(0, ls=':', color='gray',alpha=0.5)
                ax0.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax0, xLabel=(f'time centered on actual/expected \n stance onset (s)' if nFig>=8*7 else None), yLabel=('speed' if not (nFig%8) else None),Leg=(([0.1,1.1],9) if nFig==0 else None))
                nFig +=1

        ##############################
        fname = 'speed_missstep-success_psth_group_analysis_%s' % ('allSteps')

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    ################################################################################################################
    def PSTHGroupFigure_swingOnset(self, date, recordings, spikeType, PSTHSummaryAllAnimals,cellType='allMLI'):
        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0', 'C1', 'C2', 'C3']
        # tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals[cellType])
        if cellType == 'allMLI':
            nRowsCols = 8
        elif cellType == 'allPC':
            nRowsCols = 6

        fig_width = 20  # width in inches
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
        gs.update(wspace=0.25, hspace=0.25)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        # gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        nFig = 0
        nMice = len(PSTHSummaryAllAnimals[cellType])
        PSTHquantifyDict = {}
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals[cellType][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals[cellType][n]['PSTHdata']
            # pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            PSTHquantifyDict[mouse] = {}
            # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):  # cells per animal
                gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[nFig], hspace=0.1, wspace=0.5)  # , height_ratios=[1,2,1],width_ratios=[1])
                ax0 = plt.subplot(gssub2[0])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                PSTHquantifyDict[mouse][m] = {}
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAligned'][1]  #psth_swingOnsetAligned_z-scored'][1]
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAlignedMissstep'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAlignedSuccessful'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAligned'][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAligned'][1] # _z-scored'][1]
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAlignedMissstep'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_swingOnsetAlignedSuccessful'][1]
                ax0.plot(tTime, self.moving_average(tempPSTH/nRecs,3),c='C2', label='all steps')
                ax0.plot(tTime, self.moving_average(tempPSTHSucc / nRecs, 3), c='C0', label='successful steps')
                ax0.plot(tTime, self.moving_average(tempPSTHMiss / nRecs, 3), c='C1', label='miss steps')
                ax0.axvline(0, ls='--', color='gray',alpha=0.5)
                #ax0.axhline(0, ls=':', color='gray',alpha=0.5)
                ax0.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax0, xLabel=None, yLabel=(f'activity \n z-score' if not (nFig%nRowsCols) else None),Leg=(([0.1,1.1],9) if nFig==0 else None))
                #
                ax1 = plt.subplot(gssub2[1])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedSuccessful'][1]
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedMissstep'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedSuccessful'][1]
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedMissstep'][1]
                ax1.plot(tTime, tempPSTH / nRecs, c='C2', label='all steps')
                ax1.plot(tTime, tempPSTHSucc / nRecs, c='C0', label='successful steps')
                ax1.plot(tTime, tempPSTHMiss / nRecs, c='C1', label='miss steps')
                ax1.axvline(0, ls='--', color='gray', alpha=0.5)
                ax1.axhline(0, ls=':', color='gray', alpha=0.5)
                ax1.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax1, xLabel=(f'time centered on actual/expected \n swing onset (s)' if nFig >= nRowsCols*(nRowsCols-1) else None), yLabel=('speed' if not (nFig % nRowsCols) else None),
                                   Leg=(([0.1, 1.1], 9) if nFig == 0 else None))

                #
                nFig +=1

        ##############################
        fname = 'ephys_missstep-success_psth_group_analysis_swingOnset_%s_%s' % ('allSteps',cellType)

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')

    ################################################################################################################
    def SpeedGroupFigure_swingOnset(self, date, recordings, spikeType, PSTHSummaryAllAnimals):
        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0', 'C1', 'C2', 'C3']
        # tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals['allMLI'])

        fig_width = 20  # width in inches
        fig_height = 20  # height in inches
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
        gs = gridspec.GridSpec(8, 8,  # ,
                               # width_ratios=[0.1,1,5]
                               # height_ratios=[2, 1.5]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.25, hspace=0.25)
        # plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.40, 0.96, 'before stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.96, 'after stance onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.40, 0.45, 'before swing onset', clip_on=False, color='black', size=10)
        # plt.figtext(0.80, 0.45, 'after swing onset', clip_on=False, color='black', size=10)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.1)
        # gssub0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0,0], hspace=0.2)
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.15)
        nFig = 0
        nMice = len(PSTHSummaryAllAnimals['allMLI'])
        for n in range(nMice):
            mouse = PSTHSummaryAllAnimals['allMLI'][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals['allMLI'][n]['PSTHdata']
            # pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):  # cells per animal
                ax0 = plt.subplot(gs[nFig])
                nRecs = len(PSTHData[m][3])  # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempPSTH = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][1]
                        tempPSTHSucc =  PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedSuccessful'][1]
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedMissstep'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][0]
                    else:
                        tempPSTH += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAligned'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedSuccessful'][1]
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedSwingOnsetAlignedMissstep'][1]
                ax0.plot(tTime, tempPSTH/nRecs,c='C1', label='all steps')
                ax0.plot(tTime, tempPSTHSucc/nRecs, c='C0', label='successful steps')
                ax0.plot(tTime, tempPSTHMiss / nRecs, c='C2', label='miss steps')
                ax0.axvline(0, ls='--', color='gray',alpha=0.5)
                ax0.axhline(0, ls=':', color='gray',alpha=0.5)
                ax0.set_xlim(-0.3, 0.4)
                self.layoutOfPanel(ax0, xLabel=(f'time centered on actual/expected \n swing onset (s)' if nFig>=8*7 else None), yLabel=('speed' if not (nFig%8) else None),Leg=(([0.1,1.1],9) if nFig==0 else None))
                nFig +=1

        ##############################
        fname = 'speed_missstep-success_psth_group_analysis_swingOnset_%s' % ('allSteps')

        plt.savefig(self.figureDirectory + '/' + fname + '.pdf')
    ##############################################################################
    def moving_average(self, arr, x):
        kernel = np.ones(x) / x  # Create a smoothing kernel
        ma = np.convolve(arr, kernel, mode='same')  # Apply convolution
        return ma