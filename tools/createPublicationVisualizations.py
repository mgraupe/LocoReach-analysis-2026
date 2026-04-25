'''
        Class to provide images and videos
        
'''

import numpy as np
#import matplotlib
#matplotlib.use("Agg")
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import tools.groupAnalysis_psth as groupAnalysis_psth
import pandas as pd
import scipy
from mycolorpy import colorlist as mcp
#from pylab import *
import tifffile as tiff
import matplotlib as mpl
from matplotlib import rcParams
import itertools
import matplotlib.image as mpimg
from scipy.integrate import quad
import statistics
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
#import sima
#import sima.motion
#import sima.segment
from scipy.stats.stats import pearsonr
from scipy.interpolate import interp1d
#from mtspec import mt_coherence
from scipy import stats
import matplotlib.ticker as ticker
import tools.groupAnalysis as groupAnalysis
import seaborn as sns
import tools.dataAnalysis as dataAnalysis
import statsmodels.formula.api as smf
from tools.pyqtgraph.configfile import *
from matplotlib.ticker import FormatStrFormatter
# mpl.use('WxAgg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import statsmodels.api as sm
from scipy.stats import t
#import pymer4.models as models
#from pymer4.models import Lmer
#from pymer4.utils import get_resource_path
from matplotlib.colors import from_levels_and_colors
from matplotlib.collections import LineCollection
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.genmod.generalized_linear_model as glm
groupAnalysisDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary'
class createVisualizations:

    ##########################################################################################
    def __init__(self,figureDir,mouse):

        self.mouse = mouse
        self.figureDirectory = figureDir
        if not os.path.isdir(self.figureDirectory):
            os.system('mkdir %s' % self.figureDirectory)

        self.pawID = ['FL', 'FR', 'HL', 'HR']

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
    def experimentWalkingFig(self, version, sTimes,linearSpeed,cPawPos,angluarSpeed,angleTimes,frame,pawSpeed,swingStanceDict,experimentWalkingFig,wheelPawCorrCoeffs):
        pawID = 0 # use FL
        nMice = len(experimentWalkingFig)
        stanceDurDict = []
        swingDurDict = []
        for m in range(nMice):
            print(experimentWalkingFig[m][0])
            #stanceDurDict[m] =  []
            #swingDurDict[m] = []
            for sess in range(len(experimentWalkingFig[m][2])):
                for rec in range(len(experimentWalkingFig[m][2][sess][1])):
                    print(m,sess,rec)
                    stanceDurDict.extend(experimentWalkingFig[m][2][sess][1][rec][pawID]['stanceDuration'])
                    swingDurDict.extend(experimentWalkingFig[m][2][sess][1][rec][pawID]['swingDuration'])
        #pdb.set_trace()
        # figure #################################
        fig_width = 17  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 14, 'font.size': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(1, 2,  width_ratios=[1,1.3])
                               #height_ratios=[2, 1])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.1)
        plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.06)
        # plt.figtext(0.01, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.96, 'B', clip_on=False, color='black', size=22)
        # plt.figtext(0.01, 0.46, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.56, 'D', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.26, 'E', clip_on=False, color='black',  size=22)
        # plt.figtext(0.71, 0.26, 'F', clip_on=False, color='black', size=22)

        gssub0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.1, height_ratios=[2,1,0.5])

        # panel ##################
        ax0 = plt.subplot(gssub0[1])
        #pdb.set_trace()
        position = [0,-150,-250,-550]
        for i in range(4):
            ax0.plot(cPawPos[i][:,0],cPawPos[i][:,1]+position[i],label=self.pawID[i])
            # pdb.set_trace()
        self.layoutOfPanel(ax0,yLabel='x-position (pixel)',Leg=[3,9],xyInvisible=[True,False])
        # ax0.set_title('Paws x-position and wheel speed', loc='center', fontsize=14)
        #ax0.set_xlim(22,26)

        # panel ##################
        ax1 = plt.subplot(gssub0[2])
        ax1.axhline(y=0., ls=':', c='0.6')
        #linearSpeedSmooth = np.convolve(linearSpeed, np.ones(20)/20, mode='same')
        #linearSpeedSmooth1 = scipy.signal.medfilt(linearSpeed,kernel_size=9)
        ax1.plot(sTimes,-linearSpeed,c='darkorchid')
        #ax1.plot(sTimes, -linearSpeedSmooth, c='orange')
        #ax1.plot(sTimes, -linearSpeedSmooth1, c='green')
        #ax1.set_xlim(22,26)

        self.layoutOfPanel(ax1,xLabel='time (s)',yLabel='wheel speed (cm/s)')  # axL[n].append(ax)

        # panel ##################
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4, height_ratios=[1,0.3, 1, 1])
        gssubsub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[0], wspace=0.001)

        ax2 = plt.subplot(gssubsub1[0])
        ax2.imshow(np.transpose(frame),cmap='gray',vmax=np.max(frame)*0.5)
        self.layoutOfPanel(ax2, xyInvisible=[True,True])
        textx = ax2.annotate('x', xy=(90,520), annotation_clip=False,xytext=None, textcoords='data',fontsize=15,arrowprops=None,color='white')
        texty = ax2.annotate('y', xy=(30,500), annotation_clip=False,xytext=None, textcoords='data',fontsize=15,arrowprops=None,color='white')
        ax2.annotate('', xy=(160, 530), xycoords='data',xytext=(60, 530), annotation_clip=False, arrowprops=dict(arrowstyle="->", color='white', lw=2), )
        ax2.annotate('', xy=(60, 430), xycoords='data',xytext=(60, 530), annotation_clip=False,arrowprops=dict(arrowstyle="->",color='white',lw=2),)
        ###
        ax3 = plt.subplot(gssubsub1[1])
        ax3.imshow(np.transpose(frame),cmap='gray',vmax=np.max(frame)*0.5)
        startStop = [20,40]
        for i in range(4):
            timeMask = (cPawPos[i][:,0]>startStop[0])&(cPawPos[i][:,0]<startStop[1])
            ax3.plot(cPawPos[i][:,1][timeMask],cPawPos[i][:,2][timeMask],lw=0.4,alpha=0.8)
        self.layoutOfPanel(ax3, xyInvisible=[True,True])

        # panel ##################
        ax4 = plt.subplot(gssub1[2])
        pawIDtoShow = 1
        scalingFactor = 0.021
        #startStop = [14,19]
        startStop = [36,41]
        #tM1 = (sTimes>startStop[0])&(sTimes<startStop[1])
        #tM2 = (pawSpeed[1][:,0]>startStop[0])&(pawSpeed[1][:,0]<startStop[1])
        idxSwings = swingStanceDict['swingP'][pawIDtoShow][1]
        indecisiveSteps = swingStanceDict['swingP'][pawIDtoShow][3]
        recTimes = swingStanceDict['forFit'][pawIDtoShow][2]
        # pdb.set_trace()
        stanceDur = []
        swingDur = []
        ax4.axhline(y=0., ls=':', c='0.6')
        Ngradient = 20
        alphaRange = np.linspace(0.1,0.5,Ngradient,endpoint=True)
        mpl.rcParams['hatch.linewidth'] = 4
        for n in range(len(idxSwings)):
            idxStart = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[idxSwings[n][0]]))
            idxEnd = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[idxSwings[n][1]]))
            swingDur.append(recTimes[idxSwings[n][1]]-recTimes[idxSwings[n][0]])
            #print((recTimes[idxSwings[n][1]]-recTimes[idxSwings[n][0]]),(cPawPos[pawIDtoShow][idxEnd, 0]-cPawPos[pawIDtoShow][idxStart, 0]))
            if n>0:
                stanceDur.append(recTimes[idxSwings[n][0]] - recTimes[idxSwings[n - 1][1]])
                #stanceDur.append(recTimes[idxSwings[n][0]]-recTimes[idxSwings[n-1][1]])
            #ax4.plot(cPawPos[i][idxStart, 0], cPawPos[pawIDtoShow][idxStart, 1]*scalingFactor, 'x', alpha=0.5, lw=0.5)
            coloredRange = np.linspace(cPawPos[pawIDtoShow][idxStart, 0], cPawPos[pawIDtoShow][idxEnd, 0], Ngradient, endpoint=True)
            if indecisiveSteps[n][3]:  # indecisive Step
                #ax4.plot(cPawPos[pawIDtoShow][idxEnd, 0], cPawPos[pawIDtoShow][idxEnd, 1]*scalingFactor, '1', alpha=0.5, lw=0.5)
                for i in range(Ngradient-1):
                    ax4.axvspan(coloredRange[i],coloredRange[i+1],alpha=alphaRange[i],lw=0,facecolor='C0')
            else:
                #ax4.plot(cPawPos[pawIDtoShow][idxEnd, 0], cPawPos[pawIDtoShow][idxEnd, 1]*scalingFactor, '+', alpha=0.5, lw=0.5)
                for i in range(Ngradient - 1):
                    ax4.axvspan(coloredRange[i],coloredRange[i+1],alpha=alphaRange[i],lw=0, facecolor='C0')
        ax4.plot(sTimes, -linearSpeed, c='darkorchid',lw=2)
        ax4.plot(pawSpeed[pawIDtoShow][:,0],pawSpeed[pawIDtoShow][:,2]*scalingFactor,c='C0')

        self.layoutOfPanel(ax4,xLabel='time (s)',yLabel=r'speed $v_x$ (cm/s)')
        ax4.set_xlim(startStop[0],startStop[1])
        ax4.set_ylim(-30,120)

        # panel ##################
        gssub3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[3], hspace=0.3)#, height_ratios=[2, 1, 0.5])
        ax5 = plt.subplot(gssub3[0])
        pawIDtoShow = 0
        bb = np.linspace(0,0.65,25,endpoint=True)
        #for n in range(len(stanceDurDict)):
        #    aa = 0.1+0.8*n/len(stanceDurDict)
        ax5.axvline(np.median(stanceDurDict),color='gray',alpha=0.6,ls='--')
        ax5.axvline(np.median(swingDurDict),color='C0',ls='--')
        print('median swing dur:',np.percentile(swingDurDict,(25,50,75)))
        print('median stance dur:',np.percentile(stanceDurDict,(25,50,75)))
        ax5.hist(stanceDurDict,bins=bb,histtype='step',color='gray',alpha=0.6,density=True,label='%s stance' % self.pawID[pawIDtoShow])
        ax5.hist(swingDurDict,bins=bb,histtype='step',color='C0',density=True,label='%s swing' % self.pawID[pawIDtoShow])
        # ax5.set_title('Swing and stance duration', loc='center', fontsize=14)
        #ax5.hist(stanceDur, bins=bb, histtype='step', color='gray', alpha=0.4, label='%s stance' % self.pawID[pawIDtoShow])
        #ax5.hist(swingDur, bins=bb, histtype='step', color='C1', label='%s swing' % self.pawID[pawIDtoShow])
        self.layoutOfPanel(ax5, xLabel='duration (s)', yLabel='probability density',Leg=[1,9])

        ######
        # ax6 = plt.subplot(gssub3[1])
        # #for i in range(4):
        # refPaw = 0
        # idxRefSwings = swingStanceDict['swingP'][refPaw][1]
        # #indecisiveSteps = swingStanceDict['swingP'][refPaw][3]
        # recRefTimes = swingStanceDict['forFit'][refPaw][2]
        # #pdb.set_trace()
        # dt = 0.02
        # counts = np.zeros((8,int(1/dt)+1))
        # for n in range(len(idxRefSwings)-2):
        #     idxRefSwingStart = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n][0]]))
        #     refSwingStart = recRefTimes[idxRefSwings[n][0]]
        #     #idxRefSwingEnd = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n][1]]))
        #     idxRefStanceEnd = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n+1][0]]))
        #     refStandEnd = recRefTimes[idxRefSwings[n+1][0]]
        #     refStandEndNP1 = recRefTimes[idxRefSwings[n+2][0]]
        #     refDuration = refStandEnd-refSwingStart
        #     refDurationNP1 = refStandEndNP1-refStandEnd
        #     for i in range(4):
        #         idxSwings = swingStanceDict['swingP'][i][1]
        #         recTimes = swingStanceDict['forFit'][i][2]
        #         for m in range(len(idxSwings)):
        #             idxStart = np.argmin(np.abs(cPawPos[i][:,0] - recTimes[idxSwings[m][0]]))
        #             startTime = recTimes[idxSwings[m][0]]
        #             idxEnd = np.argmin(np.abs(cPawPos[i][:,0] - recTimes[idxSwings[m][1]]))
        #             endTime = recTimes[idxSwings[m][1]]
        #             if (idxStart >= idxRefSwingStart) and (idxStart<idxRefStanceEnd) and (idxEnd<idxRefStanceEnd):
        #                 iStart = int((startTime-refSwingStart)/(refDuration*dt))
        #                 iEnd = int((endTime-refSwingStart)/(refDuration*dt))
        #                 counts[2*i,iStart:iEnd]+=1
        #             elif (idxStart >= idxRefSwingStart) and (idxStart<idxRefStanceEnd) and (idxEnd>idxRefStanceEnd):
        #                 iStart = int((startTime-refSwingStart)/(refDuration*dt))
        #                 iEnd = int((endTime-refStandEnd)/(refDurationNP1*dt))
        #                 counts[2*i,iStart:]+=1
        #                 counts[2*i,:iEnd]+=1
        #
        # #ax6.imshow(counts,cmap='gist_yarg',interpolation='nearest',aspect='auto')
        # counts = counts/np.max(counts)
        # for i in range(4):
        #     ax6.plot(counts[i*2])
        # self.layoutOfPanel(ax6, xLabel='% '+self.pawID[refPaw] +' stride (norm)', yLabel='p(swing)')
        # ax6.set_title('Probability of swing (FL norm)', loc='center', fontsize=14)
        # #plt.yticks([0,2,4,6], self.pawID)
        # plt.xticks([0, 10, 20, 30, 40, 50], [0,0.2,0.4,0.6,0.8,1])

        #########
        ax6 = plt.subplot(gssub3[1])
        bb = np.linspace(np.min(wheelPawCorrCoeffs[:,(0,2,4,6,8,10)]),np.max(wheelPawCorrCoeffs[:,(0,2,4,6,8,10)]),50,endpoint=True)
        #ax6.axvline(x=0,ls=':',c='gray')
        ax6.hist(wheelPawCorrCoeffs[:,0],histtype='step',density=True,color='olive',bins=bb,label='sum')
        #ax6.hist(wheelPawCorrCoeffs[:,2],histtype='step',density=True,color='darkcyan',bins=bb,label='sum(abs)')
        ax6.hist(wheelPawCorrCoeffs[:,(4,6,8,10)].flatten(),histtype='step',density=True,color='brown',bins=bb,label='individual')
        self.layoutOfPanel(ax6, xLabel='correlation coefficient', yLabel='probability density',Leg=(1,9))
        ax6.set_xlim(-0.57,0.09)
        #ax6.set_title('Probability of swing (FL norm)', loc='center', fontsize=14)
        # plt.yticks([0,2,4,6], self.pawID)
        #plt.xticks([0, 10, 20, 30, 40, 50], [0, 0.2, 0.4, 0.6, 0.8, 1])

        ########################
        fname = 'fig_experiment-PawPositionExtraction_v%s' % version
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    ##########################################################################################
    # figVersion, sTimes,linearSpeed,cPawPos,angluarSpeed,angleTimes,frame,pawSpeed,swingStanceDict
    def experimentPhDOverviewFig(self, version, sTimes,linearSpeed,cPawPos,angluarSpeed,angleTimes,frame,pawSpeed,swingStanceDict,jointNames,suite2pFolder,stat910,ops910,F,Fneu,spks,iscell,obstacleDict,frameWhisGray,obstacleDictCa,caTimes,sTimesCa,linearSpeedCa,F820,iscell820):
        #pdb.set_trace(),
        obstacleStartEnd = np.row_stack((obstacleDict['obstacle1UPAndActivatedStartEndTime'],obstacleDict['obstacle2UPAndActivatedStartEndTime']))
        obstacleStartEnd = obstacleStartEnd[obstacleStartEnd[:, 0].argsort()]
        obstacleStartEndCa = np.row_stack((obstacleDictCa['obstacle1UPAndActivatedStartEndTime'],obstacleDictCa['obstacle2UPAndActivatedStartEndTime']))
        obstacleStartEndCa = obstacleStartEndCa[obstacleStartEndCa[:,0].argsort()]
        #pdb.set_trace()
        pawID = 0 # use FL
        jointList = [0,2,3,4,5,8,9,10,11]
        colorList = ['cyan','C0','C1','C2','C3','C0','C1','C2','C3']
        print(jointNames)
        frameHeight = np.shape(frame)[0]
        print('frameHeight : ', frameHeight)
        #pdb.set_trace()
        # figure #################################
        fig_width = 17  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 14, 'font.size': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 3)#,  width_ratios=[1,1.3])
                               #height_ratios=[2, 1])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)
        plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.06)
        # plt.figtext(0.01, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.96, 'B', clip_on=False, color='black', size=22)
        # plt.figtext(0.01, 0.46, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.56, 'D', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.26, 'E', clip_on=False, color='black',  size=22)
        # plt.figtext(0.71, 0.26, 'F', clip_on=False, color='black', size=22)

        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], hspace=0.2,width_ratios=[2,1] )#height_ratios=[2,1,0.5])
        #########################
        #gssubsub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub0[0], wspace=0.001)

        ax2 = plt.subplot(gssub0[0])
        ax2.imshow(frame, cmap='gray', vmax=np.max(frame) * 0.7)
        self.layoutOfPanel(ax2, xyInvisible=[True, True])
        textx = ax2.annotate('x', xy=(90, 520), annotation_clip=False, xytext=None, textcoords='data', fontsize=15, arrowprops=None, color='white')
        texty = ax2.annotate('y', xy=(28, 500), annotation_clip=False, xytext=None, textcoords='data', fontsize=15, arrowprops=None, color='white')
        ax2.annotate('', xy=(160, 530), xycoords='data', xytext=(60, 530), annotation_clip=False, arrowprops=dict(arrowstyle="->", color='white', lw=2), )
        ax2.annotate('', xy=(60, 430), xycoords='data', xytext=(60, 530), annotation_clip=False, arrowprops=dict(arrowstyle="->", color='white', lw=2), )
        textx = ax2.annotate('x', xy=(90, 150), annotation_clip=False, xytext=None, textcoords='data', fontsize=15, arrowprops=None, color='white')
        texty = ax2.annotate('z', xy=(28, 130), annotation_clip=False, xytext=None, textcoords='data', fontsize=15, arrowprops=None, color='white')
        ax2.annotate('', xy=(160, 160), xycoords='data', xytext=(60, 160), annotation_clip=False, arrowprops=dict(arrowstyle="->", color='white', lw=2), )
        ax2.annotate('', xy=(60, 60), xycoords='data', xytext=(60, 160), annotation_clip=False, arrowprops=dict(arrowstyle="->", color='white', lw=2), )
        ###
        ax2 = plt.subplot(gssub0[1])
        ax2.imshow(frameWhisGray, cmap='gray', vmax=np.max(frameWhisGray) * 0.7)
        self.layoutOfPanel(ax2, xyInvisible=[True, True])
        ####
        ax3 = plt.subplot(gssub0[2])
        ax3.imshow(frame, cmap='gray', vmax=np.max(frame) * 0.7)
        startStop = [20, 50]
        cN = 0
        for i in jointList : #range(len(cPawPos)):
            if i == 0:
                timeMask = (cPawPos[i][:, 0] > 0) & (cPawPos[i][:, 0] < 60)
            else:
                timeMask = (cPawPos[i][:, 0] > startStop[0]) & (cPawPos[i][:, 0] < startStop[1])
            ax3.plot(cPawPos[i][:, 1][timeMask], cPawPos[i][:, 2][timeMask], c = colorList[cN], lw=0.4, alpha=0.8)
            cN+=1
        self.layoutOfPanel(ax3, xyInvisible=[True, True])


        gssub1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.1, height_ratios=[1,1,0.7])
        # panel ##################
        startDisp = 21
        endDisp = 31
        ax0 = plt.subplot(gssub1[0])
        #pdb.set_trace()
        #position = [0,-150,-250,-550,0,-150,-250,-550,0,-150,-250,-550,0,-150,-250,-550]
        for n in range(len(obstacleStartEnd)):
            tt = obstacleStartEnd[n]
            ax0.axvspan(tt[0],tt[1],alpha=0.3,color='C5')
        cN = 0
        for i in [2,3]:
            ax0.plot(cPawPos[i][:,0],frameHeight- cPawPos[i][:,2]+cN*80,c=colorList[cN],label=jointNames[i])
            #ax0.plot(cPawPos[i][:, 0], cPawPos[i][:, ] + cN * 80, c=colorList[cN], label=jointNames[i])
            cN+=1
            # pdb.set_trace()
        self.layoutOfPanel(ax0,yLabel='z-position (pix)',Leg=[3,9],xyInvisible=[True,False])
        # ax0.set_title('Paws x-position and wheel speed', loc='center', fontsize=14)
        ax0.set_xlim(startDisp,endDisp)

        # panel ##################
        ax0 = plt.subplot(gssub1[1])
        #pdb.set_trace()
        #position = [0,-150,-250,-550,0,-150,-250,-550,0,-150,-250,-550,0,-150,-250,-550]
        for n in range(len(obstacleStartEnd)):
            tt = obstacleStartEnd[n]
            ax0.axvspan(tt[0],tt[1],alpha=0.3,color='C5')
        cN = 0
        for i in [8,9]:
            ax0.plot(cPawPos[i][:,0],cPawPos[i][:,1]+cN*180,c=colorList[cN],label=jointNames[i])
            cN+=1
            # pdb.set_trace()

        self.layoutOfPanel(ax0,yLabel='x-position (pix)',Leg=[3,9],xyInvisible=[True,False])
        # ax0.set_title('Paws x-position and wheel speed', loc='center', fontsize=14)
        ax0.set_xlim(startDisp,endDisp)

        # panel ##################
        ax1 = plt.subplot(gssub1[2])
        ax1.axhline(y=0., ls=':', c='0.6')
        #linearSpeedSmooth = np.convolve(linearSpeed, np.ones(20)/20, mode='same')
        #linearSpeedSmooth1 = scipy.signal.medfilt(linearSpeed,kernel_size=9)
        for n in range(len(obstacleStartEnd)):
            tt = obstacleStartEnd[n]
            ax1.axvspan(tt[0],tt[1],alpha=0.3,color='C5')
        ax1.plot(sTimes,-linearSpeed,c='darkorchid')
        #ax1.plot(sTimes, -linearSpeedSmooth, c='orange')
        #ax1.plot(sTimes, -linearSpeedSmooth1, c='green')
        ax1.set_xlim(startDisp,endDisp)
        #ax1.set_xlim(-21, 4)
        self.layoutOfPanel(ax1,xLabel='time (s)',yLabel='wheel speed\n (cm/s)')  # axL[n].append(ax)

        # panel ##################
        #gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4, height_ratios=[1,0.3, 1, 1])
        ax2 = plt.subplot(gs[3])
        caImg = ops910['meanImg']
        #im = np.zeros((ops910['Ly'], ops910['Lx']))
        dimensions = np.shape(caImg)
        mask = np.zeros((dimensions[0], dimensions[1]))
        #maskAmp = np.zeros((dimensions[0], dimensions[1]))
        nRois = np.sum(iscell[:,0]==1)
        nRois820 = np.sum(iscell820[:,0]==1)
        #pdb.set_trace()
        for n in range(nRois):
            #idxRoi = intersectData[idxRef]['corrDays'][d][3][n][1]
            ypix = stat910[n]['ypix']  # [~stat[n]['overlap']]
            xpix = stat910[n]['xpix']  # [~stat[n]['overlap']]
            mask[ypix, xpix] = (n + 1) / nRois + 1  # n Rois, i paw, r nDay, l recording : peakCaTransients[n,i,r,l]
            #maskAmp[ypix, xpix] = (peakCaTransients910[n, pawID, d, 0] + scaler) / (2 * scaler)  # - scaler)/(np.max(peakCaTransients[:,:,:,0])-np.min(peakCaTransients[:,:,:,0]))
        masked = np.ma.masked_where(mask == 0, mask)
        #maskedAmp = np.ma.masked_where(maskAmp == 0, maskAmp)
        # panel production : FOV with ROI ID
        #ax0 = plt.subplot(gs[d])
        #ax0.set_title('%s (%s)' % (day, idxDay))

        ax2.imshow(caImg, vmax=np.percentile(caImg.flatten(), 99), cmap=plt.get_cmap('gray'))
        ax2.imshow(masked, 'prism', interpolation='none', vmin=1, vmax=2, alpha=0.25, origin='upper')
        exampleRois = [38, 61, 77, 86, 92, 113, 117, 131, 136, 141, 152]#, 174]
        exampleRois820 = [42, 73, 98, 108, 116, 138, 144, 159, 168, 174, 190]#, 210]
        for n in range(nRois):
            if n in exampleRois:
                ypix = stat910[n]['ypix']
                xpix =  stat910[n]['xpix']
                ax2.text(np.mean(xpix),np.mean(ypix), '%s' % n, color='white', fontsize=8)
        self.layoutOfPanel(ax2, xyInvisible=[True, True])

        # panel ##################

        cc = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']
        # gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.4, height_ratios=[1,0.3, 1, 1])
        gssub4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[4], hspace=0.1, height_ratios=[2.5, 0.7])
        ax2 = plt.subplot(gssub4[0])
        # pdb.set_trace()
        timePoints = np.shape(F)[1]
        fps = 30
        tt = np.linspace(1/fps,timePoints/fps,timePoints)
        for n in range(len(obstacleStartEndCa)):
            tt = obstacleStartEndCa[n]
            if tt[0]>10 and tt[1]<52:
                ax2.axvspan(tt[0],tt[1],alpha=0.3,color='C5')
        nR = 0
        nTimePoints = len(caTimes)
        blMask = caTimes<6
        for n in range(nRois):
            if n in exampleRois:
                Ftrace = F[n][:nTimePoints]
                Fbl = np.mean(Ftrace[blMask])
                dFF = (Ftrace-Fbl)/Fbl
                ax2.plot(caTimes,dFF+nR*5,c=cc[nR])
                ax2.text(0, nR * 5+1, '%s' % n, color='0.2', fontsize=8)
                nR+=1
        nR = 0
        for n in range(nRois):
            if n in exampleRois820:
                Ftrace820 = F820[n][:nTimePoints]
                Fbl820 = np.mean(Ftrace820[blMask])
                dFF820 = (Ftrace820 - Fbl820) / Fbl820
                ax2.plot(caTimes, dFF820 + nR * 5,c=cc[nR],alpha=0.2 )
                nR += 1
        ax2.set_xlim(0,60)
        self.layoutOfPanel(ax2,xLabel='time (s)',yLabel='fluorescence ($\Delta$F/F)',xyInvisible=[True,False])
        ax2 = plt.subplot(gssub4[1])
        ax2.axhline(y=0., ls=':', c='0.6')
        #linearSpeedSmooth = np.convolve(linearSpeed, np.ones(20)/20, mode='same')
        #linearSpeedSmooth1 = scipy.signal.medfilt(linearSpeed,kernel_size=9)
        for n in range(len(obstacleStartEndCa)):
            tt = obstacleStartEndCa[n]
            if tt[0] > 10 and tt[1] < 52:
                ax2.axvspan(tt[0],tt[1],alpha=0.3,color='C5')
        ax2.plot(sTimesCa,-linearSpeedCa,c='darkorchid')
        #ax1.plot(sTimes, -linearSpeedSmooth, c='orange')
        #ax1.plot(sTimes, -linearSpeedSmooth1, c='green')
        #ax1.set_xlim(startDisp,endDisp)
        ax2.set_xlim(0,60)
        self.layoutOfPanel(ax2,xLabel='time (s)',yLabel='wheel speed\n (cm/s)')  # axL[n].append(ax)
        ######

        ax5 = plt.subplot(gs[5])
        #ax5.axvspan(x=0,ls=':',c='0.5')
        ax5.axvspan(0,0.72,alpha=0.3,color='C5')
        nO = 0
        for i in range(len(obstacleStartEndCa)):
            tt = obstacleStartEndCa[i]
            print('duration:',tt[0], tt[1]-tt[0])
            blmask = (caTimes<6.)
            mask = (caTimes>tt[0]-2)&(caTimes<tt[0]+4)
            wheelMask = (sTimesCa>tt[0]-2)&(sTimesCa<tt[0]+4)
            #beforemask = (caTimes>tt[0]-4)*(caTimes<tt[0])
            #caAvg = np.zeros(np.sum(mask))
            if (tt[0]>10) and (tt[0]<52): # in [1,2,3,4]:
                nR = 0
                nAdd=0
                nAdd820=0
                dFFmean = np.zeros(np.sum(mask))
                dFF820mean = np.zeros(np.sum(mask))
                for n in range(nRois):
                    #if n in exampleRois:
                    #idxRoi910 = np.where(np.array(exampleRois)==n)[0][0]
                    FBL = np.mean(F[n][:nTimePoints][blmask])
                    Ftrace = F[n][:nTimePoints][mask]
                    dFF = (Ftrace-FBL)/FBL
                    before = np.mean(dFF[:50])
                    #if before>3.:
                    #dFF = scipy.ndimage.gaussian_filter1d(dFF, 3)
                    dFFs = np.convolve((dFF-before), np.ones(4), 'same')
                    dFFmean+=dFFs
                    nAdd+=1
                for n in range(nRois820):
                    #nRoi810 = exampleRois820[idxRoi910]
                    #print(n,nRoi810,idxRoi910)
                    FBL820 = np.mean(F820[n][:nTimePoints][blmask])
                    Ftrace820 = F820[n][:nTimePoints][mask]
                    #pdb.set_trace()
                    #dFF = (Ftrace-FBL)/FBL
                    dFF820 = (Ftrace820-FBL820)/FBL820
                    before820 = np.mean(dFF820[:50])
                    dFF820s = np.convolve((dFF820-before820), np.ones(4), 'same')
                    dFF820mean+=dFF820s
                    #ax5.plot(caTimes[mask]-tt[0],dFF-before+nO*5,c=cc[nR],alpha=0.2)
                    #ax5.plot(caTimes[mask] - tt[0], dFF820 + nO * 8, c=cc[nR],alpha=0.4)
                    nAdd820+=1
                ax5.plot(caTimes[mask][2:-2] - tt[0], dFFmean[2:-2]/nAdd + nO * 2,lw=2,c=cc[nO])#,c='%s' % (nO/8+0.1))
                #ax5.plot(sTimesCa[wheelMask]- tt[0], -linearSpeedCa[wheelMask] + nO*5, c='darkorchid')
                #ax5.plot(caTimes[mask][2:-2] - tt[0], dFF820mean[2:-2] / nAdd820 + nO * 5,c=cc[nO],alpha=0.2)
                # ax5.plot(caTimes[mask] - tt[0], dFF820 + nO * 8, c=cc[nR],alpha=0.4)
                nR += 1
            nO+=1
        #ax2.set_xlim(0,60)
        self.layoutOfPanel(ax5, xLabel='time (s)', yLabel='fluorescence ($\Delta$F/F)')
        # ax6 = plt.subplot(gssub3[1])
        # #for i in range(4):
        # refPaw = 0
        # idxRefSwings = swingStanceDict['swingP'][refPaw][1]
        # #indecisiveSteps = swingStanceDict['swingP'][refPaw][3]
        # recRefTimes = swingStanceDict['forFit'][refPaw][2]
        # #pdb.set_trace()
        # dt = 0.02
        # counts = np.zeros((8,int(1/dt)+1))
        # for n in range(len(idxRefSwings)-2):
        #     idxRefSwingStart = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n][0]]))
        #     refSwingStart = recRefTimes[idxRefSwings[n][0]]
        #     #idxRefSwingEnd = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n][1]]))
        #     idxRefStanceEnd = np.argmin(np.abs(cPawPos[refPaw][:, 0] - recRefTimes[idxRefSwings[n+1][0]]))
        #     refStandEnd = recRefTimes[idxRefSwings[n+1][0]]
        #     refStandEndNP1 = recRefTimes[idxRefSwings[n+2][0]]
        #     refDuration = refStandEnd-refSwingStart
        #     refDurationNP1 = refStandEndNP1-refStandEnd
        #     for i in range(4):
        #         idxSwings = swingStanceDict['swingP'][i][1]
        #         recTimes = swingStanceDict['forFit'][i][2]
        #         for m in range(len(idxSwings)):
        #             idxStart = np.argmin(np.abs(cPawPos[i][:,0] - recTimes[idxSwings[m][0]]))
        #             startTime = recTimes[idxSwings[m][0]]
        #             idxEnd = np.argmin(np.abs(cPawPos[i][:,0] - recTimes[idxSwings[m][1]]))
        #             endTime = recTimes[idxSwings[m][1]]
        #             if (idxStart >= idxRefSwingStart) and (idxStart<idxRefStanceEnd) and (idxEnd<idxRefStanceEnd):
        #                 iStart = int((startTime-refSwingStart)/(refDuration*dt))
        #                 iEnd = int((endTime-refSwingStart)/(refDuration*dt))
        #                 counts[2*i,iStart:iEnd]+=1
        #             elif (idxStart >= idxRefSwingStart) and (idxStart<idxRefStanceEnd) and (idxEnd>idxRefStanceEnd):
        #                 iStart = int((startTime-refSwingStart)/(refDuration*dt))
        #                 iEnd = int((endTime-refStandEnd)/(refDurationNP1*dt))
        #                 counts[2*i,iStart:]+=1
        #                 counts[2*i,:iEnd]+=1
        #
        # #ax6.imshow(counts,cmap='gist_yarg',interpolation='nearest',aspect='auto')
        # counts = counts/np.max(counts)
        # for i in range(4):
        #     ax6.plot(counts[i*2])
        # self.layoutOfPanel(ax6, xLabel='% '+self.pawID[refPaw] +' stride (norm)', yLabel='p(swing)')
        # ax6.set_title('Probability of swing (FL norm)', loc='center', fontsize=14)
        # #plt.yticks([0,2,4,6], self.pawID)
        # plt.xticks([0, 10, 20, 30, 40, 50], [0,0.2,0.4,0.6,0.8,1])

        #########

        ########################
        fname = 'fig_experiment-phd-project_v%s' % version
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')


    ##########################################################################################
    def ephysWalkingFig(self,figVersion, ephysWalkingData, exampleTrace,ephysSummaryAllAnimals):
        nMice = len(ephysSummaryAllAnimals)
        print('%s mice in total' % nMice)
        statsAll = []
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][2])
            cellsPerMouse = []
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                temp = np.zeros((7,nRecs))
                bTemp = []
                for j in range(nRecs):
                    #print(n,j,recs)
                    #pdb.set_trace()
                    workingDB = ephysSummaryAllAnimals[n][2][c][2][j][2]
                    temp[0,j] = workingDB['ss_firingRate'][0]
                    temp[1,j] = workingDB['ss_cv'][0]
                    temp[2,j] = workingDB['ss_BaselineFiringRate'][0]
                    temp[3,j] = workingDB['ss_BaselineCV'][0]
                    temp[4,j] = workingDB['ss_WalkingFiringRate'][0]
                    temp[5,j] = workingDB['ss_WalkingCV'][0]
                    #temp[6,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_firingRate'][0])
                    #temp[7,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_cv'][0])
                    #temp[8,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineFiringRate'][0])
                    #temp[9,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineCV'][0])
                    #temp[10,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingFiringRate'][0])
                    #temp[11,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingCV'][0])
                    temp[6,j] = workingDB['ss_avgSpikeParams'][0]
                    #temp[13,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_avgSpikeParams'][0])
                    #bTemp.append(workingDB['cs_exist'][0])
                cellsPerMouse.append(temp)
            #print(n,bTemp)
            #print(temp[13])
            #if all(bTemp):
            #        cellIdentity.append('PC')
            #    else:
            #        cellIdentity.append('MLI')
            statsAll.append([n,ephysSummaryAllAnimals[n][1],cellsPerMouse])

        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(3, 1, #width_ratios=[1,1.3])
                               height_ratios=[1, 1.3,1.7])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.3)
        plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.07)
        # plt.figtext(0.01, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.27, 0.96, 'B', clip_on=False, color='black', size=22)
        # plt.figtext(0.515, 0.96, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.805, 0.96, 'D', clip_on=False, color='black',  size=22)
        # plt.figtext(0.01, 0.69, 'E', clip_on=False, color='black',  size=22)
        # plt.figtext(0.725, 0.69, 'F', clip_on=False, color='black', size=22)
        # plt.figtext(0.845, 0.69, 'G', clip_on=False, color='black', size=22)
        # plt.figtext(0.01, 0.35, 'H', clip_on=False, color='black', size=22)
        # plt.figtext(0.24, 0.35, 'I', clip_on=False, color='black', size=22)
        # plt.figtext(0.485, 0.35, 'J', clip_on=False, color='black', size=22)
        # plt.figtext(0.72, 0.35, 'K', clip_on=False, color='black', size=22)
        ##########################
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], hspace=0.15, width_ratios=[0.9,0.9,1,0.5])
        # panel ##################
        ax0 = plt.subplot(gssub0[2])
        workingDB = ephysWalkingData[exampleTrace][3][2]
        #pdb.set_trace()
        #position = [0,-150,-250,-550]
        ax0.axhline(y=0,ls='--',c='gray',alpha=0.5)
        startStop = [5,13]
        tmask = (workingDB['ch_time'] > startStop[0])&(workingDB['ch_time'] <= startStop[1])
        ax0.plot(workingDB['ch_time'][tmask],17*(workingDB['ch_data_ss'][tmask]-np.min(workingDB['ch_data_ss'][tmask]))/(np.max(workingDB['ch_data_ss'][tmask])-np.min(workingDB['ch_data_ss'][tmask])),lw=0.5,c='0.5')
        ax0.plot(ephysWalkingData[exampleTrace][1][0],ephysWalkingData[exampleTrace][1][1],c='darkorchid')
        self.layoutOfPanel(ax0,xLabel='time (s)',yLabel='wheel speed (cm/s)')
        ax0.set_xlim(startStop[0],startStop[1])
        ax0.set_ylim(-3,18)

        # panel ##################
        ax1 = plt.subplot(gssub0[3])
        #ax1.axhline(y=0., ls=':', c='0.6')
        #linearSpeedSmooth = np.convolve(linearSpeed, np.ones(20)/20, mode='same')
        #linearSpeedSmooth1 = scipy.signal.medfilt(linearSpeed,kernel_size=9)
        spikesWF = workingDB['ss_wave']
        spikesTB = workingDB['ss_wave_span']
        spikeletWaveform = np.average(spikesWF, axis=0)
        for i in range(len(spikesWF)):
            ax1.plot(spikesTB[i]*1000,spikesWF[i],c='0.7',lw=0.2)
        ax1.plot(spikesTB[0]*1000,spikeletWaveform,lw=2,c='0.3')
        #ax1.plot(sTimes, -linearSpeedSmooth, c='orange')
        #ax1.plot(sTimes, -linearSpeedSmooth1, c='green')
        ax1.set_xlim(-1,2)
        ax1.yaxis.set_ticklabels([])
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax1,xLabel='time (ms)',yLabel='current (a.u.)')

        # panel ##################
        # gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.07, width_ratios=[6,2])
        # nRecs = len(ephysWalkingData)
        # pos = [0,2]
        # spikes = []
        # ax1 = plt.subplot(gssub1[0])
        # nPlot = 0
        # for i in range(nRecs):
        #     if i in [0,4]:
        #         #spikes.append(ephysWalkingData[i][2]['ss_spikeTimes'])
        #         spikes = ephysWalkingData[i][2]['ss_spikeTimes']
        #         ax1.eventplot(spikes, lineoffsets=pos[nPlot]+1, lw=0.1, colors='0.2', linelengths=0.1)
        #         nPlot+=1
        #
        # #ax0.set_title(str(ephysData[i][0]) + ' ' + str(listOfRecordings[i][2]) + ' recs:' + str(listOfRecordings[i][3]), fontsize=9)  # +' '+str(ephysData[i][1]),fontsize=9)
        # nPlot = 0
        # for i in range(nRecs):
        #     if i in [0,4]:
        #         instFR = ephysWalkingData[i][2]['ss_instFiringRate']
        #         #ax1.plot(ephysWalkingData[i][1][0],(nRecs/4-1-i)-0.4+ephysWalkingData[i][1][1]*0.03,lw=1,c='darkorchid')
        #         #ax1.plot(instFR[:,0],(nRecs/4-1-i)-0.4+instFR[:,1]*0.005,lw=1,c='0.1')
        #         ax1.plot(ephysWalkingData[i][1][0],pos[nPlot]+ephysWalkingData[i][1][1]*0.03,lw=1,c='darkorchid')
        #         ax1.plot(instFR[:,0],pos[nPlot]+instFR[:,1]*0.005,lw=1,c='0.1')
        #         nPlot+=1
        #
        # #ax0.eventplot(statsAll[i][2], lw=0.5, colors='black', linelengths=0.8)
        # self.layoutOfPanel(ax1, xLabel='time (s)', yLabel='trials')
        # ax1.set_xlim(3, 25)
        # ax1.set_ylim(-0.7,4.7)
        # majorLocator_x = MultipleLocator(5)
        # ax1.xaxis.set_major_locator(majorLocator_x)
        # #ax1.set_xticklabels([0,1,2,3,4],[5,4,3,2,1])
        # ax1.set_yticks([0,2])
        # ax1.set_yticklabels(['5','1'])
        # for key, spine in ax1.spines.items():
        #     if key == 'left':
        #         spine.set_visible(False)
        ######
        gssub11 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1], wspace=0.1,hspace=0.1, width_ratios=[7,1,1])
        #ax1 = plt.subplot(gssub1[1])
        # pdb.set_trace()
        nFig = 0
        for i in range(nRecs):
            if i in [0,4]:
                ax0 = plt.subplot(gssub11[3*nFig])
                ax0.axhline(y=0,ls='--',c='gray',alpha=0.5)
                spikes = ephysWalkingData[i][2]['ss_spikeTimes']
                ax0.eventplot(spikes, lineoffsets=0.95, lw=0.1, colors='0.2', linelengths=0.1)
                instFR = ephysWalkingData[i][2]['ss_instFiringRate']
                # ax1.plot(ephysWalkingData[i][1][0],(nRecs/4-1-i)-0.4+ephysWalkingData[i][1][1]*0.03,lw=1,c='darkorchid')
                # ax1.plot(instFR[:,0],(nRecs/4-1-i)-0.4+instFR[:,1]*0.005,lw=1,c='0.1')
                ax0.plot(ephysWalkingData[i][1][0], ephysWalkingData[i][1][1] * 0.03, lw=1, c='darkorchid')
                ax0.plot(instFR[:, 0], instFR[:, 1] * 0.005, lw=1, c='0.1')
                ax0.set_xlim(4, 22)
                ax0.set_ylim(-0.15,1.05)
                majorLocator_x = MultipleLocator(5)
                ax0.xaxis.set_major_locator(majorLocator_x)
                self.layoutOfPanel(ax0, xLabel='time (ms)',  xyInvisible=[(True if nFig==0 else False),False])
                #nPlot += 1
                ########
                ax1 = plt.subplot(gssub11[3*nFig+1])
                wfs = ephysWalkingData[i][2]['ss_spikeWaveForm']
                ax1.plot(wfs[0] * 1000, wfs[1] / np.max(wfs[1]),lw=1.5,c='0.3')
                if i<4:
                    self.layoutOfPanel(ax1, xyInvisible=[True,True])
                else:
                    self.layoutOfPanel(ax1, xLabel='time (ms)', xyInvisible=[False,True])
                #ax1.yaxis.set_ticklabels([])
                ax1.set_xlim(-1, 2)
                #######
                ax2 = plt.subplot(gssub11[3*nFig+2])
                #pdb.set_trace()
                ac = ephysWalkingData[i][3][2]['ss_xprob']
                actime = ephysWalkingData[i][3][2]['ss_xprob_span']
                ax2.plot(actime * 1000, ac,lw=1.5,c='0.3')
                if i<4:
                    self.layoutOfPanel(ax2, xyInvisible=[True,True])
                else:
                    self.layoutOfPanel(ax2, xLabel='time (ms)', xyInvisible=[False,True])
                nFig+=1
            #ax1.yaxis.set_ticklabels([])
            #ax2.set_xlim(-1, 2)
        #ax1.set_xlim(-1, 2)
        #self.layoutOfPanel(ax1, xLabel='time (ms)', yLabel='current (a.u.)', Leg=[1, 6])

        ########################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2],width_ratios=(1,1, 0.8), wspace=0.4)
        cmap = cm.get_cmap('tab20b')
        # gssubsub20 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0],width_ratios=(4, 0.7), height_ratios=(0.7, 4),wspace=0.05, hspace=0.05)
        # ax20m = plt.subplot(gssubsub20[2])
        # allFR = []
        # allCV = []
        # for n in range(nMice):
        #     nCells = len(statsAll[n][2])
        #     for c in range(nCells):
        #         markers, caps, bars = ax20m.errorbar(np.mean(statsAll[n][2][c][0]),np.mean(statsAll[n][2][c][1]),xerr=np.std(statsAll[n][2][c][0]),yerr=np.std(statsAll[n][2][c][1]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
        #         [bar.set_alpha(0.5) for bar in bars]
        #         allFR.append(np.mean(statsAll[n][2][c][0]))
        #         allCV.append(np.mean(statsAll[n][2][c][1]))
        # self.layoutOfPanel(ax20m, xLabel='firing rate (1/s)', yLabel='CV')
        # ax20m.set_ylim(0.4,1.8)
        # ibins = 20
        # ax20top = plt.subplot(gssubsub20[0],sharex=ax20m)
        # ax20top.hist(allFR, bins=ibins,histtype='stepfilled',color='0.4')
        # ax20top.axvline(x=np.percentile(allFR,25), ls=':', c='black')
        # ax20top.axvline(x=np.percentile(allFR, 75), ls=':', c='black')
        # ax20top.axvline(x=np.median(allFR),ls='--',c='black')
        # self.layoutOfPanel(ax20top, xyInvisible=[True,True])
        # ax20right = plt.subplot(gssubsub20[3],sharey=ax20m)
        # ax20right.hist(allCV, bins=ibins, orientation='horizontal',histtype='stepfilled',color='0.4')
        # ax20right.axhline(y=np.median(allCV),ls='--',c='black')
        # ax20right.axhline(y=np.percentile(allCV,25), ls=':', c='black')
        # ax20right.axhline(y=np.percentile(allCV, 75), ls=':', c='black')
        # self.layoutOfPanel(ax20right, xyInvisible=[True, True])

        ######
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        ax21m.axline((50, 50), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            nCells = len(statsAll[n][2])
            for c in range(nCells):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAll[n][2][c][2]),np.mean(statsAll[n][2][c][4]),xerr=np.std(statsAll[n][2][c][2]),yerr=np.std(statsAll[n][2][c][4]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice),label=((statsAll[n][1][-3:]+' (%s)' % nCells) if c==0 else None))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAll[n][2][c][2]))
                allWalkFR.append(np.mean(statsAll[n][2][c][4]))
        self.layoutOfPanel(ax21m, xLabel='baseline firing rate (1/s)', yLabel='walking firing rate (1/s)', Leg=[4, 8])
        ax21m.annotate('***',(0.7,0.9),xycoords='axes fraction',color='gray',size=15)
        ax21m.set_xlim(-5,139)
        ax21m.set_ylim(-5, 139)
        majorLocator = MultipleLocator(25)
        ax21m.yaxis.set_major_locator(majorLocator)
        ax21m.xaxis.set_major_locator(majorLocator)
        ax21top = plt.subplot(gssubsub21[0],sharex=ax21m)
        ibinsFR = np.linspace(np.min(allBLFR+allWalkFR),np.max(allBLFR+allWalkFR),21,endpoint=True)
        ax21top.hist(allBLFR, bins=ibinsFR,histtype='stepfilled',color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR,25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR),ls='--',c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True,True])
        ax21right = plt.subplot(gssubsub21[3],sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibinsFR, orientation='horizontal',histtype='stepfilled',color='0.4')
        ax21right.axhline(y=np.median(allWalkFR),ls='--',c='black')
        ax21right.axhline(y=np.percentile(allWalkFR,25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])
        print('paired t-test firing rate :', stats.ttest_rel(allBLFR, allWalkFR))
        print('baseline firing rate 25, 50, 75th percentile :', np.percentile(allBLFR, (25,50,75)))
        print('walking firing rate 25, 50, 75th percentile :', np.percentile(allWalkFR, (25, 50, 75)))
        ####
        gssubsub22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax22m = plt.subplot(gssubsub22[2])
        ax22m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBaselCV = []
        allWalkCV = []
        for n in range(nMice):
            nCells = len(statsAll[n][2])
            for c in range(nCells):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAll[n][2][c][3]),np.mean(statsAll[n][2][c][5]),xerr=np.std(statsAll[n][2][c][3]),yerr=np.std(statsAll[n][2][c][5]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAll[n][2][c][3]))
                allWalkCV.append(np.mean(statsAll[n][2][c][5]))
        self.layoutOfPanel(ax22m, xLabel='baseline CV', yLabel='walking CV')
        ax22m.set_xlim(0.35,1.9)
        ax22m.set_ylim(0.35, 1.9)
        majorLocator = MultipleLocator(0.5)
        ax22m.yaxis.set_major_locator(majorLocator)
        ax22m.xaxis.set_major_locator(majorLocator)
        ax22m.annotate('n.s.', (0.7, 0.9), xycoords='axes fraction',color='gray',size=15)
        ax22top = plt.subplot(gssubsub22[0],sharex=ax22m)
        ibinsCV = np.linspace(np.min(allBaselCV + allWalkCV), np.max(allBaselCV + allWalkCV), 21, endpoint=True)
        ax22top.hist(allBaselCV, bins=ibinsCV,histtype='stepfilled',color='0.4')
        ax22top.axvline(x=np.percentile(allBaselCV,25), ls=':', c='black')
        ax22top.axvline(x=np.percentile(allBaselCV, 75), ls=':', c='black')
        ax22top.axvline(x=np.median(allBaselCV),ls='--',c='black')
        self.layoutOfPanel(ax22top, xyInvisible=[True,True])
        ax22right = plt.subplot(gssubsub22[3],sharey=ax22m)
        ax22right.hist(allWalkCV, bins=ibinsCV, orientation='horizontal',histtype='stepfilled',color='0.4')
        ax22right.axhline(y=np.median(allWalkCV),ls='--',c='black')
        ax22right.axhline(y=np.percentile(allWalkCV,25), ls=':', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax22right, xyInvisible=[True, True])
        print('paired t-test CV:',stats.ttest_rel(allBaselCV, allWalkCV))
        print('baseline CV 25, 50, 75th percentile : ', np.percentile(allBaselCV,(25,50,75)))
        print('walking CV 25, 50, 75th percentile : ', np.percentile(allWalkCV, (25, 50, 75)))

        ####
        gssubsub23 = gridspec.GridSpecFromSubplotSpec(2, 2,  subplot_spec=gssub2[2], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax23m = plt.subplot(gssubsub23[2])
        #ax23m.axline((0, 0), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        #ax23m.axhline(y=0, ls=':', color='0.7')
        #ax23m.axvline(x=0, ls=':', color='0.7')
        entPR = []
        walkPR = []
        walkPRns = []
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][2])
            #cellsPerMouse = []
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                #temp = np.zeros((7,nRecs))
                #bTemp = []
                for j in range(nRecs):
                    workingDB = ephysSummaryAllAnimals[n][2][c][2][j][2]
                    # ephysDict['%s_pearsonR'  % labels[i]] = np.array([rBL,pBL,rWP,pWP])
                    #r1 = 1+0.1*(np.random.rand()-1)
                    #r2 = 2+0.1*(np.random.rand()-1)
                    # ephysDict['%s_pearsonR' % labels[i]] = np.array([rBL, pBL, rWP, pWP,pShuffle,np.mean(rBLshuffle)])
                    #ephysDict['%s_pearsonR' % labels[i]] = np.array([rBL, pBL, rWP, pWP, pBLShuffle, np.mean(rBLshuffle), pWPShuffle, np.mean(rWPshuffle)])
                    #ax23m.plot(workingDB['ss_pearsonR'][0], workingDB['ss_pearsonR'][2], 'o',ms=2,color=cmap((n+1)/nMice),alpha=(0.2 if (workingDB['ss_pearsonR'][1]>0.05 and workingDB['ss_pearsonR'][3]>0.05) else 1.0))
                    #ax23m.plot(workingDB['ss_pearsonR'][0], workingDB['ss_pearsonR'][2], 'o',ms=2,color=cmap((n+1)/nMice),alpha=(0.2 if (workingDB['ss_pearsonR'][4]>0.05 and workingDB['ss_pearsonR'][6]>0.05) else 1.0))
                    #print(workingDB['ss_pearsonR'])
                    #ax23m.plot(r1,workingDB['ss_pearsonR'][0],'o',color='C0',alpha=(0.2 if workingDB['ss_pearsonR'][1]>0.05 else 1.0))
                    #ax23m.plot(r2,workingDB['ss_pearsonR'][2],'o',color='C1',alpha=(0.2 if workingDB['ss_pearsonR'][3]>0.05 else 1.0))
                    #print((statsAll[n][1][-3:]+' (%s)' % nCells), len(workingDB['ss_pearsonR']))
                    if workingDB['ss_pearsonR'][4]<0.05: # and workingDB['ss_pearsonR'][3]<0.05:
                        entPR.append(workingDB['ss_pearsonR'][0])
                    if workingDB['ss_pearsonR'][6]<0.05:
                        walkPR.append(workingDB['ss_pearsonR'][2])
                    else:
                        walkPRns.append(workingDB['ss_pearsonR'][2])

        ibins = np.linspace(np.min(walkPR+walkPRns),np.max(walkPR+walkPRns),20)
        ax23m.hist(walkPR, bins=ibins,histtype='stepfilled',color='0.4')
        ax23m.hist(walkPRns, bins=ibins, histtype='stepfilled', color='0.6',alpha=0.5)
        self.layoutOfPanel(ax23m, xLabel='wheel speed - firing rate correlation',yLabel='occurrence')
        #ax23m.set_ylim(-0.015,0.015)
        #majorLocator_y = MultipleLocator(0.01)
        #ax23m.yaxis.set_major_locator(majorLocator_y)
        #ax23m.set_xlim(0.5,2.5)
        #ax23m.set_xticks([1,2])
        #ax23m.set_xticklabels(['entire\n rec.','walking\n  period'],ha='right',rotation = 45)
        #pdb.set_trace()
        # ax23top = plt.subplot(gssubsub23[0],sharex=ax23m)
        # ax23top.hist(entPR, bins=ibins,histtype='stepfilled',color='0.4')
        # ax23top.axvline(x=np.percentile(entPR,25), ls=':', c='black')
        # ax23top.axvline(x=np.percentile(entPR, 75), ls=':', c='black')
        # ax23top.axvline(x=np.median(entPR),ls='--',c='black')
        # self.layoutOfPanel(ax23top, xyInvisible=[True,True])
        #
        # ax23right = plt.subplot(gssubsub23[3], sharey=ax23m)
        # #ax23right.hist(entPR, bins=ibins, orientation='horizontal', histtype='step', color='C0')
        # ax23right.hist(walkPR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        # ax23right.axhline(y=np.median(walkPR),ls='--',c='black')
        # ax23right.axhline(y=np.percentile(walkPR,25), ls=':', c='black')
        # ax23right.axhline(y=np.percentile(walkPR, 75), ls=':', c='black')
        # self.layoutOfPanel(ax23right, xyInvisible=[True, True])

        print('%s pearson rs are significat and %s are not, total : %s', (len(walkPR),len(walkPRns),len(walkPR)+len(walkPRns)))
        #print(stats.ttest_rel(entPR,walkPR))

        #     nCells = len(statsAll[n][2])
        #     for c in range(nCells):
        #         markers, caps, bars = ax2.errorbar(np.mean(statsAll[n][2][c][0]),np.mean(statsAll[n][2][c][6][0]*1000),xerr=np.std(statsAll[n][2][c][0]),yerr=np.std(statsAll[n][2][c][6][0]*1000),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
        #         [bar.set_alpha(0.5) for bar in bars]
        # self.layoutOfPanel(ax2, xLabel='baseline CV', yLabel='walking CV')


        ########################
        fname = 'fig_ephys-walking_v%s' % figVersion
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        # plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')


    ##########################################################################################
    def ephysSummaryFig(self,figVersion,ephysSummaryAllAnimals):
        nMice = len(ephysSummaryAllAnimals)
        print('%s mice in total' % nMice)
        statsAllMLI = []
        recsTotMLI = []
        totMLI = 0
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][2])
            totMLI += nCells
            cellsPerMouse = []
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                temp = np.zeros((7,nRecs))
                bTemp = []
                recsTotMLI.append(nRecs)
                for j in range(nRecs):
                    #print(n,j,recs)
                    #pdb.set_trace()
                    workingDB = ephysSummaryAllAnimals[n][2][c][2][j][2]
                    temp[0,j] = workingDB['ss_firingRate'][0]
                    temp[1,j] = workingDB['ss_cv'][0]
                    temp[2,j] = workingDB['ss_BaselineFiringRate'][0]
                    temp[3,j] = workingDB['ss_BaselineCV'][0]
                    temp[4,j] = workingDB['ss_WalkingFiringRate'][0]
                    temp[5,j] = workingDB['ss_WalkingCV'][0]
                    #temp[6,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_firingRate'][0])
                    #temp[7,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_cv'][0])
                    #temp[8,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineFiringRate'][0])
                    #temp[9,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineCV'][0])
                    #temp[10,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingFiringRate'][0])
                    #temp[11,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingCV'][0])
                    temp[6,j] = workingDB['ss_avgSpikeParams'][0]
                    #temp[13,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_avgSpikeParams'][0])
                    #bTemp.append(workingDB['cs_exist'][0])
                cellsPerMouse.append(temp)
            #print(n,bTemp)
            #print(temp[13])
            #if all(bTemp):
            #        cellIdentity.append('PC')
            #    else:
            #        cellIdentity.append('MLI')
            statsAllMLI.append([n,ephysSummaryAllAnimals[n][1],cellsPerMouse])

        statsAllPC = []
        recsTotPC = []
        totPC=0
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][3])
            cellsPerMouse = []
            totPC+=nCells
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][3][c][2])
                temp = np.zeros((14,nRecs))
                bTemp = []
                recsTotPC.append(nRecs)
                for j in range(nRecs):
                    print(n,c,j)
                    #pdb.set_trace()
                    workingDB = ephysSummaryAllAnimals[n][3][c][2][j][2]
                    temp[0,j] = workingDB['ss_firingRate'][0]
                    temp[1,j] = workingDB['ss_cv'][0]
                    temp[2,j] = workingDB['ss_BaselineFiringRate'][0]
                    temp[3,j] = workingDB['ss_BaselineCV'][0]
                    temp[4,j] = workingDB['ss_WalkingFiringRate'][0]
                    temp[5,j] = workingDB['ss_WalkingCV'][0]
                    temp[6,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_firingRate'][0])
                    temp[7,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_cv'][0])
                    temp[8,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineFiringRate'][0])
                    temp[9,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_BaselineCV'][0])
                    temp[10,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingFiringRate'][0])
                    temp[11,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_WalkingCV'][0])
                    temp[12,j] = workingDB['ss_avgSpikeParams'][0]
                    temp[13,j] = (0 if not workingDB['cs_exist'][0] else workingDB['cs_avgSpikeParams'][0])
                    bTemp.append(workingDB['cs_exist'][0])
                cellsPerMouse.append(temp)
            #print(n,bTemp)
            #print(temp[13])
            #if all(bTemp):
            #        cellIdentity.append('PC')
            #    else:
            #        cellIdentity.append('MLI')
            statsAllPC.append([n,ephysSummaryAllAnimals[n][1],cellsPerMouse])



        # figure #################################
        fig_width = 17  # width in inches
        fig_height = 18  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        #rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(4, 1, #width_ratios=[1,1.3])
                               #height_ratios=[1, 1.5,1.5])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.35)
        plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.07)
        plt.figtext(0.01, 0.96, 'A', clip_on=False, color='black',  size=22)
        plt.figtext(0.27, 0.96, 'B', clip_on=False, color='black', size=22)
        plt.figtext(0.515, 0.96, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.805, 0.96, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.01, 0.69, 'E', clip_on=False, color='black',  size=22)
        plt.figtext(0.85, 0.69, 'F', clip_on=False, color='black', size=22)
        plt.figtext(0.01, 0.35, 'G', clip_on=False, color='black', size=22)
        plt.figtext(0.24, 0.35, 'H', clip_on=False, color='black', size=22)
        plt.figtext(0.485, 0.35, 'I', clip_on=False, color='black', size=22)
        plt.figtext(0.72, 0.35, 'J', clip_on=False, color='black', size=22)
        ##########################

        ########################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.3)
        cmap = cm.get_cmap('brg')
        gssubsub20 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0],width_ratios=(4, 0.7), height_ratios=(0.7, 4),wspace=0.05, hspace=0.05)
        ax20m = plt.subplot(gssubsub20[2])
        allFR = []
        allCV = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            #nCellsPC = len(statsAllPC[n][2])
            print()
            for c in range(nCellsMLI):
                markers, caps, bars = ax20m.errorbar(np.mean(statsAllMLI[n][2][c][0]),np.mean(statsAllMLI[n][2][c][1]),xerr=np.std(statsAllMLI[n][2][c][0]),yerr=np.std(statsAllMLI[n][2][c][1]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allFR.append(np.mean(statsAllMLI[n][2][c][0]))
                allCV.append(np.mean(statsAllMLI[n][2][c][1]))
            # for c in range(nCellsPC):
            #     markers, caps, bars = ax20m.errorbar(np.mean(statsAllPC[n][2][c][0]),np.mean(statsAllPC[n][2][c][1]),xerr=np.std(statsAllPC[n][2][c][0]),yerr=np.std(statsAllPC[n][2][c][1]),fmt='^',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allFR.append(np.mean(statsAllPC[n][2][c][0]))
            #     allCV.append(np.mean(statsAllPC[n][2][c][1]))
        self.layoutOfPanel(ax20m, xLabel='simple spike firing rate (1/s)', yLabel='simple spike CV')
        #ax20m.set_ylim(0.4,1.8)
        ibins = 20
        ax20top = plt.subplot(gssubsub20[0],sharex=ax20m)
        ax20top.set_title('Simple spikes of MLI only')
        ax20top.hist(allFR, bins=ibins,histtype='stepfilled',color='0.4')
        ax20top.axvline(x=np.percentile(allFR,25), ls=':', c='black')
        ax20top.axvline(x=np.percentile(allFR, 75), ls=':', c='black')
        ax20top.axvline(x=np.median(allFR),ls='--',c='black')
        self.layoutOfPanel(ax20top, xyInvisible=[True,True])
        ax20right = plt.subplot(gssubsub20[3],sharey=ax20m)
        ax20right.hist(allCV, bins=ibins, orientation='horizontal',histtype='stepfilled',color='0.4')
        ax20right.axhline(y=np.median(allCV),ls='--',c='black')
        ax20right.axhline(y=np.percentile(allCV,25), ls=':', c='black')
        ax20right.axhline(y=np.percentile(allCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax20right, xyInvisible=[True, True])

        ######
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        ax21m.axline((50, 50), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            #nCellsPC = len(statsAllPC[n][2])
            for c in range(nCellsMLI):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllMLI[n][2][c][2]),np.mean(statsAllMLI[n][2][c][4]),xerr=np.std(statsAllMLI[n][2][c][2]),yerr=np.std(statsAllMLI[n][2][c][4]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice),label=((statsAllMLI[n][1][-3:]+' (MLI %s)' % (nCellsMLI)) if c==0 else None))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllMLI[n][2][c][2]))
                allWalkFR.append(np.mean(statsAllMLI[n][2][c][4]))
            # for c in range(nCellsPC):
            #     markers, caps, bars = ax21m.errorbar(np.mean(statsAllPC[n][2][c][2]),np.mean(statsAllPC[n][2][c][4]),xerr=np.std(statsAllPC[n][2][c][2]),yerr=np.std(statsAllPC[n][2][c][4]),fmt='^',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allFR.append(np.mean(statsAllPC[n][2][c][2]))
            #     allCV.append(np.mean(statsAllPC[n][2][c][4]))
        self.layoutOfPanel(ax21m, xLabel='simple spike baseline firing rate (1/s)', yLabel='simple spike walking firing rate (1/s)', Leg=[4, 5])
        ax21top = plt.subplot(gssubsub21[0],sharex=ax21m)
        ax21top.hist(allBLFR, bins=ibins,histtype='stepfilled',color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR,25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR),ls='--',c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True,True])
        ax21right = plt.subplot(gssubsub21[3],sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibins, orientation='horizontal',histtype='stepfilled',color='0.4')
        ax21right.axhline(y=np.median(allWalkFR),ls='--',c='black')
        ax21right.axhline(y=np.percentile(allWalkFR,25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])

        print('paired t-test firing rate MLI baseline - walking :', stats.ttest_rel(allBLFR, allWalkFR))
        ####
        gssubsub22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[2], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax22m = plt.subplot(gssubsub22[2])
        ax22m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBaselCV = []
        allWalkCV = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            #nCellsPC = len(statsAllPC[n][2])
            for c in range(nCellsMLI):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAllMLI[n][2][c][3]),np.mean(statsAllMLI[n][2][c][5]),xerr=np.std(statsAllMLI[n][2][c][3]),yerr=np.std(statsAllMLI[n][2][c][5]),fmt='o',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAllMLI[n][2][c][3]))
                allWalkCV.append(np.mean(statsAllMLI[n][2][c][5]))
            # for c in range(nCellsPC):
            #     markers, caps, bars = ax22m.errorbar(np.mean(statsAllPC[n][2][c][3]),np.mean(statsAllPC[n][2][c][5]),xerr=np.std(statsAllPC[n][2][c][3]),yerr=np.std(statsAllPC[n][2][c][5]),fmt='^',ms=3,elinewidth=1,color=cmap((n+1)/nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allFR.append(np.mean(statsAllPC[n][2][c][3]))
            #     allCV.append(np.mean(statsAllPC[n][2][c][5]))
        self.layoutOfPanel(ax22m, xLabel='simple spike baseline CV', yLabel='simple spike walking CV')
        ax22top = plt.subplot(gssubsub22[0],sharex=ax22m)
        ax22top.hist(allBaselCV, bins=ibins,histtype='stepfilled',color='0.4')
        ax22top.axvline(x=np.percentile(allBaselCV,25), ls=':', c='black')
        ax22top.axvline(x=np.percentile(allBaselCV, 75), ls=':', c='black')
        ax22top.axvline(x=np.median(allBaselCV),ls='--',c='black')
        self.layoutOfPanel(ax22top, xyInvisible=[True,True])
        ax22right = plt.subplot(gssubsub22[3],sharey=ax22m)
        ax22right.hist(allWalkCV, bins=ibins, orientation='horizontal',histtype='stepfilled',color='0.4')
        ax22right.axhline(y=np.median(allWalkCV),ls='--',c='black')
        ax22right.axhline(y=np.percentile(allWalkCV,25), ls=':', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax22right, xyInvisible=[True, True])

        ####
        gssubsub23 = gridspec.GridSpecFromSubplotSpec(2, 2,  subplot_spec=gssub2[3], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax23m = plt.subplot(gssubsub23[2])
        #ax23m.axline((0, 0), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        ax23m.axhline(y=0, ls=':', color='0.7')
        ax23m.axvline(x=0, ls=':', color='0.7')
        entPR = []
        walkPR = []
        walkPRs = []
        walkPRns = []
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][2])
            #cellsPerMouse = []
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                #temp = np.zeros((7,nRecs))
                #bTemp = []
                for j in range(nRecs):
                    workingDB = ephysSummaryAllAnimals[n][2][c][2][j][2]
                    # ephysDict['%s_pearsonR'  % labels[i]] = np.array([rBL,pBL,rWP,pWP])
                    #r1 = 1+0.1*(np.random.rand()-1)
                    #r2 = 2+0.1*(np.random.rand()-1)
                    #ax23m.plot(workingDB['ss_pearsonR'][0], workingDB['ss_pearsonR'][2], 'o',ms=2,color=cmap((n+1)/nMice),alpha=(0.2 if (workingDB['ss_pearsonR'][1]>0.05 and workingDB['ss_pearsonR'][3]>0.05) else 1.0))
                    #print(workingDB['ss_pearsonR'])
                    #ax23m.plot(r1,workingDB['ss_pearsonR'][0],'o',color='C0',alpha=(0.2 if workingDB['ss_pearsonR'][1]>0.05 else 1.0))
                    #ax23m.plot(r2,workingDB['ss_pearsonR'][2],'o',color='C1',alpha=(0.2 if workingDB['ss_pearsonR'][3]>0.05 else 1.0))
                    if workingDB['ss_pearsonR'][1]<0.05 and workingDB['ss_pearsonR'][3]<0.05:
                        entPR.append(workingDB['ss_pearsonR'][0])
                        walkPR.append(workingDB['ss_pearsonR'][2])
                    #if workingDB['ss_pearsonR'][6]<0.05:
                    #    walkPR.append(workingDB['ss_pearsonR'][2])
                    #else:
                    #    walkPRns.append(workingDB['ss_pearsonR'][2])
                    if workingDB['ss_pearsonR'][6]<0.05:
                        walkPRs.append(workingDB['ss_pearsonR'][2])
                    else:
                        walkPRns.append(workingDB['ss_pearsonR'][2])

        ibins = np.linspace(np.min(walkPRs + walkPRns), np.max(walkPRs + walkPRns), 20)
        ax23m.hist(walkPRs, bins=ibins, histtype='stepfilled', color='0.4')
        ax23m.hist(walkPRns, bins=ibins, histtype='stepfilled', color='0.6', alpha=0.5)
        self.layoutOfPanel(ax23m, xLabel='wheel speed - simple spike rate correlation', yLabel='occurrence')
        #ax23m.set_xlim(0.5,2.5)
        #ax23m.set_xticks([1,2])
        #ax23m.set_xticklabels(['entire\n rec.','walking\n  period'],ha='right',rotation = 45)
        #pdb.set_trace()
        ax23top = plt.subplot(gssubsub23[0])#,sharex=ax23m)
        ax23top.hist(entPR, bins=ibins,histtype='stepfilled',color='0.4')
        ax23top.axvline(x=np.percentile(entPR,25), ls=':', c='black')
        ax23top.axvline(x=np.percentile(entPR, 75), ls=':', c='black')
        ax23top.axvline(x=np.median(entPR),ls='--',c='black')
        self.layoutOfPanel(ax23top, xyInvisible=[True,True])

        ax23right = plt.subplot(gssubsub23[3])#, sharey=ax23m)
        #ax23right.hist(entPR, bins=ibins, orientation='horizontal', histtype='step', color='C0')
        ax23right.hist(walkPR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax23right.axhline(y=np.median(walkPR),ls='--',c='black')
        ax23right.axhline(y=np.percentile(walkPR,25), ls=':', c='black')
        ax23right.axhline(y=np.percentile(walkPR, 75), ls=':', c='black')
        self.layoutOfPanel(ax23right, xyInvisible=[True, True])


        # PC only ###############################################################################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.3)
        cmap = cm.get_cmap('brg')
        gssubsub20 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax20m = plt.subplot(gssubsub20[2])
        allFR = []
        allCV = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            print()
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax20m.errorbar(np.mean(statsAllMLI[n][2][c][0]), np.mean(statsAllMLI[n][2][c][1]), xerr=np.std(statsAllMLI[n][2][c][0]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][1]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allFR.append(np.mean(statsAllMLI[n][2][c][0]))
            #     allCV.append(np.mean(statsAllMLI[n][2][c][1]))
            for c in range(nCellsPC):
                markers, caps, bars = ax20m.errorbar(np.mean(statsAllPC[n][2][c][0]), np.mean(statsAllPC[n][2][c][1]), xerr=np.std(statsAllPC[n][2][c][0]),
                                                     yerr=np.std(statsAllPC[n][2][c][1]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allFR.append(np.mean(statsAllPC[n][2][c][0]))
                allCV.append(np.mean(statsAllPC[n][2][c][1]))
        self.layoutOfPanel(ax20m, xLabel='simple spike firing rate (1/s)', yLabel='simple spike CV')
        #ax20m.set_ylim(0.4, 1.8)
        ibins = 20
        ax20top = plt.subplot(gssubsub20[0], sharex=ax20m)
        ax20top.set_title('Simple spikes of PC only')
        ax20top.hist(allFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax20top.axvline(x=np.percentile(allFR, 25), ls=':', c='black')
        ax20top.axvline(x=np.percentile(allFR, 75), ls=':', c='black')
        ax20top.axvline(x=np.median(allFR), ls='--', c='black')
        self.layoutOfPanel(ax20top, xyInvisible=[True, True])
        ax20right = plt.subplot(gssubsub20[3], sharey=ax20m)
        ax20right.hist(allCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax20right.axhline(y=np.median(allCV), ls='--', c='black')
        ax20right.axhline(y=np.percentile(allCV, 25), ls=':', c='black')
        ax20right.axhline(y=np.percentile(allCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax20right, xyInvisible=[True, True])

        ######
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        ax21m.axline((50, 50), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax21m.errorbar(np.mean(statsAllMLI[n][2][c][2]), np.mean(statsAllMLI[n][2][c][4]), xerr=np.std(statsAllMLI[n][2][c][2]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][4]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice),
            #                                          label=((statsAllMLI[n][1][-3:] + ' (MLI %s,PC %s)' % (nCellsMLI, nCellsPC)) if c == 0 else None))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allBLFR.append(np.mean(statsAllMLI[n][2][c][2]))
            #     allWalkFR.append(np.mean(statsAllMLI[n][2][c][4]))
            for c in range(nCellsPC):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllPC[n][2][c][2]), np.mean(statsAllPC[n][2][c][4]), xerr=np.std(statsAllPC[n][2][c][2]),
                                                     yerr=np.std(statsAllPC[n][2][c][4]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice),label=((statsAllPC[n][1][-3:]+' (PC %s)' % (nCellsPC)) if c==0 else None))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllPC[n][2][c][2]))
                allWalkFR.append(np.mean(statsAllPC[n][2][c][4]))
        self.layoutOfPanel(ax21m, xLabel='simple spike baseline firing rate (1/s)', yLabel='simple spike walking firing rate (1/s)', Leg=[4, 5])
        ax21top = plt.subplot(gssubsub21[0], sharex=ax21m)
        ax21top.hist(allBLFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR, 25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR), ls='--', c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True, True])
        ax21right = plt.subplot(gssubsub21[3], sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax21right.axhline(y=np.median(allWalkFR), ls='--', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])

        print('paired t-test firing rate PC baseline - walking :', stats.ttest_rel(allBLFR, allWalkFR))
        print('PC baseline simple spike [25,50,75]th percentile :', np.percentile(allBLFR, 25),np.percentile(allBLFR, 50),np.percentile(allBLFR, 75))
        print('PC  walking simple spike [25,50,75]th percentile :', np.percentile(allWalkFR, 25), np.percentile(allWalkFR, 50), np.percentile(allWalkFR, 75))
        ####
        gssubsub22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[2], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax22m = plt.subplot(gssubsub22[2])
        ax22m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBaselCV = []
        allWalkCV = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax22m.errorbar(np.mean(statsAllMLI[n][2][c][3]), np.mean(statsAllMLI[n][2][c][5]), xerr=np.std(statsAllMLI[n][2][c][3]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][5]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allBaselCV.append(np.mean(statsAllMLI[n][2][c][3]))
            #     allWalkCV.append(np.mean(statsAllMLI[n][2][c][5]))
            for c in range(nCellsPC):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAllPC[n][2][c][3]), np.mean(statsAllPC[n][2][c][5]), xerr=np.std(statsAllPC[n][2][c][3]),
                                                     yerr=np.std(statsAllPC[n][2][c][5]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAllPC[n][2][c][3]))
                allWalkCV.append(np.mean(statsAllPC[n][2][c][5]))
        self.layoutOfPanel(ax22m, xLabel='simple spike baseline CV', yLabel='simple spike walking CV')
        ax22top = plt.subplot(gssubsub22[0], sharex=ax22m)
        ax22top.hist(allBaselCV, bins=ibins, histtype='stepfilled', color='0.4')
        ax22top.axvline(x=np.percentile(allBaselCV, 25), ls=':', c='black')
        ax22top.axvline(x=np.percentile(allBaselCV, 75), ls=':', c='black')
        ax22top.axvline(x=np.median(allBaselCV), ls='--', c='black')
        self.layoutOfPanel(ax22top, xyInvisible=[True, True])
        ax22right = plt.subplot(gssubsub22[3], sharey=ax22m)
        ax22right.hist(allWalkCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax22right.axhline(y=np.median(allWalkCV), ls='--', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 25), ls=':', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax22right, xyInvisible=[True, True])

        ####
        gssubsub23 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[3], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax23m = plt.subplot(gssubsub23[2])
        #ax23m.axline((0, 0), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        ax23m.axhline(y=0, ls=':', color='0.7')
        ax23m.axvline(x=0, ls=':', color='0.7')
        entPR = []
        walkPR = []
        walkPRs = []
        walkPRns = []
        for n in range(nMice):
            nCells = len(ephysSummaryAllAnimals[n][3])
            # cellsPerMouse = []
            for c in range(nCells):
                nRecs = len(ephysSummaryAllAnimals[n][3][c][2])
                # temp = np.zeros((7,nRecs))
                # bTemp = []
                for j in range(nRecs):
                    workingDB = ephysSummaryAllAnimals[n][3][c][2][j][2]
                    # ephysDict['%s_pearsonR'  % labels[i]] = np.array([rBL,pBL,rWP,pWP])
                    # r1 = 1+0.1*(np.random.rand()-1)
                    # r2 = 2+0.1*(np.random.rand()-1)
                    #ax23m.plot(workingDB['ss_pearsonR'][0], workingDB['ss_pearsonR'][2], 'o', ms=2, color=cmap((n + 1) / nMice),
                    #           alpha=(0.2 if (workingDB['ss_pearsonR'][1] > 0.05 and workingDB['ss_pearsonR'][3] > 0.05) else 1.0))
                    # print(workingDB['ss_pearsonR'])
                    # ax23m.plot(r1,workingDB['ss_pearsonR'][0],'o',color='C0',alpha=(0.2 if workingDB['ss_pearsonR'][1]>0.05 else 1.0))
                    # ax23m.plot(r2,workingDB['ss_pearsonR'][2],'o',color='C1',alpha=(0.2 if workingDB['ss_pearsonR'][3]>0.05 else 1.0))
                    #print(workingDB.keys())
                    #print(workingDB['ss_pearsonR'])
                    #pdb.set_trace()
                    if workingDB['ss_pearsonR'][1] < 0.05 and workingDB['ss_pearsonR'][3] < 0.05:
                        entPR.append(workingDB['ss_pearsonR'][0])
                        # if workingDB['ss_pearsonR'][3]>0.05:
                    walkPR.append(workingDB['ss_pearsonR'][2])
                    if workingDB['ss_pearsonR'][6]<0.05:
                        walkPRs.append(workingDB['ss_pearsonR'][2])
                    else:
                        walkPRns.append(workingDB['ss_pearsonR'][2])
        print('significant vs non-significant : ',len(walkPRs),len(walkPRns))
        ibins = np.linspace(np.min(walkPRs + walkPRns), np.max(walkPRs + walkPRns), 20)
        ax23m.hist(walkPRs, bins=ibins, histtype='stepfilled', color='0.4')
        ax23m.hist(walkPRns, bins=ibins, histtype='stepfilled', color='0.6', alpha=0.5)
        self.layoutOfPanel(ax23m, xLabel='wheel speed - simple spike rate correlation', yLabel='occurrence')

        #ax23m.hist(walkPRs, bins=ibins, histtype='stepfilled', color='0.4')
        #ax23m.hist(walkPRns, bins=ibins, histtype='stepfilled', color='0.8')
        #self.layoutOfPanel(ax23m, xLabel='wheel speed - simple spike rate correlation',yLabel='occurrence')
        # ax23m.set_xlim(0.5,2.5)
        # ax23m.set_xticks([1,2])
        # ax23m.set_xticklabels(['entire\n rec.','walking\n  period'],ha='right',rotation = 45)
        # pdb.set_trace()
        ax23top = plt.subplot(gssubsub23[0])#, sharex=ax23m)
        ax23top.hist(entPR, bins=ibins, histtype='stepfilled', color='0.4')
        ax23top.axvline(x=np.percentile(entPR, 25), ls=':', c='black')
        ax23top.axvline(x=np.percentile(entPR, 75), ls=':', c='black')
        ax23top.axvline(x=np.median(entPR), ls='--', c='black')
        self.layoutOfPanel(ax23top, xyInvisible=[True, True])

        ax23right = plt.subplot(gssubsub23[3])#, sharey=ax23m)
        # ax23right.hist(entPR, bins=ibins, orientation='horizontal', histtype='step', color='C0')
        ax23right.hist(walkPR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax23right.axhline(y=np.median(walkPR), ls='--', c='black')
        ax23right.axhline(y=np.percentile(walkPR, 25), ls=':', c='black')
        ax23right.axhline(y=np.percentile(walkPR, 75), ls=':', c='black')
        self.layoutOfPanel(ax23right, xyInvisible=[True, True])

        # both MLI and PC ##########################################################################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], wspace=0.3)
        cmap = cm.get_cmap('brg')
        gssubsub20 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax20m = plt.subplot(gssubsub20[2])
        allFR = []
        allCV = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            print()
            for c in range(nCellsMLI):
                markers, caps, bars = ax20m.errorbar(np.mean(statsAllMLI[n][2][c][0]), np.mean(statsAllMLI[n][2][c][1]), xerr=np.std(statsAllMLI[n][2][c][0]),
                                                     yerr=np.std(statsAllMLI[n][2][c][1]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allFR.append(np.mean(statsAllMLI[n][2][c][0]))
                allCV.append(np.mean(statsAllMLI[n][2][c][1]))
            for c in range(nCellsPC):
                markers, caps, bars = ax20m.errorbar(np.mean(statsAllPC[n][2][c][0]), np.mean(statsAllPC[n][2][c][1]), xerr=np.std(statsAllPC[n][2][c][0]),
                                                     yerr=np.std(statsAllPC[n][2][c][1]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allFR.append(np.mean(statsAllPC[n][2][c][0]))
                allCV.append(np.mean(statsAllPC[n][2][c][1]))
        self.layoutOfPanel(ax20m, xLabel='simple spike firing rate (1/s)', yLabel='simple spike CV')
        #ax20m.set_ylim(0.4, 1.8)
        ibins = 20
        ax20top = plt.subplot(gssubsub20[0], sharex=ax20m)
        ax20top.set_title('Simple spikes of MLI and PC')
        ax20top.hist(allFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax20top.axvline(x=np.percentile(allFR, 25), ls=':', c='black')
        ax20top.axvline(x=np.percentile(allFR, 75), ls=':', c='black')
        ax20top.axvline(x=np.median(allFR), ls='--', c='black')
        self.layoutOfPanel(ax20top, xyInvisible=[True, True])
        ax20right = plt.subplot(gssubsub20[3], sharey=ax20m)
        ax20right.hist(allCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax20right.axhline(y=np.median(allCV), ls='--', c='black')
        ax20right.axhline(y=np.percentile(allCV, 25), ls=':', c='black')
        ax20right.axhline(y=np.percentile(allCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax20right, xyInvisible=[True, True])

        ######
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        ax21m.axline((50, 50), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            for c in range(nCellsMLI):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllMLI[n][2][c][2]), np.mean(statsAllMLI[n][2][c][4]), xerr=np.std(statsAllMLI[n][2][c][2]),
                                                     yerr=np.std(statsAllMLI[n][2][c][4]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice),
                                                     label=((statsAllMLI[n][1][-3:] + ' (MLI %s,PC %s)' % (nCellsMLI, nCellsPC)) if c == 0 else None))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllMLI[n][2][c][2]))
                allWalkFR.append(np.mean(statsAllMLI[n][2][c][4]))
            for c in range(nCellsPC):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllPC[n][2][c][2]), np.mean(statsAllPC[n][2][c][4]), xerr=np.std(statsAllPC[n][2][c][2]),
                                                     yerr=np.std(statsAllPC[n][2][c][4]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllPC[n][2][c][2]))
                allWalkFR.append(np.mean(statsAllPC[n][2][c][4]))
        self.layoutOfPanel(ax21m, xLabel='simple spike baseline firing rate (1/s)', yLabel='simple spike walking firing rate (1/s)', Leg=[4, 5])
        ax21top = plt.subplot(gssubsub21[0], sharex=ax21m)
        ax21top.hist(allBLFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR, 25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR), ls='--', c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True, True])
        ax21right = plt.subplot(gssubsub21[3], sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax21right.axhline(y=np.median(allWalkFR), ls='--', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])

        ####
        gssubsub22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[2], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax22m = plt.subplot(gssubsub22[2])
        ax22m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBaselCV = []
        allWalkCV = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            for c in range(nCellsMLI):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAllMLI[n][2][c][3]), np.mean(statsAllMLI[n][2][c][5]), xerr=np.std(statsAllMLI[n][2][c][3]),
                                                     yerr=np.std(statsAllMLI[n][2][c][5]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAllMLI[n][2][c][3]))
                allWalkCV.append(np.mean(statsAllMLI[n][2][c][5]))
            for c in range(nCellsPC):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAllPC[n][2][c][3]), np.mean(statsAllPC[n][2][c][5]), xerr=np.std(statsAllPC[n][2][c][3]),
                                                     yerr=np.std(statsAllPC[n][2][c][5]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAllPC[n][2][c][3]))
                allWalkCV.append(np.mean(statsAllPC[n][2][c][5]))
        self.layoutOfPanel(ax22m, xLabel='simple spike baseline CV', yLabel='simple spike walking CV')
        ax22top = plt.subplot(gssubsub22[0], sharex=ax22m)
        ax22top.hist(allBaselCV, bins=ibins, histtype='stepfilled', color='0.4')
        ax22top.axvline(x=np.percentile(allBaselCV, 25), ls=':', c='black')
        ax22top.axvline(x=np.percentile(allBaselCV, 75), ls=':', c='black')
        ax22top.axvline(x=np.median(allBaselCV), ls='--', c='black')
        self.layoutOfPanel(ax22top, xyInvisible=[True, True])
        ax22right = plt.subplot(gssubsub22[3], sharey=ax22m)
        ax22right.hist(allWalkCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax22right.axhline(y=np.median(allWalkCV), ls='--', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 25), ls=':', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax22right, xyInvisible=[True, True])

        print('paired t-test simple spike CV PC baseline - walking :', stats.ttest_rel(allBaselCV, allWalkCV))
        ####
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[3], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        #ax21m.axline((50, 50), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            for c in range(nCellsMLI):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllMLI[n][2][c][0]), np.mean(statsAllMLI[n][2][c][6]*1000.), xerr=np.std(statsAllMLI[n][2][c][0]),
                                                     yerr=np.std(statsAllMLI[n][2][c][6]*1000.), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllMLI[n][2][c][0]))
                allWalkFR.append(np.mean(statsAllMLI[n][2][c][6]*1000.))
            for c in range(nCellsPC):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllPC[n][2][c][0]), np.mean(statsAllPC[n][2][c][12]*1000.), xerr=np.std(statsAllPC[n][2][c][0]),
                                                     yerr=np.std(statsAllPC[n][2][c][12]*1000.), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllPC[n][2][c][0]))
                allWalkFR.append(np.mean(statsAllPC[n][2][c][12]*1000.))
        self.layoutOfPanel(ax21m, xLabel='simple spike firing rate (1/s)', yLabel='simple spike trough-to-peak (ms)', Leg=[4, 5])
        ax21top = plt.subplot(gssubsub21[0], sharex=ax21m)
        ax21top.hist(allBLFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR, 25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR), ls='--', c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True, True])
        ax21right = plt.subplot(gssubsub21[3], sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax21right.axhline(y=np.median(allWalkFR), ls='--', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])

        # CS only ###############################################################################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3], wspace=0.3)
        cmap = cm.get_cmap('brg')
        gssubsub20 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[0], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax20m = plt.subplot(gssubsub20[2])
        allFR = []
        allCV = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            print()
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax20m.errorbar(np.mean(statsAllMLI[n][2][c][0]), np.mean(statsAllMLI[n][2][c][1]), xerr=np.std(statsAllMLI[n][2][c][0]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][1]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allFR.append(np.mean(statsAllMLI[n][2][c][0]))
            #     allCV.append(np.mean(statsAllMLI[n][2][c][1]))
            for c in range(nCellsPC):
                markers, caps, bars = ax20m.errorbar(np.mean(statsAllPC[n][2][c][6]), np.mean(statsAllPC[n][2][c][7]), xerr=np.std(statsAllPC[n][2][c][6]),
                                                     yerr=np.std(statsAllPC[n][2][c][7]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allFR.append(np.mean(statsAllPC[n][2][c][6]))
                allCV.append(np.mean(statsAllPC[n][2][c][7]))
        self.layoutOfPanel(ax20m, xLabel='complex spike firing rate (1/s)', yLabel='complex spike CV')
        #ax20m.set_ylim(0.4, 1.8)
        ibins = 20
        ax20top = plt.subplot(gssubsub20[0], sharex=ax20m)
        ax20top.set_title('Complex spikes of PCs')
        ax20top.hist(allFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax20top.axvline(x=np.percentile(allFR, 25), ls=':', c='black')
        ax20top.axvline(x=np.percentile(allFR, 75), ls=':', c='black')
        ax20top.axvline(x=np.median(allFR), ls='--', c='black')
        self.layoutOfPanel(ax20top, xyInvisible=[True, True])
        ax20right = plt.subplot(gssubsub20[3], sharey=ax20m)
        ax20right.hist(allCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax20right.axhline(y=np.median(allCV), ls='--', c='black')
        ax20right.axhline(y=np.percentile(allCV, 25), ls=':', c='black')
        ax20right.axhline(y=np.percentile(allCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax20right, xyInvisible=[True, True])

        ######
        gssubsub21 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[1], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax21m = plt.subplot(gssubsub21[2])
        ax21m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBLFR = []
        allWalkFR = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax21m.errorbar(np.mean(statsAllMLI[n][2][c][2]), np.mean(statsAllMLI[n][2][c][4]), xerr=np.std(statsAllMLI[n][2][c][2]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][4]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice),
            #                                          label=((statsAllMLI[n][1][-3:] + ' (MLI %s,PC %s)' % (nCellsMLI, nCellsPC)) if c == 0 else None))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allBLFR.append(np.mean(statsAllMLI[n][2][c][2]))
            #     allWalkFR.append(np.mean(statsAllMLI[n][2][c][4]))
            for c in range(nCellsPC):
                markers, caps, bars = ax21m.errorbar(np.mean(statsAllPC[n][2][c][8]), np.mean(statsAllPC[n][2][c][10]), xerr=np.std(statsAllPC[n][2][c][8]),
                                                     yerr=np.std(statsAllPC[n][2][c][10]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBLFR.append(np.mean(statsAllPC[n][2][c][8]))
                allWalkFR.append(np.mean(statsAllPC[n][2][c][10]))
        self.layoutOfPanel(ax21m, xLabel='complex spike baseline firing rate (1/s)', yLabel='complex spike walking firing rate (1/s)', Leg=[4, 5])
        ax21top = plt.subplot(gssubsub21[0], sharex=ax21m)
        ax21top.hist(allBLFR, bins=ibins, histtype='stepfilled', color='0.4')
        ax21top.axvline(x=np.percentile(allBLFR, 25), ls=':', c='black')
        ax21top.axvline(x=np.percentile(allBLFR, 75), ls=':', c='black')
        ax21top.axvline(x=np.median(allBLFR), ls='--', c='black')
        self.layoutOfPanel(ax21top, xyInvisible=[True, True])
        ax21right = plt.subplot(gssubsub21[3], sharey=ax21m)
        ax21right.hist(allWalkFR, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax21right.axhline(y=np.median(allWalkFR), ls='--', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 25), ls=':', c='black')
        ax21right.axhline(y=np.percentile(allWalkFR, 75), ls=':', c='black')
        self.layoutOfPanel(ax21right, xyInvisible=[True, True])

        print('paired t-test complex spike rate PC baseline - walking :', stats.ttest_rel(allBLFR, allWalkFR))
        print('PC baseline complex spike [25,50,75]th percentile :', np.percentile(allBLFR, 25), np.percentile(allBLFR, 50), np.percentile(allBLFR, 75))
        print('PC  walking complex spike [25,50,75]th percentile :', np.percentile(allWalkFR, 25), np.percentile(allWalkFR, 50), np.percentile(allWalkFR, 75))

        ####
        gssubsub22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub2[2], width_ratios=(4, 0.7), height_ratios=(0.7, 4), wspace=0.05, hspace=0.05)
        ax22m = plt.subplot(gssubsub22[2])
        ax22m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        allBaselCV = []
        allWalkCV = []
        for n in range(nMice):
            #nCellsMLI = len(statsAllMLI[n][2])
            nCellsPC = len(statsAllPC[n][2])
            # for c in range(nCellsMLI):
            #     markers, caps, bars = ax22m.errorbar(np.mean(statsAllMLI[n][2][c][3]), np.mean(statsAllMLI[n][2][c][5]), xerr=np.std(statsAllMLI[n][2][c][3]),
            #                                          yerr=np.std(statsAllMLI[n][2][c][5]), fmt='o', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
            #     [bar.set_alpha(0.2) for bar in bars]
            #     allBaselCV.append(np.mean(statsAllMLI[n][2][c][3]))
            #     allWalkCV.append(np.mean(statsAllMLI[n][2][c][5]))
            for c in range(nCellsPC):
                markers, caps, bars = ax22m.errorbar(np.mean(statsAllPC[n][2][c][9]), np.mean(statsAllPC[n][2][c][11]), xerr=np.std(statsAllPC[n][2][c][9]),
                                                     yerr=np.std(statsAllPC[n][2][c][11]), fmt='^', ms=3, elinewidth=1, color=cmap((n + 1) / nMice))
                [bar.set_alpha(0.2) for bar in bars]
                allBaselCV.append(np.mean(statsAllPC[n][2][c][9]))
                allWalkCV.append(np.mean(statsAllPC[n][2][c][11]))
        self.layoutOfPanel(ax22m, xLabel='complex spike baseline CV', yLabel='complex spike walking CV')
        ax22top = plt.subplot(gssubsub22[0], sharex=ax22m)
        ax22top.hist(allBaselCV, bins=ibins, histtype='stepfilled', color='0.4')
        ax22top.axvline(x=np.percentile(allBaselCV, 25), ls=':', c='black')
        ax22top.axvline(x=np.percentile(allBaselCV, 75), ls=':', c='black')
        ax22top.axvline(x=np.median(allBaselCV), ls='--', c='black')
        self.layoutOfPanel(ax22top, xyInvisible=[True, True])
        ax22right = plt.subplot(gssubsub22[3], sharey=ax22m)
        ax22right.hist(allWalkCV, bins=ibins, orientation='horizontal', histtype='stepfilled', color='0.4')
        ax22right.axhline(y=np.median(allWalkCV), ls='--', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 25), ls=':', c='black')
        ax22right.axhline(y=np.percentile(allWalkCV, 75), ls=':', c='black')
        self.layoutOfPanel(ax22right, xyInvisible=[True, True])

        print('paired t-test complex spike CV PC baseline - walking :', stats.ttest_rel(allBaselCV, allWalkCV))

        ##
        gssubsub23 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub2[3], wspace=0.05, hspace=0.05)
        ax23m = plt.subplot(gssubsub23[0])
        ax23m.set_title('%s mice : %s MLIs with %s recs.\nand %s PCs with %s recs.' % (nMice,totMLI,sum(recsTotMLI),totPC,sum(recsTotPC)),fontsize=10)
        #ax23m.axline((1, 1), slope=1, ls='--', color='0.6')  # , transform=plt.gca().transAxes)
        #allBaselCV = []
        #allWalkCV = []
        print('average number of recording per cells',int(np.average(np.append(recsTotMLI, recsTotPC))))
        print('STD of number of recording per cells', int(np.std(np.append(recsTotMLI, recsTotPC))))
        # pdb.set_trace()
        bbins = np.linspace(.75,9.25,8*2+2)
        #ax23m.bar(X + 0.00, data[0], color = 'b', width = 0.25)
        #ax23m.bar(X + 0.25, data[1], color = 'g', width = 0.25)
        ax23m.hist(recsTotPC,bins=bbins,histtype='step',label='PCs')
        ax23m.hist(recsTotMLI,bins=bbins,histtype='step',label='MLIs')
        majorLocator_x = MultipleLocator(1)
        ax23m.xaxis.set_major_locator(majorLocator_x)
        self.layoutOfPanel(ax23m, xLabel='number of recordings per cell', yLabel='cells',Leg=[1,9])


        ########################
        fname = 'fig_ephys-summary-MLI-PC_v%s' % figVersion
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        # plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    #######################################################
    def ephysPSTHSwing_Stance_GrantFig(self,figVersion,cellType, ephysPSTHWalkingData, ephysSummaryAllAnimals):
        nMice = len(ephysSummaryAllAnimals)
        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
                               height_ratios=[1, 2]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.93, top=0.96, bottom=0.1)
        plt.figtext(0.01, 0.955, 'A', clip_on=False, color='black',  size=22)
        plt.figtext(0.01, 0.6, 'B', clip_on=False, color='black', size=22)
        #plt.figtext(0.01, 0.585, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.5, 0.6, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.01, 0.23, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.85, 0.69, 'F', clip_on=False, color='black', size=22)
        plt.figtext(0.01, 0.35, 'G', clip_on=False, color='black', size=22)
        plt.figtext(0.24, 0.35, 'H', clip_on=False, color='black', size=22)
        plt.figtext(0.485, 0.35, 'I', clip_on=False, color='black', size=22)
        plt.figtext(0.72, 0.35, 'J', clip_on=False, color='black', size=22)

        # sub-panel enumerations
        #plt.figtext(0.06, 0.98, '%s   %s   %s' % (self.mouse, date, rec), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.2,wspace=0.1)
        col = ['C0','C1','C2','C3']
        lab = ['FL','FR','HL','HR']
        spacing = [100,200,500,500]
        #ax0 = plt.subplot(gssub[0])
        #ephysPSTHWalkingData.append([r, [foldersRecordings[f][0], foldersRecordings[f][2][r], 'ephysDataAnalyzed'], cPawPos,ephys,swingStanceDict,ephysPSTHDict])
        firingRate = []
        twoCols=['blueviolet','0.4']
        startx = [5,12]
        xLength = 9
        #pdb.set_trace()
        n = 0 # chose the trace to show here
        #for n in range(2):
        pawMax = []
        pawMin = []
        print(n,ephysPSTHWalkingData[n][0],ephysPSTHWalkingData[n][1])
        ax2 = plt.subplot(gssub0[0])
        ax0 = ax2.twinx()
        pawPos = ephysPSTHWalkingData[n][2]
        swingStanceD = ephysPSTHWalkingData[n][4]
        ephys = ephysPSTHWalkingData[n][3]
        isis = np.diff(ephys[0][(ephys[0]>10.)&(ephys[0]<=52)])
        firingRate.append([n,1./np.mean(isis)])

        binWidth = 1.E-3  # in sec
        spikecountwindow = 0.05
        nspikecountwindow = spikecountwindow / binWidth
        #tbins = np.linspace(0., len(eData) * dt, int(len(eData) * dt / binWidth) + 1)
        tbins = np.linspace(0.,60.,int(60/binWidth)+1,endpoint=True)
        spikeTimes = ephys[0]
        binnedspikes, _ = np.histogram(spikeTimes, tbins)
        spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
        # convert the convolved spike trains to units of spikes/sec
        spikesconv *= 1. / binWidth


        mmax = np.max(spikesconv)
        mmin = np.min(spikesconv)
        #ax2.eventplot(ephys[0], lineoffsets=((mmax-mmin)/2 + mmin), linelengths=(0.8*mmax-0.8*mmin),linewidths=0.1,color='0.8')
        ax2.plot((tbins[1:]+tbins[:-1])/2,spikesconv,color='0.4',lw=2)
        ax0.set_zorder(1)
        ax0.patch.set_visible(False)
        #ax2.set_ylabel('firing rate')
        ax2.set_ylim(0,110)

        self.layoutOfPanel(ax2, xLabel='time (s)', yLabel=(None if n > 0 else 'firing rate'), xyInvisible=[False, False], Leg=[1, 9])
        #pdb.set_trace()
        for i in range(4):
            #ax0 = plt.subplot(gssub[i])
            pawMin.append(np.min(pawPos[i][:,1]))
            pawMax.append(np.max(pawPos[i][:,1]))
            ax0.plot(pawPos[i][:,0],pawPos[i][:,1],c=col[i],lw=2,label=lab[i])
            idxSwings = swingStanceD['swingP'][i][1]
            indecisiveSteps = swingStanceD['swingP'][i][3]
            recTimes = swingStanceD['forFit'][i][2]
            #pdb.set_trace()

            for j in range(len(idxSwings)):
                idxStart = np.argmin(np.abs(pawPos[i][:,0] - recTimes[idxSwings[j][0]]))
                idxEnd = np.argmin(np.abs(pawPos[i][:,0] - recTimes[idxSwings[j][1]]))
                ax0.plot(pawPos[i][idxStart,0],pawPos[i][idxStart,1],'x',c=col[i],alpha=0.5,lw=0.5)
                if indecisiveSteps[n][3]: # indecisive Step
                   ax0.plot(pawPos[i][idxEnd,0],pawPos[i][idxEnd,1],'1',c=col[i],alpha=0.5,lw=0.5)
                else:
                    ax0.plot(pawPos[i][idxEnd, 0], pawPos[i][idxEnd, 1], '+', c=col[i], alpha=0.5, lw=0.5)
        mmax = np.max(pawMax)
        mmin = np.min(pawMin)
        print(n,mmax,mmin,pawMin)
        ax0.eventplot(ephys[0], lineoffsets=390, linelengths=580, linewidths=0.1, color='0.5',alpha=0.4)
        ax0.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.xaxis.set_visible(False)
        ax0.spines['left'].set_visible(False)
        #ax2.yaxis.set_visible(False)
        ax0.spines['right'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('right')
        ax0.legend(loc=2, frameon=False)
        legend = ax0.get_legend()  # plt.gca().get_legend()
        ltext = legend.get_texts()
        plt.setp(ltext, fontsize=9)
        #ax0.set_ylim(1.05*mmin,1.05*mmax)
        ax2.set_xlim(startx[n],startx[n]+xLength)
        ax0.set_ylim(100,680)



        ####################################
        #ephysPSTHWalkingData.append([r, [foldersRecordings[f][0], foldersRecordings[f][2][r], 'ephysDataAnalyzed'], cPawPos,ephys,swingStanceDict,ephysPSTHDict])
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.03, wspace=0.3)
        gssub10 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1[0], hspace=0.03, wspace=0.15)
        #swingStanceD0 = ephysPSTHWalkingData[0][4]
        #swingStanceD1 = ephysPSTHWalkingData[1][4]
        #psth0 =  ephysPSTHWalkingData[0][5]
        psth1 =  ephysPSTHWalkingData[n][5]
        yLocator0 = MultipleLocator(50)
        yLocator1 = MultipleLocator(10)
        for i in [0]:#range(4):
            ax1 = plt.subplot(gssub1[i])
            ax1.set_title('   '+lab[i],loc='left',color=col[i],fontweight='bold')
            if i ==0:
                if cellType=='MLI':
                    textx = ax1.annotate('swing onset', xy=(0.01,163), annotation_clip=False,xytext=None, textcoords='data',fontsize=10,arrowprops=None,color=twoCols[0])
                elif cellType == 'PC':
                    textx = ax1.annotate('swing onset', xy=(0.01, 190), annotation_clip=False, xytext=None, textcoords='data', fontsize=10, arrowprops=None, color=twoCols[0])
            #idxSwings = swingStanceD['swingP'][i][1]
            #ax1.set_title('%s strides' % len(idxSwings))
            ax1.axvline(x=0, ls='--', c='0.8')
            ax1.eventplot(psth1[i]['spikeTimesCenteredSwingStartSorted'],colors=twoCols[0], linewidths=1)
            ax1.eventplot(psth1[i]['swingEndSorted'], color=col[i], linewidths=2)
            ax1.eventplot(psth1[i]['strideEnd2Sorted'], color=col[i], linewidths=2)
            #ax1.eventplot(psth0[i]['spikeTimesCenteredSorted'], color=twoCols[0], linewidths=1)
            #ax1.eventplot(psth0[i]['swingStartSorted'], color=col[i], linewidths=2)
            #ax1.eventplot(psth0[i]['strideEndSorted'], color=col[i], linewidths=2)
            self.layoutOfPanel(ax1, yLabel=('strides' if i==0 else None), xyInvisible=[True, False])
            ax1.set_xlim(-0.3, 0.4)
            ax1.yaxis.set_major_locator(yLocator0)

            ax2 = plt.subplot(gssub10[0])
            if i ==0:
                if cellType=='MLI':
                    textx = ax2.annotate('stance onset', xy=(0.01,163), annotation_clip=False,xytext=None, textcoords='data',fontsize=10,arrowprops=None,color=twoCols[1])
                    textx = ax2.annotate('swing onset', xy=(-0.2,163), annotation_clip=False,xytext=None, textcoords='data',fontsize=10,arrowprops=None,color='C0')
                elif cellType == 'PC':
                    textx = ax2.annotate('stance onset', xy=(0.01, 190), annotation_clip=False, xytext=None, textcoords='data', fontsize=10, arrowprops=None, color=twoCols[1])
            #idxSwings = swingStanceD['swingP'][i][1]
            #ax2.set_title('%s strides' % len(idxSwings))
            ax2.axvline(x=0, ls='--', c='0.8')
            ax2.eventplot(psth1[i]['spikeTimesCenteredSorted'], color=twoCols[1], linewidths=1)
            ax2.eventplot(psth1[i]['swingStartSorted'], color=col[i], linewidths=2)
            ax2.eventplot(psth1[i]['strideEndSorted'], color=col[i], linewidths=2)
            self.layoutOfPanel(ax2, yLabel=('strides' if i==0 else None), xyInvisible=[True, False])
            ax2.set_xlim(-0.3, 0.4)
            ax2.yaxis.set_major_locator(yLocator0)

            ax3 = plt.subplot(gssub10[1])
            ax3.axvline(x=0, ls='--', c='0.8')
            #ax3.axhline(y=firingRate[0][1],ls=':',color=twoCols[0],alpha=0.6)
            #ax3.axhline(y=firingRate[1][1],ls=':',color=twoCols[1],alpha=0.6)
            ax3.step(psth1[i]['psth_swingOnsetAligned_allSteps'][0], psth1[i]['psth_swingOnsetAligned_allSteps'][1],ls='-', where='mid', color=twoCols[0])
            ax3.step(psth1[i]['psth_swingOnsetAligned_allSteps'][0], psth1[i]['psth_swingOnsetAligned_allSteps_median5-95perentiles'][1], where='mid', color=twoCols[1],alpha=0.3)
            ax3.fill_between(psth1[i]['psth_swingOnsetAligned_allSteps'][0],psth1[i]['psth_swingOnsetAligned_allSteps_median5-95perentiles'][0],psth1[i]['psth_swingOnsetAligned_allSteps_median5-95perentiles'][2],step='mid',color=twoCols[0],alpha=0.1)

            #ax3.step(psth0[i]['psth_stanceOnsetAligned_allSteps'][0], psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][1], where='mid', color=twoCols[0],alpha=0.3)
            #ax3.fill_between(psth0[i]['psth_stanceOnsetAligned_allSteps'][0],psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][0],psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][2],step='mid', color=twoCols[0],alpha=0.1)
            #
            ax3.step(psth1[i]['psth_stanceOnsetAligned_allSteps'][0], psth1[i]['psth_stanceOnsetAligned_allSteps'][1], where='mid', color=twoCols[1])
            ax3.step(psth1[i]['psth_stanceOnsetAligned_allSteps'][0], psth1[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][1], where='mid', color=twoCols[1],alpha=0.3)
            ax3.fill_between(psth1[i]['psth_stanceOnsetAligned_allSteps'][0],psth1[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][0],psth1[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][2],step='mid', color=twoCols[1],alpha=0.1)

            self.layoutOfPanel(ax3, xLabel='time centered on stance onset (s)', yLabel=('PSTH (spk/s)' if i==0 else None), xyInvisible=[False, False])
            ax3.set_xlim(-0.3, 0.4)
            if cellType=='MLI':
                ax3.set_ylim(15,59)
            elif cellType == 'PC':
                pass
            ax3.yaxis.set_major_locator(yLocator1)
        #binCent = psth0[i]['psth_stanceOnsetAligned_allSteps'][0]
        #pdb.set_trace()

        ####################################
        #gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], hspace=0.2, wspace=0.15)#,height_ratios=[1,2])
        cols = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        symList = ['o','P','D','+','^','4','*','x']
        oldRecDay = '770809'
        xlocator = MultipleLocator(1)
        before = [-0.1,0.]
        after = [0.,0.1]
        idxSym=0
        for i in [0]: #range(4):
            #ax3 = plt.subplot(gssub2[i])
            #ax3.axhline(y=0,ls='--',c='0.6')
            #ax3 = plt.subplot(gssub2[i])
            #ax3.axhline(y=0, ls='--', c='0.5')
            #ax3.axvline(x=0, ls='--', c='0.5')
            totalRecs = 0
            totalCells = 0
            pieCounts = np.zeros((2,9))
            modsBefore = [[], []]
            modsAfter = [[], []]
            for n in range(nMice):
                #if i == 0:
                #    ax3.set_title(ephysSummaryAllAnimals[n][1],fontsize=8,color='0.2')
                nCells = len(ephysSummaryAllAnimals[n][2])
                if i == 0: print('nCells', nCells, ephysSummaryAllAnimals[n][0],ephysSummaryAllAnimals[n][1] )
                totalCells+= nCells
                # cellsPerMouse = []
                nDay=0
                idxSym=0
                inc=0
                for c in range(nCells):
                    nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                    if i == 0: print('nRecs', nRecs, ephysSummaryAllAnimals[n][2][c][0], ephysSummaryAllAnimals[n][2][c][1] )
                    totalRecs +=nRecs
                    # temp = np.zeros((7,nRecs))
                    # bTemp = []
                    recDay = ephysSummaryAllAnimals[n][2][c][1]
                    if recDay == oldRecDay:
                        nDay+=1
                        sameDay = True
                        inc=0
                    else: sameDay = False
                    oldP = [-2,0,0]
                    modRecB = [[],[]]
                    modRecA = [[],[]]
                    for j in range(nRecs):
                        if i ==0: print(j,ephysSummaryAllAnimals[n][2][c][2][j][0],ephysSummaryAllAnimals[n][2][c][2][j][1])
                        ephysPSTHDict = ephysSummaryAllAnimals[n][2][c][2][j][2]
                        kkeys = ['psth_swingOnsetAligned_allSteps','psth_stanceOnsetAligned_allSteps']
                        for k in range(2): # loop over swing and stance onset aligned psth's
                            ttime = ephysPSTHDict[i][kkeys[k]][0]
                            beforeMask = (ttime>before[0])&(ttime<before[1])
                            afterMask = (ttime>after[0])&(ttime<after[1])
                            psth = ephysPSTHDict[i][kkeys[k]][1]
                            pqq = ephysPSTHDict[i][kkeys[k]+'_median5-95perentiles']
                            beforeEffect = True
                            afterEffect = True
                            #pdb.set_trace()
                            conditionHighB = psth[beforeMask]>pqq[2,beforeMask]
                            conditionLowB = psth[beforeMask]<pqq[0,beforeMask]
                            #pdb.set_trace()
                            consExcursions = 1 # means that at least (n+1) neighboring psth values have to cross the confidence interval
                            if (sum(conditionHighB)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighB ) if key])>consExcursions):
                                bb = np.mean(psth[beforeMask][psth[beforeMask]>pqq[2,beforeMask]]/pqq[1,beforeMask][psth[beforeMask]>pqq[2,beforeMask]])-1.
                            elif (sum(conditionLowB)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowB ) if key])>consExcursions):
                                bb = np.mean(psth[beforeMask][psth[beforeMask] < pqq[0,beforeMask]] / pqq[1,beforeMask][psth[beforeMask] < pqq[0,beforeMask]])-1.
                            else:
                                beforeEffect = False
                            conditionHighA = psth[afterMask]>pqq[2,afterMask]
                            conditionLowA = psth[afterMask]<pqq[0,afterMask]
                            if (sum(conditionHighA)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighA ) if key])>consExcursions):
                                aa =  np.mean(psth[afterMask][psth[afterMask]>pqq[2,afterMask]]/pqq[1,afterMask][psth[afterMask]>pqq[2,afterMask]])-1.
                            elif (sum(conditionLowA)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowA ) if key])>consExcursions):
                                aa = np.mean(psth[afterMask][psth[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth[afterMask] < pqq[0,afterMask]])-1.
                            else:
                                afterEffect = False
                                #aa = np.mean(psth[afterMask][psth[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth[afterMask] < pqq[0,afterMask]])-1.
                            pointPlot = [False,-1,-1]
                            if beforeEffect and afterEffect:
                                #ax3.plot(bb,aa,symList[(idxSym % len(symList))],color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,bb,aa]
                                if bb > 0 and aa > 0: pieCounts[k,0]+=1
                                elif bb>0 and aa<0: pieCounts[k,2]+=1
                                elif bb<0 and aa<0: pieCounts[k,4] += 1
                                elif bb<0 and aa>0: pieCounts[k,6] += 1
                            elif beforeEffect and not afterEffect:
                                #ax3.plot(bb, 0, symList[(idxSym % len(symList))], color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,bb,0]
                                if bb>0: pieCounts[k,1] += 1
                                elif bb<0: pieCounts[k,5] += 1
                            elif not beforeEffect and afterEffect:
                                #ax3.plot(0,aa, symList[(idxSym % len(symList))], color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,0,aa]
                                if aa>0: pieCounts[k,7] += 1
                                elif aa<0: pieCounts[k,3] += 1
                            if beforeEffect:
                                modRecB[k].append(bb)
                            else:
                                modRecB[k].append(0)
                            if afterEffect:
                                modRecA[k].append(aa)
                            else:
                                modRecA[k].append(0)
                            #if j>0 and pointPlot[0] and oldP[0]==(j-1):
                            #    #ax3.plot([pointPlot[1],oldP[1]], [pointPlot[2], oldP[2]], color=cols[n], alpha=0.2)
                            #if pointPlot[0]:
                            #    oldP = [j,pointPlot[1],pointPlot[2]]

                        #ax3.plot(nDay+inc,ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2],symList[(idxSym % len(symList))],color=cols[n],clip_on=False)
                        #if not(nDay==1 and inc==0):
                        #    ax3.plot([prev[0],nDay+inc],[prev[1],ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2]],color=cols[n],alpha=0.5)
                        #prev = [nDay+inc,ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2]]
                        inc+=1
                    for k in range(2):
                        modsAfter[k].append(modRecA[k])
                        modsBefore[k].append(modRecB[k])
                    oldRecDay = recDay
                    idxSym+=1
            #self.layoutOfPanel(ax3, xLabel='rel. change before swing-onset',yLabel='rel. change after swing-onset', xyInvisible=[False,(True if i>0 else False)])
            #ax3.spines['left'].set_position('zero')
            #ax3.spines['bottom'].set_position('zero')
            #ax3.set_ylim(-0.5,0.7)
            #ax3.set_xlim(-0.5,1)
            #pdb.set_trace()
            ########
            ax4 = plt.subplot(gssub1[1])
            #ax5 = plt.subplot(gssub2[i+4])
            #if i == 0:
            #    ax4.set_title('swing onset modulated')
            #    ax5.set_title('stance onset modulated')
            pieCounts[0,8] = totalRecs - sum(pieCounts[0,:-1])
            pieCounts[1,8] = totalRecs - sum(pieCounts[1, :-1])
            labels = '↑↑', '↑-', '↑↓', '-↓', '↓↓', '↓-', '↓↑', '-↑', '--'
            explode=(0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0)
            #pdb.set_trace()
            print(i,pieCounts)
            ccolors = ['darkred', 'orangered', 'lightcoral', 'blue', 'darkblue', 'skyblue', 'darkgreen', 'limegreen', 'C7']
            barWidth = 0.85
            startBarSwing = 0
            startBarStance = 0
            for k in range(len(pieCounts[0])-1):
                percSwing = 100*pieCounts[0,k]/totalRecs
                ax4.bar(0,percSwing,bottom=startBarSwing,color=ccolors[k],linewidth=0,edgecolor='white',width=barWidth)
                if percSwing>0:
                    ax4.annotate('%s  ' % np.round(percSwing, 1) + labels[k], xy=(-0.2, startBarSwing + 0.5*percSwing-1), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None,
                                 color='white')
                startBarSwing += percSwing
                percStance = 100*pieCounts[1, k]/totalRecs
                ax4.bar(2, percStance, bottom=startBarStance, color=ccolors[k], linewidth=0,edgecolor='white', width=barWidth)
                if percStance>0:
                    ax4.annotate('%s  ' % np.round(percStance, 1) + labels[k], xy=(1.8, startBarStance + 0.5*percStance-1), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None,
                                 color='white')
                startBarStance+=percStance

            ax4.annotate('%s %%' % np.round(startBarSwing,1), xy=(-0.2, startBarSwing+2), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None, color='black')
            ax4.annotate('%s %%' % np.round(startBarStance,1), xy=(1.8, startBarStance+2), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None, color='black')
            self.layoutOfPanel(ax4, yLabel='percentage modulated '+cellType+'s (%)', xyInvisible=[False, (True if i > 0 else False)])
            plt.xticks([0,2], ['swing onset modulated','stance onset modulated'])
            ax4.set_xlim(-1,3)
            ax4.set_ylim(0,55)
            #plt.xlabel("group")
            #ax4.pie(pieCounts[0], explode=explode, labels=labels, colors=ccolors, autopct='%1.1f%%',shadow=True, startangle=90)
            #ax4.axis('equal')
            if i ==3:
                textx = ax4.annotate('%s mice, %s %ss, %s recs' % (nMice,totalCells,cellType, totalRecs), xy=(1,47), annotation_clip=False,xytext=None, textcoords='data',fontsize=9,arrowprops=None,color='gray')
            #ax5.pie(pieCounts[1], explode=explode, labels=labels, colors=ccolors, autopct='%1.1f%%',shadow=True, startangle=90)
            #ax5.axis('equal')
            # gssub3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub2[i+4], hspace=0.1, wspace=0.15)
            # #for k in range(2):
            # ax5 = plt.subplot(gssub3[0])
            # for n in range(len(modsBefore[1])):
            #     if sum(modsBefore[1][n])>0:
            #         mm = np.asarray(modsBefore[1][n])
            #         #pdb.set_trace()
            #         ax5.plot((np.arange(len(mm)))[mm!=0]+1,mm[mm!=0],'o-',ms=2,lw=1,alpha=0.5)
            # ax6 = plt.subplot(gssub3[1])
            # for n in range(len(modsAfter[0])):
            #     if sum(modsAfter[0][n])>0:
            #         mm = np.asarray(modsAfter[0][n])
            #         ax6.plot((np.arange(len(mm)))[mm!=0]+1,mm[mm!=0],'o-',lw=1,ms=2,alpha=0.5)
            # self.layoutOfPanel(ax5,yLabel='before stance onset mod.',xyInvisible=[True,False])
            # self.layoutOfPanel(ax6,xLabel='trial #',yLabel='after swing onset mod.', xyInvisible=[False, False])
            # #ax3.xaxis.set_major_locator(xlocator)

        ########################
        fname = 'fig_ephys-psth-%s-swing-stance_grant_v%s' % (cellType,figVersion)
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        # plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')


    #######################################################
    def ephysPSTHSwing_StanceFig(self,figVersion,cellType, ephysPSTHWalkingData, ephysSummaryAllAnimals):
        nMice = len(ephysSummaryAllAnimals)
        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 18  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
                               height_ratios=[0.5, 3,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.2)

        # possibly change outer margins of the figure
        # plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.05)
        # plt.figtext(0.01, 0.955, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.01, 0.82, 'B', clip_on=False, color='black', size=22)
        # #plt.figtext(0.01, 0.585, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.01, 0.45, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.01, 0.23, 'D', clip_on=False, color='black',  size=22)
        #plt.figtext(0.85, 0.69, 'F', clip_on=False, color='black', size=22)
        #plt.figtext(0.01, 0.35, 'G', clip_on=False, color='black', size=22)
        #plt.figtext(0.24, 0.35, 'H', clip_on=False, color='black', size=22)
        #plt.figtext(0.485, 0.35, 'I', clip_on=False, color='black', size=22)
        #plt.figtext(0.72, 0.35, 'J', clip_on=False, color='black', size=22)

        # sub-panel enumerations
        #plt.figtext(0.06, 0.98, '%s   %s   %s' % (self.mouse, date, rec), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.2,wspace=0.1)
        col = ['C0','C1','C2','C3']
        lab = ['FL','FR','HL','HR']
        spacing = [100,200,500,500]
        #ax0 = plt.subplot(gssub[0])
        #ephysPSTHWalkingData.append([r, [foldersRecordings[f][0], foldersRecordings[f][2][r], 'ephysDataAnalyzed'], cPawPos,ephys,swingStanceDict,ephysPSTHDict])
        firingRate = []
        twoCols=['blueviolet','0.4']
        startx = [25,21]
        xLength = 16
        #pdb.set_trace()
        if cellType=='MLI':
            n = 1 # chose the trace to show here
        elif cellType == 'PC':
            n=0
        #for n in range(2):
        pawMax = []
        pawMin = []
        print(n,ephysPSTHWalkingData[n][0],ephysPSTHWalkingData[n][1])
        ax0 = plt.subplot(gssub0[0])
        pawPos = ephysPSTHWalkingData[n][2]
        swingStanceD = ephysPSTHWalkingData[n][4]
        ephys = ephysPSTHWalkingData[n][3]
        isis = np.diff(ephys[0][(ephys[0]>10.)&(ephys[0]<=52)])
        firingRate.append([n,1./np.mean(isis)])
        #pdb.set_trace()
        for i in range(2):
            #ax0 = plt.subplot(gssub[i])
            pawMin.append(np.min(pawPos[i][:,1]))
            pawMax.append(np.max(pawPos[i][:,1]))
            ax0.plot(pawPos[i][:,0],pawPos[i][:,1],c=col[i],lw=2,label=lab[i])
            idxSwings = swingStanceD['swingP'][i][1]
            indecisiveSteps = swingStanceD['swingP'][i][3]
            recTimes = swingStanceD['forFit'][i][2]
            #pdb.set_trace()

            for j in range(len(idxSwings)):
                idxStart = np.argmin(np.abs(pawPos[i][:,0] - recTimes[idxSwings[j][0]]))
                idxEnd = np.argmin(np.abs(pawPos[i][:,0] - recTimes[idxSwings[j][1]]))
                ax0.plot(pawPos[i][idxStart,0],pawPos[i][idxStart,1],'x',c=col[i],alpha=0.5,lw=0.5)
                #if indecisiveSteps[n][3]: # indecisive Step
                #    ax0.plot(pawPos[i][idxEnd,0],pawPos[i][idxEnd,1],'1',c=col[i],alpha=0.5,lw=0.5)
                #else:
                ax0.plot(pawPos[i][idxEnd, 0], pawPos[i][idxEnd, 1], '+', c=col[i], alpha=0.5, lw=0.5)
        mmax = np.max(pawMax)
        mmin = np.min(pawMin)
        print(n,mmax,mmin,pawMin)
        ax0.eventplot(ephys[0], lineoffsets=550, linelengths=200, linewidths=0.1, color='0.4',alpha=0.4)
        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel=(None if n>0 else 'x (pixel)'), xyInvisible=[False, False],Leg=[1,9])
        ax0.set_xlim(startx[n],startx[n]+xLength)
        ax0.set_ylim(430,680)

        ####################################
        #ephysPSTHWalkingData.append([r, [foldersRecordings[f][0], foldersRecordings[f][2][r], 'ephysDataAnalyzed'], cPawPos,ephys,swingStanceDict,ephysPSTHDict])
        gssub1 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], hspace=0.03, wspace=0.15)
        #swingStanceD0 = ephysPSTHWalkingData[0][4]
        #swingStanceD1 = ephysPSTHWalkingData[1][4]
        #psth0 =  ephysPSTHWalkingData[0][5]
        # pdb.set_trace()
        psth1 =  ephysPSTHWalkingData[n][5]
        yLocator0 = MultipleLocator(50)
        yLocator1 = MultipleLocator(10)
        for i in range(4):
            ax1 = plt.subplot(gssub1[i])
            ax1.set_title('   '+lab[i],loc='left',color=col[i],fontweight='bold')
            if i ==0:
                if cellType=='MLI':
                    textx = ax1.annotate('swing onset', xy=(0.01,163), annotation_clip=False,xytext=None, textcoords='data',fontsize=10,arrowprops=None,color=twoCols[0])
                elif cellType == 'PC':
                    textx = ax1.annotate('swing onset', xy=(0.01, 190), annotation_clip=False, xytext=None, textcoords='data', fontsize=10, arrowprops=None, color=twoCols[0])
            #idxSwings = swingStanceD['swingP'][i][1]
            #ax1.set_title('%s strides' % len(idxSwings))
            ax1.axvline(x=0, ls='--', c='0.8')

            ax1.eventplot(psth1[i]['spikeTimesCenteredSwingStartSorted'],colors=twoCols[0], linewidths=1)
            ax1.eventplot(psth1[i]['strideEndSwingCenteredSorted'], color=col[i], linewidths=2)
            ax1.eventplot(psth1[i]['stanceStartSorted'], color=col[i], linewidths=2)
            #ax1.eventplot(psth0[i]['spikeTimesCenteredSorted'], color=twoCols[0], linewidths=1)
            #ax1.eventplot(psth0[i]['swingStartSorted'], color=col[i], linewidths=2)
            #ax1.eventplot(psth0[i]['strideEndSorted'], color=col[i], linewidths=2)
            self.layoutOfPanel(ax1, yLabel=('strides' if i==0 else None), xyInvisible=[True, False])
            ax1.set_xlim(-0.3, 0.4)
            ax1.yaxis.set_major_locator(yLocator0)

            ax2 = plt.subplot(gssub1[i+4])
            if i ==0:
                if cellType=='MLI':
                    textx = ax2.annotate('stance onset', xy=(0.01,163), annotation_clip=False,xytext=None, textcoords='data',fontsize=10,arrowprops=None,color=twoCols[1])
                elif cellType == 'PC':
                    textx = ax2.annotate('stance onset', xy=(0.01, 190), annotation_clip=False, xytext=None, textcoords='data', fontsize=10, arrowprops=None, color=twoCols[1])
            #idxSwings = swingStanceD['swingP'][i][1]
            #ax2.set_title('%s strides' % len(idxSwings))
            ax2.axvline(x=0, ls='--', c='0.8')
            ax2.eventplot(psth1[i]['spikeTimesCenteredStanceStartSorted'], color=twoCols[1], linewidths=1)
            ax2.eventplot(psth1[i]['swingStartSorted'], color=col[i], linewidths=2)
            ax2.eventplot(psth1[i]['strideEndStanceCenteredSorted'], color=col[i], linewidths=2)
            self.layoutOfPanel(ax2, yLabel=('strides' if i==0 else None), xyInvisible=[True, False])
            ax2.set_xlim(-0.3, 0.4)
            ax2.yaxis.set_major_locator(yLocator0)

            ax3 = plt.subplot(gssub1[i + 8])
            ax3.axvline(x=0, ls='--', c='0.8')
            #ax3.axhline(y=firingRate[0][1],ls=':',color=twoCols[0],alpha=0.6)
            #ax3.axhline(y=firingRate[1][1],ls=':',color=twoCols[1],alpha=0.6)
            ax3.step(psth1[i]['psth_swingOnsetAligned'][0], psth1[i]['psth_swingOnsetAligned'][1],ls='-', where='mid', color=twoCols[0])
            ax3.step(psth1[i]['psth_swingOnsetAligned'][0], psth1[i]['psth_swingOnsetAligned_5-50-95perentiles'][1], where='mid', color=twoCols[1],alpha=0.3)
            ax3.fill_between(psth1[i]['psth_swingOnsetAligned'][0],psth1[i]['psth_swingOnsetAligned_5-50-95perentiles'][0],psth1[i]['psth_swingOnsetAligned_5-50-95perentiles'][2],step='mid',color=twoCols[0],alpha=0.1)

            #ax3.step(psth0[i]['psth_stanceOnsetAligned_allSteps'][0], psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][1], where='mid', color=twoCols[0],alpha=0.3)
            #ax3.fill_between(psth0[i]['psth_stanceOnsetAligned_allSteps'][0],psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][0],psth0[i]['psth_stanceOnsetAligned_allSteps_median5-95perentiles'][2],step='mid', color=twoCols[0],alpha=0.1)
            #
            ax3.step(psth1[i]['psth_stanceOnsetAligned'][0], psth1[i]['psth_stanceOnsetAligned'][1], where='mid', color=twoCols[1])
            ax3.step(psth1[i]['psth_stanceOnsetAligned'][0], psth1[i]['psth_stanceOnsetAligned_5-50-95perentiles'][1], where='mid', color=twoCols[1],alpha=0.3)
            ax3.fill_between(psth1[i]['psth_stanceOnsetAligned'][0],psth1[i]['psth_stanceOnsetAligned_5-50-95perentiles'][0],psth1[i]['psth_stanceOnsetAligned_5-50-95perentiles'][2],step='mid', color=twoCols[1],alpha=0.1)

            self.layoutOfPanel(ax3, xLabel='time centered on swing/stance onset (s)', yLabel=('PSTH (spk/s)' if i==0 else None), xyInvisible=[False, False])
            ax3.set_xlim(-0.3, 0.4)
            # if cellType=='MLI':
            #     ax3.set_ylim(15,40)
            # elif cellType == 'PC':
            #     pass
            ax3.yaxis.set_major_locator(yLocator1)
        #binCent = psth0[i]['psth_stanceOnsetAligned_allSteps'][0]
        #pdb.set_trace()

        ####################################
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], hspace=0.2, wspace=0.15)#,height_ratios=[1,2])
        cols = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        symList = ['o','P','D','+','^','4','*','x']
        oldRecDay = '770809'
        xlocator = MultipleLocator(1)
        before = [-0.1,0.]
        after = [0.,0.1]
        idxSym=0
        for i in range(4):
            #ax3 = plt.subplot(gssub2[i])
            #ax3.axhline(y=0,ls='--',c='0.6')
            #ax3 = plt.subplot(gssub2[i])
            #ax3.axhline(y=0, ls='--', c='0.5')
            #ax3.axvline(x=0, ls='--', c='0.5')
            totalRecs = 0
            totalCells = 0
            pieCounts = np.zeros((2,9))
            modsBefore = [[], []]
            modsAfter = [[], []]
            for n in range(nMice):
                #if i == 0:
                #    ax3.set_title(ephysSummaryAllAnimals[n][1],fontsize=8,color='0.2')
                nCells = len(ephysSummaryAllAnimals[n][2])
                if i == 0: print('nCells', nCells, ephysSummaryAllAnimals[n][0],ephysSummaryAllAnimals[n][1] )
                totalCells+= nCells
                # cellsPerMouse = []
                nDay=0
                idxSym=0
                inc=0
                for c in range(nCells):
                    nRecs = len(ephysSummaryAllAnimals[n][2][c][2])
                    if i == 0: print('nRecs', nRecs, ephysSummaryAllAnimals[n][2][c][0], ephysSummaryAllAnimals[n][2][c][1] )
                    totalRecs +=nRecs
                    # temp = np.zeros((7,nRecs))
                    # bTemp = []
                    recDay = ephysSummaryAllAnimals[n][2][c][1]
                    if recDay == oldRecDay:
                        nDay+=1
                        sameDay = True
                        inc=0
                    else: sameDay = False
                    oldP = [-2,0,0]
                    modRecB = [[],[]]
                    modRecA = [[],[]]
                    for j in range(nRecs):
                        if i ==0: print(j,ephysSummaryAllAnimals[n][2][c][2][j][0],ephysSummaryAllAnimals[n][2][c][2][j][1])
                        ephysPSTHDict = ephysSummaryAllAnimals[n][2][c][3][j]['allSteps']
                        kkeys = ['psth_swingOnsetAligned','psth_stanceOnsetAligned']
                        for k in range(2): # loop over swing and stance onset aligned psth's
                            ttime = ephysPSTHDict[i][kkeys[k]][0]
                            beforeMask = (ttime>before[0])&(ttime<before[1])
                            afterMask = (ttime>after[0])&(ttime<after[1])
                            psth = ephysPSTHDict[i][kkeys[k]][1]
                            pqq = ephysPSTHDict[i][kkeys[k]+'_5-50-95perentiles']
                            beforeEffect = True
                            afterEffect = True
                            #pdb.set_trace()
                            conditionHighB = psth[beforeMask]>pqq[2,beforeMask]
                            conditionLowB = psth[beforeMask]<pqq[0,beforeMask]
                            #pdb.set_trace()
                            consExcursions = 1 # means that at least (n+1) neighboring psth values have to cross the confidence interval
                            if (sum(conditionHighB)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighB ) if key])>consExcursions):
                                bb = np.mean(psth[beforeMask][psth[beforeMask]>pqq[2,beforeMask]]/pqq[1,beforeMask][psth[beforeMask]>pqq[2,beforeMask]])-1.
                            elif (sum(conditionLowB)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowB ) if key])>consExcursions):
                                bb = np.mean(psth[beforeMask][psth[beforeMask] < pqq[0,beforeMask]] / pqq[1,beforeMask][psth[beforeMask] < pqq[0,beforeMask]])-1.
                            else:
                                beforeEffect = False
                            conditionHighA = psth[afterMask]>pqq[2,afterMask]
                            conditionLowA = psth[afterMask]<pqq[0,afterMask]
                            if (sum(conditionHighA)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionHighA ) if key])>consExcursions):
                                aa =  np.mean(psth[afterMask][psth[afterMask]>pqq[2,afterMask]]/pqq[1,afterMask][psth[afterMask]>pqq[2,afterMask]])-1.
                            elif (sum(conditionLowA)>1) and any(np.array([ sum(1 for _ in group) for key, group in itertools.groupby( conditionLowA ) if key])>consExcursions):
                                aa = np.mean(psth[afterMask][psth[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth[afterMask] < pqq[0,afterMask]])-1.
                            else:
                                afterEffect = False
                                #aa = np.mean(psth[afterMask][psth[afterMask]<pqq[0,afterMask]] / pqq[1,afterMask][psth[afterMask] < pqq[0,afterMask]])-1.
                            pointPlot = [False,-1,-1]
                            if beforeEffect and afterEffect:
                                #ax3.plot(bb,aa,symList[(idxSym % len(symList))],color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,bb,aa]
                                if bb > 0 and aa > 0: pieCounts[k,0]+=1
                                elif bb>0 and aa<0: pieCounts[k,2]+=1
                                elif bb<0 and aa<0: pieCounts[k,4] += 1
                                elif bb<0 and aa>0: pieCounts[k,6] += 1
                            elif beforeEffect and not afterEffect:
                                #ax3.plot(bb, 0, symList[(idxSym % len(symList))], color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,bb,0]
                                if bb>0: pieCounts[k,1] += 1
                                elif bb<0: pieCounts[k,5] += 1
                            elif not beforeEffect and afterEffect:
                                #ax3.plot(0,aa, symList[(idxSym % len(symList))], color=cols[n],alpha=0.3+0.7*j/nRecs)
                                pointPlot = [True,0,aa]
                                if aa>0: pieCounts[k,7] += 1
                                elif aa<0: pieCounts[k,3] += 1
                            if beforeEffect:
                                modRecB[k].append(bb)
                            else:
                                modRecB[k].append(0)
                            if afterEffect:
                                modRecA[k].append(aa)
                            else:
                                modRecA[k].append(0)
                            #if j>0 and pointPlot[0] and oldP[0]==(j-1):
                            #    #ax3.plot([pointPlot[1],oldP[1]], [pointPlot[2], oldP[2]], color=cols[n], alpha=0.2)
                            #if pointPlot[0]:
                            #    oldP = [j,pointPlot[1],pointPlot[2]]

                        #ax3.plot(nDay+inc,ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2],symList[(idxSym % len(symList))],color=cols[n],clip_on=False)
                        #if not(nDay==1 and inc==0):
                        #    ax3.plot([prev[0],nDay+inc],[prev[1],ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2]],color=cols[n],alpha=0.5)
                        #prev = [nDay+inc,ephysPSTHDict[i]['psth_stanceOnsetAligned_allSteps_change'][2]]
                        inc+=1
                    for k in range(2):
                        modsAfter[k].append(modRecA[k])
                        modsBefore[k].append(modRecB[k])
                    oldRecDay = recDay
                    idxSym+=1
            #self.layoutOfPanel(ax3, xLabel='rel. change before swing-onset',yLabel='rel. change after swing-onset', xyInvisible=[False,(True if i>0 else False)])
            #ax3.spines['left'].set_position('zero')
            #ax3.spines['bottom'].set_position('zero')
            #ax3.set_ylim(-0.5,0.7)
            #ax3.set_xlim(-0.5,1)
            #pdb.set_trace()
            ########
            ax4 = plt.subplot(gssub2[i])
            #ax5 = plt.subplot(gssub2[i+4])
            #if i == 0:
            #    ax4.set_title('swing onset modulated')
            #    ax5.set_title('stance onset modulated')
            pieCounts[0,8] = totalRecs - sum(pieCounts[0,:-1])
            pieCounts[1,8] = totalRecs - sum(pieCounts[1, :-1])
            labels = '↑↑', '↑-', '↑↓', '-↓', '↓↓', '↓-', '↓↑', '-↑', '--'
            explode=(0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0)
            #pdb.set_trace()
            print(i,pieCounts)
            ccolors = ['darkred', 'orangered', 'lightcoral', 'blue', 'darkblue', 'skyblue', 'darkgreen', 'limegreen', 'C7']
            barWidth = 0.85
            startBarSwing = 0
            startBarStance = 0
            for k in range(len(pieCounts[0])-1):
                percSwing = 100*pieCounts[0,k]/totalRecs
                ax4.bar(0,percSwing,bottom=startBarSwing,color=ccolors[k],linewidth=0,edgecolor='white',width=barWidth)
                if percSwing>0:
                    ax4.annotate('%s  ' % np.round(percSwing, 1) + labels[k], xy=(-0.2, startBarSwing + 0.5*percSwing-1), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None,
                                 color='white')
                startBarSwing += percSwing
                percStance = 100*pieCounts[1, k]/totalRecs
                ax4.bar(2, percStance, bottom=startBarStance, color=ccolors[k], linewidth=0,edgecolor='white', width=barWidth)
                if percStance>0:
                    ax4.annotate('%s  ' % np.round(percStance, 1) + labels[k], xy=(1.8, startBarStance + 0.5*percStance-1), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None,
                                 color='white')
                startBarStance+=percStance

            ax4.annotate('%s %%' % np.round(startBarSwing,1), xy=(-0.2, startBarSwing+2), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None, color='black')
            ax4.annotate('%s %%' % np.round(startBarStance,1), xy=(1.8, startBarStance+2), annotation_clip=False, xytext=None, textcoords='data', fontsize=9, arrowprops=None, color='black')
            self.layoutOfPanel(ax4, yLabel='percentage modulated '+cellType+'s (%)', xyInvisible=[False, (True if i > 0 else False)])
            plt.xticks([0,2], ['swing onset modulated','stance onset modulated'])
            ax4.set_xlim(-1,3)
            ax4.set_ylim(0,55)
            #plt.xlabel("group")
            #ax4.pie(pieCounts[0], explode=explode, labels=labels, colors=ccolors, autopct='%1.1f%%',shadow=True, startangle=90)
            #ax4.axis('equal')
            if i ==3:
                textx = ax4.annotate('%s mice, %s %ss, %s recs' % (nMice,totalCells,cellType, totalRecs), xy=(1,47), annotation_clip=False,xytext=None, textcoords='data',fontsize=9,arrowprops=None,color='gray')
            #ax5.pie(pieCounts[1], explode=explode, labels=labels, colors=ccolors, autopct='%1.1f%%',shadow=True, startangle=90)
            #ax5.axis('equal')
            # gssub3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub2[i+4], hspace=0.1, wspace=0.15)
            # #for k in range(2):
            # ax5 = plt.subplot(gssub3[0])
            # for n in range(len(modsBefore[1])):
            #     if sum(modsBefore[1][n])>0:
            #         mm = np.asarray(modsBefore[1][n])
            #         #pdb.set_trace()
            #         ax5.plot((np.arange(len(mm)))[mm!=0]+1,mm[mm!=0],'o-',ms=2,lw=1,alpha=0.5)
            # ax6 = plt.subplot(gssub3[1])
            # for n in range(len(modsAfter[0])):
            #     if sum(modsAfter[0][n])>0:
            #         mm = np.asarray(modsAfter[0][n])
            #         ax6.plot((np.arange(len(mm)))[mm!=0]+1,mm[mm!=0],'o-',lw=1,ms=2,alpha=0.5)
            # self.layoutOfPanel(ax5,yLabel='before stance onset mod.',xyInvisible=[True,False])
            # self.layoutOfPanel(ax6,xLabel='trial #',yLabel='after swing onset mod.', xyInvisible=[False, False])
            # #ax3.xaxis.set_major_locator(xlocator)

        ########################
        fname = 'fig_ephys-psth-%s-swing-stance_v%s' % (cellType,figVersion)
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        # plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    ################################################################
    def fig_locomotorLearning(self,figVersion, stridePar,strideTraj,swingNumber,strideLength,strideDuration,indecisiveStrideFraction,swingSpeed, pawCoordination,allVariablesDf):

        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        # figure #################################
        fig_width = 17  # width in inches
        fig_height = 17  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 18, 'axes.titlesize': 18, 'font.size': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(3,1)

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.35, hspace=0.35)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.1)

        #panel names
        plt.figtext(0.012, 0.96, 'A', clip_on=False, color='black',  size=22)
        plt.figtext(0.35, 0.96, 'B', clip_on=False, color='black',  size=22)
        plt.figtext(0.66, 0.96, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.012, 0.66, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.35, 0.66, 'E', clip_on=False, color='black',  size=22)
        plt.figtext(0.66, 0.66, 'F', clip_on=False, color='black',  size=22)
        plt.figtext(0.012, 0.35, 'G', clip_on=False, color='black',  size=22)
        plt.figtext(0.35, 0.35, 'H', clip_on=False, color='black',  size=22)
        plt.figtext(0.66, 0.35, 'I', clip_on=False, color='black',  size=22)

        #divide figure in 3 lines
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], hspace=0.06, wspace=0.45)
        ############################Panel A : swing Number#########################
        #get average and stats
        (mean_swingNumber, std_swingNumber, sem_swingNumber)=groupAnalysis.getMeanStdNan(swingNumber[0])
        (df_swingNumber, mdf_swingNumber,md_paw_swingNumber,pawPvalues_recording_swingNumber,stars_swingNumber,tukey_paw_swingNumber)= groupAnalysis.pandaDataFrameAndMixedMLCompleteData(swingNumber[5], treatments=False, varName='swing_Number')
        df_swingNumber=df_swingNumber.dropna(how='any')
        swingNumberMean_per_Mouse = df_swingNumber.groupby(['recordingDay', 'mouse'])['measuredValue'].agg(['mean']).reset_index()

        swingNumberMeanSem = swingNumberMean_per_Mouse.groupby(['recordingDay'])['mean'].agg(['mean', 'sem']).reset_index()

        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'sex','paw', 'mouseId']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        #get data for five trials only, for visualization
        stridePar_Recordings_five_trials=stridePar_Recordings[stridePar_Recordings['trial']<=5]

        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        #perform stats Mixed Linear Model
        swingNumber_summary=groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingNumber')
        # swingNumber_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, 'swingNumber', 'mouseId', treatments=False)

        #first subplot top left (panel A)
        ax0 = plt.subplot(gssub1[0])
        nAnimals=df_swingNumber['mouse'].nunique()

        # model=Lmer('swingNumber ~ day+trial ', data=stridePar_Recordings)
        # print(model.fit())
        # pdb.set_trace()
        #per day data
        sns.lineplot(data=stridePar_Recordings_day, x='day', y='swingNumber', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax0, marker='o')
        #data for single mouse
        sns.lineplot(data=stridePar_Recordings_day, x='day', y='swingNumber', hue='mouseId', errorbar=None,
                     err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.3, legend=False, ax=ax0)
        #trial data as inset
        ax_inset = inset_axes(ax0, width="27%", height="25%",bbox_to_anchor=(0.1, 0.,0.9,0.9),
                   bbox_transform=ax0.transAxes,loc=1)

        sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y='swingNumber', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset, marker='o')


        #put N and stats pvalue stars
        ax0.text(0.95, 0.01, '(N=%s)' % nAnimals, ha='center', va='center', transform=ax0.transAxes, style='italic', fontsize=16, color='k')
        ax0.text(0.5, 1, '%s' %(swingNumber_summary['stars']['day']), ha='center', va='center', transform=ax0.transAxes, style='italic',fontfamily='serif', fontsize=16, color='k')
        ax_inset.text(0.5, 0.97, '%s'%(swingNumber_summary['stars']['trial'].replace('*','#')), ha='center', va='center', transform=ax_inset.transAxes, style='italic',fontfamily='serif', fontsize=10, color='0.4')
        #style

        ax0.xaxis.set_major_locator(MultipleLocator(1))
        ax_inset.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax_inset,xLabel='trial',yLabel='')
        self.layoutOfPanel(ax0, xLabel='session', yLabel='stride number (avg.)')


        # sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y='swingNumber', hue='mouseId', errorbar=None,
        #              err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.1, legend=False, ax=ax_inset)

        #plt.xticks([-0.5, 0, 0.5],[-0.5, 0, 0.5], size=6)
        #plt.xlim((-0.5, 0.5))
        #plt.yticks([5, 10], size=6)

        # for f in range (nAnimals):
        #     ax0.plot(np.arange(maxDays)+1,swingNumber[0][f][1], 'o-',ms=2, label=swingNumber[1][f][0], color=cmap(f/len(swingNumber[1])),alpha=0.1)
        #
        # ax0.plot(np.arange(maxDays)+1,mean_swingNumber,'-o',label=None, linewidth=1.5, ms=5,c='k')
        # sns.lineplot(swingNumberMean_per_Mouse, x='recordingDay', y='mean', hue='mouse', ls='-',ms=2, color=cmap(1/nAnimals), legend=False, alpha=0.1, ax=ax0)
        # ax0.axhline(np.percentile(df_swingNumber['measuredValue'],95), color='black', linestyle='solid', marker='s', markersize=1, markeredgewidth=1,markeredgecolor='black',alpha=0.4)

        # ax0.plot(swingNumberMeanSem['recordingDay'],swingNumberMeanSem['mean'],'-o',label=None, linewidth=1.5, ms=5,c='k')
        #ax0.fill_between(np.arange(maxDays)+1, mean_swingNumber - sem_swingNumber,mean_swingNumber + sem_swingNumber, color='0.6', alpha=0.05)

        # ax0.errorbar(np.arange(maxDays) + 1, mean_swingNumber, sem_swingNumber, capsize=3, linewidth=1, ms=5, c='k')
        # ax0.errorbar(swingNumberMeanSem['recordingDay'],swingNumberMeanSem['mean'],swingNumberMeanSem['sem'], capsize=3, linewidth=1, ms=5,c='k')

        # ax0.set_title('Stride number (all paws)',fontsize=14,color='k')
        ############################Panel B and C : rung Crossed#########################
        #get average and stats
        #rung crossed (with single paw)
        (mean_rungCrossed, std_rungCrossed, sem_rungCrossed) = groupAnalysis.getMeanStdNan(strideLength[0])
        (df_rung_crossed, mdf_strideLength, md_paw_strideLength, pawPvalues_strideLength,stars_strideLength,tukey_paw_strideLength) = groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideLength[4], treatments=False, varName='fractionstrideLength')
                #single paw
        (pawstrideLength, mean_PawstrideLength, sem_PawstrideLength)=groupAnalysis.getAverageSingleGroup(strideLength[3])
        # (swingDf, pawRungCrossedPvalues) = groupAnalysis.PawListToPandasDFAndMxLM(rungCrossed[4], trialValues=True, treatments=False)

        #plot rung crossed (all paws)
        #second subplot top middle (panel B)
        ax1 = plt.subplot(gssub1[1])
        #perform stats MLM
        strideLength_summary=groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingLength')
        # strideLength_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, 'swingLength', 'mouseId', treatments=False)
        #all animals per day
        sns.lineplot(data=stridePar_Recordings_day, x='day', y='swingLength', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax1, marker='o')
        #individual animals with alpha
        sns.lineplot(data=stridePar_Recordings_day, x='day', y='swingLength', hue='mouseId', errorbar=None,
                     err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.3, legend=False, ax=ax1)
        #trials as inset
        ax_inset_1 = inset_axes(ax1, width="27%", height="25%",bbox_to_anchor=(-0.5, 0.,0.9,0.9),
                   bbox_transform=ax1.transAxes,loc=1) #fig.add_axes([0.2, 0.2,0.2,0.2]) #plt.axes((0.6, 0.2, 0.2, 0.3), frameon=False)

        sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y='swingLength', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset_1, marker='o')
        #stars on graph
        ax1.text(0.5, 1, '%s' %(strideLength_summary['stars']['day']), ha='center', va='center', transform=ax1.transAxes, style='italic',fontfamily='serif', fontsize=16, color='k')
        ax_inset_1.text(0.5, 0.97, '%s' %strideLength_summary['stars']['trial'].replace('*','#'), ha='center', va='center', transform=ax_inset_1.transAxes, style='italic',fontfamily='serif', fontsize=10, color='0.4')

        #style
        ax_inset_1.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax_inset_1,xLabel='trial',yLabel='')
        self.layoutOfPanel(ax1, xLabel='session', yLabel='swing length (cm)')
        # majorLocator_x = MultipleLocator(1)
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        # rungCrossed_trial_average=[]
        # for f in range (nAnimals):
        #
        #     rungCrossed_trial_average.append(rungCrossed[1][f][1])
        # mice_rungCrossed_trial_average=np.nanmean(rungCrossed_trial_average, axis=0)
        # mice_rungCrossed_trial_sem= stats.sem(rungCrossed_trial_average, axis=0, nan_policy='omit')
        # # ax1.plot(np.arange(maxDays)+1,mean_rungCrossed,'-o',label=None, linewidth=2,c='k')
        # # pdb.set_trace()
        # xArray1 = []
        # for n in range(maxDays):
        #     xArray1 = np.repeat(np.arange(maxDays)+1, 5)
        #     # pdb.set_trace()
        #
        #     xArray1 = xArray1 + np.tile(np.arange(5) / 8, maxDays)
        #     xArray1 = xArray1.reshape(maxDays, 5)
        #
        #     rungCrossedAverage = mice_rungCrossed_trial_average[:,:5].flatten()
        #     rungCrossedSem=mice_rungCrossed_trial_sem[:,:5].flatten()
        #
        # ax1.plot(xArray1.flatten(), rungCrossedAverage,'-o',label=None, ms=4, linewidth=1,c='k')
        # ax1.errorbar(xArray1.flatten(),rungCrossedAverage,rungCrossedSem, ms=4, lw=0.5, capsize=1, c='k', alpha=0.2)
        # ax1.axhline(np.nanpercentile(df_rung_crossed['measuredValue'],90), color='black', linestyle='solid', marker='s', markersize=1, markeredgewidth=1,markeredgecolor='black',alpha=0.4)
        # pdb.set_trace()
        # ax.fill_between(np.arange(maxDays)+1, mean_rungCrossed - sem_rungCrossed,mean_rungCrossed + sem_rungCrossed, color='0.6', alpha=0.2)


        # ax1.set_title('Fraction of ' + r'$\geqq$' + '2 rungs crossed \n (individual trials, all paws)',fontsize=14,color='k')


        #plot rung crossed ( paws specific)
        #third subplot panel C (top right)
        ax2 = plt.subplot(gssub1[2])
        #per day all animals individual paw as hue
        sns.lineplot(data=stridePar_Recordings, x='day', y='swingLength', hue='paw',
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax2, marker='o')
        ax_inset_2 = inset_axes(ax2, width="27%", height="25%",bbox_to_anchor=(-0.5, 0.05,0.9,0.9),
                   bbox_transform=ax2.transAxes,loc=1)
        sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y='swingLength', hue='paw',
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset_2, marker='o')


        #regroup stats in star_label list, trial effect as ###
        star_label=[]
        star_label_inset = []
        for i in range(4):
            star_label.append(pawId[i]+' '+strideLength_summary['paw_stars'][pawId[i]]['day'])
            star_label_inset.append(pawId[i]+' '+strideLength_summary['paw_stars'][pawId[i]]['trial'].replace('*', '#'))
        #stars as legend
        ax2.legend(star_label, bbox_to_anchor=(0.86,0.78), frameon=False, fontsize=14)
        ax_inset_2.legend(star_label_inset,bbox_to_anchor=(0.93,0.95), frameon=False, fontsize=10)
        # ax_inset_2.text(0.5, 0.97, '%s' %strideLength_summary['stars']['trial'].replace('*','#'), ha='center', va='center', transform=ax_inset_2.transAxes, style='italic',fontfamily='serif', fontsize=10, color='0.4')
        self.layoutOfPanel(ax2, xLabel='session', yLabel='swing length (cm)')
        self.layoutOfPanel(ax_inset_2,xLabel='trial',yLabel='')
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax_inset_2.xaxis.set_major_locator(MultipleLocator(1))
            # ax2.plot(np.arange(10) + 1, mean_PawStrideLength[:,i], 'o-', label='%s %s'%(pawId[i],pawStarStrideLength), linewidth=1.5, ms=5, c='C%s'%i)
            # #ax2.fill_between(np.arange(10) + 1, mean_PawStrideLength[:,i] - sem_PawStrideLength[:,i], mean_PawStrideLength[:,i] + sem_PawStrideLength[:,i], color='C%s'%i, alpha=0.05)
            # ax2.errorbar(np.arange(maxDays) + 1, mean_PawStrideLength[:,i], sem_PawStrideLength[:,i], capsize=3, linewidth=1, ms=5,c='C%s'%i)

        # ax2.set_title('Fraction of ' + r'$\geqq$' + '2 rungs crossed \n (individual paws)', fontsize=14, color='k')

        # ax2.set_title('Fraction of ' + r'$\geqq$' +'2 rungs crossed (paw specific)')


        #
        # (pawStarStrideLength) = groupAnalysis.starMultiplier(pawPvalues_strideLength[i][0])
        ############################Panel D and E : stride Duration#########################
        #get average and stats
        # swing stance duration

        # (mean_swingDuration, std_swingDuration, sem_swingDuration) = groupAnalysis.getMeanStdNan(strideDuration[0])
        #
        # # stance
        # (mean_stanceDuration, std_stanceDuration, sem_stanceDuration) = groupAnalysis.getMeanStdNan(strideDuration[2])
        # (df_stanceDuration, mdf_stanceDuration,md_paw_stanceDuration, pawPvalues_stanceDuration,  stars_stanceDuration, tukey_paw_stanceDuration) = groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideDuration[9],  treatments=False, varName='stance_Duration')

        #2nd line of the figure, contains 3 columns : panels D,E,F
        gssub2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], hspace=0.06, wspace=0.45)

        #regroup parameters to plot in list, easier to plot
        parameters=['swingDuration', 'swingSpeed', 'indecisiveFraction']
        #Y axes labels
        parLabels=['duration (s)', 'swing speed (cm/s)', 'miss steps (fraction)']
        #plot them one by one

        for par in range(len(parameters)):
            #perform stats for each parameters
            par_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings,parameters[par])
            # par_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par], 'mouseId', treatments=False)
            #subplots each parameter as columns
            ax3 = plt.subplot(gssub2[par])
            #all animal per day
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue=None, errorbar=('se'), err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax3, marker='o')
            #individual animals
            if par!=0:
                sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId', errorbar=None, err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.3, legend=False, ax=ax3)
                ax3.text(0.5, 1, '%s' % (par_summary['stars']['day']), ha='center', va='center',transform=ax3.transAxes,style='italic', fontfamily='serif', fontsize=16, color='k')
            if par==0:
                # trials as inset
                ax3b=ax3.twinx()
                sns.lineplot(data=stridePar_Recordings_day, x='day', y='stanceDuration', hue=None,
                             errorbar=('se'), err_style='bars',
                             err_kws={'capsize': 3, 'linewidth': 1}, color='0.6', ax=ax3b, marker='o', lw=1.2)
                ax3b.spines['left'].set_visible(False)
                ax3b.spines['top'].set_visible(False)
                ax3b.spines['bottom'].set_visible(False)
                ax3b.spines['right'].set_color('0.6')
                ax3b.tick_params(axis='y', colors='0.6')
                ax3b.yaxis.label.set_color('0.6')
                ax3b.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                par_summary_stance = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'stanceDuration')
                ax3b.legend([f'stance  {par_summary_stance["stars"]["day"]} {par_summary_stance["stars"]["trial"].replace("*", "#")}'], frameon=False, loc='upper left',  bbox_to_anchor=(0.1,0.93))
                ax3.legend([f'swing {par_summary["stars"]["day"]} {par_summary["stars"]["trial"].replace("*", "#")}'], frameon=False, loc='upper left', bbox_to_anchor=(0.1,1))
                ax3b.set_ylabel('')
                ax3.yaxis.set_major_locator(MultipleLocator(1))

                # self.layoutOfPanel(ax3b, xLabel=None, yLabel='stance duration (s)')
                # ax_inset_3= inset_axes(ax3, width="27%", height="25%", bbox_to_anchor=(0.1, 0., 0.9, 0.9),
                #                         bbox_transform=ax3.transAxes,
                #                         loc=1)  # fig.add_axes([0.2, 0.2,0.2,0.2]) #plt.axes((0.6, 0.2, 0.2, 0.3), frameon=False)
                #
                # sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y=parameters[par], hue=None,
                #              errorbar=('se'), err_style='bars',
                #              err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset_3, marker='o')
                # # stars on graph
                # ax_inset_3.text(0.5, 0.97, '%s' % par_summary['stars']['trial'].replace('*', '#'), ha='center',
                #                 va='center', transform=ax_inset_3.transAxes, style='italic', fontfamily='serif',
                #                 fontsize=10, color='0.4')
                # ax_inset_3.xaxis.set_major_locator(MultipleLocator(1))
                # self.layoutOfPanel(ax_inset_3, xLabel='trial', yLabel='')
            #day effect stars

            #style
            self.layoutOfPanel(ax3, xLabel='session', yLabel=parLabels[par])
            # majorLocator_x = MultipleLocator(1)
            ax3.xaxis.set_major_locator(MultipleLocator(1))
            if par == 0:
                ax3.yaxis.set_major_locator(MultipleLocator(0.01))
            # (df_swingDuration, mdf_swingDuration, md_paw_swingDuration, pawPvalues_swingDuration, stars_swingDuration,
            #  tukey_paw_swingDuration) = groupAnalysis.pandaDataFrameAndMixedMLCompleteData(strideDuration[8],
            #                                                                                treatments=False,
            #                                                                                varName='swing_Duration')

        ###################################paw coordination panel #############################"
        #get swing counts
        swingCountNorm=pawCoordination[3]

        #third line of figure is for paw coordination and heatmap
        gssub3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], hspace=0.06, wspace=0.45)#, width_ratios=[1.9,1.9,2])
        #divide second column in two line
        gssub5 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub3[1], hspace=0.1, wspace=0.25)
        #swing onset median for FR is ax5, on top, IQR is for FL on the bottom
        ax4=plt.subplot(gssub5[1])
        ax5=plt.subplot(gssub5[0])
        #perform stats
        iqr_summary=groupAnalysis.perform_mixedlm(stridePar_Recordings, 'stanceOn_iqr_25_75_ref_FL')
        stanceOnMed_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingOnMedian_FL')

        #get data for FL and FR
        FL_df = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[0])]
        FR_df = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[1])]
        #get the data with only five trials for visualization
        FL_df_trials = stridePar_Recordings_five_trials[(stridePar_Recordings_five_trials['paw'] == pawId[0])]
        FR_df_trials = stridePar_Recordings_five_trials[(stridePar_Recordings_five_trials['paw'] == pawId[1])]
        #plot iqr per day

        sns.lineplot(data=FL_df, x='day', y='stanceOn_iqr_25_75_ref_FL', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='C0', ax=ax4, marker='o')
        #plot swing on median per day
        sns.lineplot(data=FR_df, x='day', y='swingOnMedian_FL', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='C1', ax=ax5, marker='o')

        #plot inset for trial data
        ax_inset_iqr = inset_axes(ax4, width="27%", height="25%",
                                   bbox_to_anchor=(-0.4, -0.3, 0.9, 0.9),
                                   bbox_transform=ax4.transAxes,
                                   loc=1)
        ax_inset_stanceOn_med = inset_axes(ax5, width="27%", height="25%",
                                   bbox_to_anchor=(-0.4, 0.1, 0.9, 0.9),
                                   bbox_transform=ax5.transAxes,
                                   loc=1)

        sns.lineplot(data=FL_df_trials, x='trial', y='stanceOn_iqr_25_75_ref_FL', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='C0', ax=ax_inset_iqr, marker='o')
        sns.lineplot(data=FR_df_trials, x='trial', y='swingOnMedian_FL', hue=None,
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, color='C1', ax=ax_inset_stanceOn_med, marker='o')

        #put stats stars
        ax4.text(0.5, 1, '%s' % (iqr_summary['paw_stars']['FL']['day']), ha='center', va='center', transform=ax4.transAxes,
                 style='italic', fontfamily='serif', fontsize=16, color='k')
        ax5.text(0.5, 1, '%s' % (stanceOnMed_summary['paw_stars']['FR']['day']), ha='center', va='center', transform=ax5.transAxes,
                 style='italic', fontfamily='serif', fontsize=16, color='k')

        ax_inset_iqr.text(0.5, 0.97, '%s' % (iqr_summary['paw_stars']['FL']['trial'].replace('*', '#')), ha='center',
                           va='center',
                           transform=ax_inset_iqr.transAxes, style='italic', fontfamily='serif', fontsize=8,
                           color='0.4')

        ax_inset_stanceOn_med.text(0.5, 0.97, '%s' % (stanceOnMed_summary['paw_stars']['FR']['trial'].replace('*', '#')), ha='center',
                           va='center',
                           transform=ax_inset_stanceOn_med.transAxes, style='italic', fontfamily='serif', fontsize=8,
                           color='0.4')

        self.layoutOfPanel(ax4, xLabel='session', yLabel='FL  stance onset\n IQR', xyInvisible=[False, False],Leg=[1, 9])
        self.layoutOfPanel(ax5, xLabel='session', yLabel='FR swing onset\n median', xyInvisible=[True, False],Leg=[1, 9])
        ax4.xaxis.set_major_locator(MultipleLocator(1))
        ax5.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax_inset_iqr, xLabel='trial', yLabel='')
        self.layoutOfPanel(ax_inset_stanceOn_med, xLabel='trial', yLabel='')
        ax_inset_iqr.xaxis.set_major_locator(MultipleLocator(1))
        ax_inset_stanceOn_med.xaxis.set_major_locator(MultipleLocator(1))



        #panels G with probability of swing as 1st column of third row
        gssub4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub3[0], hspace=0.15, wspace=0.25)
        #those are the p of swing for trials (ax10,11) and days (ax8, ax9)
        ax8=plt.subplot(gssub4[2])
        ax9=plt.subplot(gssub4[3])
        ax10=plt.subplot(gssub4[0])
        ax11=plt.subplot(gssub4[1])

        #get swing offset and inset
        swingOffset_mice=pawCoordination[7]
        swingOnset_mice = pawCoordination[6]
        #prepare array for average and sum of data
        swingCountNorm_sum=[]
        swingCountNorm_sum_trial=[]
        swingOnsetMedian_days=[]
        swingOnsetMedian_trials=[]
        swingOffsetMedian_days=[]
        swingOffsetMedian_trials=[]
        swingOffset_25th_days = []
        swingOffset_75th_days = []
        swingOffset_25th_trials =[]
        swingOffset_75th_trials =[]
        #get median and percentiles for swing onset and offsets
        for m in range(len(swingCountNorm)):
            swingCountNorm_sum.append(np.nansum(swingCountNorm[m][1], axis=1))
            swingCountNorm_sum_trial.append(np.nansum(swingCountNorm[m][1], axis=0))
            swingOnsetMedian_days.append(np.nanmedian(swingOnset_mice[m][1],axis=1))
            swingOnsetMedian_trials.append(np.nanmedian(swingOnset_mice[m][1],axis=0))
            swingOffsetMedian_days.append(np.nanmedian(swingOffset_mice[m][1],axis=1))
            swingOffsetMedian_trials.append(np.nanmedian(swingOffset_mice[m][1],axis=0))
            swingOffset_25th_days.append(np.nanpercentile(swingOffset_mice[m][1],25,axis=1))
            swingOffset_75th_days.append(np.nanpercentile(swingOffset_mice[m][1], 75, axis=1))
            swingOffset_25th_trials.append(np.nanpercentile(swingOffset_mice[m][1],25,axis=0))
            swingOffset_75th_trials.append(np.nanpercentile(swingOffset_mice[m][1], 75, axis=0))
        #perform median for all mice
        mice_swingCountNorm_sum=np.sum(swingCountNorm_sum, axis=0)
        mice_swingCountNorm_sum_trial = np.sum(swingCountNorm_sum_trial, axis=0)
        mice_swingOnsetMedian_days=np.nanmedian(swingOnsetMedian_days, axis=0)
        mice_swingOnsetMedian_trials=np.nanmedian(swingOnsetMedian_trials, axis=0)
        mice_swingOffsetMedian_days=np.nanmedian(swingOffsetMedian_days, axis=0)
        mice_swingOffsetMedian_trials=np.nanmedian(swingOffsetMedian_trials, axis=0)
        mice_swingOffset_25th_days=np.nanpercentile(swingOffset_25th_days,25, axis=0)
        mice_swingOffset_75th_days = np.nanpercentile(swingOffset_75th_days,75, axis=0)
        mice_swingOffset_25th_trials=np.nanpercentile(swingOffset_25th_trials,25, axis=0)
        mice_swingOffset_75th_trials =np.nanpercentile(swingOffset_75th_trials,75, axis=0)
        #get datas for trial 1/5 and day 1/10
        for i in range(4):
            if i==0 or i==1:
                swingProb_day1=mice_swingCountNorm_sum[:,i,:][0]/np.max(mice_swingCountNorm_sum[0])
                swingProb_day10=mice_swingCountNorm_sum[:,i,:][9]/np.max(mice_swingCountNorm_sum[9])
                swingProb_trial1=mice_swingCountNorm_sum_trial[:,i,:][0]/np.max(mice_swingCountNorm_sum_trial[0])
                swingProb_trial5=mice_swingCountNorm_sum_trial[:,i,:][4]/np.max(mice_swingCountNorm_sum_trial[4])
                mice_swingOnsetMedian_days1=np.nanmedian(mice_swingOnsetMedian_days[:,i,:][0])
                mice_swingOnsetMedian_days10=np.nanmedian(mice_swingOnsetMedian_days[:,i,:][9])
                mice_swingOnsetMedian_trial1=np.nanmedian(mice_swingOnsetMedian_trials[:,i,:][0])
                mice_swingOnsetMedian_trial5=np.nanmedian(mice_swingOnsetMedian_trials[:,i,:][5])
                mice_swingOffsetMedian_days1=np.nanmedian(mice_swingOffsetMedian_days[:,i,:][0])
                mice_swingOffsetMedian_days10=np.nanmedian(mice_swingOffsetMedian_days[:,i,:][9])
                mice_swingOffsetMedian_trial1=np.nanmedian(mice_swingOffsetMedian_trials[:,i,:][0])
                mice_swingOffsetMedian_trial5=np.nanmedian(mice_swingOffsetMedian_trials[:,i,:][5])
                mice_swingOffset_25th_days1=np.nanmedian(mice_swingOffset_25th_days[:,i,:][0])
                mice_swingOffset_75th_days1 = np.nanmedian(mice_swingOffset_75th_days[:, i, :][0])
                mice_swingOffset_25th_trials1=np.nanmedian(mice_swingOffset_25th_trials[:,i,:][0])
                mice_swingOffset_75th_trials1=np.nanmedian(mice_swingOffset_75th_trials[:,i,:][0])
                mice_swingOffset_25th_days10=np.nanmedian(mice_swingOffset_25th_days[:,i,:][9])
                mice_swingOffset_75th_days10 = np.nanmedian(mice_swingOffset_75th_days[:, i, :][9])
                mice_swingOffset_25th_trials5=np.nanmedian(mice_swingOffset_25th_trials[:,i,:][5])
                mice_swingOffset_75th_trials5=np.nanmedian(mice_swingOffset_75th_trials[:,i,:][5])
                # plot swing probabilities
                ax8.plot(np.linspace(0,100,51),swingProb_day1,ms=2, color='C%s'%i)
                ax9.plot(np.linspace(0,100,51),swingProb_day10, ms=2, color='C%s'%i)

                ax10.plot(np.linspace(0,100,51),swingProb_trial1,ms=2, color='C%s'%i, label=pawId[i])
                ax11.plot(np.linspace(0,100,51),swingProb_trial5, ms=2, color='C%s'%i)
                ax9.legend(frameon=False, loc='upper right', fontsize=18)

                x = np.linspace(0, 100, 51)
                #annotate with lines and fill iqr range
                if i==1:
                    ax8.axvline(mice_swingOnsetMedian_days1*100, color='C%s'%i, ls='--', lw=1, alpha=0.5)
                    ax9.axvline(mice_swingOnsetMedian_days10*100, color='C%s'%i, ls='--', lw=1, alpha=0.5)
                    # ax8.axvline(mice_swingOffsetMedian_days1*100, color='C%s'%i, ls='--', lw=1, alpha=0.5)
                    # ax9.axvline(mice_swingOffsetMedian_days10*100, color='C%s'%i, ls='--', lw=1, alpha=0.5)
                    # ax9.annotate('swing onset \n median', (53, 0.85), fontsize=12, color='C1')
                    ax10.axvline(mice_swingOnsetMedian_trial1 * 100, color='C%s' % i, ls='--', lw=1, alpha=0.5)
                    ax11.axvline(mice_swingOnsetMedian_trial5 * 100, color='C%s' % i, ls='--', lw=1, alpha=0.5)
                if i ==0:
                    ax8.fill_between(x,0,swingProb_day1, where=((x>mice_swingOffset_25th_days1*100) & (x<mice_swingOffset_75th_days1*100)),alpha=0.2)
                    ax9.fill_between(x, 0, swingProb_day10, where=((x > mice_swingOffset_25th_days10 * 100) & (x < mice_swingOffset_75th_days10 * 100)),alpha=0.2)
                    ax10.fill_between(x, 0, swingProb_trial1, where=((x > mice_swingOffset_25th_trials1 * 100) & (x < mice_swingOffset_75th_trials1 * 100)), alpha=0.2)
                    ax11.fill_between(x, 0, swingProb_trial5, where=((x > mice_swingOffset_25th_trials5 * 100) & (x < mice_swingOffset_75th_trials5 * 100)),alpha=0.2)

        ax8.xaxis.set_major_locator(MultipleLocator(50))
        ax9.xaxis.set_major_locator(MultipleLocator(50))
        self.layoutOfPanel(ax8, xLabel='% FL stride', yLabel='p(swing)', Leg=[1, 9])
        self.layoutOfPanel(ax9, xLabel='% FL stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[False, True])
        self.layoutOfPanel(ax10, xLabel='% FL stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[True, False])
        self.layoutOfPanel(ax11, xLabel='% FL stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[True, True])
        ax9.legend(loc="upper right", frameon=False, fontsize=8)
        ax10.legend(loc="upper right", frameon=False, fontsize=10)
        # ax9.legend(loc="upper left", frameon=False, fontsize=12)
        ax8.set_title('session 1')
        ax9.set_title('session 10')
        ax10.set_title('trial 1')
        ax11.set_title('trial 5')
        ##########################################Correlation map ####################################################################
        #correlation map is 3rd column of third row
        ax14=plt.subplot(gssub3[2])
        #define Y axis name
        YAxisList = ['FL swing number', 'FL swing speed',  'FL swing length',
                        'FL stance duration', 'FL swing duration','FL fraction of \n miss steps', 'FL stance onset IQR' ]#, 'wheel speed']
        #get the values to be correlated (FL paw parameters)
        FL_allVarMatrix=FL_df[['swingNumber', 'swingSpeed', 'swingLength',
             'stanceDuration',   'swingDuration','indecisiveFraction',
                        'stanceOn_iqr_25_75_ref_FL' ]]#, 'LinWheelSpeed_avg']]
        #get pvalues and mask of significance
        corr_pvalues_all, rvalues, annot, mask = groupAnalysis.calculate_correlation_pvalues(FL_allVarMatrix)
        mask2=FL_allVarMatrix.corr()!=1
        # FLcorrUpMatrix=np.triu(FL_allVarMatrix.corr())
        # mask3 = mask |FLcorrUpMatrix & mask2
        mask3 = mask  & mask2
        #plot heatmap
        FL_VarHeatMap = sns.heatmap(FL_allVarMatrix.corr(),xticklabels=YAxisList, yticklabels=YAxisList, annot=False, mask=(mask3), fmt='',cmap=sns.color_palette("PiYG_r", 50), cbar_kws={"shrink": 0.8,'label': 'correlation coefficient'},ax=ax14,square = True)
        cbar = ax14.collections[0].colorbar
        cbar.set_ticks([-0.5,0,0.5,1])
        cbar.set_ticklabels(['-0.5','0','0.5','1'])
        #cbar = fig.colorbar(ax14, ticks=[-0.5, 0, 0.5, 1])
        #background is white
        FL_VarHeatMap.set_facecolor('white')
        #colormap tick size
        FL_VarHeatMap.figure.axes[-1].tick_params(labelsize=13)
        #ticks labels parameters
        ax14.set_xticklabels(ax14.get_xticklabels(), rotation=80,fontsize=13)
        ax14.set_yticklabels(ax14.get_yticklabels(), rotation=0,fontsize=13)
        plt.setp(ax14.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor")
        # ax14.set_title('Correlation map of walking parameters (FL paw)', fontsize=14, color='k')
        # create a list to store the results
        # pdb.set_trace()
        # results_list = [mdf_swingNumber.summary().tables[1],mdf_strideLength.summary().tables[1],mdf_swingDuration.summary().tables[1], mdf_stanceDuration.summary().tables[1], mdf_swingSpeed.summary().tables[1], mdf_indecisiveStrideFraction.summary().tables[1], stanceOnsetMedian_mice_mdf.summary().tables[1] ]
        #
        # # define a list of titles for each result
        # titles = ['Swing_number',  'Fraction_of_more_than_2 rungs_crossed', 'Swing_duration',  'Stance_duration', 'Swing_Speed', 'Fraction_of_indecisive_strides', 'Median_Swing_Offset']
        #
        # # create a DataFrame from the results list
        # # df_stats = pd.DataFrame(results_list)
        # df_stats = pd.concat(results_list, keys=titles)
        # # create a MultiIndex for the DataFrame with the titles as the first level and the column names as the second level
        # # df_stats.index = pd.MultiIndex.from_product([['Mixed_Linear_Model_Results'],  df_stats.columns])
        # df_stats = df_stats.rename(index={0: "Mixed_Linear_Model_Results"})
        # ##################paw specific results##################################################
        # # save the DataFrame to a CSV file
        # paw_results_list=[[], [], [], []]
        # paw_titles=[[], [], [], []]
        # # df_stats_paw=[[], [], [], []]
        # df_stats_paw=[]
        # for p in range(4):
        #     paw_results_list[p].append([md_paw_swingNumber[p].summary().tables[1], md_paw_strideLength[p].summary().tables[1],md_paw_swingDuration[p].summary().tables[1],md_paw_stanceDuration[p].summary().tables[1],md_paw_swingSpeed[p].summary().tables[1],md_paw_indecisiveStrideFraction[p].summary().tables[1],md_paw_stanceOnsetMedian[p].summary().tables[1] ])
        # # define a list of titles for each result
        #     paw_titles[p].append(['Swing_number_%s'%(pawId[p]),  'Fraction_of_more_than_2 rungs_crossed_%s'%(pawId[p]), 'Swing_duration_%s'%(pawId[p]),  'Stance_duration_%s'%(pawId[p]), 'Swing_Speed_%s'%(pawId[p]), 'Fraction_of_indecisive_strides_%s'%(pawId[p]), 'Median_Swing_Offset_%s'%(pawId[p])])
        # # create a DataFrame from the results list for each paw
        #     df_stats_paw.append(pd.concat(paw_results_list[p][0], keys=paw_titles[p][0]))
        # df_stats_single_paw=pd.concat(df_stats_paw)
        #

        # df_stats.to_csv('ephy_behavior_results_statistics.csv')
        # df_stats_single_paw.to_csv('ephy_behavior_results_statistics_single_paw.csv')
        # with open('results_statistics.tex', 'w') as f:
        #     f.write(df_stats.style.to_latex())
        # with open('results_statistics_single_paw.tex', 'w') as f:
        #     f.write(df_stats_single_paw.style.to_latex())

        #figure name
        fname = 'fig_locomotorLearning_v%s' % figVersion
        #figure output format
        plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')
    ############################################################
    def fig_locomotorLearning_supp_FR(
            self,
            figVersion,
            stridePar,
            strideTraj,

    ):
        """
        Supplementary figure for locomotor learning (FR reference).

        Panels:
          - FR swing onset median & FR stance onset IQR (with trial insets), stats from MixedLM
          - Probability of swing across stride percentage (days 1 & 10, trials 1 & 5) with annotations
          - Correlation heatmap for FR-related variables

        Inputs are kept identical to the original function to preserve behavior.
        """
        # ---------------------------- Styling & figure scaffold ----------------------------
        paw_ids = ['FL', 'FR', 'HL', 'HR']

        fig_width, fig_height = 16, 6.5
        rcParams.update({
            'axes.labelsize': 18, 'axes.titlesize': 18, 'font.size': 18,
            'xtick.labelsize': 15, 'ytick.labelsize': 15,
            'figure.figsize': [fig_width, fig_height],
            'savefig.dpi': 600, 'axes.linewidth': 1.3,
            'ytick.major.size': 4, 'xtick.major.size': 4,
            'font.sans-serif': 'Arial'
        })

        fig = plt.figure()
        grid_main = gridspec.GridSpec(1, 1)
        grid_main.update(wspace=0.35, hspace=0.35)
        plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.2)

        # ---------------------------- Data aggregations used across panels ----------------------------
        # Trial/day/paw/mouse aggregates for plotting + MixedLM
        stride_params_by_recording = (
            stridePar.groupby(['trial', 'day', 'sex', 'paw', 'mouseId']).mean().reset_index()
        )
        # Restrict to the first five trials for insets/visualization
        stride_params_five_trials = stride_params_by_recording.query('trial <= 5')


        # ---------------------------- Panel group layout ----------------------------
        # 3 columns: (1) swing prob plots, (2) FR med/iqr with insets, (3) correlation heatmap
        grid_cols = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid_main[0], hspace=0.06, wspace=0.45)

        # Middle column (2 sub-rows): top = swing onset median (FR), bottom = stance onset IQR (FR)
        grid_fr_stats = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_cols[1], hspace=0.1, wspace=0.25)
        ax_swing_on_med = plt.subplot(grid_fr_stats[0])  # top
        ax_iqr = plt.subplot(grid_fr_stats[1])  # bottom

        # ---------------------------- MixedLM stats (kept identical) ----------------------------
        # Stats for FR stance onset IQR and FR swing onset median
        iqr_summary, iqr_csv = groupAnalysis.perform_mixedlm(
            stride_params_by_recording, 'swingOff_iqr_25_75_ref_FR',csv_folder='manuscriptFigures'
        )
        stanceOnMed_summary, stance_csv = groupAnalysis.perform_mixedlm(
            stride_params_by_recording, 'swingOnMedian_ref_FR',csv_folder='manuscriptFigures'
        )

        # ---------------------------- Data slices for FR/FL & trial-only views ----------------------------
        df_fl = stride_params_by_recording.query('paw == @paw_ids[0]')
        df_fr = stride_params_by_recording.query('paw == @paw_ids[1]')

        df_fl_trials = stride_params_five_trials.query('paw == @paw_ids[0]')
        df_fr_trials = stride_params_five_trials.query('paw == @paw_ids[1]')

        # ---------------------------- FR stats plots (day trends + trial insets) ----------------------------
        # FR stance onset IQR by day (orange)
        sns.lineplot(
            data=df_fr, x='day', y='swingOff_iqr_25_75_ref_FR', errorbar='se',
            err_style='bars', err_kws={'capsize': 3, 'linewidth': 1},
            color='C1', ax=ax_iqr, marker='o'
        )

        # FR swing onset median by day (blue)
        sns.lineplot(
            data=df_fl, x='day', y='swingOnMedian_ref_FR', errorbar='se',
            err_style='bars', err_kws={'capsize': 3, 'linewidth': 1},
            color='C0', ax=ax_swing_on_med, marker='o'
        )

        # Insets for trial-level visualization
        inset_iqr = inset_axes(
            ax_iqr, width="27%", height="25%",
            bbox_to_anchor=(-0.4, -0.3, 0.9, 0.9),
            bbox_transform=ax_iqr.transAxes, loc=1
        )
        inset_swing_on_med = inset_axes(
            ax_swing_on_med, width="27%", height="25%",
            bbox_to_anchor=(-0.4, 0.1, 0.9, 0.9),
            bbox_transform=ax_swing_on_med.transAxes, loc=1
        )

        sns.lineplot(
            data=df_fr_trials, x='trial', y='swingOff_iqr_25_75_ref_FR', errorbar='se',
            err_style='bars', err_kws={'capsize': 3, 'linewidth': 1},
            color='C1', ax=inset_iqr, marker='o'
        )
        sns.lineplot(
            data=df_fl_trials, x='trial', y='swingOnMedian_ref_FR', errorbar='se',
            err_style='bars', err_kws={'capsize': 3, 'linewidth': 1},
            color='C0', ax=inset_swing_on_med, marker='o'
        )

        # Stats stars annotations (same keys as original)
        ax_iqr.text(
            0.5, 1, f"{iqr_summary['paw_stars']['FR']['day']}",
            ha='center', va='center', transform=ax_iqr.transAxes,
            style='italic', fontfamily='serif', fontsize=16, color='k'
        )
        ax_swing_on_med.text(
            0.5, 1, f"{stanceOnMed_summary['paw_stars']['FL']['day']}",
            ha='center', va='center', transform=ax_swing_on_med.transAxes,
            style='italic', fontfamily='serif', fontsize=16, color='k'
        )

        inset_iqr.text(
            0.5, 0.97, f"{iqr_summary['paw_stars']['FR']['trial'].replace('*', '#')}",
            ha='center', va='center', transform=inset_iqr.transAxes,
            style='italic', fontfamily='serif', fontsize=8, color='0.4'
        )
        inset_swing_on_med.text(
            0.5, 0.97, f"{stanceOnMed_summary['paw_stars']['FL']['trial'].replace('*', '#')}",
            ha='center', va='center', transform=inset_swing_on_med.transAxes,
            style='italic', fontfamily='serif', fontsize=8, color='0.4'
        )

        # Axis cosmetics for FR stats
        self.layoutOfPanel(ax_iqr, xLabel='session', yLabel='FR  stance onset\n IQR', xyInvisible=[False, False],
                           Leg=[1, 9])
        self.layoutOfPanel(ax_swing_on_med, xLabel='session', yLabel='FL swing onset\n median',
                           xyInvisible=[True, False], Leg=[1, 9])
        ax_iqr.xaxis.set_major_locator(MultipleLocator(1))
        ax_swing_on_med.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(inset_iqr, xLabel='trial', yLabel='')
        self.layoutOfPanel(inset_swing_on_med, xLabel='trial', yLabel='')
        inset_iqr.xaxis.set_major_locator(MultipleLocator(1))
        inset_swing_on_med.xaxis.set_major_locator(MultipleLocator(1))

        # ---------------------------- Swing probability panels (left column) ----------------------------
        grid_prob = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid_cols[0], hspace=0.15, wspace=0.25)
        ax_day1 = plt.subplot(grid_prob[2])
        ax_day10 = plt.subplot(grid_prob[3])
        ax_trial1 = plt.subplot(grid_prob[0])
        ax_trial5 = plt.subplot(grid_prob[1])

        x_percent = np.linspace(0, 100, 51)

        # Plot for FL and FR (as in original loop over i in range(2))
        for idx in range(2):
            paw_name = paw_ids[idx]
            color = f'C{idx}'

            # Extract swing probability arrays for specific days/trials
            swing_prob_trial1 = strideTraj.query('trial == 1 and paw == @paw_name')['swingProb_ref_FR'].values
            swing_prob_trial5 = strideTraj.query('trial == 5 and paw == @paw_name')['swingProb_ref_FR'].values
            swing_prob_day1 = strideTraj.query('day == 1 and paw == @paw_name')['swingProb_ref_FR'].values
            swing_prob_day10 = strideTraj.query('day == 10 and paw == @paw_name')['swingProb_ref_FR'].values

            # Medians for annotations (swing onset / stance onset IQR bands)
            sOff_iqr_25_day1 = np.nanmedian(
                stridePar.query('day == 1 and paw == @paw_name')['swingOff_iqr_25_ref_FR'].values)
            sOff_iqr_25_day10 = np.nanmedian(
                stridePar.query('day == 10 and paw == @paw_name')['swingOff_iqr_25_ref_FR'].values)
            sOff_iqr_25_trial1 = np.nanmedian(
                stridePar.query('trial == 1 and paw == @paw_name')['swingOff_iqr_25_ref_FR'].values)
            sOff_iqr_25_trial5 = np.nanmedian(
                stridePar.query('trial == 5 and paw == @paw_name')['swingOff_iqr_25_ref_FR'].values)

            sOff_iqr_75_day1 = np.nanmedian(
                stridePar.query('day == 1 and paw == @paw_name')['swingOff_iqr_75_ref_FR'].values)
            sOff_iqr_75_day10 = np.nanmedian(
                stridePar.query('day == 10 and paw == @paw_name')['swingOff_iqr_75_ref_FR'].values)
            sOff_iqr_75_trial1 = np.nanmedian(
                stridePar.query('trial == 1 and paw == @paw_name')['swingOff_iqr_75_ref_FR'].values)
            sOff_iqr_75_trial5 = np.nanmedian(
                stridePar.query('trial == 5 and paw == @paw_name')['swingOff_iqr_75_ref_FR'].values)

            sOn_median_day1 = np.nanmedian(
                stridePar.query('day == 1 and paw == @paw_name')['swingOnMedian_ref_FR'].values)
            sOn_median_day10 = np.nanmedian(
                stridePar.query('day == 10 and paw == @paw_name')['swingOnMedian_ref_FR'].values)
            sOn_median_trial1 = np.nanmedian(
                stridePar.query('trial == 1 and paw == @paw_name')['swingOnMedian_ref_FR'].values)
            sOn_median_trial5 = np.nanmedian(
                stridePar.query('trial == 5 and paw == @paw_name')['swingOnMedian_ref_FR'].values)

            # Plot means across animals
            ax_day1.plot(x_percent, np.mean(swing_prob_day1, axis=0), ms=2, color=color)
            ax_day10.plot(x_percent, np.mean(swing_prob_day10, axis=0), ms=2, color=color)
            ax_trial1.plot(x_percent, np.mean(swing_prob_trial1, axis=0), ms=2, color=color, label=paw_name)
            ax_trial5.plot(x_percent, np.mean(swing_prob_trial5, axis=0), ms=2, color=color)

            # Add vertical lines for swing onset medians (only for FL, idx==0, to match original behavior)
            if idx == 0:
                for ax, val in [(ax_day1, sOn_median_day1), (ax_day10, sOn_median_day10),
                                (ax_trial1, sOn_median_trial1), (ax_trial5, sOn_median_trial5)]:
                    ax.axvline(val * 100, color=color, ls='--', lw=1, alpha=0.5)

            # Fill IQR bands (only for FR, idx==1, to match original behavior)
            if idx == 1:
                ax_day1.fill_between(x_percent, 0, np.mean(swing_prob_day1, axis=0),
                                     where=((x_percent > sOff_iqr_25_day1 * 100) & (
                                                 x_percent < sOff_iqr_75_day1 * 100)),
                                     alpha=0.2, color=color)
                ax_day10.fill_between(x_percent, 0, np.mean(swing_prob_day10, axis=0),
                                      where=((x_percent > sOff_iqr_25_day10 * 100) & (
                                                  x_percent < sOff_iqr_75_day10 * 100)),
                                      alpha=0.2, color=color)
                ax_trial1.fill_between(x_percent, 0, np.mean(swing_prob_trial1, axis=0),
                                       where=((x_percent > sOff_iqr_25_trial1 * 100) & (
                                                   x_percent < sOff_iqr_75_trial1 * 100)),
                                       alpha=0.2, color=color)
                ax_trial5.fill_between(x_percent, 0, np.mean(swing_prob_trial5, axis=0),
                                       where=((x_percent > sOff_iqr_25_trial5 * 100) & (
                                                   x_percent < sOff_iqr_75_trial5 * 100)),
                                       alpha=0.2, color=color)

        # Cosmetics & legends for swing prob panels
        ax_day1.xaxis.set_major_locator(MultipleLocator(50))
        ax_day10.xaxis.set_major_locator(MultipleLocator(50))

        self.layoutOfPanel(ax_day1, xLabel='% FR stride', yLabel='p(swing)', Leg=[1, 9])
        self.layoutOfPanel(ax_day10, xLabel='% FR stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[False, True])
        self.layoutOfPanel(ax_trial1, xLabel='% FR stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[True, False])
        self.layoutOfPanel(ax_trial5, xLabel='% FR stride', yLabel='p(swing)', Leg=[1, 9], xyInvisible=[True, True])

        ax_day10.legend(loc="upper right", frameon=False, fontsize=8)
        ax_trial1.legend(loc="upper right", frameon=False, fontsize=10)

        ax_day1.set_title('session 1')
        ax_day10.set_title('session 10')
        ax_trial1.set_title('trial 1')
        ax_trial5.set_title('trial 5')

        # ---------------------------- Correlation heatmap (right column) ----------------------------
        ax_corr = plt.subplot(grid_cols[2])

        # Labels shown in heatmap ticks (kept same order/names)
        y_axis_labels = [
            'FR swing number', 'FR swing speed', 'FR swing length',
            'FR stance duration', 'FR swing duration',
            'FR fraction of \n miss steps', 'FR stance onset IQR'
        ]

        # Matrix of FR variables used for correlation
        fr_vars_matrix = df_fr[
            ['swingNumber', 'swingSpeed', 'swingLength',
             'stanceDuration', 'swingDuration',
             'indecisiveFraction', 'swingOff_iqr_25_75_ref_FR']
        ]

        corr_pvalues, rvalues, annot, sig_mask = groupAnalysis.calculate_correlation_pvalues(fr_vars_matrix)
        non_diagonal_mask = fr_vars_matrix.corr() != 1
        combined_mask = sig_mask & non_diagonal_mask

        heatmap = sns.heatmap(
            fr_vars_matrix.corr(),
            xticklabels=y_axis_labels, yticklabels=y_axis_labels,
            annot=False, mask=combined_mask, fmt='',
            cmap=sns.color_palette("PiYG_r", 50),
            cbar_kws={"shrink": 0.5, 'label': 'correlation coefficient'},
            ax=ax_corr, square=True
        )
        colorbar = ax_corr.collections[0].colorbar
        colorbar.set_ticks([-0.5, 0, 0.5, 1])
        colorbar.set_ticklabels(['-0.5', '0', '0.5', '1'])

        heatmap.set_facecolor('white')
        heatmap.figure.axes[-1].tick_params(labelsize=13)

        ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=80, fontsize=13)
        ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=13)
        plt.setp(ax_corr.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor")

        # ---------------------------- Save figure ----------------------------
        fname = f'fig_locomotorLearning_supp_FR_v{figVersion}'
        plt.savefig('manuscriptFigures/' +fname + '.png')
        plt.savefig('manuscriptFigures/' +fname + '.pdf')
        plt.savefig('manuscriptFigures/' +fname + '.svg')
        print(f'figure saved in: manuscriptFigures/{fname}')

    def fig_muscimol(self,figVersion, stridePar, strideTraj,swingNumber,rungCrossed,strideDuration,indecisiveStrideFraction,swingSpeed,pawCoordination,strideLenght,musSwingStanceDic,salineSwingStanceDic):
        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        # figure #################################
        fig_width = 11  # width in inches
        fig_height = 9  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 2)

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12)
        plt.figtext(0.02, 0.93, 'A', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.93, 'B', clip_on=False, color='black',  size=20)
        # plt.figtext(0.65, 0.96, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.75, 0.96, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.015, 0.53, 'C', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.53, 'D', clip_on=False, color='black',  size=20)
        plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2], hspace=0.15, wspace=0.05, width_ratios=[1, 2])
        gssub0a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0[1], hspace=0.15, wspace=0.25)
        gssub0b= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub0[0])
        muscimol_img=mpimg.imread('MAX_Andry Mus Red Fix 011222 Mouse3 slice2_2022_12_09__13_58_17_Stitch-1.png')
        ax0a=plt.subplot(gssub0b[0])
        ax0a.imshow(muscimol_img)
        left, bottom, width, height = ax0a.get_position().bounds
        ax0a.set_position([left-0.015,bottom+0.030,width-0.025,height-0.025])  	#[left, bottom, width, height]
        ax0a.axis('off')
        ax0a.text(0.5, 0.8, 'S', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        ax0a.text(0.8, 0.5, 'IV/V', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # idxSwings =np.array(swingStanceD['swingP'][i][1])
        # recTimes = np.array(swingStanceD['forFit'][i][2])
        pawPos_mus=musSwingStanceDic['pawPos']
        pawPos_sal=salineSwingStanceDic['pawPos']
        ax0=plt.subplot(gssub0a[1])
        ax0b=plt.subplot(gssub0a[0])
        startx = 42#42
        xLength = 10#7
        for i in [0,1]:
            ax0.plot(pawPos_mus[i][:, 0], pawPos_mus[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} muscimol', ls=(0, (1, 1)))
            ax0b.plot(pawPos_sal[i][:, 0], pawPos_sal[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} saline', ls='-')

        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[True, True], Leg=[1, 9])
        self.layoutOfPanel(ax0b, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        ax0.legend(frameon=False, bbox_to_anchor=(0.85,1), loc='center')
        ax0b.legend(frameon=False, bbox_to_anchor=(0.83,0.98), loc='center')
        ax0.set_xlim(startx, startx + xLength)
        ax0b.set_xlim(startx, startx + xLength)
        ax0b.xaxis.set_major_locator(MultipleLocator(1))
        ax0b.yaxis.set_major_locator(MultipleLocator(50))
        ############################Panel A and B : timeline and image of muscimol spread or illustration image#########################
        ############################Panel C : muscimol application effect graph#########################
        ############################Panel B : Swing Number#########################

        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        #get data for five trials only, for visualization
        stridePar_Recordings_trial=stridePar_Recordings[stridePar_Recordings['trial']<=5]
        stridePar_Recordings_trial=stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day=stridePar_Recordings.groupby(['paw','day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day=stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_trial=stridePar_Recordings.groupby(['paw','trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial=stridePar_Recordings_paw_trial.reset_index()

        trials=stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList=trials.astype(str).tolist()
        daysList = days.astype(str).tolist()


        nSal=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].nunique()
        salineId=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].unique()
        muscimolId=stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].unique()
        panelList=[2,4,5]
        parameters=['swingNumber', 'indecisiveFraction']
        parameters_Y=['stride number (avg.)', 'fraction of miss steps']

        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2:4], hspace=0.15, wspace=0.35, width_ratios=[1,1])
        gssub1a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[0], hspace=0.15, wspace=0.55, width_ratios=[5,1])
        gssub1c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[1], hspace=0.15, wspace=0.55,
                                                   width_ratios=[5, 1])
        for par in range(len(parameters)):
            if par==0:
                ax1 = plt.subplot(gssub1a[0])
                ax1.set_ylim(70,160)
            elif par==1:
                ax1 = plt.subplot(gssub1c[0])
            # elif par == 2:
            #     ax1 = plt.subplot(gs[4])
            #per day data
            # perform stats Mixed Linear Model
            pars_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par],
                                                                treatments=True, exp='muscimol')
            # pars_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par],'mouseId',
            #                                                     treatments=True)
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='treatment', hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['black','red'], ax=(ax1 ), marker='o')
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId', style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)], alpha=0.1,
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['red','black'], ax=(ax1))
            ax1.text(1, 0.31, f'{pars_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            if par==0:
                ax1.text(0.7, 0.01, f'N={nSal} saline, {nMus} muscimol' , ha='center', va='center', transform=ax1.transAxes, style='italic',
                         fontsize=12, color='k')
            ax1.xaxis.set_major_locator(MultipleLocator(1))

            self.layoutOfPanel(ax1, xLabel='session', yLabel=parameters_Y[par])
            ax1.legend([f'saline {pars_summary["stars"]["saline"]["day"]} {pars_summary["stars"]["saline"]["trial"].replace("*","#")}', f'muscimol {pars_summary["stars"]["muscimol"]["day"]} {pars_summary["stars"]["muscimol"]["trial"].replace("*","#")}'],loc='upper left',frameon=False)
            df_trial_1_saline=stridePar_Recordings_trial[(stridePar_Recordings_trial['treatment']=='saline') & (stridePar_Recordings_trial['trial']==1)]
            df_trial_5_saline=stridePar_Recordings_trial[(stridePar_Recordings_trial['treatment']=='saline') & (stridePar_Recordings_trial['trial']==5)]
            df_trial_1_muscimol=stridePar_Recordings_trial[(stridePar_Recordings_trial['treatment']=='muscimol') & (stridePar_Recordings_trial['trial']==1)]
            df_trial_5_muscimol=stridePar_Recordings_trial[(stridePar_Recordings_trial['treatment']=='muscimol') & (stridePar_Recordings_trial['trial']==5)]

            mouseList=stridePar_Recordings_trial['mouseId'].unique()
            treatmentList = ['muscimol', 'saline']
            expFit_day = {}
            for t in range(len(treatmentList)):
                # expFit_trial[t] = {}
                # ax2 = plt.subplot(gssub1[t, 0])
                Id = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == treatmentList[t]][
                    'mouseId'].unique()
                for m in range(len(Id)):
                    colorMus = mcp.gen_color(cmap="Reds", n=nMus)
                    colorSal = mcp.gen_color(cmap="cubehelix", n=nSal)
                    expFit_day[Id[m]] = {}
                    xdata_day = stridePar_Recordings_day[(stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                stridePar_Recordings_day['mouseId'] == Id[m])]['day']
                    ydata_days = stridePar_Recordings_day[
                        (stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                    stridePar_Recordings_day['mouseId'] == Id[m])][parameters[par]]
                    # pdb.set_trace()
                    lm = LinearRegression()
                    lm.fit(np.array(xdata_day).reshape(-1, 1), ydata_days)
                    # popt, pcov = curve_fit(groupAnalysis.expfunc, xdata_day, ydata_days, maxfev=5000)
                    # popt, pcov = curve_fit(groupAnalysis.linfit, xdata_day, ydata_days)
                    # ax2.plot(xdata_day,ydata_days, color=(colorMus[m] if t==0 else colorSal[m]), label=Id[m])
                    # a, b, tau = popt
                    # a, b = popt
                    b = lm.intercept_
                    a = lm.coef_
                    # ax2.plot(xdata_day, groupAnalysis.linfit(xdata_day,*popt),  ls='--',label='exp fit', alpha=0.5,color=(colorMus[m] if t==0 else colorSal[m]))
                    # if t==1:
                    #     ax2.legend(frameon=False)
                    # ax2.xaxis.set_major_locator(MultipleLocator(1))
                    # self.layoutOfPanel(ax2,xLabel=('session' if t==1 else None),yLabel=f'swing number\n{treatmentList[t]}', xyInvisible=([True, False] if t==0 else [False, False]))
                    expFit_day[Id[m]]['slope'] = a
                    expFit_day[Id[m]]['intercept'] = b
                    # expFit_day[Id[m]]['tau']=tau
                    expFit_day[Id[m]]['treatment'] = treatmentList[t]
            expFit_day_df = pd.DataFrame(expFit_day)
            expFit_day_df = expFit_day_df.T
            expFit_day_df = expFit_day_df.reset_index().rename(columns={'index': 'mouseId'})
            expPar = ['intercept', 'slope']
            for e in range(len(expPar)):
                if par==0:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1a[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                else:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1c[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                ax3=plt.subplot(gssub1b[e])
                sal_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'saline'][expPar[e]])
                mus_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'muscimol'][expPar[e]])
                treatmentArray=[sal_Y, mus_Y]

                # ax3.boxplot(treatmentArray)
                ax3.bar(0, np.mean(sal_Y), color=['black'], yerr=stats.sem(sal_Y), alpha=0.8)
                ax3.bar(1, np.mean(mus_Y), color=['red'], yerr=stats.sem(mus_Y), alpha=0.8)
                ax3.scatter(np.repeat(0, nSal), sal_Y, edgecolor='black', color='white')
                ax3.scatter(np.repeat(1, nMus), mus_Y, edgecolor='red', color='white')
                ax3.set_xticks([0, 1], ['saline', 'muscimol'])
                self.layoutOfPanel(ax3, xLabel='', yLabel=expPar[e], xyInvisible=([False, False] if e==0 else [False, False]))
                if e==0:
                    ax3.get_xaxis().set_visible(False)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
                t_value, t_test_p_value = stats.ttest_ind(sal_Y, mus_Y)

                #one sample t-test
                diff0_sal_tvalue, diff0_sal_pvalue=stats.ttest_1samp(sal_Y,0)
                diff0_mus_tvalue, diff0_mus_pvalue = stats.ttest_1samp(mus_Y, 0)
                print(expPar[e],'!!!!!',t_value,t_test_p_value)
                print('saline diff 0',diff0_sal_tvalue, diff0_sal_pvalue)
                print('muscimol diff 0', diff0_mus_tvalue, diff0_mus_pvalue)
                # get the stars
                star_exp = groupAnalysis.starMultiplier(t_test_p_value)
                diff0_sal_stars=groupAnalysis.starMultiplier(diff0_sal_pvalue)
                diff0_mus_stars=groupAnalysis.starMultiplier(diff0_mus_pvalue)
                try:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value,2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10, color='k')
                    if e==1:
                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue<0.05 else f'p={diff0_sal_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue<0.05 else f'p={diff0_mus_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')
                        # ax3.annotate(, ha='center', va='center', color='0.3', fontsize=10)
                        # ax3.annotate((diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue<0.05 else f'p={round(diff0_mus_pvalue[0],2)}'),(1, np.mean(mus_Y) +(np.mean(mus_Y)/1.05)), ha='center', va='center', color='0.3', fontsize=10)
                except TypeError:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value[0],2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10,color='k')
                    if e==1:
                        # pdb.set_trace()
                        # ax3.annotate(diff0_sal_stars.replace("*",u"\u2020"),(0,np.mean(sal_Y)+(np.mean(sal_Y)/1.2)), ha='center', va='center', color='0.3')
                        # ax3.annotate(diff0_mus_stars.replace("*", u"\u2020"), (1, np.mean(mus_Y) + (np.mean(mus_Y)/1.2)), ha='center', va='center', color='0.3')
                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue[0]<0.05 else f'p={diff0_sal_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue[0]<0.05 else f'p={diff0_mus_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')




        fname = 'fig_muscimol_v%s' % figVersion
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    def fig_muscimol_filtered(self,figVersion, stridePar, strideTraj,swingNumber,rungCrossed,strideDuration,indecisiveStrideFraction,swingSpeed,pawCoordination,strideLenght,musSwingStanceDic,salineSwingStanceDic):
        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        # figure #################################
        fig_width = 11  # width in inches
        fig_height = 9  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 2)

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12)
        plt.figtext(0.02, 0.93, 'A', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.93, 'B', clip_on=False, color='black',  size=20)
        # plt.figtext(0.65, 0.96, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.75, 0.96, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.015, 0.53, 'C', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.53, 'D', clip_on=False, color='black',  size=20)
        plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2], hspace=0.15, wspace=0.05, width_ratios=[1, 2])
        gssub0a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0[1], hspace=0.15, wspace=0.25)
        gssub0b= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub0[0])
        # muscimol_img=mpimg.imread('MAX_Andry Mus Red Fix 011222 Mouse3 slice2_2022_12_09__13_58_17_Stitch-1.png')
        ax0a=plt.subplot(gssub0b[0])
        #remove frame and axis to create space for image
        ax0a.axis('off')
        # ax0a.imshow(muscimol_img)
        # left, bottom, width, height = ax0a.get_position().bounds
        # ax0a.set_position([left-0.015,bottom+0.030,width-0.025,height-0.025])  	#[left, bottom, width, height]
        # ax0a.axis('off')
        # ax0a.text(0.5, 0.8, 'S', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # ax0a.text(0.8, 0.5, 'IV/V', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # idxSwings =np.array(swingStanceD['swingP'][i][1])
        # recTimes = np.array(swingStanceD['forFit'][i][2])
        pawPos_mus=musSwingStanceDic['pawPos']
        pawPos_sal=salineSwingStanceDic['pawPos']
        ax0=plt.subplot(gssub0a[1])
        ax0b=plt.subplot(gssub0a[0])
        startx = 42#42
        xLength = 10#7
        for i in [0,1]:
            ax0.plot(pawPos_mus[i][:, 0], pawPos_mus[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} muscimol', ls=(0, (1, 1)))
            ax0b.plot(pawPos_sal[i][:, 0], pawPos_sal[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} saline', ls='-')

        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[True, True], Leg=[1, 9])
        self.layoutOfPanel(ax0b, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        ax0.legend(frameon=False, bbox_to_anchor=(0.85,1), loc='center')
        ax0b.legend(frameon=False, bbox_to_anchor=(0.83,0.98), loc='center')
        ax0.set_xlim(startx, startx + xLength)
        ax0b.set_xlim(startx, startx + xLength)
        ax0b.xaxis.set_major_locator(MultipleLocator(1))
        ax0b.yaxis.set_major_locator(MultipleLocator(50))
        ############################Panel A and B : timeline and image of muscimol spread or illustration image#########################
        ############################Panel C : muscimol application effect graph#########################
        ############################Panel B : Swing Number#########################

        print(stridePar.head())
        print(stridePar.columns)
        # pdb.set_trace()
        #drop lines with stridePar['LinWheelSpeed_avg']<6
        # stridePar=stridePar[stridePar['LinWheelSpeed_avg']<=4]
        mouse_to_exclude=["201017_m1","201017_m98", "201017_m99", "201207_f42","201207_f43"]
        stridePar=stridePar[~stridePar['mouseId'].isin(mouse_to_exclude)]
        print('after filtering')
        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        print(stridePar_Recordings)
        #get data for five trials only, for visualization
        stridePar_Recordings_trial=stridePar_Recordings[stridePar_Recordings['trial']<=5]

        stridePar_Recordings_trial=stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day=stridePar_Recordings.groupby(['paw','day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day=stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_trial=stridePar_Recordings.groupby(['paw','trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial=stridePar_Recordings_paw_trial.reset_index()

        trials=stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList=trials.astype(str).tolist()
        daysList = days.astype(str).tolist()


        nSal=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].nunique()
        salineId=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].unique()
        muscimolId=stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].unique()
        # fig = plt.figure()
        # sns.lineplot(data=stridePar_Recordings_day, x='day', y='LinWheelSpeed_avg', hue='mouseId',
        #              style='treatment', style_order=['saline', 'muscimol'],
        #              errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
        #              err_kws={'capsize': 3, 'linewidth': 1},  marker='o')
        # plt.show()


        panelList=[2,4,5]
        parameters=['swingNumber', 'indecisiveFraction']
        parameters_Y=['stride number (avg.)', 'fraction of miss steps']

        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2:4], hspace=0.15, wspace=0.35, width_ratios=[1,1])
        gssub1a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[0], hspace=0.15, wspace=0.55, width_ratios=[5,1])
        gssub1c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[1], hspace=0.15, wspace=0.55,
                                                   width_ratios=[5, 1])
        for par in range(len(parameters)):
            if par==0:
                ax1 = plt.subplot(gssub1a[0])
                ax1.set_ylim(70,120)
            elif par==1:
                ax1 = plt.subplot(gssub1c[0])
            # elif par == 2:
            #     ax1 = plt.subplot(gs[4])
            #per day data
            # perform stats Mixed Linear Model
            pars_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par],
                                                                treatments=True, exp='muscimol')
            # pars_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par],'mouseId',
            #                                                     treatments=True)
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='treatment', hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['black','red'], ax=(ax1 ), marker='o')
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId', style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)], alpha=0.1,
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['red','black'], ax=(ax1))

            treatment_comparison_star=pars_summary["stars"]["all"]["treatment[T.saline]"]
            treatment_comparison_pvalue=pars_summary["pvalues"]["all"]["treatment[T.saline]"]
            ax1.text(1, 0.31, f'{pars_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            ax1.text(1, 0.31, f'{treatment_comparison_star.replace("*", "°")}' if treatment_comparison_pvalue<0.05 else f'',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            if par==0:
                ax1.text(0.7, 0.01, f'N={nSal} saline, {nMus} muscimol' , ha='center', va='center', transform=ax1.transAxes, style='italic',
                         fontsize=12, color='k')
            ax1.xaxis.set_major_locator(MultipleLocator(1))

            self.layoutOfPanel(ax1, xLabel='session', yLabel=parameters_Y[par])
            ax1.legend(frameon=False, loc='upper left')
            if pars_summary["pvalues"]["saline"]["trial"]<0.05:
                treatment_comparison_stars_saline_trial=pars_summary["stars"]["saline"]["trial"].replace("*","#")
            else:
                treatment_comparison_stars_saline_trial=''
            if pars_summary["pvalues"]["muscimol"]["trial"]<0.05:
                treatment_comparison_stars_muscimol_trial=pars_summary["stars"]["muscimol"]["trial"].replace("*","#")
            else:
                treatment_comparison_stars_muscimol_trial=''

            ax1.legend([f'saline {pars_summary["stars"]["saline"]["day"]} {treatment_comparison_stars_saline_trial}', f'muscimol {pars_summary["stars"]["muscimol"]["day"]} {treatment_comparison_stars_muscimol_trial}'],loc='upper left',frameon=False)

            treatmentList = ['muscimol', 'saline']
            expFit_day = {}
            for t in range(len(treatmentList)):
                # expFit_trial[t] = {}
                # ax2 = plt.subplot(gssub1[t, 0])
                Id = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == treatmentList[t]][
                    'mouseId'].unique()
                for m in range(len(Id)):
                    colorMus = mcp.gen_color(cmap="Reds", n=nMus)
                    colorSal = mcp.gen_color(cmap="cubehelix", n=nSal)
                    expFit_day[Id[m]] = {}
                    xdata_day = stridePar_Recordings_day[(stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                stridePar_Recordings_day['mouseId'] == Id[m])]['day']
                    ydata_days = stridePar_Recordings_day[
                        (stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                    stridePar_Recordings_day['mouseId'] == Id[m])][parameters[par]]
                    # pdb.set_trace()
                    lm = LinearRegression()
                    lm.fit(np.array(xdata_day).reshape(-1, 1), ydata_days)

                    b = lm.intercept_
                    a = lm.coef_

                    expFit_day[Id[m]]['slope'] = a
                    expFit_day[Id[m]]['intercept'] = b
                    # expFit_day[Id[m]]['tau']=tau
                    expFit_day[Id[m]]['treatment'] = treatmentList[t]
            expFit_day_df = pd.DataFrame(expFit_day)
            expFit_day_df = expFit_day_df.T
            expFit_day_df = expFit_day_df.reset_index().rename(columns={'index': 'mouseId'})
            expPar = ['intercept', 'slope']
            for e in range(len(expPar)):
                if par==0:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1a[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                else:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1c[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                ax3=plt.subplot(gssub1b[e])
                sal_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'saline'][expPar[e]])
                mus_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'muscimol'][expPar[e]])
                treatmentArray=[sal_Y, mus_Y]

                # ax3.boxplot(treatmentArray)
                ax3.bar(0, np.mean(sal_Y), color=['black'], yerr=stats.sem(sal_Y), alpha=0.8)
                ax3.bar(1, np.mean(mus_Y), color=['red'], yerr=stats.sem(mus_Y), alpha=0.8)
                ax3.scatter(np.repeat(0, nSal), sal_Y, edgecolor='black', color='white')
                ax3.scatter(np.repeat(1, nMus), mus_Y, edgecolor='red', color='white')
                ax3.set_xticks([0, 1], ['saline', 'muscimol'])
                self.layoutOfPanel(ax3, xLabel='', yLabel=expPar[e], xyInvisible=([False, False] if e==0 else [False, False]))
                if e==0:
                    ax3.get_xaxis().set_visible(False)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
                t_value, t_test_p_value = stats.ttest_ind(sal_Y, mus_Y)

                #one sample t-test
                diff0_sal_tvalue, diff0_sal_pvalue=stats.ttest_1samp(sal_Y,0)
                diff0_mus_tvalue, diff0_mus_pvalue = stats.ttest_1samp(mus_Y, 0)
                print(expPar[e],'!!!!!',t_value,t_test_p_value)
                print('saline diff 0',diff0_sal_tvalue, diff0_sal_pvalue)
                print('muscimol diff 0', diff0_mus_tvalue, diff0_mus_pvalue)
                # get the stars
                star_exp = groupAnalysis.starMultiplier(t_test_p_value)
                diff0_sal_stars=groupAnalysis.starMultiplier(diff0_sal_pvalue)
                diff0_mus_stars=groupAnalysis.starMultiplier(diff0_mus_pvalue)
                try:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value,2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10, color='k')
                    if e==1:
                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue<0.05 else f'p={diff0_sal_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue<0.05 else f'p={diff0_mus_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')

                except TypeError:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value[0],2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10,color='k')
                    if e==1:

                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue[0]<0.05 else f'p={diff0_sal_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue[0]<0.05 else f'p={diff0_mus_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')




        fname = 'fig_muscimol_v%s_low_speed' % figVersion
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    def fig_muscimol_normalized(self,figVersion, stridePar, strideTraj,swingNumber,rungCrossed,strideDuration,indecisiveStrideFraction,swingSpeed,pawCoordination,strideLenght,musSwingStanceDic,salineSwingStanceDic):
        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        # figure #################################
        fig_width = 12  # width in inches
        fig_height = 9  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 2)

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12)
        plt.figtext(0.02, 0.93, 'A', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.93, 'B', clip_on=False, color='black',  size=20)
        # plt.figtext(0.65, 0.96, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.75, 0.96, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.015, 0.53, 'C', clip_on=False, color='black',  size=20)
        plt.figtext(0.35, 0.53, 'D', clip_on=False, color='black',  size=20)
        plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2], hspace=0.15, wspace=0.05, width_ratios=[1, 2])
        gssub0a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0[1], hspace=0.15, wspace=0.25)
        gssub0b= gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub0[0])
        # muscimol_img=mpimg.imread('MAX_Andry Mus Red Fix 011222 Mouse3 slice2_2022_12_09__13_58_17_Stitch-1.png')
        ax0a=plt.subplot(gssub0b[0])
        #remove frame and axis to create space for image
        ax0a.axis('off')
        # ax0a.imshow(muscimol_img)
        # left, bottom, width, height = ax0a.get_position().bounds
        # ax0a.set_position([left-0.015,bottom+0.030,width-0.025,height-0.025])  	#[left, bottom, width, height]
        # ax0a.axis('off')
        # ax0a.text(0.5, 0.8, 'S', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # ax0a.text(0.8, 0.5, 'IV/V', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # idxSwings =np.array(swingStanceD['swingP'][i][1])
        # recTimes = np.array(swingStanceD['forFit'][i][2])
        pawPos_mus=musSwingStanceDic['pawPos']
        pawPos_sal=salineSwingStanceDic['pawPos']
        ax0=plt.subplot(gssub0a[1])
        ax0b=plt.subplot(gssub0a[0])
        startx = 42#42
        xLength = 10#7
        for i in [0,1]:
            ax0.plot(pawPos_mus[i][:, 0], pawPos_mus[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} muscimol', ls=(0, (1, 1)))
            ax0b.plot(pawPos_sal[i][:, 0], pawPos_sal[i][:, 1], c=col[i], lw=1.4, label=f'{pawId[i]} saline', ls='-')

        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[True, True], Leg=[1, 9])
        self.layoutOfPanel(ax0b, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        ax0.legend(frameon=False, bbox_to_anchor=(0.85,1), loc='center')
        ax0b.legend(frameon=False, bbox_to_anchor=(0.83,0.98), loc='center')
        ax0.set_xlim(startx, startx + xLength)
        ax0b.set_xlim(startx, startx + xLength)
        ax0b.xaxis.set_major_locator(MultipleLocator(1))
        ax0b.yaxis.set_major_locator(MultipleLocator(50))
        ############################Panel A and B : timeline and image of muscimol spread or illustration image#########################
        ############################Panel C : muscimol application effect graph#########################
        ############################Panel B : Swing Number#########################

        print(stridePar.head())
        print(stridePar.columns)
        # pdb.set_trace()
        #drop lines with stridePar['LinWheelSpeed_avg']<6
        # stridePar=stridePar[stridePar['LinWheelSpeed_avg']<=4]
        # mouse_to_exclude=["201017_m1","201017_m98", "201017_m99", "201207_f42","201207_f43"]
        # stridePar=stridePar[~stridePar['mouseId'].isin(mouse_to_exclude)]
        # print('after filtering')
        print('before normalization', stridePar['swingNumber'])
        #normalize parameters with LinWheelSpeed_avg
        stridePar['swingNumber']=stridePar['swingNumber']/stridePar['wheelDistance']
        stridePar['indecisiveFraction']=stridePar['indecisiveFraction']/stridePar['wheelDistance']
        print('after normalization', stridePar['swingNumber'])
        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        print(stridePar_Recordings)
        #get data for five trials only, for visualization
        stridePar_Recordings_trial=stridePar_Recordings[stridePar_Recordings['trial']<=5]

        stridePar_Recordings_trial=stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day=stridePar_Recordings.groupby(['paw','day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day=stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_trial=stridePar_Recordings.groupby(['paw','trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial=stridePar_Recordings_paw_trial.reset_index()

        trials=stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList=trials.astype(str).tolist()
        daysList = days.astype(str).tolist()


        nSal=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].nunique()
        salineId=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='saline']['mouseId'].unique()
        muscimolId=stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].unique()
        # fig = plt.figure()
        # sns.lineplot(data=stridePar_Recordings_day, x='day', y='LinWheelSpeed_avg', hue='mouseId',
        #              style='treatment', style_order=['saline', 'muscimol'],
        #              errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
        #              err_kws={'capsize': 3, 'linewidth': 1},  marker='o')
        # plt.show()


        panelList=[2,4,5]
        parameters=['swingNumber', 'indecisiveFraction']
        parameters_Y=['stride number', 'fraction of miss steps']

        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2:4], hspace=0.15, wspace=0.35, width_ratios=[1,1])
        gssub1a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[0], hspace=0.15, wspace=0.55, width_ratios=[5,1])
        gssub1c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[1], hspace=0.15, wspace=0.55,
                                                   width_ratios=[5, 1])
        for par in range(len(parameters)):
            if par==0:
                ax1 = plt.subplot(gssub1a[0])
                # ax1.set_ylim(70,160)
            elif par==1:
                ax1 = plt.subplot(gssub1c[0])
            # elif par == 2:
            #     ax1 = plt.subplot(gs[4])
            #per day data
            # perform stats Mixed Linear Model
            pars_summary = groupAnalysis.perform_mixedlm(stridePar, parameters[par],
                                                                treatments=True, exp='muscimol')
            # pars_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par],'mouseId',
            #                                                     treatments=True)
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='treatment', hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['black','red'], ax=(ax1 ), marker='o')
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId', style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)], alpha=0.1,
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['red','black'], ax=(ax1))

            treatment_comparison_star=pars_summary["stars"]["all"]["treatment[T.saline]"]
            treatment_comparison_pvalue=pars_summary["pvalues"]["all"]["treatment[T.saline]"]
            ax1.text(1, 0.31, f'{pars_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            ax1.text(1, 0.31, f'{treatment_comparison_star.replace("*", "°")}' if treatment_comparison_pvalue<0.05 else f'',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            if par==0:
                ax1.text(0.7, 0.01, f'N={nSal} saline, {nMus} muscimol' , ha='center', va='center', transform=ax1.transAxes, style='italic',
                         fontsize=12, color='k')
            ax1.xaxis.set_major_locator(MultipleLocator(1))

            self.layoutOfPanel(ax1, xLabel='session', yLabel=f'{parameters_Y[par]} \n (wheel distance normalized)')
            ax1.legend(frameon=False, loc='upper left')
            if pars_summary["pvalues"]["saline"]["trial"]<0.05:
                treatment_comparison_stars_saline_trial=pars_summary["stars"]["saline"]["trial"].replace("*","#")
            else:
                treatment_comparison_stars_saline_trial=''
            if pars_summary["pvalues"]["muscimol"]["trial"]<0.05:
                treatment_comparison_stars_muscimol_trial=pars_summary["stars"]["muscimol"]["trial"].replace("*","#")
            else:
                treatment_comparison_stars_muscimol_trial=''

            ax1.legend([f'saline {pars_summary["stars"]["saline"]["day"]} {treatment_comparison_stars_saline_trial}', f'muscimol {pars_summary["stars"]["muscimol"]["day"]} {treatment_comparison_stars_muscimol_trial}'],loc='upper left',frameon=False)

            treatmentList = ['muscimol', 'saline']
            expFit_day = {}
            for t in range(len(treatmentList)):
                # expFit_trial[t] = {}
                # ax2 = plt.subplot(gssub1[t, 0])
                Id = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == treatmentList[t]][
                    'mouseId'].unique()
                for m in range(len(Id)):
                    colorMus = mcp.gen_color(cmap="Reds", n=nMus)
                    colorSal = mcp.gen_color(cmap="cubehelix", n=nSal)
                    expFit_day[Id[m]] = {}
                    xdata_day = stridePar_Recordings_day[(stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                stridePar_Recordings_day['mouseId'] == Id[m])]['day']
                    ydata_days = stridePar_Recordings_day[
                        (stridePar_Recordings_day['treatment'] == treatmentList[t]) & (
                                    stridePar_Recordings_day['mouseId'] == Id[m])][parameters[par]]
                    # pdb.set_trace()
                    lm = LinearRegression()
                    lm.fit(np.array(xdata_day).reshape(-1, 1), ydata_days)

                    b = lm.intercept_
                    a = lm.coef_

                    expFit_day[Id[m]]['slope'] = a
                    expFit_day[Id[m]]['intercept'] = b
                    # expFit_day[Id[m]]['tau']=tau
                    expFit_day[Id[m]]['treatment'] = treatmentList[t]
            expFit_day_df = pd.DataFrame(expFit_day)
            expFit_day_df = expFit_day_df.T
            expFit_day_df = expFit_day_df.reset_index().rename(columns={'index': 'mouseId'})
            expPar = ['intercept', 'slope']
            for e in range(len(expPar)):
                if par==0:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1a[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                else:
                    gssub1b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1c[1], hspace=0.35, wspace=0.5,height_ratios=[1, 1])
                ax3=plt.subplot(gssub1b[e])
                sal_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'saline'][expPar[e]])
                mus_Y = np.array(expFit_day_df[expFit_day_df['treatment'] == 'muscimol'][expPar[e]])
                treatmentArray=[sal_Y, mus_Y]

                # ax3.boxplot(treatmentArray)
                ax3.bar(0, np.mean(sal_Y), color=['black'], yerr=stats.sem(sal_Y), alpha=0.8)
                ax3.bar(1, np.mean(mus_Y), color=['red'], yerr=stats.sem(mus_Y), alpha=0.8)
                ax3.scatter(np.repeat(0, nSal), sal_Y, edgecolor='black', color='white')
                ax3.scatter(np.repeat(1, nMus), mus_Y, edgecolor='red', color='white')
                ax3.set_xticks([0, 1], ['saline', 'muscimol'])
                self.layoutOfPanel(ax3, xLabel='', yLabel=expPar[e], xyInvisible=([False, False] if e==0 else [False, False]))
                if e==0:
                    ax3.get_xaxis().set_visible(False)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
                t_value, t_test_p_value = stats.ttest_ind(sal_Y, mus_Y)

                #one sample t-test
                diff0_sal_tvalue, diff0_sal_pvalue=stats.ttest_1samp(sal_Y,0)
                diff0_mus_tvalue, diff0_mus_pvalue = stats.ttest_1samp(mus_Y, 0)
                print(expPar[e],'!!!!!',t_value,t_test_p_value)
                print('saline diff 0',diff0_sal_tvalue, diff0_sal_pvalue)
                print('muscimol diff 0', diff0_mus_tvalue, diff0_mus_pvalue)
                # get the stars
                star_exp = groupAnalysis.starMultiplier(t_test_p_value)
                diff0_sal_stars=groupAnalysis.starMultiplier(diff0_sal_pvalue)
                diff0_mus_stars=groupAnalysis.starMultiplier(diff0_mus_pvalue)
                try:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value,2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10, color='k')
                    if e==1:
                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue<0.05 else f'p={diff0_sal_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue<0.05 else f'p={diff0_mus_pvalue:.2f}'),(0,np.max(sal_Y)), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')

                except TypeError:
                    ax3.text(0.5, 1.08, f'p={round(t_test_p_value[0],2)}' if t_test_p_value > 0.05 else f'{star_exp.replace("*","¤")}',
                             ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=10,color='k')
                    if e==1:

                        ax3.text(0.23, -0.05, (diff0_sal_stars.replace("*",u"\u2020") if diff0_sal_pvalue[0]<0.05 else f'p={diff0_sal_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=6, color='0.3')
                        ax3.text(0.75, -0.05, (diff0_mus_stars.replace("*",u"\u2020") if diff0_mus_pvalue[0]<0.05 else f'p={diff0_mus_pvalue[0]:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=8, color='0.3')




        fname = 'fig_muscimol_v%s_speed_normalizd' % figVersion
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    ##########################################################################################
    def ephysTSNEFig(self,figVersion, clusterDict):

        nCells = clusterDict['nCellsRecordings'][0]
        nRecsTotal = clusterDict['nCellsRecordings'][1]
        print(nCells,' cells and ', nRecsTotal, ' recordings.')
        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,0.7])
                               #height_ratios=[1, 1.5,1.5])


        # define vertical and horizontal spacing between panels
        #gs.update(wspace=0.1, hspace=0.2)
        plt.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.08)
        plt.figtext(0.017, 0.96, 'A', clip_on=False, color='black',  size=22)
        plt.figtext(0.555, 0.96, 'B', clip_on=False, color='black', size=22)
        plt.figtext(0.017, 0.29, 'C', clip_on=False, color='black',  size=22)
        #plt.figtext(0.805, 0.96, 'D', clip_on=False, color='black',  size=22)
        #plt.figtext(0.01, 0.69, 'E', clip_on=False, color='black',  size=22)
        #plt.figtext(0.85, 0.69, 'F', clip_on=False, color='black', size=22)
        #plt.figtext(0.01, 0.35, 'G', clip_on=False, color='black', size=22)
        #plt.figtext(0.24, 0.35, 'H', clip_on=False, color='black', size=22)
        #plt.figtext(0.485, 0.35, 'I', clip_on=False, color='black', size=22)
        #plt.figtext(0.72, 0.35, 'J', clip_on=False, color='black', size=22)
        ##########################
        gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.12, width_ratios=[2,1.7])
        #gssub1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], hspace=0.1, width_ratios=[0.9,0.9,1,0.5])
        # panel ##################
        nR = 0
        ax0 = plt.subplot(gs0[0])
        #                1   2    3    4   5   6     7    8   9   10
        exampleCells = [226,100, 403, 205, 24, 229, 258, 252, 52, 264]
        nExamples = len(exampleCells)
        gssub1 = gridspec.GridSpecFromSubplotSpec(int(nExamples/2),2, subplot_spec=gs0[1], wspace=0.2, hspace=0.2)#, width_ratios=[0.9, 0.9, 1, 0.5])
        axEWave = []
        axEAC = []
        for i in range(nExamples):
            gssubsub1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gssub1[i], wspace=0.05, hspace=0.17)
            axEWave.append(plt.subplot(gssubsub1[0]))
            axEAC.append(plt.subplot(gssubsub1[1]))
        #pdb.set_trace()
        nE = 0
        nV = 0
        pcRecs = []
        mliRecs = []
        for n in range(nCells):
            nRecs = len(clusterDict['ephysDict'][n]['mouse-cell-recs'][3])
            if clusterDict['classes'][nR]==0:
                mliRecs.append(nRecs)
            else:
                pcRecs.append(nRecs)
            if clusterDict['ephysDict'][n]['visuallyGuided']:
                nV+=1
            for i in range(nRecs):
                #for i in range(nRecs):
                #print(umapDict['classes'][i])
                cc = ['C0' if clusterDict['classes'][nR]==0 else 'C1']
                aa = 0.3
                #if f38 2022.04.26_000 rec:000
                #if (clusterDict['ephysDict'][n]['mouse-cell-recs'][0][-3:] == 'f38') and (clusterDict['ephysDict'][n]['mouse-cell-recs'][1] =='2022.04.26_000'):
                #    cc = 'gray'
                #    aa = 0.9
                if clusterDict['ephysDict'][n][i]['cs_number']>0:
                    sym = '^'
                else:
                    sym = 'o'
                if clusterDict['ephysDict'][n]['visuallyGuided']:
                    #nV+=1
                    edgeC = 'C2'
                else:
                    edgeC = 'face'
                ax0.scatter(clusterDict['T-SNE']['clusterable_embedding'][nR, 0], clusterDict['T-SNE']['clusterable_embedding'][nR, 1], c=cc, marker=sym, edgecolors=edgeC,s=20,alpha=aa)#, cmap='Spectral')
                if nR in exampleCells:
                    #ee = exampleCells[nE]
                    idx = exampleCells.index(nR)
                    ff = clusterDict['ephysDict'][n][i]['ss_fr'][0]
                    dd = clusterDict['ephysDict'][n][i]['ss_avgSpikeParams'][0]*1000.
                    cvs = clusterDict['ephysDict'][n][i]['ss_spike-count_CVs']
                    ttt = '#%s f: %s d: %s CVs: %s,%s,%s,%s' %((idx+1),np.round(ff,1),np.round(dd,2),np.round(cvs[0][0],2),np.round(cvs[1][0],2),np.round(cvs[2][0],2),np.round(cvs[3][0],2))
                    ax0.annotate('#%s' % (idx+1),(clusterDict['T-SNE']['clusterable_embedding'][nR, 0],clusterDict['T-SNE']['clusterable_embedding'][nR, 1]), alpha=0.8,size=10)
                    ax0.scatter(clusterDict['T-SNE']['clusterable_embedding'][nR, 0], clusterDict['T-SNE']['clusterable_embedding'][nR, 1], c=cc, marker=sym, edgecolors=edgeC, s=30, alpha=1)  # , cmap='Spectral')
                    axEWave[idx].set_title(ttt,loc='left',size=8,pad=0)
                    wave = clusterDict['ephysDict'][n][i]['ss_wave']
                    waveS = clusterDict['ephysDict'][n][i]['ss_wave_span']
                    # wfs[j][0]*1000,(-1.)*wfs[j][1]/np.min(wfs[j][1])
                    axEWave[idx].plot(waveS*1000,(-1)*wave/np.min(wave),c=cc[0],alpha=0.5)
                    #axEWave[idx].annotate('f: %s d: %s' %(np.round(ff,1),np.round(dd,2)),(0.8,0.6),size=8)
                    axEWave[idx].plot(waveS[16:81] * 1000, (-1) * wave[16:81] / np.min(wave),c=cc[0],)
                    axEWave[idx].set_ylim(-1,1)
                    self.layoutOfPanel(axEWave[idx], xLabel='time (ms)', yLabel=None,xyInvisible=[True if idx<8 else False,True])
                    #print(nE,[True if nE<(nExamples-2) else False])
                    #
                    ac = clusterDict['ephysDict'][n][i]['ss_xprob']#[~np.isnan(umapDict['ephysDict'][n][i]['ss_xprob'])]
                    acS = clusterDict['ephysDict'][n][i]['ss_xprob_span']#[~np.isnan(umapDict['ephysDict'][n][i]['ss_xprob'])]
                    axEAC[idx].plot(acS*1000.,0.5*ac/np.mean(np.concatenate((ac[:11],ac[-11:]))),c=cc[0],alpha=0.5)
                    axEAC[idx].plot(acS[30:70] * 1000., 0.5 * ac[30:70] / np.mean(np.concatenate((ac[:11], ac[-11:]))),c=cc[0])
                    axEAC[idx].set_ylim(0, 1.5)
                    self.layoutOfPanel(axEAC[idx],xLabel='time (ms)',yLabel=None,xyInvisible=[True if idx<8 else False,True])
                    #pdb.set_trace()
                    nE+=1
                nR+=1
            #ax0.scatter()
        print('total number of cells, visually guided, example cells :', nR,len(mliRecs),len(pcRecs), nV, nE)
        #pdb.set_trace()
        pcMask = (clusterDict['classes'] == 1)
        mliMask = (clusterDict['classes'] == 0)
        ax0.annotate('MLI' , (16, -8), alpha=0.8,c='C0', size=20)
        ax0.annotate('PC' , (-17, 7), alpha=0.8,c='C1', size=20)
        ax0.annotate('%s MLIs : %s recordings' % (len(mliRecs),np.sum(mliMask)), (9, 21),  c='C0', size=10)
        ax0.annotate('%s PCs : %s recordings' % (len(pcRecs),np.sum(pcMask)), (9, 19.6), c='C1', size=10)
        self.layoutOfPanel(ax0,xLabel='t-SNE component 1',yLabel='t-SNE component 2')
        # create inset
        ax_inset = inset_axes(ax0, width="32%", height="30%",bbox_to_anchor=(0.1, 0.,0.9,0.9),
                   bbox_transform=ax0.transAxes,loc=1) #fig.add_axes([0.2, 0.2,0.2,0.2]) #plt.axes((0.6, 0.2, 0.2, 0.3), frameon=False)
        bb = np.linspace(0.75,8.25,16,endpoint=True)
        ax_inset.hist(mliRecs,rwidth=0.5,histtype='step',bins=bb)
        ax_inset.hist(pcRecs,rwidth=0.5,histtype='step',bins=bb)
        #plt.xticks([-0.5, 0, 0.5],[-0.5, 0, 0.5], size=6)
        #plt.xlim((-0.5, 0.5))
        #plt.yticks([5, 10], size=6)
        ax_inset.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax_inset,xLabel='# recordings',yLabel='occurrences')
        #ax_inset.set_xlabel("residuals", size=8)
        #ax_inset.xaxis.set_ticks_position("none")
        #ax_inset.yaxis.set_ticks_position("left")
        # histograms
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1], hspace=0.12) #, width_ratios=[0.9, 0.9, 1, 0.5])
        frmax = 117.873444384234360
        dmax = 0.00085
        dScaling = 4.

        designM = clusterDict['DesignMatrix']
        #pdb.set_trace()
        # [0.05,0.5,1.,5.]
        bb = np.linspace(0.1875,0.6125,18)
        measures = [['firing rate (spk/s)',105,frmax,[np.min(designM[:,105]),np.max(designM[:,105])]],['trough-peak delay (ms)',106,dmax*1000./dScaling,[np.min(designM[:,106]),np.max(designM[:,106])]],['spike-count CV (0.05 s)',107,1.,[np.min(designM[:,107]),1.7]],['spike-count CV (0.5 s)',108,1.,[np.min(designM[:,108]),0.7]],['spike-count CV (1 s)',109,1.,[np.min(designM[:,109]),0.7]],['spike-count CV (5 s)',110,1.,[np.min(designM[:,110]),0.7]]]
        for i in range(len(measures)):
            ax = plt.subplot(gs1[i])
            ax.hist((designM[mliMask][:,measures[i][1]])*measures[i][2],histtype='step',bins=(bb if i==1 else 30),range=(measures[i][3][0]*measures[i][2],measures[i][3][1]*measures[i][2]))
            ax.hist((designM[pcMask][:, measures[i][1]])*measures[i][2],histtype='step',bins=(bb if i==1 else 30),range=(measures[i][3][0]*measures[i][2],measures[i][3][1]*measures[i][2]))
            self.layoutOfPanel(ax,xLabel=measures[i][0],yLabel='occurrences',xyInvisible=[False,True if i>0 else False])

        ########################
        fname = 'fig_ephys-t-sne-clustering_v%s' % figVersion
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        # plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    def psth_group_analysis(self, figVersion,recordings, df_psth, df_cells,condition):
        pawList = ['FL', 'FR', 'HL', 'HR']
        col = ['C0','C1','C2','C3']
        fig_width = 20  # width in inches
        fig_height = 20  # height in inches
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
        plt.subplots_adjust(left=0.01, right=0.98, top=0.95, bottom=0.05)
        gssub0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], hspace=0.2)
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.15)
        for i in range(2):
            # paw labels panel
            ax0 = plt.subplot(gssub0[i])
            ax0.text(0.5, 0.5, pawList[i], color=col[i], fontsize=25)

            self.layoutOfPanel(ax0, xyInvisible=[True, True])
            gssub2=gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gssub1[i], hspace=0.1, wspace=0.5, height_ratios=[1,3,2,1,3,1],width_ratios=[1,1])#, height_ratios=[1,1])
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]
            events = ['stanceOnset', 'swingOnset']
            for e in range(len(events)):
                event=events[e]
                ax1=plt.subplot(gssub2[1+(e*3),0:2])
                #get the ids and df of all modulated cells
                cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df, paw_psth_df,event)
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
                gssub3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,2], hspace=0.15, height_ratios=[1,1])
                gssub4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub3[i], hspace=0.25)
                gssub5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub4[e], hspace=0.2, height_ratios=[1,12])
                times = ['before_', 'after_']
                for t in range(2):
                    catList=['↓','↑','-']
                    time = times[t]
                    ax2 = plt.subplot(gssub5[0, t])
                    print('we are here!!!!!!',time, event, i)
                    #get ids and counts of modulated cells
                    modCells_Id, modCells_count, counts=groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList, time, event)

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
                        ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),bbox_to_anchor=(0.05, 0.6, 0.85, 0.1), frameon=False, fontsize=8)#, labelcolor=label_colors_List)
                        bottom_pos += bar_width
                        ax2.get_yaxis().set_visible(False)
                        # ax2.get_xaxis().set_visible(False)
                        ax2.spines[['left','right','bottom', 'top']].set_visible(False)
                    ax2.text(0.5, 2, (f'modulation {time[:-1]} {event[:-5]} onset'), ha='center',va='center', transform=ax2.transAxes, fontsize=10)
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
                        if recordings=='MLI':
                            modulated_paw_df_visual = modulated_paw_df.drop(modulated_paw_df[modulated_paw_df['cell_global_Id'] == 25].index)
                        else:
                            modulated_paw_df_visual=modulated_paw_df

                        gssub6 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gssub5[1, t], hspace=0.7, wspace=0.3,height_ratios=[2.5, 2,2])
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

                        behavior_par = ['swingDuration', 'swingSpeed', 'swingLength', 'stanceDuration']
                        behavior_par = ['swingDuration', 'swingLength']
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
                            if (p_value > 0.05) or (r2 < 0.095) or (r2 == 1):
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
        fname = 'fig_%s_ephys_psth_group_analysis_%s_v%s' % (recordings,condition,figVersion)

        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
        # plt.savefig(fname + '.svg')
    ########################################################################################################
    def ephysPSTHSwing_StanceFig_before_After(self, figVersion, cellType, ephysPSTHDict, swingStanceD, ephys, pawPos, df_psth, df_cells,condition,event):

        def performLinearRegressionAndShowStats(data):
            x = data[:,0]
            y = data[:,1]
            # Add constant to predictor (intercept term)
            x = sm.add_constant(x)  # Adds a column of 1s for the intercept

            # Fit the regression model
            model = sm.OLS(y, x).fit()  # Ordinary Least Squares

            # Display the summary of the model
            print(model.summary())

            # Extract key values
            slope = model.params[1]
            intercept = model.params[0]
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            p_value_slope = model.pvalues[1]
            f_stat = model.fvalue
            f_pvalue = model.f_pvalue
            stderr_slope = model.bse[1]
            conf_int = model.conf_int(alpha=0.05)  # 95% Confidence Interval

            # Residual Analysis
            residuals = model.resid
            rmse = np.sqrt(np.mean(residuals ** 2))  # Root Mean Squared Error

            # Print key results
            print("\nKey Results:")
            print(f"Slope: {slope:.3f}")
            print(f"Intercept: {intercept:.3f}")
            print(f"R-squared: {r_squared:.3f}")
            print(f"Adjusted R-squared: {adj_r_squared:.3f}")
            print(f"Slope p-value: {p_value_slope:.3f}")
            print(f"F-statistic: {f_stat:.3f}, F-stat p-value: {f_pvalue:.3f}")
            print(f"Slope Standard Error: {stderr_slope:.3f}")
            print(f"95% Confidence Interval for Slope: {conf_int[1]}")

        # figure #################################
        fig_width = 18  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 16, 'axes.titlesize': 14, 'font.size': 18, 'xtick.labelsize': 16,
                  'ytick.labelsize': 16, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
                               height_ratios=[1.3, 3, 4]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.50, hspace=0.25)

        # figure panel names
        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.05)
        plt.figtext(0.01, 0.955, 'A', clip_on=False, color='black',  size=23)
        plt.figtext(0.01, 0.76, 'B', clip_on=False, color='black', size=23)
        #plt.figtext(0.01, 0.585, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.01, 0.59, 'C', clip_on=False, color='black',  size=23)
        plt.figtext(0.01, 0.34, 'D', clip_on=False, color='black',  size=23)
        plt.figtext(0.45, 0.34, 'E', clip_on=False, color='black', size=23)
        plt.figtext(0.45, 0.27, 'F', clip_on=False, color='black', size=23)

        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.25, wspace=0.1)

        #paw Ids
        pawList = ['FL', 'FR', 'HL', 'HR']
        #paw colors
        col = ['C0', 'C1', 'C2', 'C3']

        #swing and stance colors for PSTH
        twoCols = {'swing':'blueviolet', 'stance':'0.4'}
        firingRate = []

        #length of example trace
        startx = [24, 21]
        xLength = 10
        # pdb.set_trace()
        if cellType == 'MLI':
            n = 1  # chose the trace to show here
        elif cellType == 'PC':
            n = 0

        pawMax = []
        pawMin = []

        #first subplot
        ax0 = plt.subplot(gssub0[0])

        isis = np.diff(ephys[0][(ephys[0] > 10.) & (ephys[0] <= 52)])
        firingRate.append([n, 1. / np.mean(isis)])

        # plot paw position
        for i in range(2):

            pawMin.append(np.min(pawPos[i][:, 1]))
            pawMax.append(np.max(pawPos[i][:, 1]))
            ax0.plot(pawPos[i][:, 0], pawPos[i][:, 1], c=col[i], lw=2, label=pawList[i])

            idxSwings = swingStanceD['swingP'][i][1]
            indecisiveSteps = swingStanceD['swingP'][i][3]
            recTimes = swingStanceD['forFit'][i][2]

            for j in range(len(idxSwings)):
                idxStart = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[j][0]]))
                idxEnd = np.argmin(np.abs(pawPos[i][:, 0] - recTimes[idxSwings[j][1]]))
                ax0.plot(pawPos[i][idxStart, 0], pawPos[i][idxStart, 1], 'x', c=col[i], alpha=0.5, lw=0.5)
                ax0.plot(pawPos[i][idxEnd, 0], pawPos[i][idxEnd, 1], '+', c=col[i], alpha=0.5, lw=0.5)
        mmax = np.max(pawMax)
        mmin = np.min(pawMin)
        print(n, mmax, mmin, pawMin)

        #plot spiking activity
        ax0.eventplot(ephys, lineoffsets=550, linelengths=200, linewidths=0.2, color='0.4', alpha=0.7)
        ax0.legend(frameon=False, fontsize=20)
        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'x (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        ax0.set_xlim(startx[n], startx[n] + xLength)
        ax0.set_ylim(430, 680)

        ####################################
        #plot strides raster plot
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1], hspace=0.1, wspace=0.20)
        psth1 = ephysPSTHDict

        if event=='stance':
            mirrorEvent='swing'
        elif event=='swing':
            mirrorEvent='stance'
        for i in range(4):

            ax2 = plt.subplot(gssub1[i])
            if i == 0:
                if cellType == 'MLI':

                    ax2.annotate(f'{event} onset', xy=(0.01, 168), annotation_clip=False, xytext=None,
                                         textcoords='data', fontsize=12, arrowprops=None, color=twoCols[f'{event}'])
                    ax2.annotate(f'{mirrorEvent} onset', xy=(-0.18, 168), annotation_clip=False, xytext=None, textcoords='data', fontsize=12, arrowprops=None, color=col[i])
                elif cellType == 'PC':
                    textx = ax2.annotate(f'{event} onset', xy=(0.01, 190), annotation_clip=False, xytext=None,
                                         textcoords='data', fontsize=10, arrowprops=None, color=twoCols[f'{event}'])

            ax2.axvline(x=0, ls='--', c='0.8')
            ax2.set_title('   ' + pawList[i], loc='left', color=col[i], fontweight='bold', fontsize=16)
            ax2.eventplot(psth1[i][f'spikeTimesCentered{event.title()}StartSorted'], color=twoCols[f'{event}'], linewidths=1)
            if event=='swing':
                ax2.eventplot(psth1[i]['strideEndSwingCenteredSorted'], color=col[i], linewidths=2)
                ax2.eventplot(psth1[i]['stanceStartSorted'], color=col[i], linewidths=2)
            elif event=='stance':
                ax2.eventplot(psth1[i]['swingStartSorted'], color=col[i], linewidths=2)
                ax2.eventplot(psth1[i]['strideEndStanceCenteredSorted'], color=col[i], linewidths=2)
            self.layoutOfPanel(ax2, yLabel=('strides' if i == 0 else None), xyInvisible=[True, False])
            ax2.set_xlim(-0.3, 0.4)
            ax2.yaxis.set_major_locator(MultipleLocator(50))


            #plot PSTHs
            ax3 = plt.subplot(gssub1[i + 4])
            ax3.axvline(x=0, ls='--', c='0.8')

            ax3.step(psth1[i][f'psth_{event}OnsetAligned'][0], psth1[i][f'psth_{event}OnsetAligned'][1], where='mid',
                     color=twoCols[f'{event}'])
            ax3.step(psth1[i][f'psth_{event}OnsetAligned'][0],
                     psth1[i][f'psth_{event}OnsetAligned_5-50-95perentiles'][1], where='mid', color=twoCols[f'{event}'],
                     alpha=0.3)
            ax3.fill_between(psth1[i][f'psth_{event}OnsetAligned'][0],
                             psth1[i][f'psth_{event}OnsetAligned_5-50-95perentiles'][0],
                             psth1[i][f'psth_{event}OnsetAligned_5-50-95perentiles'][2], step='mid', color=twoCols[f'{event}'],
                             alpha=0.1)

            self.layoutOfPanel(ax3, xLabel=f'time centered on {event} onset (s)',
                               yLabel=('PSTH (spk/s)' if i == 0 else None), xyInvisible=[False, False])
            ax3.set_xlim(-0.3, 0.4)

            #ax3.set_ylim(15,60)
            ax3.yaxis.set_major_locator(MultipleLocator(10))


            #Divide next section in 2 columns: 1st for fraction of modulated recordings, 2nd for Z-score changes
            gssub1b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], hspace=0.4, wspace=0.2, width_ratios=[4,2])


            #time around event for keys
            times = ['before_', 'after_']

            #looking at specific paw values in the df that contains single values
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]


            #looking at specific paw values in the df that contains psth and zscore
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]

            #detect and remove empty cells, happens when there's no stride in a condition
            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)

            #the before after plot:
            gssub4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub1b[0], hspace=0.3,
                                                      wspace=0.3, width_ratios=[10,10], height_ratios=[2,2])
            # first column, the fraction of modulated recs for 4 paws, 1st column contains paw Id
            #gssub2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub4[0, 0:2])
            #iterate through time (before/after)
            #for t in range(2):

            #axis style
            # #ax3.get_yaxis().set_visible(False)
            # ax3.set_xlim(0, 100)
            # xLocator = MultipleLocator(25)
            # #ax3.xaxis.set_major_locator(xLocator)
            # ax3.get_xaxis().set_visible((False if i!=3 or t!=1 else True) )
            # if i==3 and t==1:
            #     self.layoutOfPanel(ax3, xyInvisible=[False,True])
            #     #pass
            #     #ax3.spines[['left', 'right', 'top']].set_visible(False)
            # else:
            #     self.layoutOfPanel(ax3, xyInvisible=[True,True])
            #     #pass
            #     #ax3.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
            #paw_df_FL = df_cells[(df_cells['paw'] == pawList[0])]
            ax20= plt.subplot(gssub4[0,0])
            ax21 = plt.subplot(gssub4[0,1])
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[0])]
            #pdb.set_trace()
            cell_Id = np.unique(paw_psth_df['cell_global_Id'])
            zs_stance_all = []
            zs_swing_all = []
            for n in cell_Id:
                zs_stance_psth_mean = paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_zscore'].mean()[1]
                zs_stance_all.append(zs_stance_psth_mean)
                zs_swing_psth_mean = paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_swingOnsetAligned_zscore'].mean()[1]
                zs_swing_all.append(zs_swing_psth_mean)
            zs_time = paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_time'].values[0]
            zs_stance_all = np.asarray(zs_stance_all)
            zs_swing_all = np.asarray(zs_swing_all)
            #pdb.set_trace()
            if cellType == 'PC':
                factor=0.5
            else:
                factor = 0.3
            peak_indices = np.argmax(zs_swing_all, axis=1)

            # Sort rows by the position of the positive peak
            sorted_indices = np.argsort(peak_indices)
            sorted_zs_swing_all = zs_swing_all[sorted_indices]
            heatmapStance = ax20.imshow(sorted_zs_swing_all,aspect='auto',cmap='RdBu_r',interpolation='none',extent=[zs_time[0], zs_time[-1], 0, sorted_zs_swing_all.shape[0]],
                                  vmin=-factor*np.max(np.abs(sorted_zs_swing_all)),  # Ensure 0 is white
                                  vmax=factor*np.max(np.abs(sorted_zs_swing_all))
            )

            # Add colorbar
            ax20.axvline(0, ls='--', color='k')
            cbar = plt.colorbar(heatmapStance)
            cbar.set_label('Z-score')
            ax20.set_xlim(-0.3, 0.4)
            self.layoutOfPanel(ax20, xLabel=f'time centered on swing onset (s)', yLabel='cell (N)')
            peak_indices = np.argmax(zs_stance_all, axis=1)

            # Sort rows by the position of the positive peak
            sorted_indices = np.argsort(peak_indices)
            sorted_zs_stance_all = zs_stance_all[sorted_indices]
            heatmapSwing = ax21.imshow(sorted_zs_stance_all,aspect='auto',cmap='RdBu_r',interpolation='none',extent=[zs_time[0], zs_time[-1], 0, sorted_zs_stance_all.shape[0]],
                                  vmin=-factor*np.max(np.abs(sorted_zs_stance_all)),  # Ensure 0 is white
                                  vmax=factor*np.max(np.abs(sorted_zs_stance_all))
            )
            # Add colorbar
            ax21.axvline(0,ls='--',color='k')
            cbar = plt.colorbar(heatmapSwing)
            cbar.set_label('Z-score')
            ax21.set_xlim(-0.3, 0.4)
            self.layoutOfPanel(ax21, xLabel=f'time centered on stance onset (s)', yLabel='cell (N)')

            # scatter plots of the AUC ###################################################
            #look at AUC or peak
            zscore = ['AUC', 'peak']
            zscorePar = zscore[0]
            #choose interval
            intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
            interval = intervals[0]
            #key to access the variable
            eventB = 'swing'
            zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                          'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
            zscore_keyB = ['before_%sOnset_z-score_%s_%s' % (eventB, zscorePar, interval),
                          'after_%sOnset_z-score_%s_%s' % (eventB, zscorePar, interval)]
            #list of modulation categories
            #catList = ['↓', '↑', '-']
            # label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': '0.9'}

            #go through different catefories
            #for c in range(len(catList)):
            #keys for before and after values of z-score AUC
            varKey = 'modulation_category_'
            key_after = (f'{varKey}after_{event}Onset')
            key_before= (f'{varKey}before_{event}Onset')


            #second colum and 2 lines for the before after plot
            ax40= plt.subplot(gssub4[1,0])
            ax41 = plt.subplot(gssub4[1,1])
            #only show for FL paw
            paw_df_FL = df_cells[(df_cells['paw'] == pawList[0])]
            #only look at a single category
            #modulated_paw_df = paw_df_FL[(paw_df_FL[key_before] == catList[c])]
            #get the Id of the modulated recs for that category
            modRec_Id = np.unique(paw_df_FL['rec_global_Id'])
            swingPairs = []
            stancePairs = []
            # paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_zscore'].mean()[1]
            #pdb.set_trace()
            for n in cell_Id:
                zs_time = paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_time'].values[0]
                for r in paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_zscore'].keys():
                    zsc = paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_zscore'][r][1]
                    maxIdx = np.argmax(zsc)
                    minIdx = np.argmin(zsc)

                    #ax41.plot(zs_time[maxIdx],zsc[maxIdx],'+',c='C6')
                    #ax41.plot(zs_time[minIdx], zsc[minIdx], '+', c='C7')

                for r in paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_swingOnsetAligned_zscore'].keys():
                    zsc = paw_psth_df[paw_psth_df['cell_global_Id'] == n]['psth_swingOnsetAligned_zscore'][r][1]
                    maxIdx = np.argmax(zsc)
                    minIdx = np.argmin(zsc)

                    #ax40.plot(zs_time[maxIdx], zsc[maxIdx], '+', c='C6')
                    #ax40.plot(zs_time[minIdx], zsc[minIdx], '+', c='C7')

                    #peak_indices = np.argmax(zsc, axis=1)
                    #pdb.set_trace()
                #print(len(paw_psth_df[paw_psth_df['cell_global_Id']==n]['psth_stanceOnsetAligned_zscore']))
                #print(len(paw_psth_df[paw_psth_df['cell_global_Id'] == n]['psth_swingOnsetAligned_zscore']))

            #ax40.axhspan(-3.29, 3.29, color='gray', alpha=0.3)
            #ax41.axhspan(-3.29, 3.29, color='gray', alpha=0.3)

            #pdb.set_trace()
            #for each rec of the category
            for r in range(len(paw_df_FL)):
                #get the AUC before/after
                modulationAfter=paw_df_FL[paw_df_FL['rec_global_Id'] == modRec_Id[r]][key_after]
                modulationBefore= paw_df_FL[paw_df_FL['rec_global_Id'] == modRec_Id[r]][key_before]
                #get their sign of modulation before and after
                modSignAfter=modulationAfter.values[0]
                modSignBefore = modulationBefore.values[0]
                label_colors = {'↑': 'C8', '↓': 'C4', '-': '0.9'}
                #pdb.set_trace()
                if (modSignBefore == '↑') or (modSignBefore == '↓'):
                    cc = 'C4'
                if (modSignAfter == '↑') or (modSignAfter == '↓'):
                    cc = 'C4'
                else:
                    cc = '0.9'
                # plot the peak location
                #ax41.plot(

                #plot the recs AUC as points and change color based on modulation sign
                ax41.scatter(paw_df_FL[zscore_key[0]].iloc[r],paw_df_FL[zscore_key[1]].iloc[r], color=cc,edgecolors='grey', lw=0.05, alpha=0.3, s=50)#, markerfacecolor=label_colors[modSignBefore])
                stancePairs.append([paw_df_FL[zscore_key[0]].iloc[r], paw_df_FL[zscore_key[1]].iloc[r]])

                #ax4.scatter(paw_df_FL[zscore_key[0]].iloc[r],paw_df_FL[zscore_key[1]].iloc[r], color=label_colors[modSignBefore],edgecolors='grey', lw=0.05, alpha=1, s=80)#, markerfacecolor=label_colors[modSignBefore])
                #ax4.scatter([0.5], paw_df_FL[zscore_key[1]].iloc[r], color=label_colors[modSignAfter], edgecolors='grey', lw=0.05, alpha=1, s=80)#, markerfacecolor=label_colors[modSignAfter])
                #ax4.plot([0, 0.5], [paw_df_FL[zscore_key[0]].iloc[r], modulated_paw_df[zscore_key[1]].iloc[r]],alpha=0.1,color='k')
                # also for swing onset
                key_afterB = (f'{varKey}after_{eventB}Onset')
                key_beforeB = (f'{varKey}before_{eventB}Onset')
                modulationAfter = paw_df_FL[paw_df_FL['rec_global_Id'] == modRec_Id[r]][key_afterB]
                modulationBefore = paw_df_FL[paw_df_FL['rec_global_Id'] == modRec_Id[r]][key_beforeB]
                # get their sign of modulation before and after
                modSignAfter = modulationAfter.values[0]
                modSignBefore = modulationBefore.values[0]
                label_colors = {'↑': 'C8', '↓': 'C4', '-': '0.9'}
                # pdb.set_trace()
                if (modSignBefore == '↑') or (modSignBefore == '↓'):
                    cc = 'C4'
                if (modSignAfter == '↑') or (modSignAfter == '↓'):
                    cc = 'C4'
                else:
                    cc = '0.9'
                # plot the recs AUC as points and change color based on modulation sign
                ax40.scatter(paw_df_FL[zscore_keyB[0]].iloc[r], paw_df_FL[zscore_keyB[1]].iloc[r], color=cc, edgecolors='grey', lw=0.05, alpha=0.3,s=50)  # , markerfacecolor=label_colors[modSignBefore])
                swingPairs.append([paw_df_FL[zscore_keyB[0]].iloc[r],paw_df_FL[zscore_keyB[1]].iloc[r]])

            swingPairs = np.asarray(swingPairs)
            swingPairsSorted = swingPairs[swingPairs[:, 0].argsort()]
            slope, intercept = np.polyfit(swingPairsSorted[:,0], swingPairsSorted[:,1], 1)  # Linear regression
            y_pred = slope * swingPairsSorted[:,0] + intercept
            # Calculate residuals and standard error
            residuals = swingPairsSorted[:,1] - y_pred
            s_err = np.sqrt(np.sum(residuals ** 2) / (len(swingPairsSorted[:,1]) - 2))

            # Calculate confidence intervals
            n = len(swingPairsSorted[:,0])
            t_value = t.ppf(0.975, df=n - 2)  # 95% CI, two-tailed t-test
            mean_x = np.mean(swingPairsSorted[:,0])
            ci = t_value * s_err * np.sqrt(1 / n + (swingPairsSorted[:,0] - mean_x) ** 2 / np.sum((swingPairsSorted[:,0] - mean_x) ** 2))

            # Calculate upper and lower bounds
            ci_upper = y_pred + ci
            ci_lower = y_pred - ci
            #if cellType != 'PC':
            ax40.plot(swingPairsSorted[:,0], y_pred, color='C8', label=f'Fit: y = {intercept:.2f} + {slope:.2f}x')
            ax40.fill_between(swingPairsSorted[:,0], ci_lower, ci_upper, color='C8', alpha=0.2, label='95% CI')

            # linear regression also for stance onset  ###########################
            stancePairs = np.asarray(stancePairs)
            stancePairsSorted = stancePairs[stancePairs[:, 0].argsort()]
            slope, intercept = np.polyfit(stancePairsSorted[:, 0], stancePairsSorted[:, 1], 1)  # Linear regression
            y_pred = slope * stancePairsSorted[:, 0] + intercept
            # Calculate residuals and standard error
            residuals = stancePairsSorted[:, 1] - y_pred
            s_err = np.sqrt(np.sum(residuals ** 2) / (len(stancePairsSorted[:, 1]) - 2))

            # Calculate confidence intervals
            n = len(stancePairsSorted[:, 0])
            t_value = t.ppf(0.975, df=n - 2)  # 95% CI, two-tailed t-test
            mean_x = np.mean(stancePairsSorted[:, 0])
            ci = t_value * s_err * np.sqrt(1 / n + (stancePairsSorted[:, 0] - mean_x) ** 2 / np.sum((stancePairsSorted[:, 0] - mean_x) ** 2))

            # Calculate upper and lower bounds
            ci_upper = y_pred + ci
            ci_lower = y_pred - ci
            #if cellType != 'PC':
            ax41.plot(stancePairsSorted[:, 0], y_pred, color='C8', label=f'Fit: y = {intercept:.2f} + {slope:.2f}x')
            ax41.fill_between(stancePairsSorted[:, 0], ci_lower, ci_upper, color='C8', alpha=0.2, label='95% CI')
            print('Stats for stance values:')
            performLinearRegressionAndShowStats(stancePairsSorted)
            print('Stats for swing values:')
            performLinearRegressionAndShowStats(swingPairsSorted)
            #compare before and after array with t-test
            # t_value, t_test_p_value = stats.ttest_rel(modulated_paw_df[zscore_key[0]], modulated_paw_df[zscore_key[1]])
            # #get the stars
            # star_trial = groupAnalysis.starMultiplier(t_test_p_value)
            # #show stars
            # ax4.text(0.5, (0.96), f'{star_trial} \np={t_test_p_value:.2f}' if t_test_p_value > 0.05 else f'{star_trial}',
            #          ha='center', va='center', transform=ax4.transAxes,
            #          style='italic', fontfamily='serif', fontsize=14)
            # #stylise
            #self.layoutOfPanel(ax41, xLabel=f'PSTH Z-score {zscorePar} :\n {pawList[0]} {event} onset, before', yLabel=
            #    f'{pawList[0]} {event} onset, after ')
            self.layoutOfPanel(ax41, xLabel=f'time of z-score peak :\n {pawList[0]} {event} onset', yLabel=
                f'peak amplitude (z-score)')
            #xLocator1 = MultipleLocator(0.5)
            #ax40.xaxis.set_major_locator(xLocator1)
            ax41.axhline(0, ls='--', c='0.8')
            ax41.axvline(0, ls='--', c='0.8')

            #self.layoutOfPanel(ax40, xLabel=f'PSTH Z-score {zscorePar}:\n  {pawList[0]} {eventB} onset, before', yLabel=
            #    f'PSTH Z-score {zscorePar} :\n {pawList[0]} {eventB} onset, after')
            self.layoutOfPanel(ax40, xLabel=f'time of z-score peak\n  {pawList[0]} {eventB} onset', yLabel=
                f'peak amplitude (z-score)')
            #xLocator1 = MultipleLocator(0.5)
            #ax41.xaxis.set_major_locator(xLocator1)
            ax40.axhline(0, ls='--', c='0.8')
            ax40.axvline(0, ls='--', c='0.8')
            ax40.set_xlim(-0.3,0.42)
            ax41.set_xlim(-0.3, 0.42)
            #ax4.set_xticklabels(['','before','after'])
            #ax4.tick_params(axis='x', which='major', labelsize=16)
            #ax4.set_ylim(-0.3, 1)

            #get ids of modulated recs for each paw
            modulatedRecDic={}
            modulatedRecDic_paw={}

            eventList = ['swing', 'stance']
            for ev in range(len(eventList)):
                modulatedRecDic['swing'] = np.empty(0)
                modulatedRecDic['stance'] = np.empty(0)
                modulatedRecDic_paw['swing']={}
                modulatedRecDic_paw['stance']={}
            for ev in range(len(eventList)):
                for i in range(4):
                    mod_global_key = f'modulation_category_{eventList[ev]}Onset'
                    paw_df = df_cells[(df_cells['paw'] == pawList[i])]
                    # modulatedRecDic_paw[eventList[ev]][pawList[i]]
                    # modulated means respond to at least one paw
                    # get eventList[ev]ent modulated recs for each paw:
                    modulated_paw_df = paw_df[(paw_df[mod_global_key] != '--')]
                    # get the Id of the modulated recs for that category
                    modulatedRecDic_paw[eventList[ev]][pawList[i]]= np.unique(modulated_paw_df['rec_global_Id'])
                    modulatedRecDic[eventList[ev]]=np.append(modulatedRecDic[eventList[ev]],np.unique(modulated_paw_df['rec_global_Id']))
                    tot_rec = len(np.unique(paw_df['rec_global_Id']))

            #throw away overlaps
            for ev in range(2):
                modulatedRecDic[f'mod_{eventList[ev]}'] = np.unique(modulatedRecDic[eventList[ev]])

            gssub1c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1b[1], hspace=0.2, wspace=0.2,height_ratios=[0.7,1.1])
            gssub1d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1c[0], hspace=0.025, wspace=0.2,height_ratios=[0.35, 0.5])
            gssub1e = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1c[1], hspace=0.15, wspace=0.2,
                                                       height_ratios=[0.5, 0.5])
            ax5a=plt.subplot(gssub1d[0])
            modulated_nb=len(np.unique(np.concatenate((modulatedRecDic['mod_stance'], modulatedRecDic['mod_swing']))))
            modulated_fraction=len(np.unique(np.concatenate((modulatedRecDic['mod_stance'], modulatedRecDic['mod_swing']))))/tot_rec*100

            ax5a.barh(0,100, color='0.9', edgecolor='white', alpha=0.8)
            ax5a.barh(0, modulated_fraction, color='k', edgecolor='black', alpha=0.8)

            # ax5a.annotate(f'{modulated_fraction:.1f}% ({modulated_nb}/{tot_rec})', (modulated_fraction + 2,0), fontsize=12)
            ax5a.annotate(f'{modulated_fraction:.1f}%', (modulated_fraction/2.25 + 2,0), fontsize=12, color='white', ha='center', va='center')
            ax5a.set_ylim(0.95,-0.95)
            mod_swing_and_stance = np.intersect1d(modulatedRecDic['mod_swing'], modulatedRecDic['mod_stance'])

            #pdb.set_trace()
            ax5b = plt.subplot(gssub1d[1])
            ax5b.barh([0,1,2],[100,100,100], color='0.9', edgecolor='white', alpha=0.8)
            mod_swing=len(modulatedRecDic['mod_swing'])/tot_rec*100
            mod_stance=len(modulatedRecDic['mod_stance']) / tot_rec * 100
            mod_both = len(mod_swing_and_stance) / tot_rec * 100
            ax5b.barh([2], [mod_swing], color='0.8', edgecolor='white', alpha=0.5)
            ax5b.barh([1], [mod_stance], color='0.4', edgecolor='white',alpha=0.8)
            ax5b.barh([0], [mod_both], color='0.1', edgecolor='white', alpha=0.5)
            # ax5b.annotate(f'{mod_swing:.1f}% ({len(modulatedRecDic["mod_swing"])}/{tot_rec})', (len(modulatedRecDic["mod_swing"])/tot_rec*100 + 2,1), fontsize=12)
            # ax5b.annotate(f'{mod_stance:.1f}% ({len(modulatedRecDic["mod_stance"])}/{tot_rec})', (len(modulatedRecDic["mod_stance"])/tot_rec*100 + 2, -0), fontsize=12)
            ax5b.annotate(f'{mod_swing:.1f}%', (len(modulatedRecDic["mod_swing"])/tot_rec*100 / 2.25,2), fontsize=12, ha='center', va='center')
            ax5b.annotate(f'{mod_stance:.1f}%', (len(modulatedRecDic["mod_stance"])/tot_rec*100 / 2.25, 1), fontsize=12, color='white', ha='center', va='center')
            ax5b.annotate(f'{mod_both:.1f}%', (len(mod_swing_and_stance) / tot_rec * 100 / 2.25, 0), fontsize=12, color='white', ha='center', va='center')

            ax5a.spines['left'].set_visible(False)
            ax5a.spines['top'].set_visible(False)
            ax5a.spines['right'].set_visible(False)
            ax5a.spines['bottom'].set_visible(False)
            ax5a.locator_params(axis='y', nbins=1)
            ax5a.set_yticklabels(['','all \n events'])
            ax5a.yaxis.set_major_locator(MultipleLocator(1))
            ax5a.get_xaxis().set_visible(False)
            ax5a.set_xlim(0, 100)
            ax5b.spines['left'].set_visible(False)
            ax5b.spines['top'].set_visible(False)
            ax5b.spines['right'].set_visible(False)
            ax5b.spines['bottom'].set_visible(False)
            ax5b.yaxis.set_major_locator(MultipleLocator(1))
            ax5b.get_xaxis().set_visible(False)
            ax5b.set_xlim(0, 100)
            ax5b.set_yticklabels(['','both','stance','swing'])

            for ev in  range(len(eventList)):
                ax5c = plt.subplot(gssub1e[ev])
                if ev==0:
                    alpha = 0.4
                    ax5c.get_xaxis().set_visible(False)
                    ax5c.spines['bottom'].set_visible(False)
                else:
                    alpha = 0.8
                    ax5c.text(0.5, -0.25, (f' fraction of modulated recordings (%)'),
                             ha='center', va='center',
                             transform=ax5c.transAxes, fontsize=18)
                for pw  in reversed(range(4)):
                    modulatedRec_paw=len(modulatedRecDic_paw[eventList[ev]][pawList[pw]])
                    modulatedFrac=modulatedRec_paw/tot_rec*100
                    ax5c.barh(pawList[pw], [100,100,100,100], color='0.9', edgecolor='white', alpha=alpha)
                    ax5c.barh(pawList[pw], modulatedFrac, color=f'C{pw}', edgecolor='white', alpha=alpha)
                    # ax5c.annotate(f'{modulatedFrac:.1f}% ({modulatedRec_paw}/{tot_rec})',(modulatedFrac+2,pawList[pw]), fontsize=12)
                    ax5c.annotate(f'{modulatedFrac:.1f}%',(modulatedFrac/2.5,pawList[pw]), fontsize=12, ha='center', va='center')
                    ax5c.spines['left'].set_visible(False)
                    ax5c.spines['top'].set_visible(False)
                    ax5c.spines['right'].set_visible(False)
                    ax5c.set_xlim(0,100)
                    ax5c.set_title(f'{eventList[ev]}',fontsize=16)
                    ax5c.xaxis.set_major_locator(MultipleLocator(25))
            # pdb.set_trace()

            # for i in reversed(range(4)):

                # modulatedFrac = modulatedNb / len(tot_rec) * 100
                # ax1.barh(pawList[i], [100, 100, 100, 100], color='white', edgecolor='black', alpha=alpha)
                # ax1.barh(pawList[i], modulatedFrac, color=f'C{i}', edgecolor='black', alpha=alpha)
                # ax1.annotate(f'{modulatedNb}/{total}', (modulatedFrac + 2, pawList[i]))
                # ax1.spines['left'].set_visible(False)
                # ax1.set_xlim(0, 100)
                # ax1.set_title(f'{events[e][:-5]}', fontsize=14)
            if event=='stance' :
                fig='fig'
            elif event == 'swing':
                fig='fig_supp'


            fname = f'{fig}_ephys_psth_Z-score_{zscorePar}_recording_based_{cellType}_{event}Onset_{condition}_{figVersion}'


        # plt.savefig(fname + '.png')
        # plt.show()
            plt.savefig(fname + '.pdf')
    #############################################################################################################
    def psthGroupFigure_early_vs_late(self, figVersion,cellType, df_psth, df_cells,condition,event, pawNb):
        pawList = ['FL', 'FR', 'HL', 'HR']

        #figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 21  # width in inches
        if pawNb==1:
            fig_height = 17
        else:
            fig_height=70# height in inches
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
        gs = gridspec.GridSpec(1, 1,  # ,
                               width_ratios=[ 1]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        #panel Ids
        gs.update(wspace=0.2, hspace=0.2)
        if pawNb==1:
            plt.figtext(0.035, 0.93, 'A', clip_on=False, color='black',  size=22)
            plt.figtext(0.53, 0.93, 'B', clip_on=False, color='black',  size=22)
            plt.figtext(0.035, 0.69, 'C', clip_on=False, color='black',  size=22)
            plt.figtext(0.53, 0.69, 'D', clip_on=False, color='black',  size=22)
            plt.figtext(0.035, 0.57, 'E', clip_on=False, color='black',  size=22)
            plt.figtext(0.53, 0.57, 'F', clip_on=False, color='black',  size=22)
            plt.figtext(0.035, 0.39, 'G', clip_on=False, color='black',  size=22)
            plt.figtext(0.53, 0.39, 'H', clip_on=False, color='black',  size=22)

        if pawNb==1:
            plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.07)
        else:
            plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.03)
        modCells={}
        for i in range(pawNb):
            #looking at specific paw values in the df that contains single values
            paw_df = df_cells[(df_cells['paw'] == pawList[i])]

            #looking at specific paw values in the df that contains psth and zscore
            paw_psth_df = df_psth[(df_psth['paw'] == pawList[i])]
            # detect and remove empty cells, happens when there's no stride in a condition
            empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
            # loop through the rows of the DataFrame
            for index, value in empty_cells.iteritems():
                if value:
                    print(f'Empty cell detected in line {index}')
                    paw_df = paw_df.drop(index)
                    paw_psth_df = paw_psth_df.drop(index)
            #big panels for each paw
            gssub3 = gridspec.GridSpecFromSubplotSpec(pawNb, 1, subplot_spec=gs[0], hspace=0.1)

            gssub4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub3[i], hspace=0.1)
            if pawNb==1:
                #sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gssub4[0], hspace=0.3,
                                                          height_ratios=[7,2, 20])
            else:
                # if more that one paw don't show examples z-score (supp figure) the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub4[0], hspace=0.1,
                                                          height_ratios=[1, 20])

            #time around event for keys
            times = ['before_', 'after_']
            #iterate through time (before/after)

            modCells[pawList[i]]= {}
            modCells[pawList[i]]['all']= {}
            for t in range(2):
                # modCells[pawList[i]]['before']=np.empty(0)
                # modCells[pawList[i]]['after']=np.empty(0)
                #define modulation catgories
                catList = ['↓', '↑', '-']
                #time to look at
                time = times[t]
                #change position of plot if looking at more than one paw
                if pawNb==1:
                    ax2 = plt.subplot(gssub5[1, t])
                else:
                    ax2 = plt.subplot(gssub5[0, t])


                #get Id and counts of modulated cells
                modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
                                                                                                   time, event,
                                                                                                   condition=None)
                if time=='before':
                    modCells[pawList[i]]['before']=modCells_Id
                else:
                    modCells[pawList[i]]['after'] = modCells_Id

                values = counts.values
                labels = counts.index
                percent = values / np.sum(values) * 100
                bottom_pos = 0
                label_colors = {'↑': 'indianred', '↓': 'cadetblue', '-': 'lightsteelblue'}
                label_colorsList = ['cadetblue', 'indianred', 'lightsteelblue']

                #plot fraction of modulated cells
                for l, label in enumerate(labels):

                    bar_width = percent[l]
                    bar_left = bottom_pos
                    ax2.barh(0, bar_width, left=bar_left, height=1, color=label_colors[label],
                             label=f'{label} {round(percent[l], 1)}% ({values[l]})')
                    xLocator = MultipleLocator(25)
                    ax2.xaxis.set_major_locator(xLocator)
                    ax2.set_ylim(1,-1)
                    #the text above with percent and counts bbox (left, up, spacing, right)
                    ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                               bbox_to_anchor=(0.05, 0.7, 0.9, 0.1), frameon=False,
                               fontsize=14, labelcolor='k')  # , labelcolor=label_colors_List)
                    bottom_pos += bar_width

                    #axis style
                    ax2.get_yaxis().set_visible(False)
                    ax2.set_xlim(0,100)
                    ax2.spines[['left', 'right', 'top']].set_visible(False)
                #plot title
                ax2.text(0.5, -0.6, (f'fraction of {cellType} modulated {time[:-1]} {pawList[i]} {event} onset (%)'), ha='center', va='center',
                         transform=ax2.transAxes, fontsize=14)
                #look at AUC or peak
                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                #choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[0]
                #key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]

                #go through different catefories
                for c in range(len(catList)):
                    #only look at cell modulated showing one category of modulation
                    modulated_mask = paw_df['cell_global_Id'].isin(modCells_Id[catList[c]])
                    modulated_paw_df = paw_df.loc[modulated_mask]
                    modulated_paw_psth_df=paw_psth_df.loc[modulated_mask]
                    # drop cells with only one rec
                    modulated_paw_df = modulated_paw_df.drop(
                        modulated_paw_df[modulated_paw_df['cell_trial_Nb'] == 1].index).reset_index()
                    #key for z_score and z_score time arrays
                    z_scoreKey=f'psth_{event}OnsetAligned_zscore'
                    z_scoreTimeKey = f'psth_{event}OnsetAligned_time'
                    #ids of modulated cells
                    cellsId=np.unique(modulated_paw_psth_df['cell_global_Id'])

                    #choose an example cell for each condition
                    if condition=='allSteps':
                        if pawNb==1 and i==0:
                            if event=='stance':
                                if t == 0 and c == 0:
                                    exampleCell=1 #2 is the same cell as cell #4 of t==1
                                elif t == 0 and c == 1:
                                    exampleCell=1 #1 is the same cell as cell #0 of t==1
                                elif t==1 and c==0:
                                    exampleCell=0
                                elif t==1 and c==1:
                                    exampleCell=4
                            #example cells for swing
                            else:
                                if t==0 and c==0:
                                    exampleCell=3
                                elif t==0 and c==1:
                                    exampleCell=5
                                elif t==1 and c==0:
                                    exampleCell=1
                                elif t==1 and c==1:
                                    exampleCell=0
                    elif condition == 'swingLengthLinear_lastRec_20_80':
                        if pawNb == 1 and i == 0:
                            if event == 'stance':
                                if t == 0 and c == 0:
                                    exampleCell = 7  # 2 is the same cell as cell #4 of t==1
                                elif t == 0 and c == 1:
                                    exampleCell = 8  # 1 is the same cell as cell #0 of t==1
                                elif t == 1 and c == 0:
                                    exampleCell = 15
                                elif t == 1 and c == 1:
                                    exampleCell = 4
                            # example cells for swing
                            else:
                                if t == 0 and c == 0:
                                    exampleCell = 3
                                elif t == 0 and c == 1:
                                    exampleCell = 5
                                elif t == 1 and c == 0:
                                    exampleCell = 1
                                elif t == 1 and c == 1:
                                    exampleCell = 0
                        #get the zscore of and zscore time of example cells for first and last trials
                        earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                    modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                        zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                                    modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                        lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (
                                    modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                        #subplot before and after for z-score examples
                        gssub5b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5[0,:2], hspace=0.2)
                        if t==0:
                            #for time before we need two examples for up and down modulated (c==0 and c==1)
                            gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], hspace=0.2)
                            #we don't care about non modulated (c==2)
                            if c<2:
                                ax1 = plt.subplot(gssub5c[c])
                                #vertical fill concerned time window (before -100ms to 0)
                                ax1.axvspan(-0.1, 0, color=label_colorsList[c], alpha=0.08)
                                #plot the example z-score for first and last trial
                                ax1.step(zscoreTime[0],earlyZscoreA[0][1], where='mid', color=label_colorsList[c], lw=1, alpha=0.5, label='first trial')
                                ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=label_colorsList[c], lw=1, alpha=1, label='last trial')
                                ax1.legend(frameon=False)
                        if t==1:
                            gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[1], hspace=0.2)
                            if c<2:
                                ax1 = plt.subplot(gssub5c[c])
                                # vertical fill concerned time window (after 0ms to 100)
                                ax1.axvspan(0, 0.1, color=label_colorsList[c], alpha=0.08)
                                #plot the example z-score for first and last trial for after event
                                ax1.step(zscoreTime[0],earlyZscoreA[0][1], where='mid', color=label_colorsList[c], lw=1, alpha=0.5, label='first trial')

                                ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=label_colorsList[c], lw=1, alpha=1,
                                         label='last trial')
                                ax1.legend(frameon=False)

                        #draw 0 line
                        ax1.axvline(0, ls='--', color='grey', lw=1,alpha=0.1)
                        ax1.axvline(0, ls='--', color='grey', alpha=0.1)
                        ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                        ax1.axhline(0, ls='--', color='grey', alpha=0.1)
                        self.layoutOfPanel(ax1, xLabel=f'time centered on {event} onset (s)', yLabel='PSTH Z-score', xyInvisible=[(False), False])
                    #for FL look only at those behavior parameters
                    if pawNb==1:
                        behavior_par = ['swingDuration', 'swingLengthLinear']
                        behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                        hr=np.full(len(behavior_par)+1,2)
                        hr[0]=2.8
                        gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par)+1, 3, subplot_spec=gssub5[2, t], hspace=0.5,
                                                                  wspace=0.3, height_ratios=hr)
                    # for all paw look at many more
                    else:
                        behavior_par = ['swingDuration', 'swingLengthLinear', 'stanceDuration', 'swingSpeed']
                        behavior_par_Name = ['swing duration (s)', 'swing length (cm)', 'stance duration (s)',
                                             'swing speed (cm/s)']
                        #just height ratio array
                        hr = np.full(len(behavior_par) + 1, 2)
                        hr[0] = 2.8
                        #panels for comparison and correlations of z-score AUC
                        gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par)+1, 3, subplot_spec=gssub5[1, t], hspace=0.5,
                                                                  wspace=0.3, height_ratios=hr)
                    #comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0, c])
                    #group the data per cell and trial categories
                    modulated_paw_df_first_late=modulated_paw_df.groupby(['cell_global_Id','trial_category'])[zscore_key[t]].mean().reset_index()
                    #define colors for seaborn
                    palette = sns.color_palette([label_colorsList[c]], 1)
                    grey = sns.color_palette(['grey'], 1)

                    #the invisible bar plot is to tighten things
                    sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y=zscore_key[t],  ax=ax3,
                                color=label_colorsList[c], alpha=0.001, errorbar=None)
                    #lineplot for the line between cells
                    sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y=zscore_key[t],  ax=ax3,
                                palette=grey,  legend=False, alpha=0.3, markers=True)
                    #scatterplot for the AUC values
                    sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y=zscore_key[t],
                                 ax=ax3, palette=palette, legend=False, alpha=0.8, markers=False, edgecolor='grey', lw=0.1, s=80)

                    #extract first and last trial z-score AUC arrays
                    late_days=modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category']=='first']
                    first_days = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']
                    #perform t-test
                    t_value, t_test_p_value=stats.ttest_rel(late_days[zscore_key[t]], first_days[zscore_key[t]], axis=0)




                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    #
                    ax3.text(0.5, 0.98, (f'{star_trial} \np={t_test_p_value:.2f}'), ha='center', va='center', transform=ax3.transAxes,
                             style='italic', fontfamily='serif', fontsize=11, color=label_colorsList[c])
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(f' {time[:-1]} {pawList[i]} {event} onset\n PSTH Z-score {zscorePar}' if c == 0 else ''),
                                       Leg=[1, 9])

                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)

                    #create a new column with a name without '-' to put the z-score value (necessary because statsmodel and scipy dont like '-' in the key names
                    zscore_key1 = 'zScoreAUC_' + times[t] + event
                    modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    #for each behavior parameter perform linear regression
                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p, c])

                        modulated_paw_df.dropna(subset=[zscore_key[t]], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df[zscore_key[t]]
                        #we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            r_v,p_v=stats.pearsonr(x,y)
                            r2 = np.square(r_value)

                            print(r_v, r_value)
                        except ValueError:
                            pass
                        #set a treshold to reduce alpha and highlight only pvalues<0.05 and r2>0.05
                        if (p_value > 0.05) or (r2 < 0.05) or (r2 == 1):
                            alpha = 0.1
                        else:
                            alpha = 0.8
                        #show linear reg plots
                        sns.regplot(x=x, y=y, ax=ax4, color=label_colorsList[c], scatter_kws={'alpha': alpha, 'edgecolor':'grey', 'lw':0.1},
                                    line_kws={'alpha': alpha, 'lw':1})
                        #annotate with values of r, r², p_values
                        ax4.text(0.73, 0.80, f"r = {r_value:.2f}\nr² = {r2:.2f}\np = {p_value:.2f}",
                                 transform=ax4.transAxes, fontsize=10,color='dimgrey')

                        self.layoutOfPanel(ax4, xLabel=('zscore %s' % zscorePar if p == len(behavior_par)-1 else ''),
                                           yLabel=(f'{behavior_par_Name[p]} ' if (c == 0) else ''),
                                           Leg=[1, 9])


        if pawNb==1 and event=='stance' :
            paws='FL'
            fig='fig'
        elif pawNb == 1 and event == 'swing':
            paws='FL'
            fig='fig_supp'
        else:
            paws='all_paws'
            fig='fig_supp'
        fname = f'{fig}_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paws}_{event}_onset_{condition}_{figVersion}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
    ###############################################################
    def psthGroupFigure_cell_based(self, figVersion, cellType, df_psth, df_cells, condition,  pawNb):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 16  # width in inches

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

        plt.figtext(0.01, 0.975, 'A', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.975, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.73, 'C', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.73, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.635, 'E', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.635, 'F', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.385, 'G', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.385, 'H', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.2, 'I', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.2, 'J', clip_on=False, color='black', size=26)

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
                gssub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.1)

                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub3[0], hspace=0.1,wspace=0.3)

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
                                                              wspace=0.4, height_ratios=hr)

                    # comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0,m])
                    # group the data per cell and trial categories
                    modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()
                    trialCat=['first','last']
                    # for cell in cellsId:
                    #     for tr in range(2):
                    #         modulated_paw_df_first_late['normZscoreFirst'] = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] ==trialCat[tr])]['z_score_abs']/modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] =='first')]['z_score_abs']

                    # define colors for seaborn
                    palette = sns.color_palette([f'C{i}'], 1)
                    palette_non = sns.color_palette(['0.8'], 1)
                    grey = sns.color_palette(['grey'], 1)
                    late_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']
                    first_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs']
                    for cell in cellsId:
                        late_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'last')]['z_score_abs'].values
                        first_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'first')]['z_score_abs'].values
                        if first_days_zscore_cell:
                            print('alalala', late_days_zscore_cell[0],first_days_zscore_cell[0])
                            ax3.plot(['first','last'],[first_days_zscore_cell/first_days_zscore_cell, late_days_zscore_cell/first_days_zscore_cell],lw=0.2,alpha=0.5, color='grey')
                            ax3.scatter(['first', 'last'], [first_days_zscore_cell / first_days_zscore_cell,late_days_zscore_cell / first_days_zscore_cell],edgecolor='k', facecolor=(f'C{i}' if m==0 else '0.8'),lw=0.1, s=80)
                    # the invisible bar plot is to tighten things
                    # sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y='normZscoreFirst', ax=ax3,
                    #             color='k', alpha=0.001, errorbar=None)
                    # # lineplot for the line between cells
                    # sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y='normZscoreFirst',
                    #              ax=ax3,
                    #              palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                    # # scatterplot for the AUC values
                    # sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id',
                    #                 y='normZscoreFirst',
                    #                 ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                    #                 lw=0.1, s=80)
                    ax3.set_xlim(-0.9,1.9)
                    #ax3.plot([0,1],[np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']),np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs'])],ls='-',c='k',lw=2)
                    # extract first and last trial z-score AUC arrays

                    # perform t-test
                    t_value, t_test_p_value = stats.ttest_rel(late_days_zscore, first_days_zscore,
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
                    if event == 'swing' :
                       ax3.set_ylim(0,0.815)
                    elif event == 'stance':
                       ax3.set_ylim(0,1.255)
                    # 20-80 swing length y-axis limits
                    if condition == 'allSteps' and pawNb==0:
                        if event == 'swing':
                            ax3.set_ylim(0, 8)
                        elif event == 'stance':
                            ax3.set_ylim(0, 6)
                    else:
                        if event == 'swing':
                            ax3.set_ylim(0, 8)
                        elif event == 'stance':
                            ax3.set_ylim(0, 6)

                    # create a new column with a name without '-' to put the z-score value (necessary because statsmodel and scipy dont like '-' in the key names
                    # zscore_key1 = 'zScoreAUC_' + times[t] + event
                    # modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    # for each behavior parameter perform linear regression
                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p,m])

                        modulated_paw_df.dropna(subset=['z_score_abs'], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df['z_score_abs']
                        # we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            r_v, p_v = stats.pearsonr(x, y)
                            r2 = np.square(r_value)

                            print(r_v, r_value)
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
                                           yLabel=(f'{behavior_par_Name[p]} ' if m==0 else ''),
                                           Leg=[1, 9])
                        print(event,p)
                        # allSteps AND 20-80 percentile y-axis limits
                        if behavior_par[p] == 'swingDuration':
                            ax4.set_ylim(0.065,0.36)
                        elif behavior_par[p] == 'swingLengthLinear':
                            ax4.set_ylim(1.1,4.7)

        paw=pawList[pawNb]
        fname = f'fig_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paw}_{condition}_{figVersion}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
    ##########################################################################
    def psthGroupFigure_cell_basedV2(self, figVersion, cellType, df_psth, df_cells, condition,  pawNb,allModTraces,variable):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 20  # width in inches

        fig_height = 18
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

        plt.figtext(0.01, 0.979, 'A', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.979, 'B', clip_on=False, color='black', size=26)
        #plt.figtext(0.01, 0.73, 'C', clip_on=False, color='black', size=26)
        #plt.figtext(0.51, 0.73, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.67, 'C', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.67, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.395, 'E', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.395, 'F', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.193, 'G', clip_on=False, color='black', size=26)
        plt.figtext(0.26, 0.193, 'H', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.193, 'I', clip_on=False, color='black', size=26)
        plt.figtext(0.75, 0.193, 'J', clip_on=False, color='black', size=26)
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
                gssub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0], hspace=0.1)

                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub3[0], hspace=0.1,wspace=0.22)

                    # sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub4[0,e], hspace=0.2,
                                                          height_ratios=[9, 24])

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

                    #ax2 = plt.subplot(gssub5[1])


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

                #ax2.barh([0], [100], color='0.9',  alpha=0.8)
                #ax2.barh(0, bar_width, left=bar_left, color=f'C{i}', alpha=alpha_e[e])
                #ax2.annotate(f'{bar_width:.1f}%', (bar_width/ 2, 0),
                #              fontsize=12, ha='center', va='center', c=('white' if e==1 else 'k'))

                #ax2.xaxis.set_major_locator(MultipleLocator(25))
                #ax2.set_ylim(1, -1)
                # the text above with percent and counts bbox (left, up, spacing, right)
                # ax2.legend(loc='lower left', mode='expand', ncol=len(label_colors),
                #            bbox_to_anchor=(0.05, 0.7, 0.9, 0.1), frameon=False,
                #            fontsize=14, labelcolor='k')  # , labelcolor=label_colors_List)
                bottom_pos += bar_width

                # axis style
                #ax2.get_yaxis().set_visible(False)
                #ax2.set_xlim(0, 100)
                #ax2.spines[['left', 'right', 'top']].set_visible(False)
            # plot title
                #ax2.text(0.5, -0.6, (f'fraction of  {pawList[i]} {event} onset modulated {cellType}  (%)'),
                #         ha='center', va='center',
                #         transform=ax2.transAxes)

                zscore = ['AUC', 'peak']
                zscorePar = zscore[0]
                # choose interval
                intervals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
                interval = intervals[0]
                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                modList=['all_mod', 'all_non']
                for m in [0,1]:
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
                    # exampleCell = 1
                    # choose an example cell for each condition

                    #if pawNb == 0 :
                    if e == 0 and m==0:
                        if cellType=='PC':
                            exampleCell = 4
                        else:
                            exampleCell = 21 #19 #25 #31 #41 #2 is perfect
                    elif e == 0 and m==1:
                        if cellType=='PC':
                            exampleCell = 5
                        else:
                            exampleCell = 7 #7
                    elif e==1 and m==0:
                        if cellType=='PC':
                            exampleCell=1
                        else:
                            exampleCell=2 #19 #13 #3 #21 #25 #36
                    elif e == 1 and m == 1:
                        if cellType == 'PC':
                            exampleCell = 9
                        else:
                            exampleCell=7 #2

                    #elif pawNb ==1:
                        #exampleCell=18  #16 maybe[ 1  2  4  5  6  9 11 12 13 17 19 21 26 27 29 30 32 34 35 36 37 38 40 41 42 44 45 46 47 50 52 56 58 59 61 63]
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

                                # get the zscore of and zscore time of example cells for first and last trials
                    print('we are', event, m, exampleCell)
                    print('cellsID:', cellsId)
                    earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                    lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    # subplot before and after for z-score examples
                        # subplot before and after for z-score examples
                    gssub5b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub5[0], hspace=0.2)
                    if e == 0:
                        # for time before we need two examples for up and down modulated (c==0 and c==1)
                        gssub5c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub5b[0], wspace=0.25,hspace=0.05)
                        # we don't care about non modulated (c==2)
                        if m < 2:
                            ax1a = plt.subplot(gssub5c[m])
                            ax1b = plt.subplot(gssub5c[m+2])
                            # vertical fill concerned time window (before -100ms to 0)
                            # plot the example z-score for first and last trial
                            try:
                                ax1a.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                         lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))
                                ax1b.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                         alpha=1, label='last trial')
                            except:
                                pdb.set_trace()


                            ax1a.set_ylim(-3.5,4.5)
                            ax1b.set_ylim(-3.5, 4.5)

                            # timeAfterMask=zscoreTime[0]>0
                            ax1b.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                            ax1a.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))
                            # except:
                            #     print('ALAJK', m, event)
                            #     pass

                            ax1a.legend(frameon=False, loc='upper right')
                            ax1b.legend(frameon=False, loc='upper right')
                    if e == 1:
                        gssub5c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub5b[0], hspace=0.2, wspace=0.3)
                        if m < 2:
                            ax1a = plt.subplot(gssub5c[m])
                            ax1b = plt.subplot(gssub5c[m+2])
                            # vertical fill concerned time window (after 0ms to 100)
                            # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                            # plot the example z-score for first and last trial for after event
                            if condition == 'allSteps' and pawNb==0:
                                ax1a.set_ylim(-5,10)
                                ax1b.set_ylim(-5, 10)
                            else:
                                ax1a.set_ylim(-3.5, 6)
                                ax1b.set_ylim(-3.5, 6)

                            ax1a.step(zscoreTime[0], earlyZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'),
                                     lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))

                            ax1b.step(zscoreTime[0], lateZscoreA[0][1], where='mid', color=(f'C{i}'if m==0 else '0.6'), lw=1,
                                     alpha=1,
                                     label='last trial')
                            #ax1a.legend(frameon=False)
                            #ax1b.legend(frameon=False)

                            ax1b.fill_between(zscoreTime[0], lateZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))


                            ax1a.fill_between(zscoreTime[0], earlyZscoreA[0][1],0, where=((zscoreTime[0] >= (-0.1)) & (
                                        zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0, color=(f'C{i}'if m==0 else '0.6'))

                            ax1a.legend(frameon=False, loc='upper right')
                            ax1b.legend(frameon=False, loc='upper right')
                    #draw 0 line
                    ax1a.axvline(0, ls='--', color='grey', lw=1,alpha=0.1)
                    ax1a.axvline(0, ls='--', color='grey', alpha=0.1)
                    ax1a.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    ax1a.axhline(0, ls='--', color='grey', alpha=0.1)
                    ax1b.axvline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    ax1b.axvline(0, ls='--', color='grey', alpha=0.1)
                    ax1b.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    ax1b.axhline(0, ls='--', color='grey', alpha=0.1)
                    self.layoutOfPanel(ax1a, xLabel=f'time centered on {event} onset (s)', yLabel=('PSTH Z-score' if m==0 else ''), xyInvisible=[(True), False])
                    self.layoutOfPanel(ax1b, xLabel=f'time centered on {event} onset (s)', yLabel=('PSTH Z-score' if m == 0 else ''), xyInvisible=[(False), False])
                    #ax1a.yaxis.set_major_locator(MultipleLocator(2))
                    #ax1b.yaxis.set_major_locator(MultipleLocator(2))
                    # for FL look only at those behavior parameters

                    behavior_par = ['swingDuration', 'swingLengthLinear']
                    behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                    hr = np.full(len(behavior_par) + 1, 2)
                    hr[0] = 3.2
                    hr[1] = 2.8
                    gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) + 1, 2, subplot_spec=gssub5[1],
                                                              hspace=0.5,
                                                              wspace=0.4, height_ratios=hr)

                    # comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0,m])
                    # group the data per cell and trial categories
                    modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()
                    trialCat=['first','last']
                    # for cell in cellsId:
                    #     for tr in range(2):
                    #         modulated_paw_df_first_late['normZscoreFirst'] = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] ==trialCat[tr])]['z_score_abs']/modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] =='first')]['z_score_abs']

                    # define colors for seaborn
                    palette = sns.color_palette([f'C{i}'], 1)
                    palette_non = sns.color_palette(['0.8'], 1)
                    grey = sns.color_palette(['grey'], 1)
                    late_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']
                    first_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs']
                    Nsess=0
                    for cell in cellsId:
                        late_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'last')]['z_score_abs'].values
                        first_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'first')]['z_score_abs'].values
                        if first_days_zscore_cell:
                            #print('alalala', np.shape(first_days_zscore_cell / first_days_zscore_cell,late_days_zscore_cell))
                            Nsess+=1
                            ax3.plot(['first','last'],[first_days_zscore_cell/first_days_zscore_cell, late_days_zscore_cell/first_days_zscore_cell],lw=0.2,alpha=0.5, color='grey')
                            ax3.scatter(['first', 'last'], [first_days_zscore_cell / first_days_zscore_cell,late_days_zscore_cell / first_days_zscore_cell],edgecolor='k', facecolor=(f'C{i}' if m==0 else '0.8'),lw=0.1, s=80)
                    print('N sessions :', Nsess)
                    pdb.set_trace()
                    # the invisible bar plot is to tighten things
                    # sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y='normZscoreFirst', ax=ax3,
                    #             color='k', alpha=0.001, errorbar=None)
                    # # lineplot for the line between cells
                    # sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y='normZscoreFirst',
                    #              ax=ax3,
                    #              palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                    # # scatterplot for the AUC values
                    # sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id',
                    #                 y='normZscoreFirst',
                    #                 ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                    #                 lw=0.1, s=80)
                    ax3.set_xlim(-0.9,1.9)
                    #ax3.plot([0,1],[np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']),np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs'])],ls='-',c='k',lw=2)
                    # extract first and last trial z-score AUC arrays

                    # perform t-test
                    t_value, t_test_p_value = stats.ttest_rel(late_days_zscore, first_days_zscore,
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
                    if event == 'swing' :
                       ax3.set_ylim(0,0.815)
                    elif event == 'stance':
                       ax3.set_ylim(0,1.255)
                    # 20-80 swing length y-axis limits
                    if condition == 'allSteps' and pawNb==0:
                        if event == 'swing':
                            ax3.set_ylim(0, 8)
                        elif event == 'stance':
                            ax3.set_ylim(0, 6)
                    else:
                        if event == 'swing':
                            ax3.set_ylim(0, 8)
                        elif event == 'stance':
                            ax3.set_ylim(0, 6)

                # variable='swingLengthLinear'
                possibleConditions = [f'{variable}_allRecs_percentiles_0_20',
                                      f'{variable}_allRecs_percentiles_20_40',
                                      f'{variable}_allRecs_percentiles_40_60',
                                      f'{variable}_allRecs_percentiles_60_80',
                                      f'{variable}_allRecs_percentiles_80_100'
                                      ]
                percentiles = ['0-20', '20-40', '40-60', '60-80', '80-100']
                percentilesc = ['[0-20]', '[20-40]', '[40-60]', '[60-80]', '[80-100]']
                numberArray = [[], []]
                traceAverageArray = [[], []]
                traceSemArray = [[], []]
                traceAUC = [[], []]
                var = 'peak'
                if cellType == 'PC':
                    timeInterval = 0.25
                    totCells = 34
                else:
                    timeInterval = 0.15
                    totCells = 64

                cellDicList = []

                for l, condition in enumerate(possibleConditions):
                    for g in range(2):
                        nCells = len(allModTraces[condition][events[g]]['traces'])
                        traceAUC[g].append(np.empty((nCells)))
                        # AUCdic[percentiles[l]][events[e]]=[]
                for l, condition in enumerate(possibleConditions):
                    for g in range(2):
                        time = allModTraces[condition][events[g]]['time']
                        numberArray[g].append(allModTraces[condition][events[g]]['number'])
                        eventOnsetMask = (time > -timeInterval) & (time < timeInterval)
                        dt = np.diff(time)[0]
                        for c in range(len(allModTraces[condition][events[g]]['traces'])):
                            trace = allModTraces[condition][events[g]]['traces'][c]
                            cellDic = {}
                            AUC = np.trapz(allModTraces[condition][events[g]]['traces'][c][eventOnsetMask], dx=dt)
                            cellDic['AUC'] = abs(AUC)
                            cellDic['condition'] = f'{percentilesc[l]}'  # {allModTraces[condition][events[e]]["number"]}'
                            cellDic['event'] = events[g]
                            cellDic['id'] = c
                            # Find positive peak

                            cellDicList.append(cellDic)
                        meanZscore = np.mean(allModTraces[condition][events[g]]['traces'], axis=0)
                        meanZscore = gaussian_filter1d(meanZscore, 0.8)
                        semZscore = stats.sem(allModTraces[condition][events[g]]['traces'], axis=0)
                        traceAverageArray[g].append(meanZscore)
                dfCells = pd.DataFrame(cellDicList)

            ax0b = plt.subplot(gssub6[5])
            if e == 0:
                alpha = 0.6
            else:
                alpha = 1

            numberArrayEv = np.array(numberArray[e])

            MLIfrac = (numberArrayEv / totCells) * 100
            x = np.arange(5)
            y = MLIfrac
            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                         alternative='two-sided')
            print(slope, intercept, r_value, p_value, sterr)
            ax0b.plot(percentilesc, MLIfrac, '-o', lw=2, c='C0', alpha=alpha)
            sns.regplot(x=np.arange(5), y=MLIfrac, scatter_kws={'alpha': 0.8, 'edgecolor': 'k', 'lw': 0.1},
                        line_kws={'alpha': alpha, 'lw': 1.5}, color='0.8')
            corr_star = groupAnalysis.starMultiplier(p_value)
            ax0b.set_ylim(0, 70)
            ax0b.tick_params(axis='both', labelsize=13)
            if p_value > 0.05:
                ax0b.text(0.73, 0.80, f"r = {r_value:.2f}\np = {p_value:.2f}",
                         transform=ax0b.transAxes, fontsize=10, color='dimgrey')

            else:
                ax0b.text(0.75, 0.4, f"r = {r_value:.2f}",
                         transform=ax0b.transAxes, fontsize=18, color='dimgrey')
                ax0b.text(0.5, 0.98, f"{corr_star}",
                         transform=ax0b.transAxes, fontsize=18, color='k')

            if variable != 'swingLengthLinear':
                self.layoutOfPanel(ax0b, xLabel=f'swing {variable[5:].lower()} percentile (%)',
                                   yLabel=f'{events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                   xyInvisible=[False, False])
            else:
                self.layoutOfPanel(ax0b, xLabel=f'swing length percentile (%)',
                                   yLabel=f'{events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                   xyInvisible=[False, False])
            ax1b = plt.subplot(gssub6[2:4])
            import matplotlib.colors as mcolors
            from matplotlib.colors import ColorConverter

            cc = ColorConverter()
            blue = mcolors.to_rgba('C0')

            colorList0 = ['#d2e3e4', '#b9d5e4', '#9bc7e4', '#7db9e4', '#5cace4']
            colorList = ['#8ca5bf', '#6b8dbf', '#4c74bf', '#2d5abf', '#0d40bf']
            colorList = ['#a3cef0', '#80bce5', '#5daeda', '#3a9dcf', '#167fc4']

            for a in range(5):
                intensity = (a + 1) / 5
                color = blue[:3] + (intensity,)  # adjust alpha channel\
                averageZscore = gaussian_filter1d(traceAverageArray[e][a], 0.8)
                line = ax1b.plot(time, averageZscore, color=color, lw=2,
                                label=f'[{percentiles[a]}]')

                meanAUC = np.mean(traceAUC[e][a])
                # pdb.set_trace()

                ax1b.text(0.42, 0.97, f"{events[e]} onset",
                         transform=ax1b.transAxes, fontsize=14, color='grey')
                ax1b.axvline(0, ls='--', color='grey', alpha=0.3)
                ax1b.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)

                # self.layoutOfPanel(ax1b, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                #                    xyInvisible=[False, False])
                self.layoutOfPanel(ax1b, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                                   xyInvisible=[False, False])

                legend=ax1b.legend(loc='upper left', frameon=False, fontsize=12)
                legend.set_title(f'swing {variable[5:].lower()} (%)')
                legend_title = legend.get_title()
                legend_title.set_fontsize(13)  # Change the fontsize to 12
                legend.set_bbox_to_anchor((0.002, 1.15))

            variables = ['AUC', 'PosPeak', 'PosLatency', 'NegPeak', 'NegLatency', 'TimeToPeak', 'TimeToTrough']
            variables = ['AUC']
            varNames = ['AUC (abs)', 'positive peak', 'positive peak latency', 'negative peak', 'negative peak latency',
                        'Time To Peak', 'Time To Trough']
            for v, var in enumerate(variables):
                ax2b = plt.subplot(gssub6[4 + v])
                cellDfEvent = dfCells[(dfCells['event'] == events[e])]
                cellDfEvent_80_100 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[80-100]')][var]
                cellDfEvent_0_20 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[0-20]')][var]
                cellDfEvent_20_40 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[20-40]')][var]
                cellDfEvent_40_60 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[40-60]')][var]
                cellDfEvent_60_80 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[60-80]')][var]
                dataAnova=[cellDfEvent_0_20,cellDfEvent_20_40,cellDfEvent_40_60,cellDfEvent_60_80,cellDfEvent_80_100]
                dataAnovaList = ['cellDfEvent_0_20', 'cellDfEvent_20_40', 'cellDfEvent_40_60', 'cellDfEvent_60_80',
                             'cellDfEvent_80_100']
                print('cellDfEvent info : ', cellDfEvent_0_20.info(), cellDfEvent_20_40.info(), cellDfEvent_40_60.info(), cellDfEvent_60_80.info(), cellDfEvent_80_100.info())
                pdb.set_trace()
                sns.pointplot(cellDfEvent, x='condition', y=var, hue='condition', ax=ax2b, errorbar='se',
                              palette=colorList)
                sns.lineplot(cellDfEvent, x='condition', y=var, ax=ax2b, errorbar=None, lw=2)
                if not cellType == 'PC':
                    for d in range(len(dataAnova)):
                        norm,pvalueNorm=stats.normaltest(dataAnova[d])
                        hom, pvalueHom = stats.levene(*dataAnova)
                        print(dataAnovaList[d],'normality ', pvalueNorm,'homogeneity ',pvalueHom)
                        # pdb.set_trace()
                f_statistic, p_valueAnova = stats.f_oneway(*dataAnova)
                print(events[e],f_statistic,p_valueAnova)

                t_value, t_test_p_value = stats.ttest_ind(cellDfEvent_0_20, cellDfEvent_80_100)

                star_trial = groupAnalysis.starMultiplier(p_valueAnova)
                if t_test_p_value < 0.05:
                    ax2b.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                             transform=ax2b.transAxes,
                             style='italic', fontfamily='serif', fontsize=18, color='k')
                else:
                    ax2b.text(0.5, 0.99, (f'p={p_valueAnova:.2f}'), ha='center', va='center',
                             transform=ax2b.transAxes,
                             style='italic', fontfamily='serif', fontsize=12, color='k')

                if variable != 'swingLengthLinear':
                    self.layoutOfPanel(ax2b, xLabel=f'swing {variable[5:].lower()} percentile (%)',
                                       yLabel=(f'{events[e]} onset \n PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                       xyInvisible=[False, False])
                else:
                    self.layoutOfPanel(ax2b, xLabel=f'swing length percentile (%)',
                                       yLabel=(f'{events[e]} onset \n PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                       xyInvisible=[False, False])
                ax2b.legend([], [], frameon=False)
                ax2b.tick_params(axis='both', labelsize=13)
                paw=pawList[pawNb]
        fname = f'fig_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paw}_{condition}_{figVersion}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
        # groupAnalysisFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary/'
        # plt.savefig(groupAnalysisFigDir+ fname + '.pdf')
    #########################################################################
    def fig_locomotorLearningSupp(self, figVersion, stridePar, strideTraj, swingNumber, strideLength, strideDuration,
                              indecisiveStrideFraction, swingSpeed, pawCoordination, allVariablesDf):

        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        # figure #################################
        fig_width = 15  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 18, 'axes.titlesize': 18, 'font.size': 18, 'xtick.labelsize': 15,
                  'ytick.labelsize': 15, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs.update(wspace=0.35, hspace=0.35)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.065, right=0.95, top=0.95, bottom=0.1)

        # panel names
        plt.figtext(0.012, 0.96, 'A', clip_on=False, color='black', size=22)
        plt.figtext(0.35, 0.96, 'B', clip_on=False, color='black', size=22)
        plt.figtext(0.66, 0.96, 'C', clip_on=False, color='black', size=22)
        plt.figtext(0.012, 0.66, 'D', clip_on=False, color='black', size=22)
        plt.figtext(0.35, 0.66, 'E', clip_on=False, color='black', size=22)
        plt.figtext(0.66, 0.66, 'F', clip_on=False, color='black', size=22)
        plt.figtext(0.012, 0.35, 'G', clip_on=False, color='black', size=22)
        plt.figtext(0.35, 0.35, 'H', clip_on=False, color='black', size=22)
        plt.figtext(0.66, 0.35, 'I', clip_on=False, color='black', size=22)

        # divide figure in 3 lines
        gssub1 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0], hspace=0.4, wspace=0.35)
        ############################Panel A : swing Number#########################

        # regroup data per trial, day, paw and mouse
        stridePar_Recordings = stridePar.groupby(['trial', 'day', 'sex', 'paw', 'mouseId']).mean()
        stridePar_Recordings = stridePar_Recordings.reset_index()

        # get data for five trials only, for visualization
        stridePar_Recordings_five_trials = stridePar_Recordings[stridePar_Recordings['trial'] <= 5]

        # average data per day for visualization
        stridePar_Recordings_day = stridePar_Recordings.groupby(['day', 'mouseId']).mean()
        stridePar_Recordings_day = stridePar_Recordings_day.reset_index()

        paw_parameters=['swingNumber', 'swingDuration', 'swingSpeed', 'indecisiveFraction','stanceDuration', 'twoRungsFraction']
        #regroup parameters to plot in list, easier to plot

        #Y axes labels
        par_paw_Labels=['swing number (avg.)','swing duration (s)', 'swing speed (cm/s)', 'indecisive strides fraction','stance duration (s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed']
        for p in range(len(paw_parameters)):
            par_paw_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, paw_parameters[p])
            if p==4:
                ax1 = plt.subplot(gssub1[5])
            elif p==5:
                ax1 = plt.subplot(gssub1[7])
            else:
                ax1 = plt.subplot(gssub1[p])

            # per day all animals individual paw as hue
            sns.lineplot(data=stridePar_Recordings, x='day', y=paw_parameters[p], hue='paw',
                         errorbar=('se'), err_style='bars',
                         err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax1, marker='o')

            if p==0:
                x_pos=0
                y_pos=0.15
                h_len=0.85 #
            elif p==1:
                x_pos=-0.05
                y_pos=0.15
                h_len=0.85
            elif (p == 2 or p==4 or p==5):
                x_pos = -0.45
                y_pos = 0.15
                h_len = 0.85
            elif p == 3:
                x_pos = 0
                y_pos = 0.15
                h_len = 0.85
            ax_inset_2 = inset_axes(ax1, width="27%", height="25%", bbox_to_anchor=(x_pos, y_pos, h_len, 0.9),
                                    bbox_transform=ax1.transAxes, loc=1)
            sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y=paw_parameters[p], hue='paw',
                         errorbar=('se'), err_style='bars',
                         err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset_2, marker='o',alpha=0.8)

            # regroup stats in star_label list, trial effect as ###
            star_label = []
            star_label_inset = []
            for i in range(4):
                star_label.append(pawId[i] + ' ' + par_paw_summary['paw_stars'][pawId[i]]['day'])
                star_label_inset.append(
                    pawId[i] + ' ' + par_paw_summary['paw_stars'][pawId[i]]['trial'].replace('*', '#'))
            # stars as legend
            if (p==0 or p==1 or p==3):
                ax1.legend(star_label, loc='lower left', frameon=False, fontsize=13)
            elif (p==2 or p==4 or p==5):
                ax1.legend(star_label, loc='lower right', frameon=False, fontsize=13)
                # bbox_to_anchor = (0.86, 0.78),
            # ax1.legend(star_label,loc='lower left', mode='expand', ncol=len(pawId),
            #            bbox_to_anchor=(-0.1, 0.2, 1.25, 0.1), frameon=False,
            #            fontsize=13, labelcolor='k')
            ax_inset_2.legend(star_label_inset, bbox_to_anchor=(0.93, 0.95), frameon=False, fontsize=10)
            ax_inset_2.xaxis.label.set_size(12)
            ax_inset_2.tick_params(axis='x', labelsize=12)
            ax_inset_2.tick_params(axis='y', labelsize=12)
            # ax_inset_2.text(0.5, 0.97, '%s' %strideLength_summary['stars']['trial'].replace('*','#'), ha='center', va='center', transform=ax_inset_2.transAxes, style='italic',fontfamily='serif', fontsize=10, color='0.4')
            self.layoutOfPanel(ax1, xLabel='session', yLabel=par_paw_Labels[p])
            self.layoutOfPanel(ax_inset_2, xLabel='trial', yLabel='')
            ax1.xaxis.set_major_locator(MultipleLocator(1))
            ax_inset_2.xaxis.set_major_locator(MultipleLocator(1))

        parameters = ['stanceDuration', 'twoRungsFraction']
        # Y axes labels
        parLabels = ['stance duration (s)', 'Fraction of ' + r'$\geqq$' + '2 rungs crossed']
        # plot them one by one
        for par in range(len(parameters)):
            # perform stats for each parameters
            par_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par])
            # par_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par], 'mouseId', treatments=False)
            # subplots each parameter as columns
            if par==0:
                ax0 = plt.subplot(gssub1[4])
            elif par==1:
                ax0 = plt.subplot(gssub1[6])
            # all animal per day
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue=None, errorbar=('se'),
                         err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, color='black', ax=ax0,
                         marker='o')
            # individual animals
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId', errorbar=None,
                         err_style='bars', err_kws={'capsize': 3, 'linewidth': 1}, alpha=0.1, legend=False, ax=ax0)
            if par_summary['stars']['trial']!='':
                ax_inset = inset_axes(ax0, width="27%", height="25%", bbox_to_anchor=(-0.45, 0., 0.9, 0.9),
                                      bbox_transform=ax0.transAxes, loc=1)
                sns.lineplot(data=stridePar_Recordings_five_trials, x='trial', y='swingNumber', hue=None,
                             errorbar=('se'), err_style='bars',
                             err_kws={'capsize': 3, 'linewidth': 1}, color='0.4', ax=ax_inset, marker='o')
                ax_inset.text(0.5, 0.97, '%s' % (par_summary['stars']['trial'].replace('*', '#')), ha='center',
                              va='center', transform=ax_inset.transAxes, style='italic', fontfamily='serif', fontsize=10,
                              color='0.4')
                ax_inset.xaxis.set_major_locator(MultipleLocator(1))
                self.layoutOfPanel(ax_inset, xLabel='trial', yLabel='')
                ax_inset.xaxis.label.set_size(12)
                ax_inset.tick_params(axis='x', labelsize=12)
                ax_inset.tick_params(axis='y', labelsize=12)



            # day effect stars
            ax0.text(0.5, 1, '%s' % (par_summary['stars']['day']), ha='center', va='center',
                     transform=ax0.transAxes,
                     style='italic', fontfamily='serif', fontsize=16, color='k')
            # style
            self.layoutOfPanel(ax0, xLabel='session', yLabel=parLabels[par])
            # majorLocator_x = MultipleLocator(1)
            ax0.xaxis.set_major_locator(MultipleLocator(1))

        ax2=plt.subplot(gssub1[8])
        #perform stats

        stanceOnMed_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingOnMedian_FL')

        #get data for FL and FR

        Hind_df = stridePar_Recordings[((stridePar_Recordings['paw'] == pawId[2]) |(stridePar_Recordings['paw'] == pawId[3]))]
        #get the data with only five trials for visualization
        Hind_df = stridePar_Recordings_five_trials[((stridePar_Recordings['paw'] == pawId[2]) |(stridePar_Recordings['paw'] == pawId[3]))]
        hindColors=['C2', 'C3']
        #plot swing on median per day
        sns.lineplot(data=Hind_df, x='day', y='swingOnMedian_FL', hue='paw',
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, palette=hindColors, ax=ax2, marker='o')

        #plot inset for trial data
        ax_inset_stanceOn_med = inset_axes(ax2, width="27%", height="25%",
                                   bbox_to_anchor=(-0.4, 0.1, 0.9, 0.9),
                                   bbox_transform=ax2.transAxes,
                                   loc=1)

        sns.lineplot(data=Hind_df, x='trial', y='swingOnMedian_FL', hue='paw',
                     errorbar=('se'), err_style='bars',
                     err_kws={'capsize': 3, 'linewidth': 1}, palette=hindColors, ax=ax_inset_stanceOn_med, marker='o')
        ax_inset_stanceOn_med.xaxis.label.set_size(12)
        ax_inset_stanceOn_med.tick_params(axis='x', labelsize=12)
        ax_inset_stanceOn_med.tick_params(axis='y', labelsize=12)

        star_label_stanceOn = []
        star_label_inset_stanceOn = []
        for h in [2,3]:
            star_label_stanceOn.append(pawId[h] + ' ' + stanceOnMed_summary['paw_stars'][pawId[h]]['day'])
            star_label_inset_stanceOn.append(
                pawId[h] + ' ' + stanceOnMed_summary['paw_stars'][pawId[h]]['trial'].replace('*', '#'))

        ax_inset_stanceOn_med.legend(star_label_inset_stanceOn, bbox_to_anchor=(0.93, 0.95), frameon=False, fontsize=10)
        self.layoutOfPanel(ax2, xLabel='session', yLabel='swing onset median', xyInvisible=[False, False],Leg=[1, 9])
        ax2.legend(star_label_stanceOn, loc='lower left', frameon=False, fontsize=13)
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        self.layoutOfPanel(ax_inset_stanceOn_med, xLabel='trial', yLabel='')
        ax_inset_stanceOn_med.xaxis.set_major_locator(MultipleLocator(1))
        # figure name

        # figure output format
        plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')

    def psthGroupFigure_cell_based_supp(self, figVersion, cellType, df_psth, df_cells, condition,  pawNb):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 16  # width in inches

        fig_height = 22
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
        gs = gridspec.GridSpec(2, 1,  # ,
                               height_ratios=[3,20]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.05, hspace=0.15)

        plt.figtext(0.01, 0.975, 'A', clip_on=False, color='black', size=26)
        # plt.figtext(0.51, 0.975, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.81, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.81, 'C', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.6, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.6, 'E', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.54, 'F', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.54, 'G', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.32, 'H', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.32, 'I', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.16, 'J', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.16, 'K', clip_on=False, color='black', size=26)
        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)

        events=['swing', 'stance']
        for w in range(4):
            modulated={}
            non_modulated={}
            for e in reversed(range(2)):
                event=events[e]

                # looking at specific paw values in the df that contains single values
                paw_df = df_cells[(df_cells['paw'] == pawList[w])]

                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df = df_psth[(df_psth['paw'] == pawList[w])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                        paw_psth_df = paw_psth_df.drop(index)
                # big panels for each paw

                # get the ids and df of all modulated cells
                cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df,
                                                                                                      paw_psth_df,event)

                # calculate correlation distribution
                modulated[event], non_modulated[event] = groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(cells_id,
                                                                                                      modCells_df,
                                                                                                      event)
            modulated['all_events']=np.concatenate((modulated['swing'], modulated['stance']))
            non_modulated['all_events'] = np.concatenate((non_modulated['swing'], non_modulated['stance']))

            gssub0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], hspace=0.1,wspace=0.3)
            ax0=plt.subplot(gssub0[w])
            # plot correlation distribution
            ax0.hist(modulated['all_events'], bins=30, density=True, alpha=(0.8), histtype = 'stepfilled', color=f'C{w}', label=f'{pawList[w]} modulated ')
            ax0.hist(non_modulated['all_events'], bins=30, density=True, histtype = 'stepfilled', lw=0.001, alpha=0.6, color='0.6',
                     label=(f'non modulated' if w==0 else None))

            ax0.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
            self.layoutOfPanel(ax0, xLabel=('PSTH correlation coefficient across trials'if w==0 else None),
                               yLabel=('probability density' if w==0 else None), Leg=[1, 9])
            ax0.legend(bbox_to_anchor=(0.4,0.5,0.15,0.5), loc='upper left', frameon=False, fontsize=12)
            ax0.set_xlim(-0.5, 1)
            ax0.set_ylim(0, 3)



        modCells = {}

        for e in reversed(range(2)):
            event = events[e]
            for i in [pawNb]:
                event=events[e]

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

                gssub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1], hspace=0.1)

                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub3[0], hspace=0.1,wspace=0.3)

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
                    self.layoutOfPanel(ax1, xLabel=(f'time centered on {event} onset (s)' if m==0 else ''), yLabel=('PSTH Z-score' if m==0 else ''), xyInvisible=[(False), False])
                    ax1.yaxis.set_major_locator(MultipleLocator(2))

                    # for FL look only at those behavior parameters

                    behavior_par = ['swingDuration', 'swingLengthLinear']
                    behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                    hr = np.full(len(behavior_par) + 1, 2)
                    hr[0] = 3.2
                    gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) + 1, 2, subplot_spec=gssub5[2],
                                                              hspace=0.5,
                                                              wspace=0.4, height_ratios=hr)

                    # comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0,m])
                    # group the data per cell and trial categories
                    modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()
                    # define colors for seaborn
                    palette = sns.color_palette([f'C{i}'], 1)
                    palette_non = sns.color_palette(['0.8'], 1)
                    grey = sns.color_palette(['grey'], 1)

                    late_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']
                    first_days_zscore = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs']
                    for cell in cellsId:
                        late_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'last')]['z_score_abs'].values
                        first_days_zscore_cell = modulated_paw_df_first_late[(modulated_paw_df_first_late['cell_global_Id'] ==cell)&(modulated_paw_df_first_late['trial_category'] == 'first')]['z_score_abs'].values
                        if first_days_zscore_cell:
                            # print('alalala', late_days_zscore_cell[0],first_days_zscore_cell[0])
                            ax3.plot(['first','last'],[first_days_zscore_cell/first_days_zscore_cell, late_days_zscore_cell/first_days_zscore_cell],lw=0.2,alpha=0.5, color='grey')
                            ax3.scatter(['first', 'last'], [first_days_zscore_cell / first_days_zscore_cell,late_days_zscore_cell / first_days_zscore_cell],edgecolor='k', facecolor=(f'C{i}' if m==0 else '0.8'),lw=0.1, s=80)
                    # the invisible bar plot is to tighten things
                    # sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y='z_score_abs', ax=ax3,
                    #             color='k', alpha=0.001, errorbar=None)
                    # # lineplot for the line between cells
                    # sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y='z_score_abs',
                    #              ax=ax3,
                    #              palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                    # # scatterplot for the AUC values
                    # sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id',
                    #                 y='z_score_abs',
                    #                 ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                    #                 lw=0.1, s=80)
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
                            ax3.set_ylim(0, 10)
                        elif event == 'stance':
                            ax3.set_ylim(0, 8)
                    else:
                        if event == 'swing':
                            ax3.set_ylim(0, 10)
                        elif event == 'stance':
                            ax3.set_ylim(0, 8)

                    # create a new column with a name without '-' to put the z-score value (necessary because statsmodel and scipy dont like '-' in the key names
                    # zscore_key1 = 'zScoreAUC_' + times[t] + event
                    # modulated_paw_df[zscore_key1] = modulated_paw_df.loc[:, zscore_key[t]]
                    # for each behavior parameter perform linear regression
                    for p in range(len(behavior_par)):
                        ax4 = plt.subplot(gssub6[1 + p,m])

                        modulated_paw_df.dropna(subset=['z_score_abs'], inplace=True)
                        y = modulated_paw_df[behavior_par[p]]
                        x = modulated_paw_df['z_score_abs']
                        # we use try because there are cases where you have 0 or 1 modulated cells with 0 or 1 recs and you cannot do correlation
                        try:
                            slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                                         alternative='two-sided')
                            r_v, p_v = stats.pearsonr(x, y)
                            r2 = np.square(r_value)

                            print(r_v, r_value)
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
                                           yLabel=(f'{behavior_par_Name[p]} ' if m==0 else ''),
                                           Leg=[1, 9])
                        print(event,p)
                        # allSteps AND 20-80 percentile y-axis limits
                        if behavior_par[p] == 'swingDuration':
                            ax4.set_ylim(0.065,0.36)
                        elif behavior_par[p] == 'swingLengthLinear':
                            ax4.set_ylim(1.1,4.7)

        paw=pawList[pawNb]
        fname = f'fig_supp_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paw}_{condition}_{figVersion}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')

    def psthGroupFigure_cell_based_suppV2(self, figVersion, cellType, df_psth, df_cells, condition, pawNb,allModTraces,variable):
        pawList = ['FL', 'FR', 'HL', 'HR']

        # figure generate either a figure with only FL paw or with any desired number of paw
        fig_width = 18  # width in inches

        fig_height = 16
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
        gs = gridspec.GridSpec(2, 1,  # ,
                               height_ratios=[5, 20]
                               # height_ratios=[1, 1.5,1,1.5, 1]
                               )
        # panel Ids
        gs.update(wspace=0.05, hspace=0.1)

        plt.figtext(0.01, 0.975, 'A', clip_on=False, color='black', size=26)
        # plt.figtext(0.51, 0.975, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.74, 'B', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.74, 'C', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.64, 'D', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.64, 'E', clip_on=False, color='black', size=26)
        plt.figtext(0.01, 0.22, 'F', clip_on=False, color='black', size=26)
        plt.figtext(0.24, 0.22, 'G', clip_on=False, color='black', size=26)
        plt.figtext(0.51, 0.22, 'H', clip_on=False, color='black', size=26)
        plt.figtext(0.76, 0.22, 'I', clip_on=False, color='black', size=26)
        #plt.figtext(0.91, 0.2, 'J', clip_on=False, color='black', size=26)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.05)

        events = ['swing', 'stance']
        for w in range(4):
            modulated = {}
            non_modulated = {}
            for e in reversed(range(2)):
                event = events[e]

                # looking at specific paw values in the df that contains single values
                paw_df = df_cells[(df_cells['paw'] == pawList[w])]

                # looking at specific paw values in the df that contains psth and zscore
                paw_psth_df = df_psth[(df_psth['paw'] == pawList[w])]
                # detect and remove empty cells, happens when there's no stride in a condition
                empty_cells = paw_df['before_swingOnset_z-score_AUC_0.1'].isna()
                # loop through the rows of the DataFrame
                for index, value in empty_cells.iteritems():
                    if value:
                        print(f'Empty cell detected in line {index}')
                        paw_df = paw_df.drop(index)
                        paw_psth_df = paw_psth_df.drop(index)
                # big panels for each paw

                # get the ids and df of all modulated cells
                cells_id, modCells_df = groupAnalysis_psth.get_allModulatedCells_nonModulatedCells_Id(paw_df,
                                                                                                      paw_psth_df,
                                                                                                      event)

                # calculate correlation distribution
                modulated[event], non_modulated[event] = groupAnalysis_psth.getTrials_PSTH_CorrCoeff_modulatedCells(
                    cells_id,
                    modCells_df,
                    event)
            modulated['all_events'] = np.concatenate((modulated['swing'], modulated['stance']))
            non_modulated['all_events'] = np.concatenate((non_modulated['swing'], non_modulated['stance']))

            gssub0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], hspace=0.1, wspace=0.3)
            ax0 = plt.subplot(gssub0[w])
            # plot correlation distribution
            ax0.hist(modulated['all_events'], bins=30, density=True, alpha=(0.8), histtype='stepfilled', color=f'C{w}',
                     label=f'{pawList[w]} modulated ')
            ax0.hist(non_modulated['all_events'], bins=30, density=True, histtype='stepfilled', lw=0.001, alpha=0.6,
                     color='0.6',
                     label=(f'non modulated' if w == 0 else None))

            ax0.axvline(0, ls='--', c='grey', alpha=0.2, lw=1)
            self.layoutOfPanel(ax0, xLabel=('PSTH correlation coefficient across trials' if w == 0 else None),
                               yLabel=('probability density' if w == 0 else None), Leg=[1, 9])
            ax0.legend(bbox_to_anchor=(0.4, 0.5, 0.15, 0.5), loc='upper left', frameon=False, fontsize=12)
            ax0.set_xlim(-0.5, 1)
            ax0.set_ylim(0, 3)

        modCells = {}

        for e in reversed(range(2)):
            event = events[e]
            for i in [pawNb]:
                event = events[e]

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

                gssub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1], hspace=0.1)

                gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub3[0], hspace=0.1, wspace=0.3)

                # sub panel that contains the example z-scores/ the bar plot of modulated cells/ the comparison/correlation plots
                gssub5 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub4[0, e], hspace=0.2,
                                                          height_ratios=[ 2, 20])

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

                    ax2 = plt.subplot(gssub5[0])

                    # get Id and counts of modulated cells
                    modCells_Id, modCells_count, counts = groupAnalysis_psth.getModulatedcell_Id_count(paw_df, catList,
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
                bar_width = len(modCells[pawList[i]]['all_mod']) / len(all_cells) * 100
                bar_left = bottom_pos

                ax2.barh([0], [100], color='0.9', alpha=0.8)
                ax2.barh(0, bar_width, left=bar_left, color=f'C{i}', alpha=alpha_e[e])
                ax2.annotate(f'{bar_width:.1f}%', (bar_width / 2, 0),
                             fontsize=12, ha='center', va='center', c=('white' if e == 1 else 'k'))

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
                interval = intervals[0]
                # key to access the variable (AUC or Peak)
                zscore_key = ['before_%sOnset_z-score_%s_%s' % (event, zscorePar, interval),
                              'after_%sOnset_z-score_%s_%s' % (event, zscorePar, interval)]
                modList = ['all_mod', 'all_non']
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
                    cellsId = np.unique(modulated_paw_psth_df['cell_global_Id'])
                    print('cellsID:', cellsId)
                    # exampleCell = 1
                    # choose an example cell for each condition
                    # if condition == 'allSteps':
                    #     if pawNb == 0:
                    #         if (e == 0 and m == 0):
                    #             exampleCell = 25#21  # 19 #25 #31 #41 #2 is perfect
                    #         elif (e == 0 and m == 1):
                    #             exampleCell = 7#8  # 7
                    #         elif (e == 1 and m == 0):
                    #             exampleCell = 2  # 19 #13 #3 #21 #25 #36
                    #         elif (e == 1 and m == 1):
                    #             exampleCell = 7  # 2
                    #     elif pawNb == 1:
                    #         exampleCell = 18  # 16 maybe[ 1  2  4  5  6  9 11 12 13 17 19 21 26 27 29 30 32 34 35 36 37 38 40 41 42 44 45 46 47 50 52 56 58 59 61 63]
                    # elif condition == 'swingLengthLinear_lastRec_20_80':
                    #     if pawNb == 0:
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
                    #     exampleCell = 2
                    #
                    #     # get the zscore of and zscore time of example cells for first and last trials
                    # earlyZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                    #         modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    # zscoreTime = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'first') & (
                    #         modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreTimeKey].values
                    # lateZscoreA = modulated_paw_psth_df[(modulated_paw_psth_df['trial_category'] == 'last') & (
                    #         modulated_paw_psth_df['cell_global_Id'] == cellsId[exampleCell])][z_scoreKey].values
                    # # subplot before and after for z-score examples
                    # # subplot before and after for z-score examples
                    # gssub5b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gssub5[0], hspace=0.2)
                    # if e == 0:
                    #     # for time before we need two examples for up and down modulated (c==0 and c==1)
                    #     gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], wspace=0.3)
                    #     # we don't care about non modulated (c==2)
                    #     if m < 2:
                    #         ax1 = plt.subplot(gssub5c[m])
                    #         # vertical fill concerned time window (before -100ms to 0)
                    #         # plot the example z-score for first and last trial
                    #         ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid',
                    #                  color=(f'C{i}' if m == 0 else '0.6'),
                    #                  lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))
                    #         ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid',
                    #                  color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                    #                  alpha=1, label='last trial')
                    #
                    #         ax1.set_ylim(-3.5, 4.5)
                    #
                    #         # timeAfterMask=zscoreTime[0]>0
                    #         ax1.fill_between(zscoreTime[0], lateZscoreA[0][1], 0, where=((zscoreTime[0] >= (-0.1)) & (
                    #                 zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0,
                    #                          color=(f'C{i}' if m == 0 else '0.6'))
                    #
                    #         ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1], 0, where=((zscoreTime[0] >= (-0.1)) & (
                    #                 zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0,
                    #                          color=(f'C{i}' if m == 0 else '0.6'))
                    #
                    #         ax1.legend(frameon=False, loc='upper right')
                    # if e == 1:
                    #     gssub5c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub5b[0], hspace=0.2,
                    #                                                wspace=0.3)
                    #     if m < 2:
                    #         ax1 = plt.subplot(gssub5c[m])
                    #         # vertical fill concerned time window (after 0ms to 100)
                    #         # ax1.axvspan(0, 0.1, color=f'C{i}', alpha=0.08)
                    #         # plot the example z-score for first and last trial for after event
                    #         if condition == 'allSteps' and pawNb == 0:
                    #             ax1.set_ylim(-5, 10)
                    #         else:
                    #             ax1.set_ylim(-3.5, 6)
                    #
                    #         ax1.step(zscoreTime[0], earlyZscoreA[0][1], where='mid',
                    #                  color=(f'C{i}' if m == 0 else '0.6'),
                    #                  lw=1, alpha=0.8, label='first trial', ls=(0, (1, 1)))
                    #
                    #         ax1.step(zscoreTime[0], lateZscoreA[0][1], where='mid',
                    #                  color=(f'C{i}' if m == 0 else '0.6'), lw=1,
                    #                  alpha=1,
                    #                  label='last trial')
                    #         ax1.legend(frameon=False)
                    #
                    #         ax1.fill_between(zscoreTime[0], lateZscoreA[0][1], 0, where=((zscoreTime[0] >= (-0.1)) & (
                    #                 zscoreTime[0] < 0.10)), alpha=0.2, step='mid', lw=0.0,
                    #                          color=(f'C{i}' if m == 0 else '0.6'))
                    #
                    #         ax1.fill_between(zscoreTime[0], earlyZscoreA[0][1], 0, where=((zscoreTime[0] >= (-0.1)) & (
                    #                 zscoreTime[0] < 0.10)), alpha=0.1, step='mid', lw=0.0,
                    #                          color=(f'C{i}' if m == 0 else '0.6'))
                    #
                    #         ax1.legend(frameon=False, loc='upper right')
                    # # draw 0 line
                    # ax1.axvline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    # ax1.axvline(0, ls='--', color='grey', alpha=0.1)
                    # ax1.axhline(0, ls='--', color='grey', lw=1, alpha=0.1)
                    # ax1.axhline(0, ls='--', color='grey', alpha=0.1)
                    # self.layoutOfPanel(ax1, xLabel=(f'time centered on {event} onset (s)' if m == 0 else ''),
                    #                    yLabel=('PSTH Z-score' if m == 0 else ''), xyInvisible=[(False), False])
                    # ax1.yaxis.set_major_locator(MultipleLocator(2))

                    # for FL look only at those behavior parameters

                    behavior_par = ['swingDuration', 'swingLengthLinear']
                    behavior_par_Name = ['swing duration (s)', 'swing length (cm)']
                    # hr = np.full(len(behavior_par) + 1, 2)
                    # hr[0] = 3.2
                    gssub6 = gridspec.GridSpecFromSubplotSpec(len(behavior_par) , 2, subplot_spec=gssub5[1],
                                                              hspace=0.4,
                                                              wspace=0.4, height_ratios=[2.3,1])

                    # comparison panel for each categories
                    ax3 = plt.subplot(gssub6[0, m])
                    # group the data per cell and trial categories
                    modulated_paw_df_first_late = modulated_paw_df.groupby(['cell_global_Id', 'trial_category'])[
                        'z_score_abs'].mean().reset_index()
                    # define colors for seaborn
                    palette = sns.color_palette([f'C{i}'], 1)
                    palette_non = sns.color_palette(['0.8'], 1)
                    grey = sns.color_palette(['grey'], 1)

                    late_days_zscore = \
                    modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']
                    first_days_zscore = \
                    modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs']
                    for cell in cellsId:
                        late_days_zscore_cell = modulated_paw_df_first_late[
                            (modulated_paw_df_first_late['cell_global_Id'] == cell) & (
                                        modulated_paw_df_first_late['trial_category'] == 'last')]['z_score_abs'].values
                        first_days_zscore_cell = modulated_paw_df_first_late[
                            (modulated_paw_df_first_late['cell_global_Id'] == cell) & (
                                        modulated_paw_df_first_late['trial_category'] == 'first')]['z_score_abs'].values
                        if first_days_zscore_cell:
                            # print('alalala', late_days_zscore_cell[0],first_days_zscore_cell[0])
                            ax3.plot(['first', 'last'], [first_days_zscore_cell / first_days_zscore_cell,
                                                         late_days_zscore_cell / first_days_zscore_cell], lw=0.2,
                                     alpha=0.5, color='grey')
                            ax3.scatter(['first', 'last'], [first_days_zscore_cell / first_days_zscore_cell,
                                                            late_days_zscore_cell / first_days_zscore_cell],
                                        edgecolor='k', facecolor=(f'C{i}' if m == 0 else '0.8'), lw=0.1, s=80)
                    # the invisible bar plot is to tighten things
                    # sns.barplot(modulated_paw_df_first_late, x='trial_category', hue=None, y='z_score_abs', ax=ax3,
                    #             color='k', alpha=0.001, errorbar=None)
                    # # lineplot for the line between cells
                    # sns.lineplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id', y='z_score_abs',
                    #              ax=ax3,
                    #              palette=grey, legend=False, lw=0.2,alpha=0.5, markers=True)
                    # # scatterplot for the AUC values
                    # sns.scatterplot(modulated_paw_df_first_late, x='trial_category', hue='cell_global_Id',
                    #                 y='z_score_abs',
                    #                 ax=ax3, palette=(palette if m==0 else palette_non), legend=False, alpha=alpha_e[e], markers=False, edgecolor='k',
                    #                 lw=0.1, s=80)
                    ax3.set_xlim(-0.9, 1.9)
                    # ax3.plot([0,1],[np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']['z_score_abs']),np.mean(modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']['z_score_abs'])],ls='-',c='k',lw=2)
                    # extract first and last trial z-score AUC arrays
                    late_days = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'first']
                    first_days = modulated_paw_df_first_late[modulated_paw_df_first_late['trial_category'] == 'last']
                    # perform t-test
                    t_value, t_test_p_value = stats.ttest_rel(late_days['z_score_abs'], first_days['z_score_abs'],
                                                              axis=0)
                    print('paired t-test (t-value, p): ', event, t_value, t_test_p_value)
                    star_trial = groupAnalysis.starMultiplier(t_test_p_value)
                    if t_test_p_value < 0.05:
                        ax3.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                                 transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=18, color='k')
                    else:
                        ax3.text(0.5, 0.99, (f'p={t_test_p_value:.2f}'), ha='center', va='center',
                                 transform=ax3.transAxes,
                                 style='italic', fontfamily='serif', fontsize=12, color='k')
                    self.layoutOfPanel(ax3, xLabel='Trial', yLabel=(
                        f' {event} onset PSTH \n  Z-score {zscorePar} (abs)' if m == 0 else ''),
                                       Leg=[1, 9])

                    majorLocator_x = MultipleLocator(1)
                    ax3.xaxis.set_major_locator(majorLocator_x)
                    # allSteps y-axis limits
                    # if event == 'swing' :
                    #    ax3.set_ylim(0,0.815)
                    # elif event == 'stance':
                    #    ax3.set_ylim(0,1.255)
                    # 20-80 swing length y-axis limits
                    if condition == 'allSteps' and pawNb == 0:
                        if event == 'swing':
                            ax3.set_ylim(0, 10)
                        elif event == 'stance':
                            ax3.set_ylim(0, 8)
                    else:
                        if event == 'swing':
                            ax3.set_ylim(0, 10)
                        elif event == 'stance':
                            ax3.set_ylim(0, 8)

                    possibleConditions = [f'{variable}_allRecs_percentiles_0_20',
                                          f'{variable}_allRecs_percentiles_20_40',
                                          f'{variable}_allRecs_percentiles_40_60',
                                          f'{variable}_allRecs_percentiles_60_80',
                                          f'{variable}_allRecs_percentiles_80_100'
                                          ]
                    percentiles = ['0-20', '20-40', '40-60', '60-80', '80-100']
                    percentilesc = ['[0-20]', '[20-40]', '[40-60]', '[60-80]', '[80-100]']
                    numberArray = [[], []]
                    traceAverageArray = [[], []]
                    traceSemArray = [[], []]
                    traceAUC = [[], []]
                    var = 'peak'
                    if cellType == 'PC':
                        timeInterval = 0.25
                        totCells = 34
                    else:
                        timeInterval = 0.15
                        totCells = 64

                    cellDicList = []

                    for l, condition in enumerate(possibleConditions):
                        for g in range(2):
                            nCells = len(allModTraces[condition][events[g]]['traces'])
                            traceAUC[g].append(np.empty((nCells)))
                            # AUCdic[percentiles[l]][events[e]]=[]
                    for l, condition in enumerate(possibleConditions):
                        for g in range(2):
                            time = allModTraces[condition][events[g]]['time']
                            numberArray[g].append(allModTraces[condition][events[g]]['number'])
                            eventOnsetMask = (time > -timeInterval) & (time < timeInterval)
                            dt = np.diff(time)[0]
                            for c in range(len(allModTraces[condition][events[g]]['traces'])):
                                trace = allModTraces[condition][events[g]]['traces'][c]
                                cellDic = {}
                                AUC = np.trapz(allModTraces[condition][events[g]]['traces'][c][eventOnsetMask], dx=dt)
                                cellDic['AUC'] = abs(AUC)
                                cellDic[
                                    'condition'] = f'{percentilesc[l]}'  # {allModTraces[condition][events[e]]["number"]}'
                                cellDic['event'] = events[g]
                                cellDic['id'] = c
                                # Find positive peak

                                cellDicList.append(cellDic)
                            meanZscore = np.mean(allModTraces[condition][events[g]]['traces'], axis=0)
                            meanZscore = gaussian_filter1d(meanZscore, 0.8)
                            semZscore = stats.sem(allModTraces[condition][events[g]]['traces'], axis=0)
                            traceAverageArray[g].append(meanZscore)
                    dfCells = pd.DataFrame(cellDicList)

                ax0b = plt.subplot(gssub6[3])
                if e == 0:
                    alpha = 0.6
                else:
                    alpha = 1

                numberArrayEv = np.array(numberArray[e])

                MLIfrac = (numberArrayEv / totCells) * 100
                x = np.arange(5)
                y = MLIfrac
                slope, intercept, r_value, p_value, sterr = stats.linregress(x, y,
                                                                             alternative='two-sided')
                ax0b.plot(percentilesc, MLIfrac, '-o', lw=2, c='C1', alpha=alpha)
                sns.regplot(x=np.arange(5), y=MLIfrac, scatter_kws={'alpha': 0.8, 'edgecolor': 'k', 'lw': 0.1},
                            line_kws={'alpha': alpha, 'lw': 1.5}, color='0.8')
                corr_star = groupAnalysis.starMultiplier(p_value)
                ax0b.set_ylim(0, 70)
                ax0b.tick_params(axis='both', labelsize=13)
                if p_value > 0.05:
                    ax0b.text(0.73, 0.80, f"r = {r_value:.2f}\np = {p_value:.2f}",
                              transform=ax0b.transAxes, fontsize=10, color='dimgrey')

                else:
                    ax0b.text(0.75, 0.4, f"r = {r_value:.2f}",
                              transform=ax0b.transAxes, fontsize=18, color='dimgrey')
                    ax0b.text(0.5, 0.98, f"{corr_star}",
                              transform=ax0b.transAxes, fontsize=18, color='k')

                if variable != 'swingLengthLinear':
                    self.layoutOfPanel(ax0b, xLabel=f'swing {variable[5:].lower()} percentile (%)',
                                       yLabel=f'{events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                       xyInvisible=[False, False])
                else:
                    self.layoutOfPanel(ax0b, xLabel=f'swing length percentile (%)',
                                       yLabel=f'{events[e]} modulated {cellType} (%)', Leg=[0, 9],
                                       xyInvisible=[False, False])
                # ax1b = plt.subplot(gssub6[2:4])
                import matplotlib.colors as mcolors
                from matplotlib.colors import ColorConverter

                cc = ColorConverter()
                blue = mcolors.to_rgba('C1')

                colorList0 = ['#d2e3e4', '#b9d5e4', '#9bc7e4', '#7db9e4', '#5cace4']
                colorList = ['#8ca5bf', '#6b8dbf', '#4c74bf', '#2d5abf', '#0d40bf']
                colorList = ['#a3cef0', '#80bce5', '#5daeda', '#3a9dcf', '#167fc4']
                colorList = ['#FFDDC1', '#FFC397', '#FFA76D', '#FF8B43', '#FF6F19']
                # for a in range(5):
                #     intensity = (a + 1) / 5
                #     color = blue[:3] + (intensity,)  # adjust alpha channel\
                #     averageZscore = gaussian_filter1d(traceAverageArray[e][a], 0.8)
                #     line = ax1b.plot(time, averageZscore, color=color, lw=2,
                #                      label=f'[{percentiles[a]}]')
                #
                #     meanAUC = np.mean(traceAUC[e][a])
                #     # pdb.set_trace()
                #
                #     ax1b.text(0.42, 0.97, f"{events[e]} onset",
                #               transform=ax1b.transAxes, fontsize=14, color='grey')
                #     ax1b.axvline(0, ls='--', color='grey', alpha=0.3)
                #     ax1b.axhline(0, ls='--', color='grey', lw=1, alpha=0.3)
                #
                #     # self.layoutOfPanel(ax1b, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                #     #                    xyInvisible=[False, False])
                #     self.layoutOfPanel(ax1b, xLabel='time (s)', yLabel=' average PSTH Z-score', Leg=[0, 9],
                #                        xyInvisible=[False, False])
                #
                #     legend = ax1b.legend(loc='upper left', frameon=False, fontsize=12)
                #     legend.set_title(f'swing {variable[5:].lower()} (%)')
                #     legend_title = legend.get_title()
                #     legend_title.set_fontsize(13)  # Change the fontsize to 12
                #     legend.set_bbox_to_anchor((0.002, 1.15))

                variables = ['AUC', 'PosPeak', 'PosLatency', 'NegPeak', 'NegLatency', 'TimeToPeak', 'TimeToTrough']
                variables = ['AUC']
                varNames = ['AUC (abs)', 'positive peak', 'positive peak latency', 'negative peak',
                            'negative peak latency',
                            'Time To Peak', 'Time To Trough']
                for v, var in enumerate(variables):
                    ax2b = plt.subplot(gssub6[2 + v])
                    cellDfEvent = dfCells[(dfCells['event'] == events[e])]
                    cellDfEvent_80_100 = \
                    dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[80-100]')][var]
                    cellDfEvent_0_20 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[0-20]')][
                        var]
                    cellDfEvent_20_40 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[20-40]')][
                        var]
                    cellDfEvent_40_60 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[40-60]')][
                        var]
                    cellDfEvent_60_80 = dfCells[(dfCells['event'] == events[e]) & (dfCells['condition'] == '[60-80]')][
                        var]
                    dataAnova = [cellDfEvent_0_20, cellDfEvent_20_40, cellDfEvent_40_60, cellDfEvent_60_80,
                                 cellDfEvent_80_100]
                    sns.pointplot(cellDfEvent, x='condition', y=var, hue='condition', ax=ax2b, errorbar='se',
                                  palette=colorList)
                    sns.lineplot(cellDfEvent, x='condition', y=var, ax=ax2b, errorbar=None, lw=2, color='C1')
                    f_statistic, p_valueAnova = stats.f_oneway(*dataAnova)
                    print(events[e], f_statistic, p_valueAnova)

                    t_value, t_test_p_value = stats.ttest_ind(cellDfEvent_0_20, cellDfEvent_80_100)

                    star_trial = groupAnalysis.starMultiplier(p_valueAnova)
                    if t_test_p_value < 0.05:
                        ax2b.text(0.5, 0.99, (f'{star_trial} '), ha='center', va='center',
                                  transform=ax2b.transAxes,
                                  style='italic', fontfamily='serif', fontsize=18, color='k')
                    else:
                        ax2b.text(0.5, 0.99, (f'p={p_valueAnova:.2f}'), ha='center', va='center',
                                  transform=ax2b.transAxes,
                                  style='italic', fontfamily='serif', fontsize=12, color='k')

                    if variable != 'swingLengthLinear':
                        self.layoutOfPanel(ax2b, xLabel=f'swing {variable[5:].lower()} percentile (%)',
                                           yLabel=(f'{events[e]} onset \n PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                           xyInvisible=[False, False])
                    else:
                        self.layoutOfPanel(ax2b, xLabel=f'swing length percentile (%)',
                                           yLabel=(f'{events[e]} onset \n PSTH Z-score {varNames[v]}'), Leg=[0, 9],
                                           xyInvisible=[False, False])
                    ax2b.legend([], [], frameon=False)
                    ax2b.tick_params(axis='both', labelsize=13)
                    paw = pawList[pawNb]
        fname = f'fig_supp_ephys_psth_Z-score_{zscorePar}_cell_based_{cellType}_{paw}_{condition}_{figVersion}'
        # plt.savefig(fname + '.png')
        # plt.show()
        plt.savefig(fname + '.pdf')
        # groupAnalysisFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary/'
        # plt.savefig(groupAnalysisFigDir+ fname + '.pdf')
    def fig_supp_muscimol(self, figVersion, stridePar, strideTraj, swingNumber, rungCrossed, strideDuration,
                     indecisiveStrideFraction, swingSpeed, pawCoordination, strideLenght, musSwingStanceDic,
                     salineSwingStanceDic):
        cmap = cm.get_cmap('tab20')
        colors = ['steelblue', 'darkorange', 'yellowgreen', 'salmon']
        pawId = ['FL', 'FR', 'HL', 'HR']
        col = ['C0', 'C1', 'C2', 'C3']
        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 9  # height in inches
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
        gs = gridspec.GridSpec(2,3)

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.38, hspace=0.35)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12)
        plt.figtext(0.02, 0.95, 'A', clip_on=False, color='black', size=20)
        plt.figtext(0.37, 0.95, 'B', clip_on=False, color='black', size=20)
        # plt.figtext(0.65, 0.96, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.75, 0.96, 'D', clip_on=False, color='black',  size=22)
        plt.figtext(0.67, 0.95, 'C', clip_on=False, color='black', size=20)
        plt.figtext(0.02, 0.48, 'D', clip_on=False, color='black', size=20)
        plt.figtext(0.37, 0.48, 'E', clip_on=False, color='black', size=20)
        plt.figtext(0.67, 0.48, 'F', clip_on=False, color='black', size=20)
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0:2], hspace=0.15, wspace=0.05,
                                                  width_ratios=[1, 2])

        # regroup data per trial, day, paw and mouse
        stridePar_Recordings = stridePar.groupby(['trial', 'day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings = stridePar_Recordings.reset_index()

        # get data for five trials only, for visualization
        stridePar_Recordings_trial = stridePar_Recordings[stridePar_Recordings['trial'] <= 5]
        stridePar_Recordings_trial = stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        # average data per day for visualization
        stridePar_Recordings_day = stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day = stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day = stridePar_Recordings.groupby(['paw', 'day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day = stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_trial = stridePar_Recordings.groupby(
            ['paw', 'trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial = stridePar_Recordings_paw_trial.reset_index()

        trials = stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList = trials.astype(str).tolist()
        daysList = days.astype(str).tolist()

        nSal = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'saline']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol']['mouseId'].nunique()
        salineId = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'saline']['mouseId'].unique()
        muscimolId = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'muscimol'][
            'mouseId'].unique()
        panelList = [2, 4, 5]
        parameters = ['swingLength', 'swingDuration', 'stanceDuration', 'swingSpeed']
        parameters_Y = ['swing length (cm)', 'swing duration (s)', 'stance duration (s)','swing speed (cm/s)']


        for par in range(len(parameters)):
            ax1 = plt.subplot(gs[par+1])
            # elif par == 2:
            #     ax1 = plt.subplot(gs[4])
            # per day data
            # perform stats Mixed Linear Model
            pars_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, parameters[par],
                                                         treatments=True)
            # pars_summary = groupAnalysis.performLMMANOVA(stridePar_Recordings, parameters[par],'mouseId',
            #                                                     treatments=True)
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='treatment',
                         hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['black', 'red'], ax=(ax1), marker='o')
            sns.lineplot(data=stridePar_Recordings_day, x='day', y=parameters[par], hue='mouseId',
                         style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)], alpha=0.1,
                         err_kws={'capsize': 3, 'linewidth': 1}, palette=['red', 'black'], ax=(ax1))
            ax1.text(1, 0.4, f'{pars_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
                     ha='center', va='center', transform=ax1.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            # if par == 0:
            #     ax1.text(0.7, 0.01, f'N={nSal} saline, {nMus} muscimol', ha='center', va='center',
            #              transform=ax1.transAxes, style='italic',
            #              fontsize=12, color='k')
            ax1.xaxis.set_major_locator(MultipleLocator(1))

            self.layoutOfPanel(ax1, xLabel='session', yLabel=parameters_Y[par])
            ax1.legend([
                           f'saline {pars_summary["stars"]["saline"]["day"]} {pars_summary["stars"]["saline"]["trial"].replace("*", "#")}',
                           f'muscimol {pars_summary["stars"]["muscimol"]["day"]} {pars_summary["stars"]["muscimol"]["trial"].replace("*", "#")}'],
                       loc='upper left', frameon=False)
        # swingNumber_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingNumber',
        #                                              treatments=True)
        for i in range(4):
            gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], hspace=0.4, wspace=0.35)
            ax2=plt.subplot(gssub0[i])
            pawDf=stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[i])]
            swingNumber_summary = groupAnalysis.perform_mixedlm_treatment_single_paw(pawDf, 'swingNumber')
            sns.lineplot(data=pawDf, x='day', y='swingNumber', hue=None,
                         hue_order=['saline', 'muscimol'], style='treatment', style_order=['saline', 'muscimol'],
                         errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
                         err_kws={'capsize': 3, 'linewidth': 1}, color=f'C{i}', ax=(ax2), marker='o')
            ax2.text(1, 0.33, f'{swingNumber_summary["stars"]["all"]["treatment[T.saline]"].replace("*", "°")}',
                     ha='center', va='center', transform=ax2.transAxes, style='italic', fontfamily='serif',
                     fontsize=15, color='k')
            if i==2:
                self.layoutOfPanel(ax2, xLabel='session', yLabel='stride number (avg.)', xyInvisible=([False, False]))
                ax2.set_ylim(75,155)
            elif i==1 :
                self.layoutOfPanel(ax2, xLabel='session', yLabel='', xyInvisible=([True, True]))
                ax2.set_ylim(70,110)
            elif i==0 :
                self.layoutOfPanel(ax2, xLabel='session', yLabel='', xyInvisible=([True, False]))
                ax2.set_ylim(70,110)
            elif i==3 :
                self.layoutOfPanel(ax2, xLabel='session', yLabel='', xyInvisible=([False, True]))
                ax2.set_ylim(75,155)
            ax2.xaxis.set_major_locator(MultipleLocator(2))
            ax2.legend([f'saline',f'muscimol'], bbox_to_anchor=(0.35,0.7,0.5,0.5),loc='upper left', frameon=False, fontsize=10)
            # ax2.legend([f'saline {swingNumber_summary["stars"]["saline"]["day"]}',f'muscimol {swingNumber_summary["stars"]["muscimol"]["day"]}'], bbox_to_anchor=(0.35,0.7,0.5,0.5),loc='upper left', frameon=False, fontsize=10)


        iqr_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'stanceOn_iqr_25_75_ref_FL', treatments=True)
        swingOnMed_summary = groupAnalysis.perform_mixedlm(stridePar_Recordings, 'swingOnMedian_FL', treatments=True)
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[5], width_ratios=[1], hspace=0.15,
                                                  wspace=0.80)
        for i in range(2):
            col_paw=[f'C{i}']
            for t in ['saline', 'muscimol']:
                paw_treatment_data = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[i]) & (stridePar_Recordings['treatment'] == t)]
                if i==0:
                    ax1b = plt.subplot(gssub1[1])
                    stars_iqr = f'{pawId[i]} {t} {iqr_summary["paw_stars"][pawId[i]][t]["day"]} {iqr_summary["paw_stars"][pawId[i]][t]["trial"].replace("*", "#")}'
                    sns.lineplot(data=paw_treatment_data, x='day', y='stanceOn_iqr_25_75_ref_FL', hue='paw', style='treatment', style_order=['saline', 'muscimol'],
                                 errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                                 err_kws={'capsize': 3, 'linewidth': 1}, palette=col_paw, ax=ax1b, marker='o', legend=False, label=stars_iqr)
                    self.layoutOfPanel(ax1b, xLabel='session', yLabel= 'FL  stance onset\n IQR')

                    # ax1b.legend(f'{pawId[i]} {iqr_summary["paw_stars"][pawId[i]]["day"]} {iqr_summary["paw_stars"][pawId[i]]["day"].replace("*","#")}')

                elif i==1:
                    ax1b = plt.subplot(gssub1[0])

                    stars_swingOn= f'{pawId[i]} {t} {swingOnMed_summary["paw_stars"][pawId[i]][t]["day"]} {swingOnMed_summary["paw_stars"][pawId[i]][t]["trial"].replace("*", "#")}'

                    sns.lineplot(data=paw_treatment_data, x='day', y='swingOnMedian_FL', hue='paw', style='treatment', style_order=['saline', 'muscimol'],
                                 errorbar=('se'), err_style='bars',dashes=[(1, 0), (1, 1)],
                                 err_kws={'capsize': 3, 'linewidth': 1}, palette=col_paw, ax=ax1b, marker='o', legend=False, label=stars_swingOn)
                    self.layoutOfPanel(ax1b, xLabel='session', yLabel= 'FR swing onset\n median',xyInvisible=[True, False])
                ax1b.xaxis.set_major_locator(MultipleLocator(1))
                ax1b.legend(loc='upper right', frameon=False, fontsize=10)
          #regroup stats in star_label list, trial effect as ###
        # star_label=[]
        # star_label_inset = []
        # for i in range(4):
        #     star_label.append(pawId[i]+' '+swingNumber_summary['paw_stars'][pawId[i]]['day'])
        #     star_label_inset.append(pawId[i]+' '+swingNumber_summary['paw_stars'][pawId[i]]['trial'].replace('*', '#'))
        #stars as legend
        # ax2.legend(star_label, bbox_to_anchor=(0.86,0.78), frameon=False, fontsize=14)







        fname = 'fig_supp_muscimol_v%s' % figVersion
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')
    ###############################################################################
    def fig_real_time_experiment(self,figVersion, stridePar, strideTraj, triggerDict,opsValues,tomValues,data): #DictopsinSwingStanceDic,tdTomatoSwingStanceDic, trigger):
        #############################
        def determineLaserActivationDistribution(case, successRate, idxSwings, recTimes, pawPos, pawID, laserDict):
            #successRate[case]['normStepCycleBins'] = np.zeros(201)
            #successRate[case]['normTimeStepCycle'] =  np.linspace(0,2,201)
            normTimeStepCycle = successRate[case]['normTimeStepCycle']
            for j in range(len(idxSwings) - 2):
                idxStart = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][0]]))
                idxEnd = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][1]]))
                idxStartNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][0]]))
                idxEndNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][1]]))
                idxStartNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][0]]))
                idxEndNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][1]]))
                swingNStart = pawPos[pawID][idxStart, 0]
                swingNEnd = pawPos[pawID][idxEnd, 0]
                swingNPOStart = pawPos[pawID][idxStartNext, 0]
                swingNPOEnd = pawPos[pawID][idxEndNext, 0]
                swingNPTStart = pawPos[pawID][idxStartNNext, 0]
                swingNPTEnd = pawPos[pawID][idxEndNNext, 0]
                laserActivationTimeSwing = np.argwhere((laserDict['laserStartTime'] >= swingNStart) & (laserDict['laserStartTime'] < swingNEnd))
                laserActivationTimeStance = np.argwhere((laserDict['laserStartTime'] >= swingNEnd) & (laserDict['laserStartTime'] < swingNPOStart))
                if len(laserActivationTimeSwing) > 0:
                    successRate[case]['swingWithLaserActivation'] += 1
                else:
                    successRate[case]['swingNoLaserActivation'] += 1
                # if len(laserActivationTimeSwing)>1 :
                #     print(laserActivationTimeSwing)
                #     print('more than one activations during the swing')
                #     pdb.set_trace()
                if len(laserActivationTimeStance) > 0: successRate[case]['stanceWithLaserActivation'] += 1
                # if len(laserActivationTimeStance)>1:
                #     print(laserActivationTimeStance)
                #     print('more than one activations during the stance')
                #     pdb.set_trace()
                swingDuration = swingNEnd - swingNStart
                stanceDuration = swingNPOStart - swingNEnd
                swingNextDuration = swingNPOEnd - swingNPOStart
                stanceNextDuration = swingNPTStart - swingNPOEnd
                for k in laserActivationTimeSwing:
                    normStart = (laserDict['laserStartTime'][k] - swingNStart) / swingDuration
                    start_bin = np.digitize(normStart, normTimeStepCycle[:101]) - 1
                    if laserDict['laserEndTime'][k] < swingNEnd:
                        normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / swingDuration
                        assert normDuration < 1, print('The normalized duration in case one should be smaller than one!')
                        normEnd = normStart + normDuration
                        end_bin = np.digitize(normEnd, normTimeStepCycle) - 1  # unique_bins = np.arange(start_bin, end_bin + 1)  # normStepCycleBins[np.unique(unique_bins)] += 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] < swingNPOStart):
                        # normDuration = (laserDict['laserEndTime'][k]-laserDict['laserStartTime'][k])/swingDuration
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNEnd) / stanceDuration
                        assert normDurationStance < 1, print('The normalized duration in case two should be smaller than one!')
                        #if normDurationStance < 1 : print('It\'s happening again in case two!')
                        # start_bin = np.digitize(0, normTimeStepCycle[100:]) - 1 + 100
                        end_bin = np.digitize(1.+ normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case twoB should be smaller than one!' % normDurationSwing)
                        #if  normDurationStance < 1 : print('It\'s happening again in case twoB !')
                        end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 101), np.arange(0, end_bin)))
                        successRate[case]['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case twoC should be smaller than one!' % normDurationSwing)
                        end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                        successRate[case]['normStepCycleBins'] += 1  # in this condition, laser is active during a whole stance and swing
                        #successRate[case]['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
                    successRate[case]['normStepCycleBins'][np.unique(unique_bins)] += 1
                for k in laserActivationTimeStance:
                    normStart = (laserDict['laserStartTime'][k] - swingNEnd) / stanceDuration
                    start_bin = np.digitize(normStart + 1., normTimeStepCycle) - 1
                    if laserDict['laserEndTime'][k] < swingNPOStart:
                        normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / stanceDuration
                        assert normDuration < 1, print('The normalized duration in case three should be smaller than one!')
                        normEnd = 1 + normStart + normDuration
                        end_bin = np.digitize(normEnd, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                        normDurationSwing = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                        assert normDurationSwing < 1, print('The normalized duration %s in case four should be smaller than one!' % normDurationSwing)
                        end_bin = np.digitize(normDurationSwing, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(0, end_bin+1))) # np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case five should be smaller than one!' % normDurationStance)
                        end_bin = np.digitize(normDurationStance + 1, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(100, end_bin + 1)))
                        successRate[case]['normStepCycleBins'][np.arange(0, 101)] += 1  # in this condition, laser is active during a whole swing
                    # unique_bins = np.arange(start_bin, end_bin + 1)
                    successRate[case]['normStepCycleBins'][np.unique(unique_bins)] += 1


        #############################
        def getLaserActivationStartAndEnd(time,data):
            laserDict = {}
            difference = np.diff(data)  # calculate difference
            laserDict['laserStartIdx'] = np.arange(len(data))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
            laserDict['laserEndIdx'] = np.arange(len(data))[np.concatenate((np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
            laserDict['laserStartTime'] = time[laserDict['laserStartIdx']]
            laserDict['laserEndTime'] = time[laserDict['laserEndIdx']]
            return laserDict

        displayRec = 4
        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        # figure #################################
        fig_width = 14  # width in inches
        fig_height = 13  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1.2] )#),width_ratios=[1,0.9])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.34, hspace=0.35)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07)
        plt.figtext(0.03, 0.96, 'A', clip_on=False, color='black',  size=20)
        plt.figtext(0.53, 0.96, 'B', clip_on=False, color='black',  size=22)
        plt.figtext(0.03, 0.53, 'C', clip_on=False, color='black', size=22)
        plt.figtext(0.48, 0.53, 'D', clip_on=False, color='black',  size=20)
        plt.figtext(0.48, 0.28, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)

        gssub0a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.35, wspace=0.3, width_ratios=[4,4])
        gssub0a1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub0a[0], hspace=0.35, wspace=0.1,width_ratios=[7,4])

        #opsin_img_merge=mpimg.imread('Andry F3 ACR red fix by Ali Mosaic2 20x_Stitch-1-very_red_small_optic_fiber_tract.png')
        #opsin_img_red = mpimg.imread('Andry F3 ACR red fix by Ali Mosaic2 20x_Stitch-1-very_red_small.png')
        # opsin_img_merge=mpimg.imread('AVG_AVG_f65-d3-cell-pushed_00002_filtered.png')
        # opsin_img_red =mpimg.imread('AVG_AVG_f65-d3-cell-pushed_00002_filtered.png')
        ax0a=plt.subplot(gssub0a1[0])
        ax0a1 = plt.subplot(gssub0a1[1])
        # ax0a.imshow(opsin_img_merge)
        # ax0a1.imshow(opsin_img_red)
        left, bottom, width, height = ax0a.get_position().bounds
        left1, bottom1, width1, height1 = ax0a1.get_position().bounds
        ax0a.set_position([left-0.03,bottom-0.04,width+0.05,height+0.05])  	#[left, bottom, width, height]
        ax0a.axis('off')
        ax0a.text(0.9, 0.96, 'GtACR2', fontsize=11, color='#FF0000', ha='center', va='center', transform=ax0a.transAxes, alpha=0.9)
        ax0a1.set_position([left1-0.,bottom1-0.04,width1+0.05,height1+0.05])  	#[left, bottom, width, height]
        ax0a1.axis('off')
        ax0a1.text(0.86, 0.95, 'GtACR2', fontsize=10, color='#FF0000', ha='center', va='center', transform=ax0a1.transAxes, alpha=0.9)
        ax0a.text(0.1, 0.2, 'IV/V', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        ax0a.text(0.5, 0.5, 'S', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # idxSwings =np.array(swingStanceD['swingP'][i][1])
        # recTimes = np.array(swingStanceD['forFit'][i][2])
        pawPos_opsin=triggerDict[displayRec]['ops']['swingStance']['pawPos'] #opsinSwingStanceDic['pawPos']
        pawPos_tdTomato= triggerDict[displayRec]['tom']['swingStance']['pawPos'] #tdTomatoSwingStanceDic['pawPos']
        gssub0d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0a[1], hspace=0.18, wspace=0.1,
                                                   height_ratios=[2, 2])
        ax0=plt.subplot(gssub0d[1])
        ax0b=plt.subplot(gssub0d[0])
        startx = 34#42
        xLength = 10#7

        # to extract swing and stance onsets
        swingStanceD = triggerDict[displayRec]['ops']['swingStance']
        pawID = 0
        idxSwings = swingStanceD['swingP'][pawID][1]
        #indecisiveSteps = swingStanceD['swingP'][i][3]
        recTimes = swingStanceD['forFit'][pawID][2]


        # end of

        #for i in [0]:
        t_emg_ops = triggerDict[displayRec]['ops']['time'] #trigger['time']  # time points for EMG data
        t_emg_tom = triggerDict[displayRec]['tom']['time']

        laserDict = getLaserActivationStartAndEnd(t_emg_ops, triggerDict[displayRec]['ops']['Trigger'])

        pos_interp_opsin = np.interp(t_emg_ops, pawPos_opsin[pawID][:, 0], pawPos_opsin[pawID][:, 1])
        pos_interp_tdTomato = np.interp(t_emg_tom, pawPos_tdTomato[pawID][:, 0], pawPos_tdTomato[pawID][:, 1])
        ax0.plot(t_emg_ops, pos_interp_opsin, c=col[pawID], lw=1.4, label=f'{pawId[pawID]} GtACR2', ls=(0, (1, 1)))
        ax0.axhline(y=340,ls='--',c='0.8',alpha=0.5)
        ax0.axhline(y=340+120,ls='--',c='0.8',alpha=0.5)
        ax0b.plot(t_emg_tom, pos_interp_tdTomato, c=col[pawID], lw=1.4, label=f'{pawId[pawID]} tdTomato', ls='-')
        ax0b.axhline(y=320,ls='--',c='0.8',alpha=0.5)
        ax0b.axhline(y=320+120,ls='--',c='0.8',alpha=0.5)
        #ax0e = ax0.twinx()
        #ax0f = ax0b.twinx()
        #ax0b.axis('off')
        #pdb.set_trace()
        ax0.fill_between(t_emg_ops, 200 + triggerDict[displayRec]['ops']['Trigger']*200, color='C6', alpha=0.3,edgecolor=None)
        ax0b.fill_between(t_emg_tom, 200 + triggerDict[displayRec]['tom']['Trigger']*200, color='C6', alpha=0.3,edgecolor=None)
        #ax0e.axis('off')
        #ax0f.axis('off')
        ax0.set_ylim(310,510)
        ax0b.set_ylim(290,490)

        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'FL x  (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        self.layoutOfPanel(ax0b, xLabel='time (s)', yLabel= 'FL x (pixel)', Leg=[1, 9])
        ax0.legend(frameon=False, bbox_to_anchor=(0.85,1.05), loc='center')
        ax0b.legend(frameon=False, bbox_to_anchor=(0.83,1.05), loc='center')
        ax0.set_xlim(startx, startx + xLength)
        ax0b.set_xlim(startx, startx + xLength)
        ax0b.xaxis.set_major_locator(MultipleLocator(1))
        #ax0b.yaxis.set_major_locator(MultipleLocator(50))

        # for j in range(len(idxSwings)):
        #     idxStart = np.argmin(np.abs(pawPos_opsin[pawID][:, 0] - recTimes[idxSwings[j][0]]))
        #     idxEnd = np.argmin(np.abs(pawPos_opsin[pawID][:, 0] - recTimes[idxSwings[j][1]]))
        #     ax0.plot(pawPos_opsin[pawID][idxStart, 0], pawPos_opsin[pawID][idxStart, 1], 'x', c=col[pawID], alpha=0.5, lw=0.5)
        #     ax0.plot(pawPos_opsin[pawID][idxEnd, 0], pawPos_opsin[pawID][idxEnd, 1], '+', c=col[pawID], alpha=0.5, lw=0.5)
        # for n in range(len(laserDict['laserStartIdx'])):
        #     ax0e.plot(laserDict['laserStartTime'][n], 0.5,'x')
        #     ax0e.plot(laserDict['laserEndTime'][n], 0.5, '+')


        # generate a plot to quantify efficiency of laser activation ###############################

        #normStepCycleBinsTom = np.zeros(201)
        #normTimeStepCycleTom = np.linspace(0,2,201)
        successRate = {}  # np.zeros(4)
        cases = {'tom':[0,1,2,3,4],'ops':[2,3,4]}
        #tomRecs = [0,1,2,3,4]
        #opsRecs = [2,3,4]
        pawID=0
        for key,values in cases.items():
            successRate[key] = {}
            successRate[key]['normStepCycleBins'] = np.zeros(201)
            successRate[key]['normTimeStepCycle'] =  np.linspace(0,2,201)
            successRate[key]['totalNumberOfSteps'] = 0 #len(idxSwings)
            successRate[key]['swingWithLaserActivation'] = 0
            successRate[key]['swingNoLaserActivation'] = 0
            successRate[key]['stanceWithLaserActivation'] = 0

            for r in values:
                t_emg = triggerDict[r][key]['time']  # trigger['time']  # time points for EMG data
                pawPos = triggerDict[r][key]['swingStance']['pawPos']  # tdTomatoSwingStanceDic['pawPos']
                laserDict = getLaserActivationStartAndEnd(t_emg, triggerDict[r][key]['Trigger'])
                swingStanceD = triggerDict[r][key]['swingStance']

                pawID = 0
                idxSwings = swingStanceD['swingP'][pawID][1]
                # indecisiveSteps = swingStanceD['swingP'][i][3]
                recTimes = swingStanceD['forFit'][pawID][2]
                successRate[key]['totalNumberOfSteps'] += len(idxSwings)
                determineLaserActivationDistribution(key, successRate, idxSwings, recTimes, pawPos, pawID, laserDict)

        #pdb.set_trace()
        # Draw shaded red vertical bars for light on periods
        # light_on_periods_ops = np.where(trigger['ops'] > 1)[0]
        # light_on_periods_tom = np.where(trigger['tom'] > 1)[0]
        #
        # for period in light_on_periods_ops:
        #     ax0.axvspan(t_emg[period], t_emg[period + 1], alpha=0.005, color='C6')
        #
        # for period in light_on_periods_tom:
        #     ax0b.axvspan(t_emg[period], t_emg[period + 1], alpha=0.005, color='C6')


        ############################Panel A and B : image of opsin spread or illustration image#########################
        ############################Panel C : opsin application effect graph#########################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.35, wspace=0.3, width_ratios=[4, 4])
        gssub1b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[1], hspace=0.4, wspace=0.35, width_ratios=[9, 4])

        ax11a=plt.subplot(gssub1b[0])
        ax11b=plt.subplot(gssub1b[1])
        totalStepsTom = successRate['tom']['totalNumberOfSteps'] #(successRate['tdTomoato']['swingWithLaserActivation']+successRate['tdTomoato']['swingNoLaserActivation'])
        totalStepsOps = successRate['ops']['totalNumberOfSteps'] #(successRate['opsin']['swingWithLaserActivation'] + successRate['opsin']['swingNoLaserActivation'])
        ax11a.step(successRate['tom']['normTimeStepCycle'],successRate['tom']['normStepCycleBins']/totalStepsTom,c='C6',where='pre')
        ax11a.step(successRate['ops']['normTimeStepCycle'],successRate['ops']['normStepCycleBins'] / totalStepsOps, c='C6', where='pre', ls=(0, (1, 1)))
        ax11a.axvline(1,ls='--',c='0.2')

        self.layoutOfPanel(ax11a, xLabel='normalized time', yLabel= 'probability of laser ON')

        ###################
        x_pos = [1, 1.2, 2, 2.2]
        #ax11b.bar()
        #values = []
        #for case in cases:
        #    v1 = successRate[case]['swingWithLaserActivation']/(successRate[case]['swingWithLaserActivation']+successRate[case]['swingNoLaserActivation'])
        #    values.append(v1)
        #    v2 = successRate[case]['stanceWithLaserActivation']/(successRate[case]['swingWithLaserActivation']+successRate[case]['swingNoLaserActivation'])
        #    values.append(v2)
        opsValues = np.asarray(opsValues)
        tomValues = np.asarray(tomValues)
        sns.stripplot(data=[opsValues[:,0],opsValues[:,1],tomValues[:,0],tomValues[:,1]],jitter=True, size=6, alpha=0.3,ax=ax11b,color='0.1')
        sns.boxplot(data=[opsValues[:,0],opsValues[:,1],tomValues[:,0],tomValues[:,1]], ax=ax11b, width=0.3, showcaps=False,showfliers=False, boxprops={'facecolor':'None'},whiskerprops={'linewidth':1.5}, medianprops={'color':'black'}) #, positions=[0.8, 2.2])

        #print(np.shape(opsValues))
        #print(np.shape(tomValues))
        #pdb.set_trace()

        ccc = ['swing','stance']
        for  i in range(2):
            print('%s activation (25,50,75) : ' % ccc[i])
            print('  opsin  : ',end='')
            median = np.median(opsValues[:,i])
            q1 = np.percentile(opsValues[:,i], 25)
            q3 = np.percentile(opsValues[:,i], 75)
            print(q1,median,q3)
            print('  tdtomato  : ',end='')
            median = np.median(tomValues[:,i])
            q1 = np.percentile(tomValues[:,i], 25)
            q3 = np.percentile(tomValues[:,i], 75)
            print(q1,median,q3)
        #pdb.set_trace()
        #sns.stripplot(x=np.repeat(x_pos[1], len(opsValues[:, 1])), y=opsValues[:, 1], jitter=True, size=6, alpha=0.3, ax=ax11b,color='0.8')
        #sns.boxplot(data=opsValues[:, 1], ax=ax11b, width=0.3, showcaps=False, boxprops={'facecolor': 'None'}, whiskerprops={'linewidth': 1.5}, medianprops={'color': 'black'})
        #ax11b.plot(x_pos[0]+(0.1*(np.random.rand(len(opsValues))-0.5)),opsValues[:,0], 'o',color='0.1')
        #ax11b.plot(x_pos[1]+(0.1*(np.random.rand(len(opsValues))-0.5)), opsValues[:, 1], 'o', color='0.8')
        #sns.stripplot(x=np.repeat(x_pos[2], len(tomValues[:, 0])), y=tomValues[:, 0], jitter=True, size=6, alpha=0.3, ax=ax11b, color='0.1')
        #sns.stripplot(x=np.repeat(x_pos[3], len(tomValues[:, 1])), y=tomValues[:, 1], jitter=True, size=6, alpha=0.3, ax=ax11b, color='0.8')
        #ax11b.plot(x_pos[2]+(0.1*(np.random.rand(len(tomValues))-0.5)),tomValues[:,0], 'o',color='0.1')
        #ax11b.plot(x_pos[3]+(0.1*(np.random.rand(len(tomValues))-0.5)), tomValues[:, 1], 'o', color='0.8')
        #, '0.8', '0.1', '0.9'])
        cases = ['GtACR2','dtTomato']
        # Set custom x-labels at appropriate positions
        plt.xticks([0.5, 2.5], cases, rotation=45, ha="right")
        self.layoutOfPanel(ax11b, yLabel='laser activations (%)')
        print(successRate)
        #pdb.set_trace()
        ##############################################################################################
        # ephysTimes = data[0][10]
        # dt = np.mean(ephysTimes[1:] - ephysTimes[:-1])
        # recStart = ephysTimes[0]
        # recEnd = ephysTimes[-1] + dt
        #
        # for i in range(len(data)):
        #     trace = data[i][9][0]
        #     timeTrace = data[i][10]
        #     rescaledTrace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
        #     difference = np.diff(rescaledTrace)  # calculate difference
        #     expStart = np.arange(len(rescaledTrace))[
        #         np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
        #     expEnd = np.arange(len(rescaledTrace))[np.concatenate(
        #         (np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
        #     # print(expStart,expEnd,timeTrace[expStart],timeTrace[expEnd])
        #     stimStart = timeTrace[expStart]
        #     stimEnd = timeTrace[expEnd]
        #     print('Stimulation start and end : ', stimStart, stimEnd)
        # gssub0c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0a[1], hspace=0.15, wspace=0.2,
        #                                            height_ratios=[3, 2])
        # ax0c = plt.subplot(gssub0c[0])
        # allSpikes = []
        # if len(stimStart) > 1:
        #     for n in range(len(stimStart)):
        #         ax0c.axvspan(stimStart[n], stimEnd[n], alpha=0.3, facecolor='C6', linewidth=None)
        # else:
        #     ax0c.axvspan(stimStart, stimEnd, alpha=0.3, facecolor='C6', linewidth=None)
        # for i in range(len(data)):
        #     # ax0c.plot(data[i][13],data[i][12]*0.8/maxSpeed+i-0.4,lw=0.2,c='C1')
        #     ax0c.vlines(data[i][7],i-0.4, i+0.4,lw=0.05,color='0.5')
        #     allSpikes.extend(data[i][7])
        #
        # self.layoutOfPanel(ax0c,xLabel='time (s)',yLabel='trial',xyInvisible=[True,False])  # axL[n].append(ax)
        # majorLocator_y = plt.MultipleLocator(2)
        # ax0c.yaxis.set_major_locator(majorLocator_y)
        # ax0c.set_xlim(3, 8)
        # ax0d= plt.subplot(gssub0c[1])
        # binWidth = 0.1 # in sec
        # ephysTimes = data[0][3]
        # dt = np.mean(ephysTimes[1:] - ephysTimes[:-1])
        # tbins =  np.linspace(0.,len(ephysTimes)*dt,int(len(ephysTimes)*dt/binWidth)+1)
        # binnedspikes, _ = np.histogram(allSpikes, tbins)
        # binnedNormalizedSpikes = binnedspikes/(binWidth*len(data))
        # #spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
        # #convert the convolved spike trains to units of spikes/sec
        # #spikesconv *= 1. / binWidth
        #
        # if len(stimStart)>1:
        #     for n in range(len(stimStart)):
        #        ax0d.axvspan(stimStart[n], stimEnd[n], alpha=0.3, facecolor='C6', linewidth=None)
        #     else:
        #         try:
        #             ax0d.axvspan(stimStart, stimEnd, alpha=0.3, facecolor='C6', linewidth=None)
        #         except:
        #             pass
        #     #ax1.axvspan(stimStart, stimEnd, alpha=0.3, color='green')
        # ax0d.step(tbins[1:],binnedNormalizedSpikes, color='k')
        # ax0d.xaxis.set_major_locator(MultipleLocator(1))
        # ax0d.set_ylim(0,60)
        # ax0d.set_xlim(3, 8)
        # self.layoutOfPanel(ax0d, xLabel='time (s)', yLabel='average firing rate (spk/s)')  # axL[n].append(ax)
        ############################Panel B : Swing Number#########################
        stridePar = stridePar.drop(stridePar[(stridePar['day'] > 9)].index).reset_index()
        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        #get data for five trials only, for visualization
        stridePar_Recordings_trial=stridePar_Recordings[stridePar_Recordings['trial']<=5]
        stridePar_Recordings_trial=stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day=stridePar_Recordings.groupby(['paw','day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day=stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_day.to_csv('opto_results_statistics.csv')

        stridePar_Recordings_paw_trial=stridePar_Recordings.groupby(['paw','trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial=stridePar_Recordings_paw_trial.reset_index()

        trials=stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList=trials.astype(str).tolist()
        daysList = days.astype(str).tolist()


        nSal=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='tdTomato']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'opsin']['mouseId'].nunique()

        FL_swingLen_tdTomatoId=stridePar_Recordings_paw_day[(stridePar_Recordings_paw_day['treatment']=='tdTomato') & (stridePar_Recordings_paw_day['paw']=='FL')]['swingLength']
        FL_swingLen_opsinId=stridePar_Recordings_paw_day[(stridePar_Recordings_paw_day['treatment'] == 'opsin')& (stridePar_Recordings_paw_day['paw']=='FL')]['swingLength']

        print('tomato', 'mean', np.mean(FL_swingLen_tdTomatoId)*0.025, 'SD', np.std(FL_swingLen_tdTomatoId)*0.025)
        print('opsin', 'mean', np.mean(FL_swingLen_opsinId)*0.025, 'SD', np.std(FL_swingLen_opsinId)*0.025)
        #pdb.set_trace()
        panelList=[2,4,5]
        parameters=['swingLength', 'stanceDuration', 'indecisiveFraction','stanceOn_iqr_25_75_ref_FL']
        parameters_Y=['swing length (cm)','stance duration (s)', 'fraction of miss steps', 'stance onset iqr']
        parameters=['swingLength', 'indecisiveFraction','stanceOn_iqr_25_75_ref_FL']
        parameters_Y=['swing length (cm)','fraction of miss steps', 'stance onset IQR']
        hr=np.repeat(3,len(parameters))
        hr[0]=6
        gssub0b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], hspace=0.5)
        for p in range(len(parameters)):
            if parameters[p]!='swingLength':
                gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub0b[1], hspace=0.4, wspace=0.35)
                pawNb=2
            else:
                gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub0b[0], hspace=0.35, wspace=0.15)
                pawNb=4
            for i in range(pawNb):
                if parameters[p] != 'swingLength':
                    ax2 = plt.subplot(gssub0[p-1,i])
                else:
                    ax2 = plt.subplot(gssub0[i])
                pawDf = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[i])]
                pawDf.loc[:, 'swingLength'] = pawDf['swingLength'] * 0.025
                swingLength_summary = groupAnalysis.perform_mixedlm_treatment_single_paw(pawDf, parameters[p])
                sns.lineplot(data=pawDf, x='day', y=parameters[p], hue=None,
                             hue_order=['tdTomato', 'opsin'], style='treatment', style_order=['tdTomato', 'opsin'],
                             errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
                             err_kws={'capsize': 3, 'linewidth': 1}, color=f'C{i}', ax=(ax2), marker='o')
                interactionTerm = f'day:treatment[T.tdTomato]'
                ax2.text(1.05, (0.4 if p==len(parameters)-1 else 0.57), f'{swingLength_summary["stars"]["all"]["treatment[T.tdTomato]"].replace("*", "°")}',
                         ha='center', va='center', transform=ax2.transAxes, style='italic', fontfamily='serif',
                         fontsize=15, color='k')
                if parameters[p] == 'swingLength':
                    if i==0 or i==1:
                        ax2.set_ylim(1.8, 3.2)
                    else:
                        ax2.set_ylim(0.8, 2.2)
                elif parameters[p] == 'stanceDuration':
                    ax2.set_ylim(0.4, 0.8)
                elif parameters[p] == 'indecisiveFraction':
                    ax2.set_ylim(0, 0.3)
                elif parameters[p] == 'stanceOn_iqr_25_75_ref_FL':
                    ax2.set_ylim(0, 0.4)
                if i == 0 and parameters[p]!='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel=f'{parameters_Y[p]}', xyInvisible=([(True if p==1 else False), False]))
                    # ax2.set_ylim(70, 110)
                elif i == 1 and parameters[p]!='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel='', xyInvisible=([( True if p==1 else False), True]))


                    # ax2.set_ylim(70, 110)
                elif i == 1 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' if p==len(parameters)-1 else ''), yLabel=f'{parameters_Y[p]}', xyInvisible=([True, True]))
                    # ax2.set_ylim(70, 110)
                elif i == 0 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' if p==len(parameters)-1 else ''), yLabel='', xyInvisible=([True, False]))
                elif i == 3 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel='', xyInvisible=([False, True]))
                elif i == 2 and parameters[p] == 'swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' ), yLabel=parameters_Y[p], xyInvisible=([False, False]))
                # elif parameters[p]=='stanceOn_iqr_25_75_ref_FL':
                #     self.layoutOfPanel(ax2, xLabel='session', yLabel=parameters_Y[p], xyInvisible=([False, False]))
                    # ax2.set_ylim(75, 155)
                ax2.xaxis.set_major_locator(MultipleLocator(1))
                print(parameters[p],  pawId[i], 'all',swingLength_summary['table']['all'].summary())
                #pdb.set_trace()
                print(parameters[p], pawId[i], 'tomato', swingLength_summary['table']['tdTomato'].summary())
                #pdb.set_trace()
                print(parameters[p], pawId[i], 'opsin', swingLength_summary['table']['opsin'].summary())
                #pdb.set_trace()
                ax2.legend([f'{pawId[i]}  tdTomato {swingLength_summary["stars"]["tdTomato"]["day"]} {swingLength_summary["stars"]["tdTomato"]["trial"].replace("*", "#")}',f'{pawId[i]} GtACR2 {swingLength_summary["stars"]["opsin"]["day"]} {swingLength_summary["stars"]["opsin"]["trial"].replace("*", "#")}'], bbox_to_anchor=(0.1, (0.7 if p==0 else 0.7), 0.5, 0.5), loc='upper left', frameon=False,
                           fontsize=9)


        fname = 'fig_real-time-experiment_v%s' % figVersion
        # plt.savefig(fname + '.pdf')
        # plt.savefig(fname + '.svg')
        #groupAnalysisFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary/'
        #plt.savefig(groupAnalysisFigDir+fname + '.pdf')
        plt.savefig(fname + '.pdf')

        plt.savefig(fname + '.svg')
    ###############################################################################
    def fig_real_time_experimentB(self,figVersion, stridePar, strideTraj, triggerDict,opsValues,tomValues,data): #DictopsinSwingStanceDic,tdTomatoSwingStanceDic, trigger):
        #############################
        def determineLaserActivationDistribution(case, successRate, idxSwings, recTimes, pawPos, pawID, laserDict):
            #successRate[case]['normStepCycleBins'] = np.zeros(201)
            #successRate[case]['normTimeStepCycle'] =  np.linspace(0,2,201)
            normTimeStepCycle = successRate[case]['normTimeStepCycle']
            for j in range(len(idxSwings) - 2):
                idxStart = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][0]]))
                idxEnd = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j][1]]))
                idxStartNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][0]]))
                idxEndNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 1][1]]))
                idxStartNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][0]]))
                idxEndNNext = np.argmin(np.abs(pawPos[pawID][:, 0] - recTimes[idxSwings[j + 2][1]]))
                swingNStart = pawPos[pawID][idxStart, 0]
                swingNEnd = pawPos[pawID][idxEnd, 0]
                swingNPOStart = pawPos[pawID][idxStartNext, 0]
                swingNPOEnd = pawPos[pawID][idxEndNext, 0]
                swingNPTStart = pawPos[pawID][idxStartNNext, 0]
                swingNPTEnd = pawPos[pawID][idxEndNNext, 0]
                laserActivationTimeSwing = np.argwhere((laserDict['laserStartTime'] >= swingNStart) & (laserDict['laserStartTime'] < swingNEnd))
                laserActivationTimeStance = np.argwhere((laserDict['laserStartTime'] >= swingNEnd) & (laserDict['laserStartTime'] < swingNPOStart))
                if len(laserActivationTimeSwing) > 0:
                    successRate[case]['swingWithLaserActivation'] += 1
                else:
                    successRate[case]['swingNoLaserActivation'] += 1
                # if len(laserActivationTimeSwing)>1 :
                #     print(laserActivationTimeSwing)
                #     print('more than one activations during the swing')
                #     pdb.set_trace()
                if len(laserActivationTimeStance) > 0: successRate[case]['stanceWithLaserActivation'] += 1
                # if len(laserActivationTimeStance)>1:
                #     print(laserActivationTimeStance)
                #     print('more than one activations during the stance')
                #     pdb.set_trace()
                swingDuration = swingNEnd - swingNStart
                stanceDuration = swingNPOStart - swingNEnd
                swingNextDuration = swingNPOEnd - swingNPOStart
                stanceNextDuration = swingNPTStart - swingNPOEnd
                for k in laserActivationTimeSwing:
                    normStart = (laserDict['laserStartTime'][k] - swingNStart) / swingDuration
                    start_bin = np.digitize(normStart, normTimeStepCycle[:101]) - 1
                    if laserDict['laserEndTime'][k] < swingNEnd:
                        normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / swingDuration
                        assert normDuration < 1, print('The normalized duration in case one should be smaller than one!')
                        normEnd = normStart + normDuration
                        end_bin = np.digitize(normEnd, normTimeStepCycle) - 1  # unique_bins = np.arange(start_bin, end_bin + 1)  # normStepCycleBins[np.unique(unique_bins)] += 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] < swingNPOStart):
                        # normDuration = (laserDict['laserEndTime'][k]-laserDict['laserStartTime'][k])/swingDuration
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNEnd) / stanceDuration
                        assert normDurationStance < 1, print('The normalized duration in case two should be smaller than one!')
                        #if normDurationStance < 1 : print('It\'s happening again in case two!')
                        # start_bin = np.digitize(0, normTimeStepCycle[100:]) - 1 + 100
                        end_bin = np.digitize(1.+ normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case twoB should be smaller than one!' % normDurationSwing)
                        #if  normDurationStance < 1 : print('It\'s happening again in case twoB !')
                        end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 101), np.arange(0, end_bin)))
                        successRate[case]['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
                    elif (laserDict['laserEndTime'][k] >= swingNEnd) and (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case twoC should be smaller than one!' % normDurationSwing)
                        end_bin = np.digitize(normDurationStance, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                        successRate[case]['normStepCycleBins'] += 1  # in this condition, laser is active during a whole stance and swing
                        #successRate[case]['normStepCycleBins'][100:201] += 1  # in this condition, laser is active during a whole stance
                    successRate[case]['normStepCycleBins'][np.unique(unique_bins)] += 1
                for k in laserActivationTimeStance:
                    normStart = (laserDict['laserStartTime'][k] - swingNEnd) / stanceDuration
                    start_bin = np.digitize(normStart + 1., normTimeStepCycle) - 1
                    if laserDict['laserEndTime'][k] < swingNPOStart:
                        normDuration = (laserDict['laserEndTime'][k] - laserDict['laserStartTime'][k]) / stanceDuration
                        assert normDuration < 1, print('The normalized duration in case three should be smaller than one!')
                        normEnd = 1 + normStart + normDuration
                        end_bin = np.digitize(normEnd, normTimeStepCycle) - 1
                        unique_bins = np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] < swingNPOEnd):
                        normDurationSwing = (laserDict['laserEndTime'][k] - swingNPOStart) / swingNextDuration
                        assert normDurationSwing < 1, print('The normalized duration %s in case four should be smaller than one!' % normDurationSwing)
                        end_bin = np.digitize(normDurationSwing, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(0, end_bin+1))) # np.arange(start_bin, end_bin + 1)
                    elif (laserDict['laserEndTime'][k] >= swingNPOStart) and (laserDict['laserEndTime'][k] > swingNPOEnd):
                        normDurationStance = (laserDict['laserEndTime'][k] - swingNPOEnd) / stanceNextDuration
                        assert normDurationStance < 1, print('The normalized duration %s in case five should be smaller than one!' % normDurationStance)
                        end_bin = np.digitize(normDurationStance + 1, normTimeStepCycle) - 1
                        unique_bins = np.concatenate((np.arange(start_bin, 201), np.arange(100, end_bin + 1)))
                        successRate[case]['normStepCycleBins'][np.arange(0, 101)] += 1  # in this condition, laser is active during a whole swing
                    # unique_bins = np.arange(start_bin, end_bin + 1)
                    successRate[case]['normStepCycleBins'][np.unique(unique_bins)] += 1



        #############################

        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        # figure #################################
        fig_width = 20  # width in inches
        fig_height = 13  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        #rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1.2] )#),width_ratios=[1,0.9])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.34, hspace=0.35)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07)
        plt.figtext(0.03, 0.96, 'A', clip_on=False, color='black',  size=20)
        plt.figtext(0.53, 0.96, 'B', clip_on=False, color='black',  size=22)
        plt.figtext(0.03, 0.53, 'C', clip_on=False, color='black', size=22)
        plt.figtext(0.48, 0.53, 'D', clip_on=False, color='black',  size=20)
        plt.figtext(0.48, 0.28, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)

        gssub0a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.35, wspace=0.3, width_ratios=[4,4])
        gssub0a1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub0a[0], hspace=0.35, wspace=0.1,width_ratios=[7,4])

        #opsin_img_merge=mpimg.imread('Andry F3 ACR red fix by Ali Mosaic2 20x_Stitch-1-very_red_small_optic_fiber_tract.png')
        #opsin_img_red = mpimg.imread('Andry F3 ACR red fix by Ali Mosaic2 20x_Stitch-1-very_red_small.png')
        # opsin_img_merge=mpimg.imread('AVG_AVG_f65-d3-cell-pushed_00002_filtered.png')
        # opsin_img_red =mpimg.imread('AVG_AVG_f65-d3-cell-pushed_00002_filtered.png')
        ax0a=plt.subplot(gssub0a1[0])
        ax0a1 = plt.subplot(gssub0a1[1])
        # ax0a.imshow(opsin_img_merge)
        # ax0a1.imshow(opsin_img_red)
        left, bottom, width, height = ax0a.get_position().bounds
        left1, bottom1, width1, height1 = ax0a1.get_position().bounds
        ax0a.set_position([left-0.03,bottom-0.04,width+0.05,height+0.05])  	#[left, bottom, width, height]
        ax0a.axis('off')
        ax0a.text(0.9, 0.96, 'GtACR2', fontsize=11, color='#FF0000', ha='center', va='center', transform=ax0a.transAxes, alpha=0.9)
        ax0a1.set_position([left1-0.,bottom1-0.04,width1+0.05,height1+0.05])  	#[left, bottom, width, height]
        ax0a1.axis('off')
        ax0a1.text(0.86, 0.95, 'GtACR2', fontsize=10, color='#FF0000', ha='center', va='center', transform=ax0a1.transAxes, alpha=0.9)
        ax0a.text(0.1, 0.2, 'IV/V', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        ax0a.text(0.5, 0.5, 'S', fontsize=16, color='white', ha='center', va='center', transform=ax0a.transAxes)
        # idxSwings =np.array(swingStanceD['swingP'][i][1])
        # recTimes = np.array(swingStanceD['forFit'][i][2])
        pawPos_opsin=triggerDict[displayRec]['ops']['swingStance']['pawPos'] #opsinSwingStanceDic['pawPos']
        pawPos_tdTomato= triggerDict[displayRec]['tom']['swingStance']['pawPos'] #tdTomatoSwingStanceDic['pawPos']


        ############################Panel A and B : image of opsin spread or illustration image#########################
        ############################Panel C : opsin application effect graph#########################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.3, wspace=0.2, width_ratios=[2,1.5])


        ##############################################################################################
        ephysTimes = data[0][10]
        dt = np.mean(ephysTimes[1:] - ephysTimes[:-1])
        recStart = ephysTimes[0]
        recEnd = ephysTimes[-1] + dt

        for i in range(len(data)):
            trace = data[i][9][0]
            timeTrace = data[i][10]
            rescaledTrace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
            difference = np.diff(rescaledTrace)  # calculate difference
            expStart = np.arange(len(rescaledTrace))[
                np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
            expEnd = np.arange(len(rescaledTrace))[np.concatenate(
                (np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
            # print(expStart,expEnd,timeTrace[expStart],timeTrace[expEnd])
            stimStart = timeTrace[expStart]
            stimEnd = timeTrace[expEnd]
            print('Stimulation start and end : ', stimStart, stimEnd)
        gssub0c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[0], hspace=0.2, wspace=0.2)#height_ratios=[3, 2])
        ax0c = plt.subplot(gssub0c[0])
        allSpikes = []
        if len(stimStart) > 1:
            for n in range(len(stimStart)):
                ax0c.axvspan(stimStart[n]-5, stimEnd[n]-5, alpha=0.3, facecolor='C6', linewidth=None)
        else:
            ax0c.axvspan(stimStart-5, stimEnd-5, alpha=0.3, facecolor='C6', linewidth=None)
        for i in range(len(data)):
            # ax0c.plot(data[i][13],data[i][12]*0.8/maxSpeed+i-0.4,lw=0.2,c='C1')
            ax0c.vlines(data[i][7]-5,i-0.4, i+0.4,lw=0.05,color='0.5')
            allSpikes.extend(data[i][7])

        self.layoutOfPanel(ax0c,xLabel='time (s)',yLabel='trial',xyInvisible=[False,False])  # axL[n].append(ax)
        majorLocator_y = plt.MultipleLocator(2)
        ax0c.yaxis.set_major_locator(majorLocator_y)
        ax0c.set_xlim(-2, 3)

        ax0d= plt.subplot(gssub0c[1])
        binWidth = 0.1 # in sec
        ephysTimes = data[0][3]
        dt = np.mean(ephysTimes[1:] - ephysTimes[:-1])
        tbins =  np.linspace(0.,len(ephysTimes)*dt,int(len(ephysTimes)*dt/binWidth)+1)
        binnedspikes, _ = np.histogram(allSpikes, tbins)
        binnedNormalizedSpikes = binnedspikes/(binWidth*len(data))
        #spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
        #convert the convolved spike trains to units of spikes/sec
        #spikesconv *= 1. / binWidth

        if len(stimStart)>1:
            for n in range(len(stimStart)):
               ax0d.axvspan(stimStart[n]-5, stimEnd[n]-5, alpha=0.3, facecolor='C6', linewidth=None)
            else:
                try:
                    ax0d.axvspan(stimStart-5, stimEnd-5, alpha=0.3, facecolor='C6', linewidth=None)
                except:
                    pass
            #ax1.axvspan(stimStart, stimEnd, alpha=0.3, color='green')
        ax0d.step(tbins[1:]-5,binnedNormalizedSpikes, color='k')
        ax0d.xaxis.set_major_locator(MultipleLocator(1))
        ax0d.set_ylim(0,60)
        ax0d.set_xlim(-2, 3)
        self.layoutOfPanel(ax0d, xLabel='time (s)', yLabel='average firing rate (spk/s)')  # axL[n].append(ax)

        ###############################################
        gssub0d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub1[1], hspace=0.18, wspace=0.1,
                                                   height_ratios=[2, 2])
        ax0=plt.subplot(gssub0d[1])
        ax0b=plt.subplot(gssub0d[0])
        startx = 34#42
        xLength = 10#7

        # to extract swing and stance onsets
        swingStanceD = triggerDict[displayRec]['ops']['swingStance']
        pawID = 0
        idxSwings = swingStanceD['swingP'][pawID][1]
        #indecisiveSteps = swingStanceD['swingP'][i][3]
        recTimes = swingStanceD['forFit'][pawID][2]


        # end of

        #for i in [0]:
        t_emg_ops = triggerDict[displayRec]['ops']['time'] #trigger['time']  # time points for EMG data
        t_emg_tom = triggerDict[displayRec]['tom']['time']

        laserDict = getLaserActivationStartAndEnd(t_emg_ops, triggerDict[displayRec]['ops']['Trigger'])

        pos_interp_opsin = np.interp(t_emg_ops, pawPos_opsin[pawID][:, 0], pawPos_opsin[pawID][:, 1])
        pos_interp_tdTomato = np.interp(t_emg_tom, pawPos_tdTomato[pawID][:, 0], pawPos_tdTomato[pawID][:, 1])
        ax0.plot(t_emg_ops, pos_interp_opsin, c=col[pawID], lw=1.4, label=f'{pawId[pawID]} GtACR2', ls=(0, (1, 1)))
        ax0.axhline(y=340,ls='--',c='0.8',alpha=0.5)
        ax0.axhline(y=340+120,ls='--',c='0.8',alpha=0.5)
        ax0b.plot(t_emg_tom, pos_interp_tdTomato, c=col[pawID], lw=1.4, label=f'{pawId[pawID]} tdTomato', ls='-')
        ax0b.axhline(y=320,ls='--',c='0.8',alpha=0.5)
        ax0b.axhline(y=320+120,ls='--',c='0.8',alpha=0.5)
        #ax0e = ax0.twinx()
        #ax0f = ax0b.twinx()
        #ax0b.axis('off')
        #pdb.set_trace()
        ax0.fill_between(t_emg_ops, 200 + triggerDict[displayRec]['ops']['Trigger']*200, color='C6', alpha=0.3,edgecolor=None)
        ax0b.fill_between(t_emg_tom, 200 + triggerDict[displayRec]['tom']['Trigger']*200, color='C6', alpha=0.3,edgecolor=None)
        #ax0e.axis('off')
        #ax0f.axis('off')
        ax0.set_ylim(310,510)
        ax0b.set_ylim(290,490)

        self.layoutOfPanel(ax0, xLabel='time (s)', yLabel= 'FL x  (pixel)',
                           xyInvisible=[False, False], Leg=[1, 9])
        self.layoutOfPanel(ax0b, xLabel='time (s)', yLabel= 'FL x (pixel)', Leg=[1, 9])
        ax0.legend(frameon=False, bbox_to_anchor=(0.85,1.05), loc='center')
        ax0b.legend(frameon=False, bbox_to_anchor=(0.83,1.05), loc='center')
        ax0.set_xlim(startx, startx + xLength)
        ax0b.set_xlim(startx, startx + xLength)
        ax0b.xaxis.set_major_locator(MultipleLocator(1))
        #ax0b.yaxis.set_major_locator(MultipleLocator(50))

        # for j in range(len(idxSwings)):
        #     idxStart = np.argmin(np.abs(pawPos_opsin[pawID][:, 0] - recTimes[idxSwings[j][0]]))
        #     idxEnd = np.argmin(np.abs(pawPos_opsin[pawID][:, 0] - recTimes[idxSwings[j][1]]))
        #     ax0.plot(pawPos_opsin[pawID][idxStart, 0], pawPos_opsin[pawID][idxStart, 1], 'x', c=col[pawID], alpha=0.5, lw=0.5)
        #     ax0.plot(pawPos_opsin[pawID][idxEnd, 0], pawPos_opsin[pawID][idxEnd, 1], '+', c=col[pawID], alpha=0.5, lw=0.5)
        # for n in range(len(laserDict['laserStartIdx'])):
        #     ax0e.plot(laserDict['laserStartTime'][n], 0.5,'x')
        #     ax0e.plot(laserDict['laserEndTime'][n], 0.5, '+')


        # generate a plot to quantify efficiency of laser activation ###############################

        #normStepCycleBinsTom = np.zeros(201)
        #normTimeStepCycleTom = np.linspace(0,2,201)
        successRate = {}  # np.zeros(4)
        cases = {'tom':[0,1,2,3,4],'ops':[2,3,4]}
        #tomRecs = [0,1,2,3,4]
        #opsRecs = [2,3,4]
        pawID=0
        for key,values in cases.items():
            successRate[key] = {}
            successRate[key]['normStepCycleBins'] = np.zeros(201)
            successRate[key]['normTimeStepCycle'] =  np.linspace(0,2,201)
            successRate[key]['totalNumberOfSteps'] = 0 #len(idxSwings)
            successRate[key]['swingWithLaserActivation'] = 0
            successRate[key]['swingNoLaserActivation'] = 0
            successRate[key]['stanceWithLaserActivation'] = 0

            for r in values:
                t_emg = triggerDict[r][key]['time']  # trigger['time']  # time points for EMG data
                pawPos = triggerDict[r][key]['swingStance']['pawPos']  # tdTomatoSwingStanceDic['pawPos']
                laserDict = getLaserActivationStartAndEnd(t_emg, triggerDict[r][key]['Trigger'])
                swingStanceD = triggerDict[r][key]['swingStance']

                pawID = 0
                idxSwings = swingStanceD['swingP'][pawID][1]
                # indecisiveSteps = swingStanceD['swingP'][i][3]
                recTimes = swingStanceD['forFit'][pawID][2]
                successRate[key]['totalNumberOfSteps'] += len(idxSwings)
                determineLaserActivationDistribution(key, successRate, idxSwings, recTimes, pawPos, pawID, laserDict)

        #pdb.set_trace()
        # Draw shaded red vertical bars for light on periods
        # light_on_periods_ops = np.where(trigger['ops'] > 1)[0]
        # light_on_periods_tom = np.where(trigger['tom'] > 1)[0]
        #
        # for period in light_on_periods_ops:
        #     ax0.axvspan(t_emg[period], t_emg[period + 1], alpha=0.005, color='C6')
        #
        # for period in light_on_periods_tom:
        #     ax0b.axvspan(t_emg[period], t_emg[period + 1], alpha=0.005, color='C6')




        #####################################
        gssub0b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.25, width_ratios=[0.8, 1, 1])
        gssub1b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub0b[0], hspace=0.4, wspace=0.35, width_ratios=[6, 4])

        ax11a=plt.subplot(gssub1b[0])
        ax11b=plt.subplot(gssub1b[1])
        totalStepsTom = successRate['tom']['totalNumberOfSteps'] #(successRate['tdTomoato']['swingWithLaserActivation']+successRate['tdTomoato']['swingNoLaserActivation'])
        totalStepsOps = successRate['ops']['totalNumberOfSteps'] #(successRate['opsin']['swingWithLaserActivation'] + successRate['opsin']['swingNoLaserActivation'])
        ax11a.step(successRate['tom']['normTimeStepCycle'],successRate['tom']['normStepCycleBins']/totalStepsTom,c='C6',where='pre')
        ax11a.step(successRate['ops']['normTimeStepCycle'],successRate['ops']['normStepCycleBins'] / totalStepsOps, c='C6', where='pre', ls=(0, (1, 1)))
        ax11a.axvline(1,ls='--',c='0.2')

        self.layoutOfPanel(ax11a, xLabel='normalized time', yLabel= 'probability of laser ON')

        ###################
        x_pos = [1, 1.2, 2, 2.2]
        #ax11b.bar()
        #values = []
        #for case in cases:
        #    v1 = successRate[case]['swingWithLaserActivation']/(successRate[case]['swingWithLaserActivation']+successRate[case]['swingNoLaserActivation'])
        #    values.append(v1)
        #    v2 = successRate[case]['stanceWithLaserActivation']/(successRate[case]['swingWithLaserActivation']+successRate[case]['swingNoLaserActivation'])
        #    values.append(v2)
        opsValues = np.asarray(opsValues)
        tomValues = np.asarray(tomValues)
        sns.stripplot(data=[opsValues[:,0],opsValues[:,1],tomValues[:,0],tomValues[:,1]],jitter=True, size=6, alpha=0.3,ax=ax11b,color='0.1')
        sns.boxplot(data=[opsValues[:,0],opsValues[:,1],tomValues[:,0],tomValues[:,1]], ax=ax11b, width=0.3, showcaps=False,showfliers=False, boxprops={'facecolor':'None'},whiskerprops={'linewidth':1.5}, medianprops={'color':'black'}) #, positions=[0.8, 2.2])

        ccc = ['swing','stance']
        for  i in range(2):
            print('%s activation (25,50,75) : ' % ccc[i])
            print('  opsin  : ',end='')
            median = np.median(opsValues[:,i])
            q1 = np.percentile(opsValues[:,i], 25)
            q3 = np.percentile(opsValues[:,i], 75)
            print(q1,median,q3)
            print('  tdtomato  : ',end='')
            median = np.median(tomValues[:,i])
            q1 = np.percentile(tomValues[:,i], 25)
            q3 = np.percentile(tomValues[:,i], 75)
            print(q1,median,q3)
        pdb.set_trace()
        #sns.stripplot(x=np.repeat(x_pos[1], len(opsValues[:, 1])), y=opsValues[:, 1], jitter=True, size=6, alpha=0.3, ax=ax11b,color='0.8')
        #sns.boxplot(data=opsValues[:, 1], ax=ax11b, width=0.3, showcaps=False, boxprops={'facecolor': 'None'}, whiskerprops={'linewidth': 1.5}, medianprops={'color': 'black'})
        #ax11b.plot(x_pos[0]+(0.1*(np.random.rand(len(opsValues))-0.5)),opsValues[:,0], 'o',color='0.1')
        #ax11b.plot(x_pos[1]+(0.1*(np.random.rand(len(opsValues))-0.5)), opsValues[:, 1], 'o', color='0.8')
        #sns.stripplot(x=np.repeat(x_pos[2], len(tomValues[:, 0])), y=tomValues[:, 0], jitter=True, size=6, alpha=0.3, ax=ax11b, color='0.1')
        #sns.stripplot(x=np.repeat(x_pos[3], len(tomValues[:, 1])), y=tomValues[:, 1], jitter=True, size=6, alpha=0.3, ax=ax11b, color='0.8')
        #ax11b.plot(x_pos[2]+(0.1*(np.random.rand(len(tomValues))-0.5)),tomValues[:,0], 'o',color='0.1')
        #ax11b.plot(x_pos[3]+(0.1*(np.random.rand(len(tomValues))-0.5)), tomValues[:, 1], 'o', color='0.8')
        #, '0.8', '0.1', '0.9'])
        cases = ['GtACR2','dtTomato']
        # Set custom x-labels at appropriate positions
        plt.xticks([0.5, 2.5], cases, rotation=45, ha="right")
        self.layoutOfPanel(ax11b, yLabel='laser activations (%)')
        print(successRate)
        #pdb.set_trace()



        ############################Panel B : Swing Number#########################
        stridePar = stridePar.drop(stridePar[(stridePar['day'] > 9)].index).reset_index()
        #regroup data per trial, day, paw and mouse
        stridePar_Recordings=stridePar.groupby(['trial','day', 'paw', 'mouseId', 'treatment']).mean()
        stridePar_Recordings=stridePar_Recordings.reset_index()

        #get data for five trials only, for visualization
        stridePar_Recordings_trial=stridePar_Recordings[stridePar_Recordings['trial']<=5]
        stridePar_Recordings_trial=stridePar_Recordings_trial.groupby(['trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_trial = stridePar_Recordings_trial.reset_index()
        #average data per day for visualization
        stridePar_Recordings_day=stridePar_Recordings.groupby(['day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_day=stridePar_Recordings_day.reset_index()

        stridePar_Recordings_paw_day=stridePar_Recordings.groupby(['paw','day', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_day=stridePar_Recordings_paw_day.reset_index()
        stridePar_Recordings_paw_day.to_csv('opto_results_statistics.csv')

        stridePar_Recordings_paw_trial=stridePar_Recordings.groupby(['paw','trial', 'mouseId', 'treatment']).mean()
        stridePar_Recordings_paw_trial=stridePar_Recordings_paw_trial.reset_index()

        trials=stridePar_Recordings['trial'].unique()
        days = stridePar_Recordings['day'].unique()
        trialList=trials.astype(str).tolist()
        daysList = days.astype(str).tolist()


        nSal=stridePar_Recordings_day[stridePar_Recordings_day['treatment']=='tdTomato']['mouseId'].nunique()
        nMus = stridePar_Recordings_day[stridePar_Recordings_day['treatment'] == 'opsin']['mouseId'].nunique()

        FL_swingLen_tdTomatoId=stridePar_Recordings_paw_day[(stridePar_Recordings_paw_day['treatment']=='tdTomato') & (stridePar_Recordings_paw_day['paw']=='FL')]['swingLength']
        FL_swingLen_opsinId=stridePar_Recordings_paw_day[(stridePar_Recordings_paw_day['treatment'] == 'opsin')& (stridePar_Recordings_paw_day['paw']=='FL')]['swingLength']

        print('tomato', 'mean', np.mean(FL_swingLen_tdTomatoId)*0.025, 'SD', np.std(FL_swingLen_tdTomatoId)*0.025)
        print('opsin', 'mean', np.mean(FL_swingLen_opsinId)*0.025, 'SD', np.std(FL_swingLen_opsinId)*0.025)
        #pdb.set_trace()
        panelList=[2,4,5]
        parameters=['swingLength', 'stanceDuration', 'indecisiveFraction','stanceOn_iqr_25_75_ref_FL']
        parameters_Y=['swing length (cm)','stance duration (s)', 'fraction of miss steps', 'stance onset iqr']
        parameters=['swingLength', 'indecisiveFraction','stanceOn_iqr_25_75_ref_FL']
        parameters_Y=['swing length (cm)','fraction of miss steps', 'stance onset IQR']
        hr=np.repeat(3,len(parameters))
        hr[0]=6

        for p in range(len(parameters)):
            if parameters[p]!='swingLength':
                gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub0b[2], hspace=0.3, wspace=0.15)
                pawNb=2
            else:
                gssub0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub0b[1], hspace=0.3, wspace=0.15)
                pawNb=4
            for i in range(pawNb):
                if parameters[p] != 'swingLength':
                    ax2 = plt.subplot(gssub0[p-1,i])
                else:
                    ax2 = plt.subplot(gssub0[i])
                pawDf = stridePar_Recordings[(stridePar_Recordings['paw'] == pawId[i])]
                pawDf.loc[:, 'swingLength'] = pawDf['swingLength'] * 0.025
                swingLength_summary = groupAnalysis.perform_mixedlm_treatment_single_paw(pawDf, parameters[p])
                sns.lineplot(data=pawDf, x='day', y=parameters[p], hue=None,
                             hue_order=['tdTomato', 'opsin'], style='treatment', style_order=['tdTomato', 'opsin'],
                             errorbar=('se'), err_style='bars', dashes=[(1, 0), (1, 1)],
                             err_kws={'capsize': 3, 'linewidth': 1}, color=f'C{i}', ax=(ax2), marker='o')
                interactionTerm = f'day:treatment[T.tdTomato]'
                ax2.text(1.05, (0.4 if p==len(parameters)-1 else 0.57), f'{swingLength_summary["stars"]["all"]["treatment[T.tdTomato]"].replace("*", "°")}',
                         ha='center', va='center', transform=ax2.transAxes, style='italic', fontfamily='serif',
                         fontsize=15, color='k')
                if parameters[p] == 'swingLength':
                    if i==0 or i==1:
                        ax2.set_ylim(1.8, 3.2)
                    else:
                        ax2.set_ylim(0.8, 2.2)
                elif parameters[p] == 'stanceDuration':
                    ax2.set_ylim(0.4, 0.8)
                elif parameters[p] == 'indecisiveFraction':
                    ax2.set_ylim(0, 0.3)
                elif parameters[p] == 'stanceOn_iqr_25_75_ref_FL':
                    ax2.set_ylim(0, 0.4)
                if i == 0 and parameters[p]!='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel=f'{parameters_Y[p]}', xyInvisible=([(True if p==1 else False), False]))
                    # ax2.set_ylim(70, 110)
                elif i == 1 and parameters[p]!='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel='', xyInvisible=([( True if p==1 else False), True]))


                    # ax2.set_ylim(70, 110)
                elif i == 1 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' if p==len(parameters)-1 else ''), yLabel=f'{parameters_Y[p]}', xyInvisible=([True, True]))
                    # ax2.set_ylim(70, 110)
                elif i == 0 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' if p==len(parameters)-1 else ''), yLabel='', xyInvisible=([True, False]))
                elif i == 3 and parameters[p]=='swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session'), yLabel='', xyInvisible=([False, True]))
                elif i == 2 and parameters[p] == 'swingLength':
                    self.layoutOfPanel(ax2, xLabel=('session' ), yLabel=parameters_Y[p], xyInvisible=([False, False]))
                # elif parameters[p]=='stanceOn_iqr_25_75_ref_FL':
                #     self.layoutOfPanel(ax2, xLabel='session', yLabel=parameters_Y[p], xyInvisible=([False, False]))
                    # ax2.set_ylim(75, 155)
                ax2.xaxis.set_major_locator(MultipleLocator(1))
                print(parameters[p],  pawId[i], 'all',swingLength_summary['table']['all'].summary())
                #pdb.set_trace()
                print(parameters[p], pawId[i], 'tomato', swingLength_summary['table']['tdTomato'].summary())
                #pdb.set_trace()
                print(parameters[p], pawId[i], 'opsin', swingLength_summary['table']['opsin'].summary())
                #pdb.set_trace()
                ax2.legend([f'{pawId[i]}  tdTomato {swingLength_summary["stars"]["tdTomato"]["day"]} {swingLength_summary["stars"]["tdTomato"]["trial"].replace("*", "#")}',f'{pawId[i]} GtACR2 {swingLength_summary["stars"]["opsin"]["day"]} {swingLength_summary["stars"]["opsin"]["trial"].replace("*", "#")}'], bbox_to_anchor=(0.1, (0.7 if p==0 else 0.7), 0.5, 0.5), loc='upper left', frameon=False,
                           fontsize=9)


        fname = 'fig_real-time-experiment_v%s' % figVersion
        # plt.savefig(fname + '.pdf')
        # plt.savefig(fname + '.svg')
        #groupAnalysisFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary/'
        #plt.savefig(groupAnalysisFigDir+fname + '.pdf')
        plt.savefig(fname + '.pdf')
        # plt.savefig(groupAnalysisFigDir+fname + '.svg')

    ###############################################################################
    def fig_real_time_experimentOverview(self,figVersion, opsinMice, tomMice): #DictopsinSwingStanceDic,tdTomatoSwingStanceDic, trigger):
        cmap=cm.get_cmap('tab20')
        colors=['steelblue','darkorange','yellowgreen','salmon']
        pawId=['FL','FR','HL','HR']
        col=['C0','C1','C2','C3']
        opsValues = []
        tomValues = []
        # figure #################################
        fig_width = 16  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 15, 'axes.titlesize': 12, 'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(5, 4, wspace=[0.2,0.3,0.2]) #, height_ratios=[1,1,1.2] )#),width_ratios=[1,0.9])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.34, hspace=0.5)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07)
        #plt.figtext(0.03, 0.96, 'A', clip_on=False, color='black',  size=20)
        #plt.figtext(0.53, 0.96, 'B', clip_on=False, color='black',  size=22)
        #plt.figtext(0.03, 0.53, 'C', clip_on=False, color='black', size=22)
        #plt.figtext(0.48, 0.53, 'D', clip_on=False, color='black',  size=20)
        #plt.figtext(0.48, 0.28, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.53, 0.53, 'E', clip_on=False, color='black',  size=20)
        # plt.figtext(0.85, 0.53, 'F', clip_on=False, color='black',  size=20)

        #gssub0a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.35, wspace=0.3, width_ratios=[4,4])
        #gssub0a1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub0a[0], hspace=0.35, wspace=0.1,width_ratios=[7,4])


        ############################Panel C : opsin application effect graph#########################
        #gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.35, wspace=0.3, width_ratios=[4, 4])
        #gssub1b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gssub1[1], hspace=0.4, wspace=0.35, width_ratios=[9, 4])
        opsFig = 0
        for m in opsinMice:
            ax0 = plt.subplot(gs[opsFig])
            ax1 = plt.subplot(gs[opsFig+1])
            ax0.set_title('mouse : %s' % m)
            inc = 1./(len(opsinMice[m]['recordings'])+1)
            ai = 0.1
            for session in opsinMice[m]['recordings']:
                successRateSession = opsinMice[m][session]['scucessRateSession']
                totalSteps = successRateSession['totalNumberOfSteps']  # (successRate['tdTomoato']['swingWithLaserActivation']+successRate['tdTomoato']['swingNoLaserActivation'])
                ax0.step(successRateSession['normTimeStepCycle'], successRateSession['normStepCycleBins'] / totalSteps, c='C6', where='pre', alpha = ai,label=session)
                v1 = (0 if (successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation'])==0 else successRateSession['swingWithLaserActivation']/(successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation']))
                v2 = (0 if (successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation'])==0 else successRateSession['stanceWithLaserActivation']/(successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation']))
                opsValues.append([v1,v2])
                ax1.plot(1+(0.1*(np.random.rand()-0.5)),v1,'o',c='0.2',alpha=ai)
                ax1.plot(2 + (0.1 * (np.random.rand() - 0.5)), v2, 'o', c='0.8',alpha=ai)
                ai+=inc
            ax0.axvline(1, ls='--', c='0.2')
            self.layoutOfPanel(ax0, xLabel='normalized time', yLabel='probability of laser ON',Leg=[1,5])
            cases = ['swing','stance']
            # Set custom x-labels at appropriate positions
            ax1.set_xticks([1, 2], cases, rotation=45, ha="right")
            self.layoutOfPanel(ax1, yLabel='laser activations (%)')
            opsFig+=4

        tomFig = 0
        for m in tomMice:
            ax0 = plt.subplot(gs[tomFig+2])
            ax1 = plt.subplot(gs[tomFig+3])
            ax0.set_title('mouse : %s' % m)
            inc = 1./(len(tomMice[m]['recordings'])+1)
            ai = 1
            for session in tomMice[m]['recordings']:
                successRateSession = tomMice[m][session]['scucessRateSession']
                totalSteps = successRateSession['totalNumberOfSteps']  # (successRate['tdTomoato']['swingWithLaserActivation']+successRate['tdTomoato']['swingNoLaserActivation'])
                ax0.step(successRateSession['normTimeStepCycle'], successRateSession['normStepCycleBins'] / totalSteps, c='C6', alpha = ai, where='pre',label=session)
                v1 = (0 if (successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation'])==0 else successRateSession['swingWithLaserActivation']/(successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation']))
                v2 = (0 if (successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation'])==0 else successRateSession['stanceWithLaserActivation']/(successRateSession['swingWithLaserActivation']+successRateSession['swingNoLaserActivation']))
                tomValues.append([v1, v2])
                ax1.plot(1+(0.1*(np.random.rand()-0.5)),v1,'o',c='0.2',alpha=ai)
                ax1.plot(2 + (0.1 * (np.random.rand() - 0.5)), v2, 'o', c='0.9', alpha=ai)
                ai-=inc
            ax0.axvline(1, ls='--', c='0.2')
            self.layoutOfPanel(ax0, xLabel='normalized time', yLabel='probability of laser ON',Leg=[1,5])
            cases = ['swing','stance']
            # Set custom x-labels at appropriate positions
            ax1.set_ylim(0,1)
            ax1.set_xticks([1, 2], cases, rotation=45, ha="right")
            self.layoutOfPanel(ax1, yLabel='laser activations (%)')
            tomFig+=4

        #############################
        fname = 'fig_real-time-experiment_LaserActivationSummary_v%s' % figVersion
        # plt.savefig(fname + '.pdf')
        # plt.savefig(fname + '.svg')
        #groupAnalysisFigDir = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/simplexSummary/'
        #plt.savefig(groupAnalysisFigDir+fname + '.pdf')
        plt.savefig(fname + '.pdf')
        # plt.savefig(groupAnalysisFigDir+fname + '.svg')
        return opsValues,tomValues

    ##########################################################################################
    def swingAnalysisFig(self, figVersion,mouseList, allSwingStanceDict):
        def calculateDistanceBtwLineAndPoint(x1,y1,x2,y2,x0,y0):
            nenner   = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            zaehler  = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1
            dist = zaehler/nenner
            return dist

        ##############################################################################################################
        # calculate paw-rung distance
        def calculatePawRungDistances(forFit,rungMotion,pawPos,obstacle=False):
            pawRungDistances = []
            obsRungs = []
            #pdb.set_trace()
            for i in range(4):
                rungInd = []

                for n in forFit[i][4]:

                    #pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameters])
                    # (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5))

                    #if
                    if obstacle:
                        obsRungs.extend(rungMotion[n][2][rungMotion[n][6]>0])

                    rungLocs = rungMotion[n][3]
                    #xPaw = pawTracks[6][0] + pawTracks[0][n,(i*3+1)]
                    #yPaw = pawTracks[6][2] + pawTracks[0][n,(i*3+2)]
                    #size of array dont match!!!!!!!!!!!!!!!!
                    try:
                        xPaw =  pawPos[i][n,1] # pawTracks[5][i][n,1]
                        yPaw =  pawPos[i][n,2]  #pawTracks[5][i][n,2]
                    except:
                        pass

                    distances = calculateDistanceBtwLineAndPoint(rungLocs[:,0],rungLocs[:,1],rungLocs[:,2],rungLocs[:,3],xPaw,yPaw)
                    sortedArguments  = np.argsort(np.abs(distances))

                    #closestRungIdx = np.argmin(np.abs(distances))
                    closestRungNumber = rungMotion[n][2][sortedArguments[0]]
                    closestDist = distances[sortedArguments[0]]

                    secondClosestRungNumber = rungMotion[n][2][sortedArguments[1]]
                    secondClosestDist = distances[sortedArguments[1]]
                    #pdb.set_trace()
                    if obstacle:
                        closestRungObsIdentity=int(rungMotion[n][6][sortedArguments[0]])
                        # if closestRungObsIdentity >0:
                        #     # print(closestRungObsIdentity)
                        secondClosestRungObsIdentity=int(rungMotion[n][6][sortedArguments[1]])
                        rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw,closestRungObsIdentity,secondClosestRungObsIdentity])
                    else:
                        rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw])
                    #pdb.set_trace()

                rungInd = np.asarray(rungInd)
                pawRungDistances.append([i,rungInd])
            return (rungInd, pawRungDistances) #pdb.set_trace()
        ##############################################

        pawIDtoShow = 0
        scalingFactor = 0.021
        figsInRow = 5
        figRows = len(mouseList)

        pawBeAf = 10
        wheelBeAf = 40
        Ngradient = 20
        alphaRange = np.linspace(0.1, 0.5, Ngradient, endpoint=True)
        # figure #################################
        fig_width = 30  # width in inches
        fig_height = figRows*5 + 3 #12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 10, 'axes.titlesize': 10, 'font.size': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        #rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(figRows, figsInRow) #,  width_ratios=[1,1.3])
                               #height_ratios=[2, 1])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.2)
        plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.06)
        # plt.figtext(0.01, 0.96, 'A', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.96, 'B', clip_on=False, color='black', size=22)
        # plt.figtext(0.01, 0.46, 'C', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.56, 'D', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.26, 'E', clip_on=False, color='black',  size=22)
        # plt.figtext(0.71, 0.26, 'F', clip_on=False, color='black', size=22)

        #gssub0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.1, height_ratios=[2,1,0.5])
        nFig = 0
        figFull = False
        for m in range(len(mouseList)): # loop over mice

            # panel ##################
            # for f in range(len(foldersRecordings)):
            # allSwingStanceDict[m][f] = {} #
            mouse = mouseList[m]
            allSessions = np.array([key for key in allSwingStanceDict[mouse].keys()])
            indices = np.linspace(0, len(allSessions) - 1, 5, dtype=int)
            #pdb.set_trace()
            showSessions = allSessions[indices]
            showSessions[-1] = allSessions[-1] # make sure that the last shown session is the last overall session
            #for f in allSwingStanceDict[mouse].keys():  # loop over sessions
            for f in showSessions:  # loop over sessions
                # one 3 x 3 panel per session
                gssub0 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[nFig], hspace=0.1, wspace=0.1)
                allRecordings = np.array([key for key in allSwingStanceDict[mouse][f].keys()])
                indicesR = np.linspace(0, len(allRecordings) - 1, 3, dtype=int)
                showRecs = allRecordings[indicesR]
                showRecs[-1] = allRecordings[-1]  # make sure that the last shown recording is the last overall recording
                #for r in allSwingStanceDict[mouse][f].keys(): # loop over recordings
                nPanel = 0
                for r in showRecs:
                    #ax0 = plt.subplot(gs[nFig])
                    #gssub0 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[nFig], hspace=0.1, wspace=0.1)
                    print(mouse,', session: ', f, ', recording: ',r,', figure number: ',nFig)
                    swingStanceDict = allSwingStanceDict[mouse][f][r]['swingStanceDict']
                    cPawPos = allSwingStanceDict[mouse][f][r]['cPawPos']
                    pawSpeed = allSwingStanceDict[mouse][f][r]['pawSpeed']
                    sTimes =  allSwingStanceDict[mouse][f][r]['sTimes']
                    linearSpeed = allSwingStanceDict[mouse][f][r]['linearSpeed']
                    #
                    #ax0.plot(cPawPos[pawIDtoShow][:, 0], cPawPos[pawIDtoShow][:, 1] - 500, c='C0')
                    #ax0.plot(pawSpeed[pawIDtoShow][:, 0], pawSpeed[pawIDtoShow][:, 2] * scalingFactor, c='C1')
                    #ax0.plot(sTimes, -linearSpeed, c='darkorchid',lw=2)
                    idxSwings = np.asarray(swingStanceDict['swingP'][pawIDtoShow][1])
                    indecisiveSteps = swingStanceDict['swingP'][pawIDtoShow][3]

                    #pdb.set_trace()
                    stepCharacter = swingStanceDict['swingP'][pawIDtoShow][3]
                    forFit = swingStanceDict['forFit']
                    #pawPos =  swingStanceDict['pawPos']
                    rungMotion = allSwingStanceDict[mouse][f][r]['rungMotion']
                    (rungInd, pawRungDistances) = calculatePawRungDistances(forFit,rungMotion,cPawPos) # calculatePawRungDistances(forFit,rungMotion,pawPos,obstacle=False):
                    recTimes = swingStanceDict['forFit'][pawIDtoShow][2]
                    idxSwingsMotorization = np.arange(len(idxSwings))[(recTimes[idxSwings[:,0]]>10.)&(recTimes[idxSwings[:,0]]<52.)]
                    randomIdxSwings = np.sort(np.random.choice(idxSwingsMotorization, size=3, replace=False))

                    #pdb.set_trace()
                    #for n in range(len(idxSwings)):
                    for n in range(len(randomIdxSwings)): # selection of random swings
                        indIdxSwing = idxSwings[randomIdxSwings[n]]
                        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub0[nPanel], hspace=0.05)
                        #ax0 = plt.subplot(gssub0[nPanel])
                        ax0 = plt.subplot(gssub1[0])
                        ax1 = plt.subplot(gssub1[1])
                        if nPanel==0 :
                            ax0.set_title('%s sess. %s rec. %s ' %(mouse,f,r),fontsize=8)
                        elif (not (nPanel%3)):
                            ax0.set_title('rec. %s ' % r, fontsize=8)
                        idxStart = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[indIdxSwing[0]]))
                        idxEnd = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[indIdxSwing[1]]))
                        idxWheelStart = np.argmin(np.abs(sTimes - recTimes[indIdxSwing[0]]))
                        idxWheelEnd = np.argmin(np.abs(sTimes - recTimes[indIdxSwing[1]]))
                        swingMask = np.arange(idxStart,idxEnd+1)
                        startSpeedPaw = pawSpeed[pawIDtoShow][:,2][swingMask[0]]
                        startSpeedWheel = -linearSpeed[idxWheelStart]
                        startPosPaw = cPawPos[pawIDtoShow][:,1][swingMask[0]]
                        startTimeSwing = cPawPos[pawIDtoShow][:,0][swingMask[0]]
                        rungDistMask = (pawRungDistances[pawIDtoShow][1][:, 0] >= swingMask[0]-pawBeAf) & (pawRungDistances[pawIDtoShow][1][:, 0] < swingMask[-1]+pawBeAf)
                        closestDist = pawRungDistances[pawIDtoShow][1][rungDistMask][:, 1]
                        secondCloestDist = pawRungDistances[pawIDtoShow][1][rungDistMask][:, 4]
                        ax0.axhline(y=0,ls='--',c='0.5',alpha=0.5)
                        ax1.axhline(y=0, ls='--', c='0.5', alpha=0.5)
                        timeOfRungDistPlot = cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                        #pdb.set_trace()
                        jumps = np.arange(len(closestDist))[np.concatenate((([False]),np.abs(np.diff(closestDist))>30))]
                        #pdb.set_trace()
                        if (len(timeOfRungDistPlot) != len(closestDist)):
                            shortestLength = np.min((len(timeOfRungDistPlot), len(closestDist)))
                        else:
                            shortestLength = len(timeOfRungDistPlot)
                        if len(jumps)>0:
                            ax0.plot(timeOfRungDistPlot[:jumps[0]],closestDist[:jumps[0]],c='C1')
                            for k in range(len(jumps)-1):
                                ax0.plot(timeOfRungDistPlot[jumps[k]:jumps[k+1]],closestDist[jumps[k]:jumps[k+1]],c='C1')
                            ax0.plot(timeOfRungDistPlot[jumps[-1]:shortestLength], closestDist[jumps[-1]:shortestLength], c='C1')
                        else:
                            ax0.plot(timeOfRungDistPlot[:shortestLength],closestDist[:shortestLength],c='C1')
                        #ax0.plot(cPawPos[pawIDtoShow][:, 0][swingMask[0] - pawBeAf:swingMask[-1] + pawBeAf] - startTimeSwing, secondCloestDist)
                        ax1.plot(sTimes[idxWheelStart-wheelBeAf:idxWheelEnd+wheelBeAf] - startTimeSwing, -linearSpeed[idxWheelStart-wheelBeAf:idxWheelEnd+wheelBeAf]-startSpeedWheel, c='darkorchid', lw=2)
                        #ax0.axvspan(cPawPos[pawIDtoShow][:,0][swingMask[0]-10], cPawPos[pawIDtoShow][:,0][swingMask[0]], facecolor='gray', alpha=0.3, edgecolor=None)  # First 10 points
                        #ax0.axvspan(cPawPos[pawIDtoShow][:,0][swingMask[-1]], cPawPos[pawIDtoShow][:,0][swingMask[-1]+10], facecolor='gray', alpha=0.3, edgecolor=None)  # Last 10 points
                        ax0.plot(cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing,cPawPos[pawIDtoShow][:,1][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]-startPosPaw,c='k',alpha=0.4)
                        ax1.plot(pawSpeed[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing,(pawSpeed[pawIDtoShow][:,2][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf])*scalingFactor-startSpeedWheel,c='C0',alpha=0.4)
                        #
                        ax0.plot(cPawPos[pawIDtoShow][:,0][swingMask]- startTimeSwing,cPawPos[pawIDtoShow][:,1][swingMask]-startPosPaw,c='k')
                        ax1.plot(pawSpeed[pawIDtoShow][:,0][swingMask]- startTimeSwing,(pawSpeed[pawIDtoShow][:,2][swingMask])*scalingFactor-startSpeedWheel,c='C0')
                        #ax0.set_xlim(20,25)
                        coloredRange = np.linspace(cPawPos[pawIDtoShow][idxStart, 0]- startTimeSwing, cPawPos[pawIDtoShow][idxEnd+1, 0]- startTimeSwing, Ngradient, endpoint=True)
                        #if indecisiveSteps[n][3]:  # indecisive Step
                        # ax4.plot(cPawPos[pawIDtoShow][idxEnd, 0], cPawPos[pawIDtoShow][idxEnd, 1]*scalingFactor, '1', alpha=0.5, lw=0.5)
                        if indecisiveSteps[randomIdxSwings[n]][3]:
                            closeIndicies = indecisiveSteps[randomIdxSwings[n]][4]
                            firstExpectedStancePeriod = np.where((closeIndicies[:,1]-closeIndicies[:,0])>=2)[0]
                            if len(firstExpectedStancePeriod)>0:
                                expectedStanceIdx = closeIndicies[firstExpectedStancePeriod[0],0]
                            else:
                                expectedStanceIdx = closeIndicies[0,0]
                            ax0.axvline(x=cPawPos[pawIDtoShow][:, 0][swingMask][expectedStanceIdx] - startTimeSwing, ls=':', c='C8')
                            ax1.axvline(x=cPawPos[pawIDtoShow][:, 0][swingMask][expectedStanceIdx] - startTimeSwing, ls=':', c='C8')

                            #pdb.set_trace()
                        for i in range(Ngradient - 1):
                            #if indecisiveSteps[randomIdxSwings[n]][3]
                            #pdb.set_trace()
                            ax0.axvspan(coloredRange[i], coloredRange[i + 1], alpha=alphaRange[i], lw=0, facecolor=('crimson' if indecisiveSteps[randomIdxSwings[n]][3] else 'C0'))
                            ax1.axvspan(coloredRange[i], coloredRange[i + 1], alpha=alphaRange[i], lw=0, facecolor=('crimson' if indecisiveSteps[randomIdxSwings[n]][3] else 'C0'))
                        #ax0.set_xlim(-0.1,0.6)
                        # decide on axis limits
                        posTime = cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                        speedTime = pawSpeed[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                        startPlotTime = (posTime[0] if posTime[0]>speedTime[0] else speedTime[0])
                        endPlotTime = (posTime[-1] if posTime[-1]<speedTime[-1] else speedTime[-1])
                        ax0.set_xlim(startPlotTime,endPlotTime)
                        ax1.set_xlim(startPlotTime, endPlotTime)
                        # panel layouts
                        self.layoutOfPanel(ax0,xLabel='time (s)', yLabel=('position \n(pixel)' if not (nPanel%3) else None), xyInvisible=[True,(False if not (nPanel%3) else True)])
                        self.layoutOfPanel(ax1, xLabel='time (s)', yLabel=('speed \n(cm/s)' if not (nPanel % 3) else None),xyInvisible=[(False if (nPanel > 5) else True), (False if not (nPanel % 3) else True)])
                        nPanel+=1
                nFig +=1
                #if nFig == 10:
                #        figFull = True
                #        break
                #if figFull:
                #    break # show only the first session for the moment
            #if figFull:
            #    break # show only the first session for the moment
        #pdb.set_trace()


        ########################
        fname = 'fig_swing-phase-analysis_v%s' % figVersion
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')


    ##########################################################################################
    def missStepAnalysisFig(self, figVersion,mouseList, allSwingStanceDict,PSTHSummaryAllAnimals,cellType='allMLI'):
        def performLinearRegressionAndShowStats(data):
            x = data[:,0]
            y = data[:,1]
            # Add constant to predictor (intercept term)
            x = sm.add_constant(x)  # Adds a column of 1s for the intercept

            # Fit the regression model
            model = sm.OLS(y, x).fit()  # Ordinary Least Squares

            # Display the summary of the model
            print(model.summary())

            # Extract key values
            slope = model.params[1]
            intercept = model.params[0]
            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            p_value_slope = model.pvalues[1]
            f_stat = model.fvalue
            f_pvalue = model.f_pvalue
            stderr_slope = model.bse[1]
            conf_int = model.conf_int(alpha=0.05)  # 95% Confidence Interval

            # Residual Analysis
            residuals = model.resid
            rmse = np.sqrt(np.mean(residuals ** 2))  # Root Mean Squared Error

            # Print key results
            print("\nKey Results:")
            print(f"Slope: {slope:.3f}")
            print(f"Intercept: {intercept:.3f}")
            print(f"R-squared: {r_squared:.3f}")
            print(f"Adjusted R-squared: {adj_r_squared:.3f}")
            print(f"Slope p-value: {p_value_slope:.3f}")
            print(f"F-statistic: {f_stat:.3f}, F-stat p-value: {f_pvalue:.3f}")
            print(f"Slope Standard Error: {stderr_slope:.3f}")
            print(f"95% Confidence Interval for Slope: {conf_int[1]}")

        def performLinearRegressionPlotStats(data,ax):
            stancePairs = np.asarray(data)
            stancePairsSorted = stancePairs[stancePairs[:, 0].argsort()]
            #pdb.set_trace()
            slope, intercept = np.polyfit(stancePairsSorted[:, 0], stancePairsSorted[:, 1], 1)  # Linear regression
            y_pred = slope * stancePairsSorted[:, 0] + intercept
            # Calculate residuals and standard error
            residuals = stancePairsSorted[:, 1] - y_pred
            s_err = np.sqrt(np.sum(residuals ** 2) / (len(stancePairsSorted[:, 1]) - 2))

            # Calculate confidence intervals
            n = len(stancePairsSorted[:, 0])
            t_value = t.ppf(0.975, df=n - 2)  # 95% CI, two-tailed t-test
            mean_x = np.mean(stancePairsSorted[:, 0])
            ci = t_value * s_err * np.sqrt(1 / n + (stancePairsSorted[:, 0] - mean_x) ** 2 / np.sum((stancePairsSorted[:, 0] - mean_x) ** 2))

            # Calculate upper and lower bounds
            ci_upper = y_pred + ci
            ci_lower = y_pred - ci
            ax.plot(stancePairsSorted[:, 0], y_pred, color='C8', label=f'Fit: y = {intercept:.2f} + {slope:.2f}x')
            ax.fill_between(stancePairsSorted[:, 0], ci_lower, ci_upper, color='C8', alpha=0.2, label='95% CI')
            print('Stats for stance values')
            performLinearRegressionAndShowStats(stancePairsSorted)

        ######################################################################
        def calculateDistanceBtwLineAndPoint(x1,y1,x2,y2,x0,y0):
            nenner   = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            zaehler  = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1
            dist = zaehler/nenner
            return dist

        #######################################################################
        # calculate paw-rung distance
        def calculatePawRungDistances(forFit,rungMotion,pawPos,obstacle=False):
            pawRungDistances = []
            obsRungs = []
            #pdb.set_trace()
            for i in range(4):
                rungInd = []

                for n in forFit[i][4]:

                    #pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameters])
                    # (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5))

                    #if
                    if obstacle:
                        obsRungs.extend(rungMotion[n][2][rungMotion[n][6]>0])

                    rungLocs = rungMotion[n][3]
                    #xPaw = pawTracks[6][0] + pawTracks[0][n,(i*3+1)]
                    #yPaw = pawTracks[6][2] + pawTracks[0][n,(i*3+2)]
                    #size of array dont match!!!!!!!!!!!!!!!!
                    try:
                        xPaw =  pawPos[i][n,1] # pawTracks[5][i][n,1]
                        yPaw =  pawPos[i][n,2]  #pawTracks[5][i][n,2]
                    except:
                        pass

                    distances = calculateDistanceBtwLineAndPoint(rungLocs[:,0],rungLocs[:,1],rungLocs[:,2],rungLocs[:,3],xPaw,yPaw)
                    sortedArguments  = np.argsort(np.abs(distances))

                    #closestRungIdx = np.argmin(np.abs(distances))
                    closestRungNumber = rungMotion[n][2][sortedArguments[0]]
                    closestDist = distances[sortedArguments[0]]

                    secondClosestRungNumber = rungMotion[n][2][sortedArguments[1]]
                    secondClosestDist = distances[sortedArguments[1]]
                    #pdb.set_trace()
                    if obstacle:
                        closestRungObsIdentity=int(rungMotion[n][6][sortedArguments[0]])
                        # if closestRungObsIdentity >0:
                        #     # print(closestRungObsIdentity)
                        secondClosestRungObsIdentity=int(rungMotion[n][6][sortedArguments[1]])
                        rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw,closestRungObsIdentity,secondClosestRungObsIdentity])
                    else:
                        rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw])
                    #pdb.set_trace()

                rungInd = np.asarray(rungInd)
                pawRungDistances.append([i,rungInd])
            return (rungInd, pawRungDistances) #pdb.set_trace()
        ##############################################

        np.random.seed(43)
        pawIDtoShow = 0
        scalingFactor = 0.021
        figsInRow = 5
        figRows = len(mouseList)

        pawBeAf = 100
        wheelBeAf = 400 # 40
        Ngradient = 20
        alphaRange = np.linspace(0.1, 0.5, Ngradient, endpoint=True)
        # PSTH part ################################
        pawList = ['FL', 'FR', 'HL', 'HR']
        pawIdxToshow = 0
        col = ['C0', 'C1', 'C2', 'C3']
        # tbinsRes = np.linspace(0,2,int((0+2)/dtRes)+1,endpoint=True)

        nMice = len(PSTHSummaryAllAnimals[cellType])
        if cellType == 'allMLI':
            nRowsCols = 8
        elif cellType == 'allPC':
            nRowsCols = 6

        # figure #################################
        fig_width = 18  # width in inches
        fig_height = 20 #12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 10, 'axes.titlesize': 10, 'font.size': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        #rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1) #,  width_ratios=[1,1.3])
                               #height_ratios=[2, 1])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.2, hspace=0.15)
        plt.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.07)
        plt.figtext(0.01, 0.98, 'A', clip_on=False, color='black',  size=22)
        plt.figtext(0.5, 0.98, 'B', clip_on=False, color='black', size=22)
        plt.figtext(0.01, 0.495, 'C', clip_on=False, color='black',  size=22)
        plt.figtext(0.3, 0.495, 'D', clip_on=False, color='black',  size=22)
        # plt.figtext(0.46, 0.26, 'E', clip_on=False, color='black',  size=22)
        # plt.figtext(0.71, 0.26, 'F', clip_on=False, color='black', size=22)

        #gssub0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.1, height_ratios=[2,1,0.5])
        nFig = 0
        #for m in range(len(mouseList)): # loop over mice

        # panel ##################
        # for f in range(len(foldersRecordings)):
        # allSwingStanceDict[m][f] = {} #
        mouse = mouseList[4]
        allSessions = np.array([key for key in allSwingStanceDict[mouse].keys()])
        indices = np.linspace(0, len(allSessions) - 1, 4, dtype=int)
        showSessions = allSessions[indices]
        showSessions[-1] = allSessions[-1] # make sure that the last shown session is the last overall session
        #for f in allSwingStanceDict[mouse].keys():  # loop over sessions
        print('Sessions shown : ', showSessions)
        showSessions = showSessions[:-1]
        #pdb.set_trace()
        gssub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.08, wspace=0.16)
        gssub00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gssub0[0], hspace=0.2, wspace=0.08)
        nPanel = 0
        #[  3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
        #  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
        #  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56
        #  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
        #  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
        #  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110
        # 111 112 113 114 115 116 117 118 119 120]

        # [ 11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28
        #29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46
        #47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64
        #65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82
        #83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100
        #101 102 103]

        #[  4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21
        #22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39
        #40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57
        #58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75
        #76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93
        #94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
        #112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129
        #130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147
        #148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165
        #166 167 168 169]

        #[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
        #19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
        #37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
        #55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
        #73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
        #91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108
        #109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126
        #127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144
        #145 146 147 148 149 150 151 152 153 154]
        #randomStepIdx = [[58,14, 42, 82], [8,36,51,105], [78,60,51,118], [48,14,30,141]]
        #randomStepIdx = [[58, 14, 82], [8, 36, 105], [78, 60, 118],]# [48, 14, 30, 141]] = currently in manuscript March-24-2026
        randomStepIdx = [[58, 70, 88], [8, 36, 105], [78, 60, 118], ]  # [48, 14, 30, 141]]
        for f in showSessions:  # loop over sessions
            # one 3 x 3 panel per session

            allRecordings = np.array([key for key in allSwingStanceDict[mouse][f].keys()])
            indicesR = np.linspace(0, len(allRecordings) - 1, 3, dtype=int)
            showRec = np.random.choice(len(allRecordings), size=1)[0]
            #showRecs = allRecordings[indicesR]
            #showRecs[-1] = allRecordings[-1]  # make sure that the last shown recording is the last overall recording
            #for r in allSwingStanceDict[mouse][f].keys(): # loop over recordings
            #nPanel = 0
            #for r in showRecs:
            #ax0 = plt.subplot(gs[nFig])
            #gssub0 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[nFig], hspace=0.1, wspace=0.1)
            print(mouse,', session: ', f, ', recording: ',showRec,', figure number: ',nFig)
            swingStanceDict = allSwingStanceDict[mouse][f][showRec]['swingStanceDict']
            cPawPos = allSwingStanceDict[mouse][f][showRec]['cPawPos']
            pawSpeed = allSwingStanceDict[mouse][f][showRec]['pawSpeed']
            sTimes =  allSwingStanceDict[mouse][f][showRec]['sTimes']
            linearSpeed = allSwingStanceDict[mouse][f][showRec]['linearSpeed']
            #
            #ax0.plot(cPawPos[pawIDtoShow][:, 0], cPawPos[pawIDtoShow][:, 1] - 500, c='C0')
            #ax0.plot(pawSpeed[pawIDtoShow][:, 0], pawSpeed[pawIDtoShow][:, 2] * scalingFactor, c='C1')
            #ax0.plot(sTimes, -linearSpeed, c='darkorchid',lw=2)
            idxSwings = np.asarray(swingStanceDict['swingP'][pawIDtoShow][1])
            indecisiveSteps = swingStanceDict['swingP'][pawIDtoShow][3]

            #pdb.set_trace()
            stepCharacter = swingStanceDict['swingP'][pawIDtoShow][3]
            forFit = swingStanceDict['forFit']
            #pawPos =  swingStanceDict['pawPos']
            rungMotion = allSwingStanceDict[mouse][f][showRec]['rungMotion']
            (rungInd, pawRungDistances) = calculatePawRungDistances(forFit,rungMotion,cPawPos) # calculatePawRungDistances(forFit,rungMotion,pawPos,obstacle=False):
            recTimes = swingStanceDict['forFit'][pawIDtoShow][2]
            idxSwingsMotorization = np.arange(len(idxSwings))[(recTimes[idxSwings[:,0]]>10.)&(recTimes[idxSwings[:,0]]<52.)]
            #randomIdxSwings = np.sort(np.random.choice(idxSwingsMotorization, size=4, replace=False))
            print(idxSwingsMotorization)
            #print(randomIdxSwings)
            for l in range(len(idxSwingsMotorization)):
                print(idxSwingsMotorization[l], ':', indecisiveSteps[idxSwingsMotorization[l]][3], ' ',end='')
            print()
            #pdb.set_trace()
            #for n in range(len(idxSwings)):

            for n in range(len(randomStepIdx[nFig])): # selection of random swings
                indIdxSwing = idxSwings[randomStepIdx[nFig][n]]

                gssub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub00[nPanel], hspace=0.05)
                #ax0 = plt.subplot(gssub0[nPanel])
                ax0 = plt.subplot(gssub1[0])
                ax1 = plt.subplot(gssub1[1])
                #if nPanel==0 :
                #    ax0.set_title('%s sess. %s rec. %s ' %(mouse,f,showRec),fontsize=8)
                #elif (not (nPanel%3)):
                #    ax0.set_title('rec. %s ' % showRec, fontsize=8)
                idxStart = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[indIdxSwing[0]]))
                idxEnd = np.argmin(np.abs(cPawPos[pawIDtoShow][:, 0] - recTimes[indIdxSwing[1]]))
                idxWheelStart = np.argmin(np.abs(sTimes - recTimes[indIdxSwing[0]]))
                idxWheelEnd = np.argmin(np.abs(sTimes - recTimes[indIdxSwing[1]]))
                swingMask = np.arange(idxStart,idxEnd+1)
                startSpeedPaw = pawSpeed[pawIDtoShow][:,2][swingMask[0]]
                startSpeedWheel = -linearSpeed[idxWheelStart]
                startPosPaw = cPawPos[pawIDtoShow][:,1][swingMask[0]]
                startTimeSwing = cPawPos[pawIDtoShow][:,0][swingMask[0]]
                rungDistMask = (pawRungDistances[pawIDtoShow][1][:, 0] >= swingMask[0]-pawBeAf) & (pawRungDistances[pawIDtoShow][1][:, 0] < swingMask[-1]+pawBeAf)
                closestDist = pawRungDistances[pawIDtoShow][1][rungDistMask][:, 1]
                secondCloestDist = pawRungDistances[pawIDtoShow][1][rungDistMask][:, 4]
                ax0.axhline(y=0,ls='--',c='0.5',alpha=0.5)
                ax1.axhline(y=0, ls='--', c='0.5', alpha=0.5)
                timeOfRungDistPlot = cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                #pdb.set_trace()
                jumps = np.arange(len(closestDist))[np.concatenate((([False]),np.abs(np.diff(closestDist))>30))]
                #pdb.set_trace()
                if (len(timeOfRungDistPlot) != len(closestDist)):
                    shortestLength = np.min((len(timeOfRungDistPlot), len(closestDist)))
                else:
                    shortestLength = len(timeOfRungDistPlot)
                if len(jumps)>0:
                    ax0.plot(timeOfRungDistPlot[:jumps[0]],(closestDist[:jumps[0]])*scalingFactor,c='C5')
                    for k in range(len(jumps)-1):
                        ax0.plot(timeOfRungDistPlot[jumps[k]:jumps[k+1]],(closestDist[jumps[k]:jumps[k+1]])*scalingFactor,c='C5')
                    ax0.plot(timeOfRungDistPlot[jumps[-1]:shortestLength], (closestDist[jumps[-1]:shortestLength])*scalingFactor, c='C5')
                else:
                    ax0.plot(timeOfRungDistPlot[:shortestLength],closestDist[:shortestLength],c='C5')
                #ax0.plot(cPawPos[pawIDtoShow][:, 0][swingMask[0] - pawBeAf:swingMask[-1] + pawBeAf] - startTimeSwing, secondCloestDist)
                ax1.plot(sTimes[idxWheelStart-wheelBeAf:idxWheelEnd+wheelBeAf] - startTimeSwing, -linearSpeed[idxWheelStart-wheelBeAf:idxWheelEnd+wheelBeAf], c='darkorchid', lw=2)
                #ax0.axvspan(cPawPos[pawIDtoShow][:,0][swingMask[0]-10], cPawPos[pawIDtoShow][:,0][swingMask[0]], facecolor='gray', alpha=0.3, edgecolor=None)  # First 10 points
                #ax0.axvspan(cPawPos[pawIDtoShow][:,0][swingMask[-1]], cPawPos[pawIDtoShow][:,0][swingMask[-1]+10], facecolor='gray', alpha=0.3, edgecolor=None)  # Last 10 points
                ax0.plot(cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing,(cPawPos[pawIDtoShow][:,1][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]-startPosPaw)*scalingFactor,c='k',alpha=0.4)
                ax1.plot(pawSpeed[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing,(pawSpeed[pawIDtoShow][:,2][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf])*scalingFactor,c='C0',alpha=0.4)
                #
                ax0.plot(cPawPos[pawIDtoShow][:,0][swingMask]- startTimeSwing,(cPawPos[pawIDtoShow][:,1][swingMask]-startPosPaw)*scalingFactor,c='k')
                ax1.plot(pawSpeed[pawIDtoShow][:,0][swingMask]- startTimeSwing,(pawSpeed[pawIDtoShow][:,2][swingMask])*scalingFactor,c='C0')
                #ax0.set_xlim(20,25)
                coloredRange = np.linspace(cPawPos[pawIDtoShow][idxStart, 0]- startTimeSwing, cPawPos[pawIDtoShow][idxEnd+1, 0]- startTimeSwing, Ngradient, endpoint=True)
                #if indecisiveSteps[n][3]:  # indecisive Step
                # ax4.plot(cPawPos[pawIDtoShow][idxEnd, 0], cPawPos[pawIDtoShow][idxEnd, 1]*scalingFactor, '1', alpha=0.5, lw=0.5)
                if indecisiveSteps[randomStepIdx[nFig][n]][3]:
                    closeIndicies = indecisiveSteps[randomStepIdx[nFig][n]][4]
                    firstExpectedStancePeriod = np.where((closeIndicies[:,1]-closeIndicies[:,0])>=2)[0]
                    if len(firstExpectedStancePeriod)>0:
                        expectedStanceIdx = closeIndicies[firstExpectedStancePeriod[0],0]
                    else:
                        expectedStanceIdx = closeIndicies[0,0]
                    ax0.axvline(x=cPawPos[pawIDtoShow][:, 0][swingMask][expectedStanceIdx] - startTimeSwing, ls='--', c='C8')
                    ax1.axvline(x=cPawPos[pawIDtoShow][:, 0][swingMask][expectedStanceIdx] - startTimeSwing, ls='--', c='C8')

                    #pdb.set_trace()
                for i in range(Ngradient - 1):
                    #if indecisiveSteps[randomIdxSwings[n]][3]
                    #pdb.set_trace()
                    ax0.axvspan(coloredRange[i], coloredRange[i + 1], alpha=alphaRange[i], lw=0, facecolor='C0')#('crimson' if indecisiveSteps[randomStepIdx[nFig][n]][3] else 'C0'))
                    ax1.axvspan(coloredRange[i], coloredRange[i + 1], alpha=alphaRange[i], lw=0, facecolor='C0')#('crimson' if indecisiveSteps[randomStepIdx[nFig][n]][3] else 'C0'))
                #ax0.set_xlim(-0.1,0.6)
                # decide on axis limits
                posTime = cPawPos[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                speedTime = pawSpeed[pawIDtoShow][:,0][swingMask[0]-pawBeAf:swingMask[-1]+pawBeAf]- startTimeSwing
                startPlotTime = (posTime[0] if posTime[0]>speedTime[0] else speedTime[0])
                endPlotTime = (posTime[-1] if posTime[-1]<speedTime[-1] else speedTime[-1])
                #ax0.set_xlim(startPlotTime,endPlotTime)
                #ax1.set_xlim(startPlotTime, endPlotTime)
                ax0.set_xlim(-0.05, 0.25)
                ax1.set_xlim(-0.05, 0.25)
                ax0.set_ylim(-1,3.15)
                ax1.set_ylim(-40,130)
                # panel layouts
                self.layoutOfPanel(ax0,xLabel=None, yLabel=('rel. position \n(cm)' if not (nPanel%4) else None), xyInvisible=[True,(True if (nPanel % 3) else False)])
                self.layoutOfPanel(ax1, xLabel=('time (s)' if (nPanel > 11) else None), yLabel=('speed \n(cm/s)' if not (nPanel % 3) else None),xyInvisible=[False, (True if (nPanel % 3) else False)])
                nPanel+=1
            nFig +=1
        # PSTH part #######################################
        #pdb.set_trace()
        nFig1 = 0
        nCell = 0
        cellList = [3,9,15,46]
        nRowsCols = 2
        nMice = len(PSTHSummaryAllAnimals[cellType])
        gssub01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gssub0[1], hspace=0.18, wspace=0.18)
        PSTHquantifyDict = {}
        pearsonR = []
        for n in range(nMice):
            #print(nCell, cellList)
            mouse = PSTHSummaryAllAnimals[cellType][n]['mouse']
            PSTHData = PSTHSummaryAllAnimals[cellType][n]['PSTHdata']
            # pdb.set_trace()
            nCellPerAnimal = len(PSTHData)
            PSTHquantifyDict[mouse] = {}
            # gssub2=gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[n], hspace=0.1, wspace=0.5) #, height_ratios=[1,2,1],width_ratios=[1])
            for m in range(nCellPerAnimal):  # cells per animal
                print(n, nCell, cellList)
                nRecs = len(PSTHData[m][3])  # recordings per cell
                PSTHquantifyDict[mouse][m] = {}
                for k in range(nRecs):
                    if k == 0:
                        tempPSTHMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHExpStanceMis = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_expectedStanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful_z-scored'][1]
                        tTime = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful'][0]
                    else:
                        tempPSTHMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHExpStanceMis += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_expectedStanceOnsetAlignedMissstep_z-scored'][1]
                        tempPSTHSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_stanceOnsetAlignedSuccessful_z-scored'][1]

                # psths.append([tTime,tempPSTHSucc/nRecs,tempPSTHMiss/nRecs,tempExpStanceMis / nRecs])
                PSTHquantifyDict[mouse][m]['timePSTH'] = tTime
                PSTHquantifyDict[mouse][m]['psthSucc'] = tempPSTHSucc / nRecs
                PSTHquantifyDict[mouse][m]['psthMiss'] = tempPSTHMiss / nRecs
                PSTHquantifyDict[mouse][m]['psthExp'] = tempPSTHExpStanceMis / nRecs
                PSTHquantifyDict[mouse][m]['peakTimeSuccessfulStep'] = tTime[np.argmax(tempPSTHSucc / nRecs)]
                PSTHquantifyDict[mouse][m]['peakTimeMissStep'] = tTime[np.argmax(tempPSTHMiss / nRecs)]
                PSTHquantifyDict[mouse][m]['peakTimeExpectedStanceMissStep'] = tTime[np.argmax(tempPSTHExpStanceMis / nRecs)]
                PSTHquantifyDict[mouse][m]['psthStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['stanceStart']

                nRecs = len(PSTHData[m][3])  # recordings per cell
                for k in range(nRecs):
                    if k == 0:
                        tempSpeedMiss = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempSpeedExpStanceMis = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempSpeedSucc = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                        tTimeSpeed = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][0]
                    else:
                        tempSpeedMiss += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedMissstep'][1]
                        tempSpeedExpStanceMis += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedExpectedStanceOnsetAlignedMissstep'][1]
                        tempSpeedSucc += PSTHData[m][3][k]['allSteps'][pawIdxToshow]['psth_speedStanceOnsetAlignedSuccessful'][1]
                PSTHquantifyDict[mouse][m]['timeSpeed'] = tTimeSpeed
                PSTHquantifyDict[mouse][m]['speedSucc'] = tempSpeedSucc / nRecs
                PSTHquantifyDict[mouse][m]['speedMiss'] = tempSpeedMiss / nRecs
                PSTHquantifyDict[mouse][m]['speedExp'] = tempSpeedExpStanceMis / nRecs
                PSTHquantifyDict[mouse][m]['speedStanceStart'] = PSTHData[m][3][k]['allSteps'][pawIdxToshow]['speedCenteredStanceStart']
                #print(tTime, tTimeSpeed)
                #pdb.set_trace()
                interpSucc = interp1d(tTimeSpeed, tempSpeedSucc / nRecs, bounds_error=False)
                speedAtPHSTTimeSucc = interpSucc(tTime)
                interpMiss = interp1d(tTimeSpeed, tempSpeedMiss / nRecs, bounds_error=False)
                speedAtPHSTTimeMiss = interpSucc(tTime)
                interpExp = interp1d(tTimeSpeed, tempSpeedExpStanceMis / nRecs, bounds_error=False)
                speedAtPHSTTimeExp = interpSucc(tTime)
                print('Correlation is compuated for the time :', tTime)
                PSTHquantifyDict[mouse][m]['cross-corrSucc'] = scipy.stats.pearsonr(speedAtPHSTTimeSucc,tempPSTHSucc / nRecs)
                PSTHquantifyDict[mouse][m]['cross-corrMiss'] = scipy.stats.pearsonr(speedAtPHSTTimeMiss,tempPSTHMiss / nRecs)
                PSTHquantifyDict[mouse][m]['cross-corrExp'] = scipy.stats.pearsonr(speedAtPHSTTimeExp,tempPSTHExpStanceMis / nRecs)
                pearsonR.append([PSTHquantifyDict[mouse][m]['cross-corrSucc'],PSTHquantifyDict[mouse][m]['cross-corrMiss'],PSTHquantifyDict[mouse][m]['cross-corrExp']])
                if nCell in cellList :
                    gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gssub01[nFig1], hspace=0.1, wspace=0.5)  # , height_ratios=[1,2,1],width_ratios=[1])
                    ax0 = plt.subplot(gssub2[1])
                    ax0.plot(tTime, tempPSTHSucc / nRecs, c='black', label='success, actual stance')
                    ax0.plot(tTime, tempPSTHMiss / nRecs, c='gray', label='miss, actual stance')
                    ax0.plot(tTime, tempPSTHExpStanceMis / nRecs, c='C8', label='miss, expected stance')
                    ax0.axvline(0, ls='--', color='gray', alpha=0.5)
                    ax0.axhline(0, ls=':', color='gray', alpha=0.5)
                    ax0.set_xlim(-0.3, 0.3)
                    self.layoutOfPanel(ax0, xLabel=(f'time centered on actual/expected \n stance onset (s)' if nFig1 >= nRowsCols * (nRowsCols - 1) else None), yLabel=(f'activity (z-score)' if not (nFig1 % nRowsCols) else None))
                    #
                    ax1 = plt.subplot(gssub2[0])
                    ax1.plot(tTimeSpeed, (tempSpeedMiss / nRecs)*scalingFactor, c='gray', label='miss, actual stance')
                    ax1.plot(tTimeSpeed, (tempSpeedSucc / nRecs)*scalingFactor, c='black', label='success, actual stance')
                    ax1.plot(tTimeSpeed, (tempSpeedExpStanceMis / nRecs)*scalingFactor, c='C8', label='miss, expected stance')
                    ax1.axvline(0, ls=':', color='gray', alpha=0.5)
                    ax1.axhline(0, ls='--', color='gray', alpha=0.5)
                    ax1.set_xlim(-0.3, 0.3)
                    self.layoutOfPanel(ax1, xLabel=None,yLabel=('speed (cm/s)' if not (nFig1 % nRowsCols) else None),xyInvisible=[True,False],Leg=(([0.5, 0.75], 9) if nFig1 == 0 else None))

                    nFig1+=1
                #
                nCell += 1
        # second row ###################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], hspace=0.1, wspace=0.1,width_ratios=[1.1,2,2])
        gssub10 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gssub1[0], hspace=0.25, wspace=0.2)
        ax10 = plt.subplot(gssub10[0])
        ax11 = plt.subplot(gssub10[1])
        ax12 = plt.subplot(gssub10[2])
        #cols = ['C0','C1','C2']
        rr01 = []
        rr02 = []
        rr12 = []
        for n in range(len(pearsonR)):
            #for i in range(3):
            ax10.plot(pearsonR[n][0][0],pearsonR[n][1][0],'o',c=('purple' if (pearsonR[n][0][1]<0.05 and pearsonR[n][1][1]<0.05) else '0.5'),alpha=0.3)
            ax11.plot(pearsonR[n][0][0], pearsonR[n][2][0], 'o', c=('purple' if (pearsonR[n][0][1]<0.05 and pearsonR[n][1][1]<0.05) else '0.5'), alpha=0.3)
            ax12.plot(pearsonR[n][1][0], pearsonR[n][2][0], 'o', c=('purple' if (pearsonR[n][0][1]<0.05 and pearsonR[n][1][1]<0.05) else '0.5'), alpha=0.3)
            rr01.append([pearsonR[n][0][0],pearsonR[n][1][0]])
            rr02.append([pearsonR[n][0][0], pearsonR[n][2][0]])
            rr12.append([pearsonR[n][1][0], pearsonR[n][2][0]])
            #ax10.plot([0,1],[pearsonR[n][0][0],pearsonR[n][1][0]],'-',c='0.5',alpha=0.5)
            #ax10.plot([1,2],[pearsonR[n][1][0],pearsonR[n][2][0]],'-',c='0.5',alpha=0.5)

        # linear regression also for stance onset  ###########################
        performLinearRegressionPlotStats(rr01, ax10)
        performLinearRegressionPlotStats(rr02, ax11)
        performLinearRegressionPlotStats(rr12, ax12)
        self.layoutOfPanel(ax10, xLabel='corr.: success, actual stance',yLabel='corr.: miss, actual stance')
        self.layoutOfPanel(ax11, xLabel='corr.: success, actual stance', yLabel='corr.: miss, expected stance')
        self.layoutOfPanel(ax12, xLabel='corr.: miss, actual stance', yLabel='corr.: miss, expected stance')
        #ax10.set_xticks([0,1,2])
        ax10.plot([-.65,1],[-0.65,1], ls='--', color='gray', alpha=0.5)
        ax10.axhline(y=0,color='gray',ls=':')
        ax10.axvline(x=0, color='gray', ls=':')
        ax11.plot([-.65, 1], [-0.65, 1], ls='--', color='gray', alpha=0.5)
        ax11.axhline(y=0,color='gray',ls=':')
        ax11.axvline(x=0, color='gray', ls=':')
        ax12.plot([-.65, 1], [-0.65, 1], ls='--', color='gray', alpha=0.5)
        ax12.axhline(y=0,color='gray',ls=':')
        ax12.axvline(x=0, color='gray', ls=':')
        majorLocator_all = MultipleLocator(0.5)
        ax10.xaxis.set_major_locator(majorLocator_all)
        ax10.yaxis.set_major_locator(majorLocator_all)
        ax11.xaxis.set_major_locator(majorLocator_all)
        ax11.yaxis.set_major_locator(majorLocator_all)
        ax12.xaxis.set_major_locator(majorLocator_all)
        ax12.yaxis.set_major_locator(majorLocator_all)
        ax10.set_box_aspect(1)
        ax11.set_box_aspect(1)
        ax12.set_box_aspect(1)
        #ax10.set_xticklabels(['success, actual stance','miss, actual stance','miss, expected stance'], rotation=45, ha='right')
        ########################
        fname = 'fig_miss-step-analysis_v%s' % figVersion
        #plt.savefig('caTriggeredAverages_%s.pdf' % caTriggeredAverages[nDays][0])  # define vertical and horizontal spacing between panels  # plt.show()
        #plt.savefig(fname + '.png')
        #plt.show()
        plt.savefig(fname + '.pdf')
        plt.savefig(fname + '.svg')
