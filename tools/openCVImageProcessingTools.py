import cv2
import sys
import pdb
import pickle
#from imutils import perspective
#from imutils import contours
import numpy as np
import pandas as pd
#import imutils
from scipy.spatial import distance as dist
from scipy import optimize
import math
import scipy
from scipy.interpolate import interp1d
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import time
mp.use('TkAgg')

import tools.dataAnalysis as dataAnalysis

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
import random
class openCVImageProcessingTools:
    def __init__(self,analysisLoc, figureLoc, ff, showI = False):
        #  "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/170606_f37/170606_f37_2017.07.12_001_behavingMLI_000_raw_behavior.avi"
        self.analysisLocation = analysisLoc
        self.figureLocation = figureLoc
        self.f = ff
        self.showImages = showI
        self.Vwidth = 816
        self.Vheight = 616

        self.testAboveBarGenerated = False
        self.testBelowBarGenerated = False

    ############################################################
    def __del__(self):
        try :
            self.video.release()
        except:
            pass

        # cv2.destroyAllWindows()
        print('on exit')


    ############################################################
    def openVideo(self,pathAndFileName,outputProps=True):

        # get properties
        self.video = cv2.VideoCapture(pathAndFileName)
        self.Vlength = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.Vwidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Vheight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Vfps = self.video.get(cv2.CAP_PROP_FPS)
        if outputProps:
            print('Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (self.Vlength, self.Vwidth, self.Vheight, self.Vfps))

        if not self.video.isOpened():
            print('Could not open video')
            sys.exit()
        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # Return an array representing the indices of a grid.
        self.imgGrid = np.indices((self.Vheight, self.Vwidth))

        return (ok,img)
    ############################################################
    def generateTestBarArray(self, yPos, areaWidth, location,shifts,recordingStruc=None,recordingBatch=None):

        yLocation = yPos + areaWidth / 2.
        #yRefBelow = self.Vheight - 0.899
        #yRefAbove = self.Vheight - 381.119
        print(yPos,yLocation,location,recordingStruc)
        if recordingStruc is None:
            yRefBelow = self.Vheight - 68.791 #61.205 #58.367
            yRefAbove = self.Vheight - 272.268 #264.679 #257.599
            polyCoeffsBelow = np.array([-2.38644362e-08,  4.93041183e-05, -3.00048499e-02,  9.12569758e+01])#np.array([-6.39624280e-08,  1.00446943e-04, -4.79610092e-02,  8.72256601e+01]) #np.array([ 3.24055344e-08, -4.24100323e-05,  1.58473987e-02,  7.44691553e+01])#np.array([5.29427827e-10, -5.44334567e-07, 2.33618937e-04, -6.14393512e-02, 9.09112549e+01])
            polyCoeffsAbove = np.array([-5.22160118e-08,  8.53704035e-05, -4.33461673e-02,  9.22485586e+01])#np.array([-2.22655550e-10,  1.20388522e-05, -1.18899809e-02,  8.33872114e+01]) #np.array([-1.30470866e-08,  3.73803147e-05, -2.54318037e-02,  7.99139595e+01])#np.array([7.56895295e-10, -8.82197309e-07, 3.89379907e-04, -8.41450517e-02, 8.73406737e+01])
        elif (recordingStruc == 'simplexBehavior') and (recordingBatch is None):
            yRefBelow = self.Vheight - 86.242
            yRefAbove = self.Vheight - 405.545
            polyCoeffsBelow = np.array([ 5.77980631e-08, -5.77805502e-05,  1.21589718e-02,  8.34963092e+01])#np.array([-2.82445578e-10,  5.55470017e-07, -3.56808443e-04,  8.24484996e-02, 7.82924544e+01])
            polyCoeffsAbove = np.array([ 2.04566542e-08, -5.48136664e-06, -8.77195245e-03,  8.04143685e+01]) #[-1.13037300e-10, 2.17339955e-07, -1.23679201e-04 , 1.95090152e-02, 7.82087407e+01]) #[-2.56275881e-10,  4.89392643e-07, -2.96872276e-04,  6.07987027e-02, 7.53825337e+01])
        elif (recordingStruc == 'simplexBehavior') and (recordingBatch == 'B'):
            yRefBelow = self.Vheight - 86.242
            yRefAbove = self.Vheight - 405.545
            polyCoeffsBelow = np.array([-7.07900126e-08,  1.18970766e-04, -5.69764360e-02,  1.01025512e+02]) #np.array([ 5.77980631e-08, -5.77805502e-05,  1.21589718e-02,  8.34963092e+01])#np.array([-2.82445578e-10,  5.55470017e-07, -3.56808443e-04,  8.24484996e-02, 7.82924544e+01])
            polyCoeffsAbove = np.array([ 1.64889836e-08,  1.27960101e-05, -2.14594279e-02, 9.42392013e+01]) #np.array([ 2.04566542e-08, -5.48136664e-06, -8.77195245e-03,  8.04143685e+01]) #[-1.13037300e-10, 2.17339955e-07, -1.23679201e-04 , 1.95090152e-02, 7.82087407e+01]) #[-2.56275881e-10,  4.89392643e-07, -2.96872276e-04,  6.07987027e-02, 7.53825337e+01])
        elif recordingStruc == 'simplexNew':
            yRefBelow = self.Vheight - 141.13041968085105
            yRefAbove = self.Vheight - 330.28578026595744
            polyCoeffsBelow = np.array( [-1.00317930e-07,  1.55855102e-04, -7.48134828e-02, 9.13227764e+01])#np.array([ 1.00282581e-08,  6.00177385e-06, -1.18852681e-02,  9.13424242e+01])  # np.array([-2.82445578e-10,  5.55470017e-07, -3.56808443e-04,  8.24484996e-02, 7.82924544e+01])
            polyCoeffsAbove = np.array([-5.34826974e-08,  8.25106659e-05, -3.87005977e-02,  8.57617865e+01])#np.array([ 5.94058018e-09,  1.93450068e-05, -2.25353790e-02,  9.15595770e+01])  # [-1.13037300e-10, 2.17339955e-07, -1.23679201e-04 , 1.950
        elif recordingStruc == 'simplexNPX':
            yRefBelow = self.Vheight - 38.88929873786406
            yRefAbove = self.Vheight - 239.96968097087364
            polyCoeffsBelow = np.array([-4.69990436e-08,  6.91322685e-05, -3.37792247e-02,  8.22936319e+01])#np.array([5.29427827e-10, -5.44334567e-07, 2.33618937e-04, -6.14393512e-02, 9.09112549e+01])
            polyCoeffsAbove = np.array([-4.67336641e-08,  7.69711392e-05, -3.98929634e-02,  8.28060976e+01])#np.array([7.56895295e-10, -8.82197309e-07, 3.89379907e-04, -8.41450517e-02, 8.73406737e+01])
        #coeff4 = polyCoeffsBelow[4] + (yLocation - yRefBelow) * (polyCoeffsAbove[4] - polyCoeffsBelow[4]) / (yRefAbove - yRefBelow)
        #if location == 'below':
        #    polyCoeffs = np.copy(polyCoeffsBelow)
        #elif location == 'above':
        #    polyCoeffs = np.copy(polyCoeffsAbove)
        #polyCoeffs[4] = coeff4
        #print(yPos, yLocation, yRefBelow, yRefAbove, location, recordingStruc,polyCoeffs,coeff4)

        testArray = np.zeros((shifts, self.Vwidth))
        midBarArray = []
        nRungs = []
        for i in range(shifts):
            # def generateBarArrayForTest(startIdx):
            # testArray = np.zeros(self.Vwidth)
            midBars = []
            locIdx = i
            nBars = 0
            while ((locIdx + 18) < self.Vwidth) and nBars<9:
                midBars.append(locIdx + 9)
                testArray[i, (locIdx + 1):(locIdx + 3)] = np.array([0.33, 0.66])
                testArray[i, (locIdx + 3):(locIdx + 16)] = 1.
                testArray[i, (locIdx + 16):(locIdx + 18)] = np.array([0.66, 0.33])
                distanceAbove = scipy.polyval(polyCoeffsAbove, locIdx)
                distanceBelow = scipy.polyval(polyCoeffsBelow, locIdx)
                distance = distanceBelow + (yLocation - yRefBelow)*(distanceAbove - distanceBelow)/(yRefAbove - yRefBelow)
                #polyCoeffsBelow[4] + (yLocation - yRefBelow) * (polyCoeffsAbove[4] - polyCoeffsBelow[4]) / (yRefAbove - yRefBelow)
                locIdx += int(distance + 0.5)
                nBars+=1
            midBarArray.append(midBars)  # return(testArray,midBars)
            nRungs.append(nBars)
        return (testArray,midBarArray,nRungs)

    ############################################################
    #self.findBarsDiffSum(imgGray, yPosAbove, barWidthAbove, loc='above', recStr=recStruc)
    def findBarsDiffSum(self, firstImgGray, yPos, areaWidth, loc='above', recStr=None):

        self.display = False
        bestDiffSum = None
        shifts = 100

        #yLocation = yPos+areaWidth/2.
        #yRefBelow = self.Vheight - 0.899
        #yRefAbove = self.Vheight - 381.119
        #polyCoeffsBelow = np.array([5.29427827e-10, -5.44334567e-07, 2.33618937e-04, -6.14393512e-02, 9.09112549e+01])
        #polyCoeffsAbove = np.array([7.56895295e-10, -8.82197309e-07, 3.89379907e-04, -8.41450517e-02, 8.73406737e+01])
        #coeff4 = polyCoeffsBelow[4] + (yLocation-yRefBelow)*(polyCoeffsAbove[4] - polyCoeffsBelow[4])/(yRefAbove-yRefBelow)
        if loc == 'below':
            if not self.testBelowBarGenerated :
                (self.testArrayBelow,self.midBarArrayBelow) = self.generateTestBarArray( yPos, areaWidth, loc,shifts,recordingStruc=recStr)
                #print('below test array generated')
                self.testBelowBarGenerated = True
            testArr = self.testArrayBelow
            midBars = self.midBarArrayBelow

        elif loc == 'above':
            if not self.testAboveBarGenerated :
                (self.testArrayAbove,self.midBarArrayAbove) = self.generateTestBarArray( yPos, areaWidth, loc,shifts,recordingStruc=recStr)
                #print('above test array generated')
                self.testAboveBarGenerated = True
            testArr = self.testArrayAbove
            midBars = self.midBarArrayAbove

        nPoints = 100
        intensity = np.log(np.average(firstImgGray[yPos:(yPos+areaWidth)],0))
        differenceInt = np.abs(np.diff(intensity))
        differenceIntConv = np.convolve(differenceInt,np.ones(6)/6,mode='same')
        intensityConv =  np.convolve(intensity,np.ones(nPoints)/nPoints,mode='same')
        intensityAvgSub = np.abs(intensity - intensityConv) + np.concatenate((np.array([0]),differenceInt)) # + np.max(intensity)
        # remove largest peak from array
        idxPeak = np.argmax(intensityAvgSub)
        intensityAvgSub[idxPeak-10:idxPeak+10]=0.
        if loc == 'below':
            intensityAvgSub[176:200] = 0
            intensityAvgSub[176:200] = 0
        shiftResults = []
        for i in range(shifts):
            #(testArr,midBars) = generateBarArrayForTest(i)
            diffSum = np.sum(np.abs((intensityAvgSub - testArr[i]*max(intensityAvgSub)) ** 2))
            if bestDiffSum is None or diffSum < bestDiffSum:
                barLocs = midBars[i]
                bestDiffSum = diffSum
                #bestOffset = i
                bestBarArray = testArr[i]
            shiftResults.append([i,testArr,diffSum,midBars[i]])


        #pdb.set_trace()
        #print()
        if self.display:
            fig = plt.figure()
            ax0 = fig.add_subplot(2,1,1)
            ax0.plot(intensity)
            ax0.plot(intensityAvgSub)
            ax0.plot(intensityConv)
            ax0.plot(bestBarArray*max(intensityAvgSub))

            ax1 = fig.add_subplot(2,1,2)
            for i in range(shifts):
                ax1.plot(shiftResults[i][0],shiftResults[i][2],'o',c='C0')
            plt.show()
            pdb.set_trace()

        return np.asarray(barLocs)
    ############################################################
    ############################################################
    def videoToArray(self,videoFileName,limits=None):
        # get properties
        video = cv2.VideoCapture(videoFileName)
        Vlength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        Vwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        Vheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Vfps = video.get(cv2.CAP_PROP_FPS)
        #if outputProps:
        print('%s' % videoFileName)
        print('  Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (Vlength, Vwidth, Vheight, Vfps))

        if not video.isOpened():
            print('Could not open video')
            sys.exit()
        if limits is None:
            length = Vlength
            includeBool = np.full(length, True)
        else:
            length = limits[1]-limits[0]
            includeBool = np.full(Vlength, False)
            includeBool[limits[0]:limits[1]] = True
        frames = np.empty((length, Vwidth, Vheight))
        #pdb.set_trace()
        # read first video frame
        nFrame = 0
        nRead = 0
        ok, img = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        if includeBool[nRead]:
            frames[nFrame] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            nFrame+= 1
        nRead+=1
        while True:
            ok, img = video.read()
            if ok:
                if includeBool[nRead]:
                    frames[nFrame] = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    nFrame += 1
                nRead+=1
            else:
                break
            if nRead>limits[1]:
                break
        video.release()
        print(nFrame, ' where added to array ')
        return frames
    ############################################################
    #self.findBarsDiffSum(imgGray, yPosAbove, barWidthAbove, loc='above', recStr=recStruc)
    # (imgGray, yPosAbove, yPosBelow, barWidthBelow,rungPositions[-1],recStr=recStruc)
    def findBarsDiffSumBoth(self, firstImgGray, yPosAbove, yPosBelow, areaWidth, lastRungPosition, recStr=None):

        def getIntensityArray(img,yPos,loc):
            intensity = np.log(np.average(img[yPos:(yPos + areaWidth)], 0))
            differenceInt = np.abs(np.diff(intensity))
            differenceIntConv = np.convolve(differenceInt, np.ones(4) / 4, mode='same')
            intensityConv = np.convolve(intensity, np.ones(nPoints) / nPoints, mode='same')
            intensityAvgSub = np.abs(intensity - intensityConv) + 5*np.concatenate((np.array([0]), differenceIntConv))  # + np.max(intensity)
            # remove largest peak from array
            # idxPeak = np.argmax(intensityAvgSub)
            # intensityAvgSub[idxPeak-10:idxPeak+10]=0.
            if loc == 'above':
                intensityAvgSub[173:272] = 0
                intensityAvgSub[272:317] = 0
            elif loc == 'below': # because of the two vertical bar of the side-wall holder
                intensityAvgSub[176:230] = 0
                intensityAvgSub[273:322] = 0

            intensityAvgSub[:43] = 0
            intensityAvgSub[-43:] = 0
            return intensityAvgSub

        self.display = False
        bestDiffSumAbove = None
        bestDiffSumBelow = None
        shifts = 100

        #yLocation = yPos+areaWidth/2.
        #yRefBelow = self.Vheight - 0.899
        #yRefAbove = self.Vheight - 381.119
        #polyCoeffsBelow = np.array([5.29427827e-10, -5.44334567e-07, 2.33618937e-04, -6.14393512e-02, 9.09112549e+01])
        #polyCoeffsAbove = np.array([7.56895295e-10, -8.82197309e-07, 3.89379907e-04, -8.41450517e-02, 8.73406737e+01])
        #coeff4 = polyCoeffsBelow[4] + (yLocation-yRefBelow)*(polyCoeffsAbove[4] - polyCoeffsBelow[4])/(yRefAbove-yRefBelow)

        if not self.testBelowBarGenerated :
            (self.testArrayBelow,self.midBarArrayBelow,self.nRungs) = self.generateTestBarArray( yPosBelow, areaWidth, 'below',shifts,recordingStruc=recStr)
            #print('below test array generated')
            self.testBelowBarGenerated = True
        testArrBelow = self.testArrayBelow
        midBarsBelow = self.midBarArrayBelow
        numberOfRungsBelow = self.nRungs

        #elif loc == 'above':
        if not self.testAboveBarGenerated :
            (self.testArrayAbove,self.midBarArrayAbove,self.nRungs) = self.generateTestBarArray( yPosAbove, areaWidth, 'above',shifts,recordingStruc=recStr)
            #print('above test array generated')
            self.testAboveBarGenerated = True
        testArrAbove = self.testArrayAbove
        midBarsAbove = self.midBarArrayAbove
        numberOfRungsAbove = self.nRungs

        nPoints = 100
        intensityAvgSubBelow = getIntensityArray(firstImgGray,yPosBelow,'below')
        intensityAvgSubAbove = getIntensityArray(firstImgGray, yPosAbove, 'above')

        shiftResults = []
        for i in range(shifts):
            #(testArr,midBars) = generateBarArrayForTest(i)
            diffSumAbove = np.sum(((intensityAvgSubAbove - testArrAbove[i]*max(intensityAvgSubAbove)) ** 2))
            if bestDiffSumAbove is None or diffSumAbove < bestDiffSumAbove:
                barLocsAbove = midBarsAbove[i]
                bestShiftAbove = i
                bestDiffSumAbove = diffSumAbove
                #bestOffset = i
                bestBarArrayAbove = testArrAbove[i]
        for i in range(bestShiftAbove-30,((bestShiftAbove+30) if (bestShiftAbove+30)<100 else 99)):
            diffSumBelow = np.sum(((intensityAvgSubBelow - testArrBelow[i] * max(intensityAvgSubBelow)) ** 2))
            if bestDiffSumBelow is None or diffSumBelow < bestDiffSumBelow:
                barLocsBelow = midBarsBelow[i]
                bestShiftBelow = i
                bestDiffSumBelow = diffSumBelow
                #bestOffset = i
                bestBarArrayBelow = testArrBelow[i]
            shiftResults.append([i,testArrAbove[i],testArrBelow[i],intensityAvgSubAbove,intensityAvgSubBelow,diffSumAbove,diffSumBelow,diffSumBelow,midBarsAbove[i],midBarsBelow[i]])
        #print(bestShiftAbove,bestShiftBelow,(bestShiftAbove-bestShiftBelow),barLocsAbove,barLocsBelow)

        #pdb.set_trace()
        #print()
        if self.display:
            mp.use('WxAgg')
            fig = plt.figure()
            ax0 = fig.add_subplot(2,2,1)
            #ax0.plot(intensity)
            ax0.plot(intensityAvgSubAbove)
            #ax0.plot(intensityConv)
            ax0.plot(shiftResults[bestShiftAbove-2][1]*max(intensityAvgSubAbove),lw=0.5,c='green')
            ax0.plot(bestBarArrayAbove*max(intensityAvgSubAbove))
            ax0.plot(shiftResults[bestShiftAbove+2][1]*max(intensityAvgSubAbove),lw=0.5, c='red')

            ax1 = fig.add_subplot(2,2,2)
            for i in range(shifts):
                ax1.plot(shiftResults[i][0],shiftResults[i][5],'o',c='C0')

            ax2 = fig.add_subplot(2,2,3)
            #ax0.plot(intensity)
            ax2.plot(intensityAvgSubBelow)
            #ax0.plot(intensityConv)
            ax2.plot(bestBarArrayBelow*max(intensityAvgSubBelow))

            ax3 = fig.add_subplot(2,2,4)
            for i in range(shifts):
                ax3.plot(shiftResults[i][0],shiftResults[i][6],'o',c='C0')

            plt.show()
            pdb.set_trace()

        return (np.asarray(barLocsAbove),np.asarray(barLocsBelow),bestShiftAbove,bestShiftBelow)



    ############################################################
    def findBarsCrossCorr(self, firstImgGray, yPos, areaWidth, location):

        self.display = False

        yLocation = int(yPos + areaWidth / 2.)

        if location == 'below':
            polyCoeffs = np.array([ 5.29427827e-10, -5.44334567e-07, 2.33618937e-04, -6.14393512e-02, 9.09112549e+01])
        elif location == 'above':
            polyCoeffs = np.array([ 7.56895295e-10, -8.82197309e-07, 3.89379907e-04, -8.41450517e-02, 8.73406737e+01])


        barWidth = 15

        testArray = np.zeros(self.Vwidth)
        locIdx = 0
        while locIdx<self.Vwidth:
            testArray[(locIdx+1):(locIdx+3)] = np.array([0.33,0.66])
            testArray[(locIdx+3):(locIdx+16)] = 1.
            testArray[(locIdx+16):(locIdx+18)] = np.array([0.66,0.33])
            distance = scipy.polyval(polyCoeffs, locIdx)
            locIdx += int(distance+0.5)

        intensity = np.log(np.average(firstImgGray[yPos:(yPos+areaWidth)],0))
        #intensityBelow = np.average(firstImgGray[yPosBelow:(yPosBelow+barWidthBelow)],0)

        corrBars = dataAnalysis.crosscorr(1,intensity,testArray,100)

        maximaIdx = scipy.signal.argrelextrema(corrBars[:,1],np.greater)
        #minimaIdx = scipy.signal.argrelextrema(corrBars[:,1],np.less)
        maximaLocations = corrBars[:, 0][maximaIdx[0]].astype(int)
        maxima          = corrBars[:, 1][maximaIdx[0]]
        maximum = np.max(maxima)
        maximumLoc = maximaLocations[np.argmax(maxima)]
        #loc = np.argmax(corrBars[:, 1][maximaIdx[0]].astype(int))
        barLocation = []
        if maximumLoc < 0:
            bL = maximumLoc#+9
        elif maximumLoc >= 0:
            bL = (-maximumLoc)
        barLocation.append(bL)
        while barLocation[-1]<self.Vwidth:
            distance = scipy.polyval(polyCoeffs, barLocation[-1])
            nL = distance + barLocation[-1]
            barLocation.append(nL)

        barLocation=np.asarray(barLocation)

        #print(maximaStats)
        #pdb.set_trace()
        #print()
        if self.display:
            fig = plt.figure()
            ax0 = fig.add_subplot(2,1,1)
            ax0.plot(intensity)
            for i in maximaLocations:
                x = np.arange(len(testArray))
                y = testArray
                if i < 0:
                    x = x[abs(i):]
                    y = y[:-abs(i)]
                elif i >= 0:
                    x = x[:-abs(i)]
                    y = y[abs(i):]
                ax0.plot(x,y*max(intensity),label='%s, %s' % (i,i))
            plt.legend()
            ax1 = fig.add_subplot(2,1,2)
            ax1.plot(corrBars[:,0],corrBars[:,1])

            plt.show()
            pdb.set_trace()
        return barLocation

    ############################################################
    def findHorizontalArea(self, img, coordinates=None,orientation='horizontal'):

        if coordinates is None:
            pos = 300
            areaWidth = 10
        else:
            pos = coordinates[0]
            areaWidth = coordinates[1]

        Npix = 5
        continueLoop = True
        # optimize with keyboard
        while continueLoop:
            #rungs = []
            imgLine = img.copy()
            if orientation=='horizontal':
                cv2.line(imgLine, (0, pos), (self.Vwidth, pos), (255, 0, 255), 2)
                cv2.line(imgLine, (0, pos+areaWidth), (self.Vwidth, pos+areaWidth), (255, 0, 255), 2)
            elif orientation == 'vertical':
                cv2.line(imgLine, (pos, 0), (pos, self.Vheight), (255, 0, 255), 2)
                cv2.line(imgLine, (pos+areaWidth, 0), (pos+areaWidth, self.Vheight), (255, 0, 255), 2)

            cv2.imshow("Rungs", imgLine)
            #print 'current xPosition, yPostion : ', xPosition, yPosition
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                pos -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                pos += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                areaWidth += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                areaWidth -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                continueLoop = False
            elif PressedKey == 27: # Escape
                continueLoop = False
            else:
                pass
        cv2.destroyWindow("Rungs")

        if orientation == 'horizontal':
            print('y Pos, width  :', pos,areaWidth)
        elif orientation == 'vertical':
            print('x Pos, width  :', pos,areaWidth)
        #mask = np.zeros((self.Vheight, self.Vwidth))
        return (pos,areaWidth)

    ############################################################
    # (frames,coordinates=SavedLEDcoordinates,currentCoordExist=currentCoodinatesExist)
    def findLEDNumberArea(self, frames, whichCamera, coordinates=None, currentCoordExist=False, determineAgain=False, videoRec=None, auto=False):
        if whichCamera == 'LEDinAniVideo':
            if videoRec['vidRecSoftware'] == 'ACQ4':
                initialValues = [730, 30, 20, 43, 20]
                typicalLEDnumber = True
            elif videoRec['vidRecSoftware'] == 'Bonsai':
                initialValues = [712, 27, 19, 43, 16]
                typicalLEDnumber = True
        elif whichCamera == 'LEDinWhisVideo':
            initialValues = [529, 409, 28, 66,-6]
            typicalLEDnumber = True

        # auxilary functions
        mp.use('TkAgg')
        # pdb.set_trace()
        if type(frames)==np.ndarray : # contains the actual raw frames
            avgFrame = np.average(frames[5000:6000], axis=0)
        elif type(frames) == str:  # contains the pointer to the frame stream
            videoFrames = self.videoToArray(frames,limits=[5000,5200])
            avgFrame = np.average(videoFrames, axis=0)
        #cv2.imwrite('image_%s.png' % videoType, np.transpose(avgFrame), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        def detect_circles(img):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Set minimum and maximum circle sizes
            min_radius = 3
            max_radius = 50
            # Set minimum distance between circle centers
            min_distance = round(9)
            # Set the number of circles expected to be found
            num_circles = 4

            # Initialize the list to store the detected circles
            circles_list = []

            breakout = False

            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[5:120, 667:800] = 255
            masked_gray = cv2.bitwise_and(gray, mask)
            max_guess_accumulator_array_threshold = 100
            guess_accumulator_array_threshold = max_guess_accumulator_array_threshold
            # Loop over different parameters to detect circles
            while guess_accumulator_array_threshold > 1 and breakout == False:
                    circles = cv2.HoughCircles(masked_gray,
                                               cv2.HOUGH_GRADIENT,
                                               dp=1,  # resolution of accumulator array.
                                               minDist=45,
                                               # number of pixels center of circles should be from each other, hardcode
                                               param1=16,
                                               param2=guess_accumulator_array_threshold,
                                               # minRadius=guess_radius - 1,
                                               minRadius=13,
                                               # HoughCircles will look for circles at minimum this size
                                               maxRadius=35,
                                               # HoughCircles will look for circles at maximum this size

                                               )
                    if circles is not None:
                        if len(circles[0]) == num_circles:
                            circles_list.append(circles)
                            print('good circles found !!!!!!!!')
                            g=0
                            for c in range(4):
                                if (circles[0][c][1]<120) and (circles[0][c][0]>650):
                                    g+=1

                    guess_accumulator_array_threshold -= 1

                                # for (x, y, r) in cir:

            random_index = random.randint(0, len(circles_list) - 1)
            cir=circles_list[random_index]
            # convert the (x, y) coordinates and radius of the circles to integers
            output = np.copy(img)
            print(cir[0, :])
            cir = np.round(cir[0, :]).astype("int")
            cirSum=np.sum(cir, axis=1)
            order=np.argsort(cirSum)
            orderedCir=cir[order]
            print('auto circle', orderedCir)
            # for (x, y, r) in cir:
            #     # pdb.set_trace()
            #     cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            #     # cv2.circle(output, (x, y), r, (0, 255, 255), 10)
            #     cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # cv2.imshow("output", np.hstack([img, output]))

            # cv2.imshow("output", output)
            # # cv2.waitKey(0).hstack([imgList[i], output])
            # cv2.waitKey(0)
            # # pdb.set_trace()
            LEDcoordinates=[4,orderedCir[:,0], orderedCir[:,1], 18]
            return LEDcoordinates
        def rotatePoints(theta,x,y):
            x = x.astype(float)
            y = y.astype(float)
            posXTemp = x[1:] - x[0]  # move the rotation point to the origin
            posYTemp = y[1:] - y[0]  # move the rotation point to the origin
            rotMatrix = np.array([[np.cos(theta*np.pi/180.), -np.sin(theta*np.pi/180.)], [np.sin(theta*np.pi/180.), np.cos(theta*np.pi/180.)]])
            posRotTemp = np.matmul(rotMatrix, np.row_stack((posXTemp, posYTemp)))
            #print('before')
            #pdb.set_trace()
            #posRotTemp = (posRotTemp+0.5).astype(int)
            x[1:] = posRotTemp[0] + x[0]
            y[1:] = posRotTemp[1] + y[0]
            #print('after')
            #pdb.set_trace()
            return (x,y) # 0.5 is added to get the correct rounding
        def changeSpacing(nLED,spacing,x,y):
            for i in range(1, nLED):
                x[i] = x[0] + (i % 2) * spacing
                y[i] = y[0] + (i // 2) * spacing
            return (x,y)
        ##############################
        if determineAgain:
            doLEDROIdetermination = True
        else:
            doLEDROIdetermination = (not currentCoordExist)
        # don't check location if recording already exists
        if doLEDROIdetermination :
            #pdb.set_trace()

            # the below in the clause allows to set the ROI on the LED location
            frame8bit = np.array(np.transpose(avgFrame), dtype=np.uint8)
            img = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            cv2.imwrite('averageImg.jpg', img)

            if auto:
                coordinates=detect_circles(img)
                nLED = coordinates[0]
                posX = coordinates[1]
                posY = coordinates[2]
                circleRadius=coordinates[3]
                continueCountLEDLoop = True
                # while continueCountLEDLoop:
                imgCircle = img.copy()
                for i in range(nLED):
                    cv2.circle(imgCircle, (int(posX[i]+0.5), int(posY[i]+0.5)), circleRadius, (255, 0, 255), 2)
                    cv2.putText(imgCircle,'%s' % i, (int(posX[i]+0.5), int(posY[i]+0.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6,color=(255,0,0))
                cv2.imshow("ImageWithLEDCircles", imgCircle)
                cv2.waitKey(2000)  # Wait for 2000 milliseconds (2 seconds)
                cv2.destroyWindow("ImageWithLEDCircles")
                cv2.destroyAllWindows()
            else:
                # first let's decide on how many LED's (if any) are present in the FOV
                if (coordinates is None) and (not typicalLEDnumber):
                    continueCountLEDLoop = True
                    while continueCountLEDLoop:
                        #rungs = []
                        imgPure = img.copy()
                        cv2.imshow("PureImage", imgPure)
                        print('specify how many LEDs are present in the field of view (e.g., 1,4 or 0 if none) :')
                        PressedKey = cv2.waitKey(0)
                        print(PressedKey)
                        if PressedKey == 49:
                            nLED = 1
                        elif PressedKey == 52 :
                            nLED = 4
                        elif PressedKey == 48:
                            nLED = 0
                        try:
                            print('Number of LEDs in image :',nLED)
                        except:
                            pass
                        else:
                            continueCountLEDLoop = False
                        cv2.destroyWindow("PureImage")
                elif (coordinates is None) and typicalLEDnumber:
                    nLED = 4
                else:
                    nLED = coordinates[0]
                ## sets the location of the ROIs for all LEDs
                if nLED > 0:
                    movePixels = 1
                    #spacing = 50
                    rotAngle = 2.
                    continueLoop = True
                    # optimize with keyboard
                    if (coordinates is None) or any(coordinates[1]>800) or any(coordinates[2]>800):
                        posX = np.full(nLED,0)
                        posY = np.full(nLED,0)
                        posX[0] = initialValues[0] # 730
                        posY[0] = initialValues[1] # 30
                        circleRadius = initialValues[2] #20
                        spacing = initialValues[3]
                        theta = initialValues[4]
                        (posX,posY) = changeSpacing(nLED,spacing,posX,posY)
                        (posX,posY) = rotatePoints(theta,posX,posY)
                    else:
                        nLED = coordinates[0]
                        posX = coordinates[1]
                        posY = coordinates[2]
                        circleRadius = coordinates[3]
                        spacing =  coordinates[4]
                        theta = coordinates[5]
                        #(posX, posY) = changeSpacing(nLED, spacing, posX, posY)
                        #(posX, posY) = rotatePoints(theta, posX, posY)
                    print(posX,posY)
                    while continueLoop:
                        #rungs = []
                        imgCircle = img.copy()
                        for i in range(nLED):
                            cv2.circle(imgCircle, (int(posX[i]+0.5), int(posY[i]+0.5)), circleRadius, (255, 0, 255), 2)
                            cv2.putText(imgCircle,'%s' % i, (int(posX[i]+0.5), int(posY[i]+0.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, thickness=1, fontScale=0.6,color=(255,0,0))
                        cv2.imshow("ImageWithLEDCircles", imgCircle)
                        print('change circle position with arrow buttons; circle size with + or - buttons; rotation with r, l button; spacing with w,c buttons; exit loop with space/enter or ESC :')
                        PressedKey = cv2.waitKey(0)
                        print(PressedKey)
                        print(posX,posY,circleRadius,spacing,theta)
                        if PressedKey == 56 or PressedKey ==82: #UP arrow
                            posY -= movePixels
                        elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                            posY += movePixels
                        elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                            posX += movePixels
                        elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                            posX -= movePixels
                        elif PressedKey == 61 : #+ button
                            circleRadius += movePixels
                        elif PressedKey == 45 : #- button
                            circleRadius -= movePixels
                        elif PressedKey == 114 : # r button
                            theta += rotAngle
                            (posX,posY) = rotatePoints(rotAngle,posX,posY)
                        elif PressedKey == 108 : # l button
                            theta -=rotAngle
                            (posX, posY) = rotatePoints(-rotAngle, posX, posY)
                        elif PressedKey == 119 : # w button
                            spacing +=movePixels
                            (posX,posY) = changeSpacing(nLED,spacing,posX,posY)
                        elif PressedKey == 99 : # c button
                            spacing -=movePixels
                            (posX, posY) = changeSpacing(nLED,spacing, posX, posY)
                        elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                            continueLoop = False
                        elif PressedKey == 27: # Escape
                            continueLoop = False
                        else:
                            pass
                    cv2.destroyWindow("ImageWithLEDCircles")

                    print('LED positions : ',posX,posY,circleRadius,spacing,theta)
                else:
                    print('No LED in field of view.')
                    posX = None
                    posY = None
                    circleRadius = None
        else:
            if not auto:
                (nLEDs,posX,posY,circleRadius,spacing,theta) = (coordinates[0],coordinates[1],coordinates[2],coordinates[3],coordinates[4],coordinates[5])
            else:
                (nLEDs, posX, posY, circleRadius) = (
                coordinates[0], coordinates[1], coordinates[2], coordinates[3])


        if not auto:
            print('coordinates used for extraction :',nLED,posX,posY,circleRadius,spacing,theta)
            coordinates = np.array([nLED, posX, posY, circleRadius, spacing, theta], dtype=object)
        else:
            print('coordinates used for extraction :',nLED,posX,posY,circleRadius)
            coordinates = np.array([nLED, posX, posY, circleRadius], dtype=object)


        return coordinates

    ############################################################
    # extract temporal trace of LED area mask
    def extractLEDtraces(self, frames, coordinates, optimize=False):
        (nLEDs, posX, posY, circleRadius) = (coordinates[0], coordinates[1], coordinates[2], coordinates[3])
        LEDtraces = []
        if type(frames) == np.ndarray:
            # get mask for circular area comprising the LED
            dims = np.shape(np.transpose(frames[0]))
            maskGrid = np.indices((dims[0],dims[1]))
            framesNew = np.transpose(frames, axes=(0, 2, 1))  # permutate last two axes as for the image depiction
            for i in range(nLEDs):
                maskCircle = np.sqrt((maskGrid[1] - posX[i]) ** 2 + (maskGrid[0] - posY[i]) ** 2) < circleRadius


                # apply mask to the frame array and extract mean brigthness of the LED ROI
                LEDtr = np.mean(framesNew[:,maskCircle],axis=1)
                LEDtraces.append(LEDtr)

                # print('LED number', i)
                #plt.plot(LEDtr)

                #plt.show()
            LEDtraces = np.asarray(LEDtraces)
        elif type(frames) == str:
            def getValuesBasedOnMask(frame,posX,posY, optimize):
                vals = []
                for i in range(nLEDs):
                    maskCircle = np.sqrt((maskGrid[1] - posX[i])**2 + (maskGrid[0] - posY[i])**2) < circleRadius*0.3
                    vals.append(np.mean(frame[maskCircle]))
                return vals
            #video = frames
            video = cv2.VideoCapture(frames)
            Vlength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            Vwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            Vheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            Vfps = video.get(cv2.CAP_PROP_FPS)
            #if outputProps:
            print('  Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (Vlength, Vwidth, Vheight, Vfps))
            maskGrid = np.indices((Vheight,Vwidth))
            # read first video frame
            LEDtraces = []
            if optimize:
                ####################################################################################\
                # Read a few test frames to find the optimal mask position
                videoTest = cv2.VideoCapture(frames)
                num_test_frames=60
                test_frames = []
                frame_count=0
                total_frames = int(videoTest.get(cv2.CAP_PROP_FRAME_COUNT))
                random_divisor = random.randint(2, 7)
                start_frame = (total_frames - num_test_frames) // random_divisor # Calculate the starting frame index
                print(f'lets take frame {start_frame} - {start_frame+num_test_frames} for optimization...')
                videoTest.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set the video capture to the starting frame
                for _ in range(num_test_frames):
                    okTest, imgTest = videoTest.read()
                    if okTest:
                        gray_frameTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
                        test_frames.append(gray_frameTest)
                        frame_count += 1
                        progress = frame_count / num_test_frames * 100  # Calculate the loading progress percentage
                        #sys.stdout.write('\r')
                        #sys.stdout.write(f'Loading test frames: {frame_count}/{num_test_frames} - {progress:.2f}% complete')
                        #sys.stdout.flush()

                optimal_mask_positions = []
                print('\nTest frames loaded')

                print('\nSearching for the optimal mask position')
                start_timeMask = time.time()
                # Find the optimal mask position for each LED based on luminosity differences
                # for i in range(nLEDs):
                #     luminosity_diff = np.zeros((2 * circleRadius + 1, 2 * circleRadius + 1))
                #     for x in range(int(posX[i]) - circleRadius, int(posX[i]) + circleRadius + 1):
                #         for y in range(int(posY[i]) - circleRadius, int(posY[i]) + circleRadius + 1):
                #             if 0 <= x < Vwidth and 0 <= y < Vheight:
                #                 maskCircle = np.sqrt((maskGrid[1] - x) ** 2 + (maskGrid[0] - y) ** 2) < circleRadius
                #                 LED_luminosity = np.array([np.mean(frame[maskCircle]) for frame in test_frames])
                #                 luminosity_diff[x - int(posX[i]) + circleRadius, y - int(posY[i]) + circleRadius] = np.max(LED_luminosity) - np.min(LED_luminosity)
                #
                #     # Find the pixel with the maximum luminosity difference
                #     best_pixel_idx = np.unravel_index(np.argmax(luminosity_diff), luminosity_diff.shape)
                #     best_x = best_pixel_idx[0] + int(posX[i]) - circleRadius
                #     best_y = best_pixel_idx[1] + int(posY[i]) - circleRadius
                #     optimal_mask_positions.append((best_x, best_y))
                # print('\noptimal mask position found')

                for i in range(nLEDs):
                    max_luminosity_diff = 0
                    optimal_x = None
                    optimal_y = None
                    compareFrameIdx = 12  # Use luminosity differences across 7 consecutive frames

                    for x in range(int(posX[i]) - circleRadius, int(posX[i]) + circleRadius + 1):
                        for y in range(int(posY[i]) - circleRadius, int(posY[i]) + circleRadius + 1):
                            elapsed_timeMask = time.time() - start_timeMask
                            #sys.stdout.write('\r')
                            #sys.stdout.write(
                            #    f'Elapsed Time: {elapsed_timeMask:.2f} s | {elapsed_timeMask / 110 * 100:.2f}% complete ')
                            #sys.stdout.flush()

                            if 0 <= x < Vwidth and 0 <= y < Vheight:
                                maskCircle = np.sqrt((maskGrid[1] - x) ** 2 + (maskGrid[0] - y) ** 2) < circleRadius

                                # Calculate luminosity differences across compareFrameIdx consecutive frames
                                luminosity_diff_frames = [
                                    np.abs(np.mean(frame[maskCircle]) - np.mean(prev_frame[maskCircle])) for
                                    frame, prev_frame in
                                    zip(test_frames[compareFrameIdx:], test_frames[:-compareFrameIdx])]

                                max_diff = max(luminosity_diff_frames)
                                if max_diff > max_luminosity_diff:
                                    max_luminosity_diff = max_diff
                                    optimal_x = x
                                    optimal_y = y

                    optimal_mask_positions.append((optimal_x, optimal_y))
                # Continue with processing frames using the optimal mask positions

                frame_count = 0
                start_time = time.time()
                print('\nextracting LED traces...')
                while True:
                    ok, img = video.read()
                    if ok:
                        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        LED_vals = getValuesBasedOnMask(gray_frame, [pos[0] for pos in optimal_mask_positions],
                                                        [pos[1] for pos in optimal_mask_positions], optimize)
                        frame_count += 1
                        #progress = (frame_count / Vlength * 100)  # Calculate the loading progress percentage
                        #elapsed_time = time.time() - start_time
                        # Calculate remaining time
                        #remaining_frames = Vlength - frame_count
                        #frames_per_second = frame_count / elapsed_time
                        #estimated_remaining_time = remaining_frames / frames_per_second
                        #sys.stdout.write('\r')
                        #sys.stdout.write(f'{videoId} | Loading  frames: {frame_count}/{Vlength} - {progress:.2f}% complete | Elapsed Time: {elapsed_time:.2f} s | Estimated Remaining Time: {estimated_remaining_time:.2f} s')
                        #sys.stdout.flush()
                        LEDtraces.append(LED_vals)
                        # Display the frame with optimal masks for one second
                        if frame_count==200:
                            for pos in optimal_mask_positions:
                                x, y = pos
                                maskCircle = np.sqrt((maskGrid[1] - x) ** 2 + (maskGrid[0] - y) ** 2) < circleRadius*0.3
                                img[maskCircle] = [0, 0,
                                                   255]  # Set mask area to red color (you can adjust the color as needed)

                            cv2.imshow("Optimal Mask Frame", img)
                            cv2.waitKey(5000)  # Display for 1 second
                            cv2.destroyAllWindows()


                        # ok, img = video.read()



                    else:
                        print('')
                        print('breaking at nFrame : ', frame_count)
                        break
                print('  extracted LED values for %s frames' % len(LEDtraces))
                # Convert the LEDtraces list to a NumPy array and transpose it

                video.release()
                print('LED traces extracted')
            else:
                ##################################################
                ok, img = video.read()
                LEDtraces.append(getValuesBasedOnMask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),posX,posY, optimize))
                nFrame = 1
                start_time = time.time()
                while True:
                    ok, img = video.read()
                    if ok:
                        LEDtraces.append(getValuesBasedOnMask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),posX,posY, optimize))
                        nFrame += 1
                        #progress = (nFrame / Vlength * 100)  # Calculate the loading progress percentage
                        #elapsed_time = time.time() - start_time
                        # Calculate remaining time
                        #remaining_frames = Vlength - nFrame
                        #frames_per_second = nFrame / elapsed_time
                        #estimated_remaining_time = remaining_frames / frames_per_second
                        #sys.stdout.write('\r')
                        #sys.stdout.write(f'{videoId} |Loading  frames: {nFrame}/{Vlength} - {progress:.2f}% complete | Elapsed Time: {elapsed_time:.2f} s | Estimated Remaining Time: {estimated_remaining_time:.2f} s')
                        #sys.stdout.flush()
                    else:
                        print('breaking at nFrame : ', nFrame)
                        break
                print('  extracted LED values for %s frames' % nFrame)
                if nFrame != Vlength:
                    print('Discrepancy : video length and extracted ROI values do not match!')
                    pdb.set_trace()
                video.release()
            LEDtraces = np.transpose(np.asarray(LEDtraces))
        print('LED traces extracted')
        return (LEDtraces)

    ############################################################
    def extractFramesFromVideoFile(self,fileName):
        video = cv2.VideoCapture(fileName)
        Vlength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        Vwidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        Vheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Vfps = video.get(cv2.CAP_PROP_FPS)
        # if outputProps:
        print('  Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (Vlength, Vwidth, Vheight, Vfps))
        #maskGrid = np.indices((Vheight, Vwidth))
        # read first video frame
        frames = np.empty((Vlength,Vheight,Vwidth),dtype=np.uint8)
        ok, img = video.read()
        frames[0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #LEDtraces.append(getValuesBasedOnMask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), posX, posY))
        nFrame = 1
        while True:
            ok, img = video.read()
            if ok:
                frames[nFrame]=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                nFrame += 1
            else:
                print('breaking at nFrame : ', nFrame)
                break
        print('  generated numpy array for %s frames' % nFrame)
        if nFrame != Vlength:
            print('Discrepancy : video length and extracted ROI values do not match!')
            pdb.set_trace()
        video.release()

        return frames


    ############################################################
    def cropImg(self, img,Ycoordinates=None):

        if Ycoordinates is None:
            leftY = 215
            rightY = 215
            Npix = 5
            continueLoop = True
            # optimize with keyboard
            while continueLoop:
                #rungs = []
                imgLine = img.copy()
                cv2.line(imgLine, (0, leftY), (self.Vwidth, rightY), (255, 0, 0), 2)

                cv2.imshow("Rungs", imgLine)
                #print 'current xPosition, yPostion : ', xPosition, yPosition
                PressedKey = cv2.waitKey(0)
                if PressedKey == 56 or PressedKey ==82: #UP arrow
                    leftY -= Npix
                elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                    leftY += Npix
                elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                    rightY += Npix
                elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                    rightY -= Npix
                elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                    continueLoop = False
                elif PressedKey == 27: # Escape
                    continueLoop = False
                else:
                    pass
                cv2.destroyWindow("Rungs")
        else:
            leftY = Ycoordinates[0]
            rightY = Ycoordinates[1]

        print('Left, right Y :', leftY,rightY)
        mask = np.zeros((self.Vheight, self.Vwidth))

        # create masks for mouse area and for lower area
        maskGrid = ((self.imgGrid[1] - 0.)*(rightY-leftY) - (self.imgGrid[0] -leftY)*(self.Vwidth - 0)) < 0
        #maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        mask[maskGrid] = 1
        #wheelMaskInv[maskInv] = 1
        mask = np.array(mask, dtype=np.uint8)
        #wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)
        return (mask)

    ############################################################
    def getIntensityArray(self, image, yPos, loc, barWidth,recStruc,recBatch):
        nPoints = 100
        nDiffConv = 10
        intensity = np.log(np.average(image[yPos:(yPos + barWidth)], 0))
        differenceInt = np.abs(np.diff(intensity))
        differenceIntConv = np.convolve(differenceInt, np.ones(nDiffConv) / nDiffConv, mode='same')
        intensityConv = np.convolve(intensity, np.ones(nPoints) / nPoints, mode='same')
        intensityAvgSub = np.abs(intensity - intensityConv) + 5 * np.concatenate((np.array([0]), differenceIntConv))  # + np.max(intensity)
        # remove largest peak from array
        # idxPeak = np.argmax(intensityAvgSub)
        # intensityAvgSub[idxPeak-10:idxPeak+10]=0.
        if (recStruc == 'simplexBehavior') and (recBatch is None):
            if loc == 'above':
                intensityAvgSub[173:272] = 0
                intensityAvgSub[272:317] = 0
            elif loc == 'below':  # because of the two vertical bar of the side-wall holder
                intensityAvgSub[176:230] = 0
                intensityAvgSub[273:322] = 0

        intensityAvgSub[:43] = 0
        intensityAvgSub[-43:] = 0
        return intensityAvgSub

    ############################################################
    def trackRungs(self, mouse, date, rec, videoIdx, timeArray, obstacle=None, defineROI=False,recStruc=None,recBatch=None, resize=False):

        def findClosestIdxToTime(newTime, recordedTime):
            indicies = []
            for z in range(len(newTime)):
                indicies.append(np.argmin(np.abs(recordedTime - newTime[z])))
            return np.asarray(indicies)
        # if obstacle[0] is not 'normal':
        if obstacle[0]=='animal' or obstacle[0]=='animalObstacle':
            obstacleUPdic=obstacle[1]
            obsZeroMask1=((obstacleUPdic['obstacle1UPAndActivated'][1][:,0]>timeArray[0]) & (obstacleUPdic['obstacle1UPAndActivated'][1][:,0]<timeArray[-1]))
            obsZeroMask2=((obstacleUPdic['obstacle2UPAndActivated'][1][:,0]>timeArray[0]) & (obstacleUPdic['obstacle2UPAndActivated'][1][:,0]<timeArray[-1]))


            obsZeroTime1 = obstacleUPdic['obstacle1UPAndActivated'][1][:,0][obsZeroMask1]
            obsZeroTime2=obstacleUPdic['obstacle2UPAndActivated'][1][:,0][obsZeroMask2]


            obsZeroIdx1=findClosestIdxToTime(obsZeroTime1,timeArray)
            obsZeroIdx2 = findClosestIdxToTime(obsZeroTime2,timeArray )
            obsIdx=[obsZeroIdx1, obsZeroIdx2]
            # pdb.set_trace()
            obsRung=544
            obsId=0

        shifts = 110
        if recStruc is None:
            barPositionAbove = 325#255#270
            barPositionBelow = 510# 475#565
        elif recStruc == 'simplexBehavior':
            barPositionAbove = 250
            barPositionBelow = 480
        elif recStruc == 'simplexNew':
            barPositionAbove = 300
            barPositionBelow = 420
        else:
            print('Verify the position of the bar ROIs in the data-set under investigation!')
        showVideo = False
        displayOptimization = False
        badVideo = 0
        stopProgram = False
        jumpThresholdAbove =150
        jumpThresholdBelow = 160
        # tracking parameters #########################
        self.thresholdValue = 0.7  # in %
        self.minContourArea = 40  # square pixels

        self.scoreWeights = {'distanceWeight': 5, 'areaWeight': 1}
        ###############################################
        rec = rec.replace('/', '-')
        if obstacle[0]=='animal' or obstacle[0]=='normal':
            videoFileName = self.analysisLocation + '%s_%s_%s_processed-animal-video.avi' % (mouse, date, rec)
        elif obstacle[0]=='obstacle':
            videoFileName = self.analysisLocation + '%s_%s_%s_processed-animalObstacle-video.avi' % (mouse, date, rec)
        else:
            videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        (ok,firstImg) = self.openVideo(videoFileName)
        videoLen=self.Vlength
        firstImgGray = cv2.cvtColor(firstImg, cv2.COLOR_BGR2GRAY)
        print('image dims :', np.shape(firstImgGray))
        # create video output streams
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

        if obstacle[0] is not None:
            videoOutputName=self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec)
            self.outRung = cv2.VideoWriter(videoOutputName, fourcc, 40.0, (self.Vwidth, self.Vheight),0)
        else:
            videoOutputName = self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec)
            self.outRung = cv2.VideoWriter(videoOutputName,
                                           fourcc, 40.0, (self.Vwidth, self.Vheight),0)

        # Find horizontal areas for paw position extraction ###
        if defineROI:
            (yPosAbove,barWidthAbove) = self.findHorizontalArea(firstImgGray,[barPositionAbove,30],orientation='horizontal')
            (yPosBelow,barWidthBelow) = self.findHorizontalArea(firstImgGray,[barPositionBelow,30],orientation='horizontal')
        else:
            (yPosAbove,barWidthAbove) = [barPositionAbove,30]
            (yPosBelow,barWidthBelow) = [barPositionBelow,30]
        #(xPos,barVerticalWidth) = self.findHorizontalArea(firstImgGray,[285,340],orientation='vertical')

        if not self.testBelowBarGenerated:
            (self.testArrayBelow, self.midBarArrayBelow,self.nRungs) = self.generateTestBarArray(yPosBelow, barWidthBelow, 'below', shifts, recordingStruc=recStruc,recordingBatch=recBatch)
            # print('below test array generated')
            self.testBelowBarGenerated = True
        testArrBelow = self.testArrayBelow
        midBarsBelow = self.midBarArrayBelow
        numberOfRungsBelow = self.nRungs

        # elif loc == 'above':
        if not self.testAboveBarGenerated:
            (self.testArrayAbove, self.midBarArrayAbove,self.nRungs) = self.generateTestBarArray(yPosAbove, barWidthAbove, 'above', shifts, recordingStruc=recStruc,recordingBatch=recBatch)
            # print('above test array generated')
            self.testAboveBarGenerated = True
        testArrAbove = self.testArrayAbove
        midBarsAbove = self.midBarArrayAbove
        numberOfRungsAbove = self.nRungs

        #above = (np.average(firstImgGray[yPosAbove:(yPosAbove+barWidthAbove)],0))
        #below = (np.average(firstImgGray[yPosBelow:(yPosBelow + barWidthBelow)], 0))
        #np.save('above.npy',above)
        #np.save('below.npy',below)self
        #plt.show()
        #pdb.set_trace()
        nImg = 0
        #saveImgIdx = [0, 1000, 2000, 3000, 4000, 5000]
        rungPositions = []
        diffs=[]
        alignResults = []
        rungIdx = 0
        pixelsMovedTotal = 0
        MaxDiffAbove = 0
        MaxDiffBelow = 0
        jumpDisplayOptimization = False
        start_time = time.time()
        while True:
            nFrame=nImg+1
            remaining_frames = videoLen - nFrame
            progress = (nFrame / videoLen * 100)  # Calculate the loading progress percentage
            elapsed_time = time.time() - start_time
            frames_per_second = nFrame / elapsed_time
            estimated_remaining_time = remaining_frames / frames_per_second
            sys.stdout.write('\r')
            sys.stdout.write(
                f' {mouse}, {date}, {rec}, writing  frames: {nFrame}/{videoLen} - {progress:.2f}% complete | Elapsed Time: {elapsed_time:.2f} s | Estimated Remaining Time: {estimated_remaining_time:.2f} s')
            sys.stdout.flush()
            # if not (nImg%100):
            #     print(nImg)
            if nImg == 0:
                img = firstImg.copy()
            else:
                ok, img = self.video.read()
            if not ok:
                break
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe =cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imgGray=clahe.apply(imgGray)
            # if nImg % 500 == 0:
            #     cv2.imshow('Adjusted Image', imgGray)
            #     cv2.waitKey(2000)  # Wait for 2000 milliseconds (2 seconds)
            #     cv2.destroyAllWindows()
            intensityAvgSubBelow = self.getIntensityArray(imgGray, yPosBelow, 'below',barWidthBelow,recStruc,recBatch)
            intensityAvgSubAbove = self.getIntensityArray(imgGray, yPosAbove, 'above',barWidthAbove,recStruc,recBatch)
            #print(np.mean(intensityAvgSubBelow),np.mean(intensityAvgSubAbove))
            shiftResults = []
            bestDiffSumAbove = None
            bestDiffSumBelow = None
            for i in range(shifts):
                # (testArr,midBars) = generateBarArrayForTest(i)
                #nBarsA = numberOfRungsAbove[i]
                #nBarsB = numberOfRungsBelow[i]
                diffSumAbove = np.sum(((intensityAvgSubAbove - testArrAbove[i] * max(intensityAvgSubAbove)) ** 2))/numberOfRungsAbove[i]
                diffSumBelow = np.sum(((intensityAvgSubBelow - testArrBelow[i] * max(intensityAvgSubBelow)) ** 2))/numberOfRungsBelow[i]
                if bestDiffSumAbove is None or diffSumAbove < bestDiffSumAbove:
                    barLocsAbove = midBarsAbove[i]
                    bestShiftAbove = i
                    #bestShiftAbodisplayOptimizationve = i
                    bestDiffSumAbove = diffSumAbove
                    # bestOffset = i
                    bestBarArrayAbove = testArrAbove[i]
                #for i in range(bestShiftAbove - 30, ((bestShiftAbove + 30) if (bestShiftAbove + 30) < 100 else 99)):
                if (recStruc == 'simplexBehavior') and (recBatch is None): # use fix shift, basically used the above position for below as well
                    #diffSumBelow = np.sum(((intensityAvgSubBelow - testArrBelow[i] * max(intensityAvgSubBelow)) ** 2))
                    #if bestDiffSumBelow is None or diffSumBelow < bestDiffSumBelow:
                    barLocsBelow = midBarsBelow[bestShiftAbove-13]
                    bestShiftBelow = bestShiftAbove-13 #i
                    bestDiffSumBelow = diffSumBelow
                    # bestOffset = i
                    bestBarArrayBelow = testArrBelow[bestShiftAbove-13]
                else:
                    if bestDiffSumBelow is None or diffSumBelow < bestDiffSumBelow:
                        barLocsBelow = midBarsBelow[i]
                        bestShiftBelow = i #i
                        bestDiffSumBelow = diffSumBelow
                        # bestOffset = i
                        bestBarArrayBelow = testArrBelow[i]

                shiftResults.append([i, testArrAbove[i], testArrBelow[i], intensityAvgSubAbove, intensityAvgSubBelow, diffSumAbove, diffSumBelow, diffSumBelow, midBarsAbove[i], midBarsBelow[i]])
            #print(bestShiftAbove,bestShiftBelow,(bestShiftAbove-bestShiftBelow),barLocsAbove,barLocsBelow)
            alignResults.append([nImg,bestShiftAbove,bestShiftBelow,(bestShiftAbove-bestShiftBelow)])

            barAbove = np.asarray(barLocsAbove)
            barBelow = np.asarray(barLocsBelow)

            # pdb.set_trace()
            # print()
            #(barAbove,barBelow,bestShiftAbove,bestShiftBelow) = self.findBarsDiffSumBoth(imgGray, yPosAbove, yPosBelow, barWidthBelow,rungPositions,recStr=recStruc)

            # align both detected bar arrays
            midPointAbove = int(len(barAbove)/2.)
            midPointBelow = np.argmin(abs(barBelow-barAbove[midPointAbove]))
            # bring both arrays to the same length
            if midPointAbove < midPointBelow:
                barBelow = barBelow[(midPointBelow-midPointAbove):]
            elif midPointAbove > midPointBelow:
                barAbove = barAbove[(midPointAbove-midPointBelow):]
            barAbove = barAbove[:min(len(barAbove),len(barBelow))]
            barBelow = barBelow[:min(len(barAbove),len(barBelow))]
            barLocs = np.column_stack((barAbove,np.repeat(int(yPosAbove+barWidthAbove/2.),len(barAbove)),barBelow,np.repeat(int(yPosBelow + barWidthBelow/2.),len(barAbove))))
            #for i in range(len(barAbove)):
            #    barLocs = .append([barAbove[i],int(yPosAbove+barWidthAbove/2.),barBelow[i],int(yPosBelow + barWidthBelow/2.)])
            # save extracted rung positions
            # rung movement is based on the bar detection above
            if nImg == 0:
                barLocsOld = barLocs[0,0]
            pixelDifference = barLocs[0,0]-barLocsOld
            pixelsMovedTotal -= pixelDifference
            if pixelDifference < -20.:  # a bar moved into the frame from the left
                rungIdx -=1
                pixelsMovedTotal -= (barAbove[1] - barAbove[0])
            elif pixelDifference > 20.: # a bar left the frame on the left corner
                rungIdx +=1
                pixelsMovedTotal += (barAbove[1] - barAbove[0])
            #dD.append(degreeDifference[0])
            #rungsNumbered.append([i,frameNumbers[i],len(ppF),d1,degreeDifference[0],rungCounter,numberedR,ppF[:,:2]])
            rungIdentity = np.arange(rungIdx,rungIdx+len(barAbove))
            # if obstacle[0] is not 'normal':
            if obstacle[0] == 'animal' or obstacle[0] =='animalObstacle':
                obsIdentity = np.zeros(len(rungIdentity))
                #rungID =

                for ob in range(2):
                    if len(obsIdx[ob])>0:
                        obs=len(rungIdentity)-1
                        for o1 in range(len(obsIdx[ob])):
                            if nImg==obsIdx[ob][o1]:
                                obsRung=rungIdentity[-1]
                                if ob==0:
                                    obsId=1
                                else:
                                    obsId=2
                    obsIndex=np.where(rungIdentity==obsRung)[0]
                    obsIdentity[obsIndex]=obsId

                # print('frame', nImg, 'rungIdentity',rungIdentity,'rungIdx',rungIdx,'barAbove',barAbove)
                # print('obsIdentity',obsIdentity)
                rungPositions.append([nImg, len(rungIdentity), rungIdentity, barLocs, pixelDifference, pixelsMovedTotal, obsIdentity])
            else:
                rungPositions.append([nImg, len(rungIdentity), rungIdentity, barLocs, pixelDifference, pixelsMovedTotal])

                            # obsIdentity[obs]=1
                            # obs-=rungIdx



            # if nImg==400:
            #     pdb.set_trace()

            # print('obstacle id', obsIdentity)
            # check if large jump happend
            if len(rungPositions)>1:
                #print(rungPositions[-1],rungPositions[-2])
                (_,intersectIdxLast,intersectIdxSecondtoLast) = np.intersect1d(rungPositions[-1][2],rungPositions[-2][2], return_indices=True)
                #print(intersectIdxLast,intersectIdxSecondtoLast)
                diffAbove = np.sum(np.abs(rungPositions[-1][3][intersectIdxLast][:,0]-rungPositions[-2][3][intersectIdxSecondtoLast][:,0]))/(videoIdx[nImg]-videoIdx[nImg-1])
                diffBelow = np.sum(np.abs(rungPositions[-1][3][intersectIdxLast][:,2]-rungPositions[-2][3][intersectIdxSecondtoLast][:,2]))/(videoIdx[nImg]-videoIdx[nImg-1])
                #print(MaxDiffAbove,diffAbove,MaxDiffBelow,diffBelow)
                if diffAbove>MaxDiffAbove:
                    MaxDiffAbove=diffAbove
                if diffBelow>MaxDiffBelow:
                    MaxDiffBelow = diffBelow
                if diffAbove>jumpThresholdAbove:
                    #print('jump above')
                    jumpDisplayOptimization = False
                    print('difference in movement above : ',diffAbove, ' at frame ', nImg)
                    #print(rungPositions[-1],rungPositions[-2],intersectIdxLast,intersectIdxSecondtoLast)
                    #pdb.set_trace()
                else:
                    jumpDisplayOptimization=False
                if diffBelow > jumpThresholdBelow:
                    #print('jump below')
                    jumpDisplayOptimization = False
                    print('difference in movement below : ', diffBelow, ' at frame ',nImg)
                    #print(rungPositions[-1], rungPositions[-2], intersectIdxLast, intersectIdxSecondtoLast)
                    #pdb.set_trace()
                else:
                    jumpDisplayOptimization = False
                diffs.append([nImg,diffAbove,diffBelow])
                rungPositions[-1].append([diffAbove,diffBelow])
            #pdb.set_trace()
            #print(nImg,rungIdentity,pixelDifference,pixelsMovedTotal,barLocs[0,0])
            barLocsOld = barLocs[0,0]
            #pdb.set_trace()
            if (displayOptimization and (nImg % 100 == 0) or jumpDisplayOptimization ):# or (nImg>5478):
                mp.use('WxAgg')
                fig = plt.figure()
                ax0 = fig.add_subplot(2, 2, 1)
                # ax0.plot(intensity)
                ax0.set_title('above')
                ax0.plot(intensityAvgSubAbove)
                # ax0.plot(intensityConv)
                ax0.plot(shiftResults[bestShiftAbove - 2][1] * max(intensityAvgSubAbove), lw=0.5, c='green')
                ax0.plot(bestBarArrayAbove * max(intensityAvgSubAbove))
                ax0.plot(shiftResults[bestShiftAbove + 2][1] * max(intensityAvgSubAbove), lw=0.5, c='red')

                ax1 = fig.add_subplot(2, 2, 2)
                for i in range(shifts):
                    ax1.plot(shiftResults[i][0], shiftResults[i][5], 'o', c='C0')
                ax1.plot(shiftResults[bestShiftAbove][0], shiftResults[bestShiftAbove][5], 'o', c='C1')

                ax2 = fig.add_subplot(2, 2, 3)
                ax0.set_title('below')
                # ax0.plot(intensity)
                ax2.plot(intensityAvgSubBelow)
                # ax0.plot(intensityConv)
                ax2.plot(bestBarArrayBelow * max(intensityAvgSubBelow))

                ax3 = fig.add_subplot(2, 2, 4)
                for i in range(shifts):
                    ax3.plot(shiftResults[i][0], shiftResults[i][6], 'o', c='C0')
                ax3.plot(shiftResults[bestShiftBelow][0], shiftResults[bestShiftBelow][6],'o',c='C1')

                plt.show()
                #pdb.set_trace()

            # generate video where rungs are marked by lines
            img2= img.copy()
            # L, A, B = cv2.split(imgRungs)
            imgRungsGray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            clahe =cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imgRungs=clahe.apply(imgRungsGray)

            for i in range(len(barAbove)):
                # if obstacle[0] != 'normal':
                if obstacle [0]=='animal' or obstacle[0]=='animalObstacle':
                    rungIdP = obsIdentity[i]
                else:
                    rungIdP = np.zeros((len(rungIdentity)))[i]
                rungId = int(rungIdentity[i])
                if rungIdP == 1:
                    color = 200  # Light gray for rungIdP == 1
                elif rungIdP == 2:
                    color = 100  # Dark gray for rungIdP == 2
                else:
                    color = 250  # Medium gray for other cases

                cv2.line(imgRungs, (int(barAbove[i]), int(yPosAbove + barWidthAbove / 2.)),
                         (int(barBelow[i]), int(yPosBelow + barWidthBelow / 2.)), color, 2)
                cv2.putText(imgRungs, str(rungId), (int(barBelow[i]), int(yPosBelow + barWidthBelow / 2.) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            #time.sleep(0.1)
            if showVideo:
                #print('in display')
                cv2.imshow('Rungs', imgRungs)
                k = cv2.waitKey(2) & 0xff
                if k == 27: break
            #PressedKey = cv2.waitKey(0)
            #cv2.destroyWindow('Rungs')
            #imgRungsColor = cv2.cvtColor(imgRungs, cv2.COLOR_GRAY2BGR)
            self.outRung.write(imgRungs)
            nImg+=1


        self.outRung.release()
        # cv2.destroyAllWindows()
        # pickle.dump(rungPositions, open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb'))
        return (rungPositions,diffs,np.asarray(alignResults))
    def trackRungsObstacle(self, mouse, date, rec, videoIdx, timeArray,angleTimes,obstacleUPdic,obstacleVideo=False, defineROI=False,recStruc='simplexNPX',recBatch=None, resize=False):
        # obsZeroIdx=[[],[]]
        #
        # obsAtZeroIdx = np.where(obsAngleArr< 0)[1][0]
        # zeroIdx1=np.where(obsAngleArr[:, obsAtZeroIdx] < 0)[0][-1]
        # zeroAngle1=obsAngleArr[zeroIdx1,obsAtZeroIdx]
        # if obsAtZeroIdx==0:
        #     oppObsIx=1
        # else:
        #     oppObsIx = 0
        # while zeroAngle1< obsAngleArr[-1,obsAtZeroIdx]:
        #
        #     obsZeroIdx[obsAtZeroIdx].append(np.where(obsAngleArr[:,obsAtZeroIdx] < zeroAngle1)[0][-1])
        #     zeroAngle1 += 360
        #
        # obsZeroTime=[timeArray[obsZeroIdx[0]],timeArray[obsZeroIdx[1]]]
        # pdb.set_trace()
        def findClosestIdxToTime(newTime, recordedTime):
            indicies = []
            for z in range(len(newTime)):
                indicies.append(np.argmin(np.abs(recordedTime - newTime[z])))
            return np.asarray(indicies)

        obsZeroMask1=((obstacleUPdic['obstacle1UPAndActivated'][1][:,0]>timeArray[0]) & (obstacleUPdic['obstacle1UPAndActivated'][1][:,0]<timeArray[-1]))
        obsZeroMask2=((obstacleUPdic['obstacle2UPAndActivated'][1][:,0]>timeArray[0]) & (obstacleUPdic['obstacle2UPAndActivated'][1][:,0]<timeArray[-1]))
        obsZeroTime1 = obstacleUPdic['obstacle1UPAndActivated'][1][:,0][obsZeroMask1]
        obsZeroTime2=obstacleUPdic['obstacle2UPAndActivated'][1][:,0][obsZeroMask2]
        obsZeroIdx1=findClosestIdxToTime(obsZeroTime1,timeArray)
        obsZeroIdx2 = findClosestIdxToTime(obsZeroTime2,timeArray )
        obsIdx=[obsZeroIdx1, obsZeroIdx2]
        shifts = 110
        if recStruc is None:
            barPositionAbove = 325#255#270
            barPositionBelow = 510# 475#565
        elif recStruc == 'simplexBehavior':
            barPositionAbove = 250
            barPositionBelow = 480
        elif recStruc == 'simplexNew':
            barPositionAbove = 255
            barPositionBelow = 475
        elif recStruc == 'simplexNPX':
            barPositionAbove = 350
            barPositionBelow = 550
        else:
            print('Verify the position of the bar ROIs in the data-set under investigation!')
        showVideo = False
        displayOptimization = False
        badVideo = 0
        stopProgram = False
        jumpThresholdAbove =150
        jumpThresholdBelow = 160
        # tracking parameters #########################
        self.thresholdValue = 0.7  # in %
        self.minContourArea = 40  # square pixels

        self.scoreWeights = {'distanceWeight': 5, 'areaWeight': 1}
        ###############################################
        rec = rec.replace('/', '-')
        if obstacleVideo:
            videoFileName = self.analysisLocation + '%s_%s_%s_processed-animalObstacle-video.avi' % (mouse, date, rec)
        else:
            videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        (ok,firstImg) = self.openVideo(videoFileName)
        firstImgGray = cv2.cvtColor(firstImg, cv2.COLOR_BGR2GRAY)
        print('image dims :', np.shape(firstImgGray))
        # create video output streams
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec), fourcc, 40.0, (self.Vwidth, self.Vheight))
        if obstacleVideo:
            self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_Obstacle_rungTracking.avi' % (mouse, date, rec), fourcc, 40.0, (self.Vwidth, self.Vheight))
        else:
            self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec),
                                           fourcc, 40.0, (self.Vwidth, self.Vheight))

        # Find horizontal areas for paw position extraction ###
        if defineROI:
            (yPosAbove,barWidthAbove) = self.findHorizontalArea(firstImgGray,[barPositionAbove,30],orientation='horizontal')
            (yPosBelow,barWidthBelow) = self.findHorizontalArea(firstImgGray,[barPositionBelow,30],orientation='horizontal')
        else:
            (yPosAbove,barWidthAbove) = [barPositionAbove,30]
            (yPosBelow,barWidthBelow) = [barPositionBelow,30]
        #(xPos,barVerticalWidth) = self.findHorizontalArea(firstImgGray,[285,340],orientation='vertical')

        if not self.testBelowBarGenerated:
            (self.testArrayBelow, self.midBarArrayBelow,self.nRungs) = self.generateTestBarArray(yPosBelow, barWidthBelow, 'below', shifts, recordingStruc=recStruc,recordingBatch=recBatch)
            # print('below test array generated')
            self.testBelowBarGenerated = True
        testArrBelow = self.testArrayBelow
        midBarsBelow = self.midBarArrayBelow
        numberOfRungsBelow = self.nRungs

        # elif loc == 'above':
        if not self.testAboveBarGenerated:
            (self.testArrayAbove, self.midBarArrayAbove,self.nRungs) = self.generateTestBarArray(yPosAbove, barWidthAbove, 'above', shifts, recordingStruc=recStruc,recordingBatch=recBatch)
            # print('above test array generated')
            self.testAboveBarGenerated = True
        testArrAbove = self.testArrayAbove
        midBarsAbove = self.midBarArrayAbove
        numberOfRungsAbove = self.nRungs

        #above = (np.average(firstImgGray[yPosAbove:(yPosAbove+barWidthAbove)],0))
        #below = (np.average(firstImgGray[yPosBelow:(yPosBelow + barWidthBelow)], 0))
        #np.save('above.npy',above)
        #np.save('below.npy',below)self


        nImg = 0
        #saveImgIdx = [0, 1000, 2000, 3000, 4000, 5000]
        rungPositions = []
        diffs=[]
        alignResults = []
        rungIdx = 0
        pixelsMovedTotal = 0
        MaxDiffAbove = 0
        MaxDiffBelow = 0
        jumpDisplayOptimization = False
        while True:
            if not (nImg%100):
                print(nImg)
            if nImg == 0:
                img = firstImg.copy()
            else:
                ok, img = self.video.read()
            if not ok:
                break
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            intensityAvgSubBelow = self.getIntensityArray(imgGray, yPosBelow, 'below',barWidthBelow,recStruc,recBatch)
            intensityAvgSubAbove = self.getIntensityArray(imgGray, yPosAbove, 'above',barWidthAbove,recStruc,recBatch)
            #print(np.mean(intensityAvgSubBelow),np.mean(intensityAvgSubAbove))
            shiftResults = []
            bestDiffSumAbove = None
            bestDiffSumBelow = None
            for i in range(shifts):
                # (testArr,midBars) = generateBarArrayForTest(i)
                #nBarsA = numberOfRungsAbove[i]
                #nBarsB = numberOfRungsBelow[i]
                diffSumAbove = np.sum(((intensityAvgSubAbove - testArrAbove[i] * max(intensityAvgSubAbove)) ** 2))/numberOfRungsAbove[i]
                diffSumBelow = np.sum(((intensityAvgSubBelow - testArrBelow[i] * max(intensityAvgSubBelow)) ** 2))/numberOfRungsBelow[i]
                if bestDiffSumAbove is None or diffSumAbove < bestDiffSumAbove:
                    barLocsAbove = midBarsAbove[i]
                    bestShiftAbove = i
                    #bestShiftAbodisplayOptimizationve = i
                    bestDiffSumAbove = diffSumAbove
                    # bestOffset = i
                    bestBarArrayAbove = testArrAbove[i]
                #for i in range(bestShiftAbove - 30, ((bestShiftAbove + 30) if (bestShiftAbove + 30) < 100 else 99)):
                if (recStruc == 'simplexBehavior') and (recBatch is None): # use fix shift, basically used the above position for below as well
                    #diffSumBelow = np.sum(((intensityAvgSubBelow - testArrBelow[i] * max(intensityAvgSubBelow)) ** 2))
                    #if bestDiffSumBelow is None or diffSumBelow < bestDiffSumBelow:
                    barLocsBelow = midBarsBelow[bestShiftAbove-13]
                    bestShiftBelow = bestShiftAbove-13 #i
                    bestDiffSumBelow = diffSumBelow
                    # bestOffset = i
                    bestBarArrayBelow = testArrBelow[bestShiftAbove-13]
                else:
                    if bestDiffSumBelow is None or diffSumBelow < bestDiffSumBelow:
                        barLocsBelow = midBarsBelow[i]
                        bestShiftBelow = i #i
                        bestDiffSumBelow = diffSumBelow
                        # bestOffset = i
                        bestBarArrayBelow = testArrBelow[i]

                shiftResults.append([i, testArrAbove[i], testArrBelow[i], intensityAvgSubAbove, intensityAvgSubBelow, diffSumAbove, diffSumBelow, diffSumBelow, midBarsAbove[i], midBarsBelow[i]])
            #print(bestShiftAbove,bestShiftBelow,(bestShiftAbove-bestShiftBelow),barLocsAbove,barLocsBelow)
            alignResults.append([nImg,bestShiftAbove,bestShiftBelow,(bestShiftAbove-bestShiftBelow)])


            # print()
            #(barAbove,barBelow,bestShiftAbove,bestShiftBelow) = self.findBarsDiffSumBoth(imgGray, yPosAbove, yPosBelow, barWidthBelow,rungPositions,recStr=recStruc)

            # align both detected bar arrays
            midPointAbove = int(len(barAbove)/2.)
            midPointBelow = np.argmin(abs(barBelow-barAbove[midPointAbove]))
            # bring both arrays to the same length
            if midPointAbove < midPointBelow:
                barBelow = barBelow[(midPointBelow-midPointAbove):]
            elif midPointAbove > midPointBelow:
                barAbove = barAbove[(midPointAbove-midPointBelow):]
            barAbove = barAbove[:min(len(barAbove),len(barBelow))]
            barBelow = barBelow[:min(len(barAbove),len(barBelow))]
            barLocs = np.column_stack((barAbove,np.repeat(int(yPosAbove+barWidthAbove/2.),len(barAbove)),barBelow,np.repeat(int(yPosBelow + barWidthBelow/2.),len(barAbove))))
            #for i in range(len(barAbove)):
            #    barLocs = .append([barAbove[i],int(yPosAbove+barWidthAbove/2.),barBelow[i],int(yPosBelow + barWidthBelow/2.)])
            # save extracted rung positions
            # rung movement is based on the bar detection above
            if nImg == 0:
                barLocsOld = barLocs[0,0]
            pixelDifference = barLocs[0,0]-barLocsOld
            pixelsMovedTotal -= pixelDifference
            if pixelDifference < -20.:  # a bar moved into the frame from the left
                rungIdx -=1
                pixelsMovedTotal -= (barAbove[1] - barAbove[0])
            elif pixelDifference > 20.: # a bar left the frame on the left corner
                rungIdx +=1
                pixelsMovedTotal += (barAbove[1] - barAbove[0])
            #dD.append(degreeDifference[0])

            #rungsNumbered.append([i,frameNumbers[i],len(ppF),d1,degreeDifference[0],rungCounter,numberedR,ppF[:,:2]])
            rungIdentity = np.arange(rungIdx,rungIdx+len(barAbove))
            obsIdentity=rungIdentity
            if len(obsIdx[0])>0:
                obs=len(rungIdentity)
                for o1 in range(len(obsIdx[0])):
                    if (nImg==obsIdx[0][o1]) and (nImg<=nImg+len(rungIdentity)):
                        #for id in range(len(rungIdentity)):
                        obsIdentity= np.zeros(len(rungIdentity))
                        obsIdentity[obs]=1
                        obs-=rungIdx
            print('obstacle id',obsIdentity)
            rungPositions.append([nImg,len(rungIdentity),rungIdentity,barLocs,pixelDifference,pixelsMovedTotal, obsIdentity])

            barAbove = np.asarray(barLocsAbove)
            barBelow = np.asarray(barLocsBelow)



            ###################################################################################################




            # check if large jump happend
            if len(rungPositions)>1:
                #print(rungPositions[-1],rungPositions[-2])
                (_,intersectIdxLast,intersectIdxSecondtoLast) = np.intersect1d(rungPositions[-1][2],rungPositions[-2][2], return_indices=True)
                #print(intersectIdxLast,intersectIdxSecondtoLast)
                diffAbove = np.sum(np.abs(rungPositions[-1][3][intersectIdxLast][:,0]-rungPositions[-2][3][intersectIdxSecondtoLast][:,0]))/(videoIdx[nImg]-videoIdx[nImg-1])
                diffBelow = np.sum(np.abs(rungPositions[-1][3][intersectIdxLast][:,2]-rungPositions[-2][3][intersectIdxSecondtoLast][:,2]))/(videoIdx[nImg]-videoIdx[nImg-1])
                #print(MaxDiffAbove,diffAbove,MaxDiffBelow,diffBelow)
                if diffAbove>MaxDiffAbove:
                    MaxDiffAbove=diffAbove
                if diffBelow>MaxDiffBelow:
                    MaxDiffBelow = diffBelow
                if diffAbove>jumpThresholdAbove:
                    #print('jump above')
                    jumpDisplayOptimization = False
                    print('difference in movement above : ',diffAbove, ' at frame ', nImg)
                    #print(rungPositions[-1],rungPositions[-2],intersectIdxLast,intersectIdxSecondtoLast)
                    #pdb.set_trace()
                else:
                    jumpDisplayOptimization=False
                if diffBelow > jumpThresholdBelow:
                    #print('jump below')
                    jumpDisplayOptimization = False
                    print('difference in movement below : ', diffBelow, ' at frame ',nImg)
                    #print(rungPositions[-1], rungPositions[-2], intersectIdxLast, intersectIdxSecondtoLast)
                    #pdb.set_trace()
                else:
                    jumpDisplayOptimization = False
                diffs.append([nImg,diffAbove,diffBelow])
                rungPositions[-1].append([diffAbove,diffBelow])
            #pdb.set_trace()
            #print(nImg,rungIdentity,pixelDifference,pixelsMovedTotal,barLocs[0,0])
            barLocsOld = barLocs[0,0]
            #pdb.set_trace()
            if displayOptimization or jumpDisplayOptimization:# or (nImg>5478):
                mp.use('WxAgg')
                fig = plt.figure()
                ax0 = fig.add_subplot(2, 2, 1)
                # ax0.plot(intensity)
                ax0.set_title('above')
                ax0.plot(intensityAvgSubAbove)
                # ax0.plot(intensityConv)
                ax0.plot(shiftResults[bestShiftAbove - 2][1] * max(intensityAvgSubAbove), lw=0.5, c='green')
                ax0.plot(bestBarArrayAbove * max(intensityAvgSubAbove))
                ax0.plot(shiftResults[bestShiftAbove + 2][1] * max(intensityAvgSubAbove), lw=0.5, c='red')

                ax1 = fig.add_subplot(2, 2, 2)
                for i in range(shifts):
                    ax1.plot(shiftResults[i][0], shiftResults[i][5], 'o', c='C0')
                ax1.plot(shiftResults[bestShiftAbove][0], shiftResults[bestShiftAbove][5], 'o', c='C1')

                ax2 = fig.add_subplot(2, 2, 3)
                ax0.set_title('below')
                # ax0.plot(intensity)
                ax2.plot(intensityAvgSubBelow)
                # ax0.plot(intensityConv)
                ax2.plot(bestBarArrayBelow * max(intensityAvgSubBelow))

                ax3 = fig.add_subplot(2, 2, 4)
                for i in range(shifts):
                    ax3.plot(shiftResults[i][0], shiftResults[i][6], 'o', c='C0')
                ax3.plot(shiftResults[bestShiftBelow][0], shiftResults[bestShiftBelow][6],'o',c='C1')

                plt.show()
                #pdb.set_trace()

            # generate video where rungs are marked by lines
            imgRungs = img.copy()
            for i in range(len(barAbove)):
                #print(i)
                cv2.line(imgRungs,(int(barAbove[i]),int(yPosAbove+barWidthAbove/2.)),(int(barBelow[i]),int(yPosBelow+barWidthBelow/2.)),(255,0,255),2)
                cv2.putText(imgRungs,(int(barBelow[i]),int(yPosBelow+barWidthBelow/2.)-10),1,(255,0,0),2)
            #time.sleep(0.1)
            if showVideo:
                #print('in display')
                cv2.imshow('Rungs', imgRungs)
                k = cv2.waitKey(2) & 0xff
                if k == 27: break
            #PressedKey = cv2.waitKey(0)
            #cv2.destroyWindow('Rungs')
            #imgRungsColor = cv2.cvtColor(imgRungs, cv2.COLOR_GRAY2BGR)
            self.outRung.write(imgRungs)
            nImg+=1


        self.outRung.release()
        # cv2.destroyAllWindows()
        # pickle.dump(rungPositions, open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb'))
        return (rungPositions,diffs,np.asarray(alignResults))
    ############################################################
    def trackPawsAndRungs(self,mouse,date,rec, **kwargs):
        badVideo = 0
        stopProgram = False
        # tracking parameters #########################
        self.thresholdValue = 0.7 # in %
        self.minContourArea = 40 # square pixels

        self.scoreWeights = {'distanceWeight':5,'areaWeight':1}
        ###############################################
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        firstImg = self.openVideo(videoFileName)

        # create video output streams
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.outPaw = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        self.outRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))

        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        
        # Return an array representing the indices of a grid.
        imgGrid = np.indices((self.Vheight, self.Vwidth))

        ########################################################################
        # loop to find correct wheel mask
        Radius = 1500  # 1400
        xCenter = 1205  # 1485
        yCenter = 1625  # 1545
        xPosition = 190
        yPosition = 0
        Npix = 5
        nIt = 0
        ConfirmedMask=False

        if 'WheelMask' in kwargs:
            Radius = kwargs['WheelMask'][0]
            xCenter = kwargs['WheelMask'][1]
            yCenter = kwargs['WheelMask'][2]

        
        while not ConfirmedMask and not stopProgram:

            imgCircle = img.copy()
            cv2.circle(imgCircle, (xCenter, yCenter), Radius, (0, 0, 255), 2)
            #if nIt > 0:
                #cv2.circle(imgCircle, (oldxCenter, oldyCenter), oldRadius, (0, 0, 100), 2)
                #cv2.putText(imgCircle,'now',(10,10),color=(0, 0, 255))
                #cv2.putText(imgCircle,'before',(10,20),fontScale=4,color=(0, 0, 150),thickness=2)
            cv2.imshow("Wheel mask", imgCircle)
            #print 'current radius, xCenter, yCenter : ' , Radius, xCenter, yCenter
            #print('Adjust the wheel mask using the arrows and +/- \n Press Space or Enter to confirm')
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                yCenter -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                yCenter += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                xCenter += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                xCenter -= Npix
            elif PressedKey == 43: # + Button
                Radius += Npix
            elif PressedKey == 45: # - Button
                Radius -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                cv2.destroyWindow("Wheel mask")
                ConfirmedMask = True
            elif PressedKey == 27: # Escape
                cv2.destroyWindow("Wheel mask")
                stopProgram=True
                #sys.exit()
            elif PressedKey ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyWindow("Wheel mask")
                #cv2.destroyAllWindows()
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            else:
                pass
            #nIt +=1
        print('masking after loop, Radius = %s, xCenter %s, yCenter = %s' % (Radius, xCenter, yCenter))

        wheelMask = np.zeros((self.Vheight, self.Vwidth))
        wheelMaskInv = np.zeros((self.Vheight, self.Vwidth))

        # create masks for mouse area and for lower area
        mask = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) > Radius
        maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        wheelMask[mask] = 1
        wheelMaskInv[maskInv] = 1
        wheelMask = np.array(wheelMask, dtype=np.uint8)
        wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)

        ########################################################################
        xPosition = 190
        yPosition = 0

        if 'RungsLoc' in kwargs:
            xPosition = kwargs['RungsLoc'][0]
            yPosition = kwargs['RungsLoc'][1]
            


        # loop to find correct rung lines
        imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
        # convert image to gray-scale
        imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
        nIt = 0
        ConfirmedRungALignement = False
        while not ConfirmedRungALignement and not stopProgram:
            rungs = []
            imgLines = imgCircle.copy()
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=50, param2=15, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                cLoc = []
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(imgLines, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(imgLines, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.line(imgLines, (i[0], i[1]), (xPosition,yPosition), (255, 0, 0), 2)
                    rungs.append([0, i[0], i[1], xPosition, yPosition])
                    if nIt > 0:
                        cv2.line(imgLines, (i[0], i[1]), (oldxPosition, oldyPosition), (100, 0, 0), 2)
                    cLoc.append([i[0],i[1]])
            cLoc =np.asarray(cLoc)
            a = np.sum(np.sqrt((cLoc[0]-cLoc[1])**2)) #np.linalg.norm(cLoc[0]-cLoc[1])
            b = np.sum(np.sqrt((cLoc[0]-cLoc[2])**2))
            c = np.sum(np.sqrt((cLoc[1]-cLoc[2])**2))
            #pdb.set_trace()
            #print 'argLengths : ', a ,b, c
            cv2.imshow("Rungs", imgLines)
            #print 'current xPosition, yPostion : ', xPosition, yPosition
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                yPosition -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                yPosition += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                xPosition += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                xPosition -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                cv2.destroyWindow("Rungs")
                ConfirmedRungALignement = True
            elif PressedKey == 27: # Escape
                cv2.destroyWindow("Rungs")
                stopProgram=True
                #sys.exit()
            elif PressedKey ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyWindow("Wheel mask")
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            else:
                pass
        #########################################################################

        recs = []
        if not stopProgram:
            bboxFront = cv2.selectROI("Select dot for FRONT paw \n mouse %s - rec = %s // %s" % (mouse, date, rec), img, False)
            print(recs)
            bboxHind = cv2.selectROI("Select dot for HIND paw", img, False)
            cv2.destroyAllWindows()
            #print 'front, hind paw bounding boxes : ', bboxFront, bboxHind
            pointLoc = bboxFront[:2]
            #print 'bounding box area : ', bboxFront[2] * bboxFront[3]
            frontpawPos = []
            hindpawPos = []
            # append first paw postions to list : [0 number of image, 1 success or failure, 2 location of paw, 3 all roi - ellipse - info, 4 area ]
            frontpawPos.append([0, 's', bboxFront[:2], [], np.pi * bboxFront[2] * bboxFront[3] / 4.])
            hindpawPos.append([0, 's', bboxHind[:2], [], np.pi * bboxHind[2] * bboxHind[3] / 4.])
            #hindPawPos.append([0, [], bboxHind[:2], np.pi * bboxHind[2] * bboxHind[3] / 4.])
            fcheckPos = -1

        ########################################################################

        
        hcheckPos = -1
        nF = 1
        #########################################################################
        # loop over all images in video
        print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
        print("Running the tracking algorithm...")
        while not stopProgram:
            #os.system('clear')
            #print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
            # Read a new frame
            thresholdV = self.thresholdValue
            ok, img = self.video.read()
            if not ok:
                break
            orig = img.copy()
            origCL = img.copy()
            #orig3 = img.copy()
            # while (1):
            # ret, frame = cap.read()

            # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
            imgMouse = cv2.bitwise_and(img, img, mask=wheelMask)

            # convert image to gray-scale
            imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
            imgGMouse = cv2.cvtColor(imgMouse, cv2.COLOR_BGR2GRAY)
            ###############################################################################################
            # find location of rungs


            # find circles in the lower part of the image, i.e., find screws to determine paw positions,
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=50, param2=15, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(origCL, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(origCL, (i[0], i[1]), 2, (0, 0, 255), 3)
                    #cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.line(origCL, (i[0], i[1]), (xPosition, yPosition), (255, 0, 0), 3)
                    rungs.append([nF, i[0], i[1], xPosition, yPosition])
            # find lines in the upper part of the image, i.e., the rungs
            #edges = cv2.Canny(imgGMouse,10,150,apertureSize = 3)
            #minLineLength = 50
            #maxLineGap = 10
            #cv2.imshow('edges detection', edges)
            #pdb.set_trace()
            #lines = cv2.HoughLinesP(edges,1,np.pi/(2*180),50,minLineLength,maxLineGap)
            #if lines is not None:
            #    for x1,y1,x2,y2 in lines[0]:
            #        cv2.line(origCL,(x1,y1),(x2,y2),(0,255,0),2)

            self.outPawRung.write(origCL)
            if self.showImages:
                cv2.imshow("detected circles  - mouse : %s   rec : %s/%s" % (mouse, date, rec), origCL)

            #################################################################################################
            # find contours based on maximal illumination

            # blur image and apply threshold
            blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
            minMaxL = cv2.minMaxLoc(blur)
            #mask = np.zeros(imgGMouse.shape,dtype="uint8")
            #cv2.drawContours(mask, [contour], -1, 255, -1)
            #mean,stddev = cv2.meanStdDev(image,mask=mask)
            while True:
                ret, th1 = cv2.threshold(blur, minMaxL[1] * thresholdV, 255, cv2.THRESH_BINARY)
                #print ret, th1
                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                edged = cv2.Canny(th1, 50, 100)
                edged = cv2.dilate(edged, None, iterations=1)
                edged = cv2.erode(edged, None, iterations=1)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                if len(cnts)>0:
                    # sort the contours from left-to-right and initialize the
                    # 'pixels per metric' calibration variable
                    (cnts, _) = contours.sort_contours(cnts)
                nLarge = 0
                for c in cnts:
                    if cv2.contourArea(c) > self.minContourArea:
                        nLarge += 1
                if nLarge >= 4 :
                    break
                else:
                    thresholdV = thresholdV - 0.05
            #print mmean, sstddev
            #for c in cnts:
            #    mask = np.zeros(imgGMouse.shape, dtype="uint8")
            #    cv2.drawContours(mask, c, 0, 255, 2)
            #    mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
            #    print cv2.contourArea(c), mmean, sstddev
            #pdb.set_trace()

            #cntDistances = []
            statsRois = []
            rois = []
            for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < self.minContourArea:
                    continue

                # print 'contourArea : ',cv2.contourArea(c)
                # compute the rotated bounding box of the contour
                ell = cv2.fitEllipse(c)
                # print ell
                #cntDistances.append(dist.euclidean(pawPos[checkPos][3], ell[0]))
                #cntArea.append(np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0))
                # get statistics of contour
                mask = np.zeros(imgGMouse.shape, dtype="uint8")
                cv2.drawContours(mask, c, -1, (255), 1)  # cv.drawContours(mask, contours, -1, (255),1)
                mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
                statsRois.append([cv2.contourArea(c), mmean[0][0], sstddev[0][0]])

                rois.append([ell,np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0)])
                orig = cv2.ellipse(orig, ell, (255, 0, 0), 2)


            #pdb.set_trace()

            # find ellipse which is the best continuation of the previous ones
            # print 'nContours, Dist, Areaa : ', nCnts, cntDistances, cntArea
            #print 'frame ', nF, len(rois),
            # if rois were detected
            if len(rois) > 0:
                cornerDist = []
                frontDist  = []
                hindDist   = []
                frontAreaChange = []
                hindAreaChange  = []
                fpScore = np.zeros(len(rois))
                hpScore = np.zeros(len(rois))
                #pdb.set_trace()
                for i in range(len(rois)):
                    cornerDist.append(dist.euclidean((0,self.Vheight), rois[i][0][0]))
                    frontDist.append(dist.euclidean(frontpawPos[fcheckPos][2], rois[i][0][0]))
                    hindDist.append(dist.euclidean(hindpawPos[hcheckPos][2], rois[i][0][0]))
                    frontAreaChange.append(np.abs(frontpawPos[fcheckPos][2] - statsRois[i][0]))
                    hindAreaChange.append(np.abs(hindpawPos[hcheckPos][2] - statsRois[i][0]))
                    #pdb.set_trace()
                    cv2.putText(orig, '%s , %s , %s' % (statsRois[i][0],statsRois[i][1],statsRois[i][2]), (int(rois[i][0][0][0]),int(rois[i][0][0][1])), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                # score based on the distance to last paw position, smallest distance -> highest score
                fpScore+= self.scoreWeights['distanceWeight']/np.asarray(frontDist)
                hpScore+= self.scoreWeights['distanceWeight']/np.asarray(hindDist)
                # score based on change of contour area, smallest change -> highest score
                fpScore+= self.scoreWeights['areaWeight']/np.asarray(frontAreaChange)
                hpScore+= self.scoreWeights['areaWeight']/np.asarray(hindAreaChange)
                # score based on distance from lower left corner : small dist -> high score for hind
                #if len(cornerDist) == 2:
                #    hindIdx =  np.argmin(np.asarray(cornerDist))
                #    frontIdx = np.argmax(np.asarray(cornerDist))
                #else :

                highestSTD = np.argsort(1./(np.asarray(statsRois)[:,2]))
                hindSortIdx = np.argsort(np.asarray(hindDist))
                frontSortIdx = np.argsort(np.asarray(frontDist))
                cornerSortIdx = np.argsort(np.asarray(cornerDist))
                #frontIdx = highestSTD[0] # int(np.where(highestSTD == 0)[0])
                #hindIdx = highestSTD[1] # int(np.where(highestSTD == 1)[0])
                #pdb.set_trace()
                # if STD of first two points is very large : certain that they are the paws
                if statsRois[highestSTD[1]][2]/statsRois[highestSTD[2]][2] > 3. :
                    if np.where(highestSTD[0] == hindSortIdx) < np.where(highestSTD[0] == frontSortIdx):
                        hindIdx  = highestSTD[0]
                        frontIdx = highestSTD[1]
                    else:
                        hindIdx  = highestSTD[1]
                        frontIdx = highestSTD[0]
                elif np.where(hindSortIdx[0] == cornerSortIdx) < np.where(frontSortIdx[0] == cornerSortIdx):
                    hindIdx  = hindSortIdx[0]
                    frontIdx = frontSortIdx[0]
                else:
                    hindIdx  = 0
                    frontIdx = 0
                #if hindSortIdx[0] in highestSTD[:2] :
                #    hindIdx = hindSortIdx[0]
                if (frontDist[frontIdx] < abs(fcheckPos) * 50.): # frontIdx in np.where(highestSTD<=1)[0] :
                    #print 'front, hind index ', frontIdx, hindIdx
                    #pdb.set_trace()
                    # if (frontDist[frontIdx] < abs(fcheckPos) * 50.):
                    Str = 'frontDist success : %s' % frontDist[frontIdx]
                    frontpawPos.append([nF,rois[frontIdx][0][0],rois[frontIdx][0],rois[frontIdx][1],statsRois[frontIdx][0]])
                    fcheckPos = -1
                else:
                    Str = 'frontDist failure : %s' % frontDist[frontIdx]
                    frontpawPos.append([nF,rois[frontIdx][0][0],rois[frontIdx][0],rois[frontIdx][1],statsRois[frontIdx][0]])
                    fcheckPos -= 1
                #
                if (hindDist[hindIdx] < abs(hcheckPos) * 50.): # hindIdx in np.where(highestSTD<=1)[0] : #hindIdx in np.where(highestSTD<=1)[0] : #(hindDist[hindIdx] < abs(hcheckPos) * 50.):

                    Str2 = '    ///     hindDist success : %s' % hindDist[hindIdx]
                    hindpawPos.append([nF,rois[hindIdx][0][0], rois[hindIdx],rois[hindIdx][1],statsRois[hindIdx][0]] )
                    hcheckPos = -1
                else:
                    Str2 = 'hindDist failure : %s' % hindDist[hindIdx]
                    hindpawPos.append([nF, 'f', rois[hindIdx][0][0], rois[hindIdx],rois[hindIdx][1],statsRois[hindIdx][0]] )
                    hcheckPos -=1
                ##
                if self.showImages:
                    FrameStr =  'frame %s (len(rois) = %s)' % (nF, len(rois))
                    x =  int(self.Vwidth * (nF/float(self.Vlength)))
                    cv2.rectangle(orig, (0, self.Vheight), (x, self.Vheight-15), (100, 100, 100), thickness=-1)
                    cv2.putText(orig, FrameStr, (0, self.Vheight-20), cv2.QT_FONT_NORMAL, 0.45, color=(255, 255, 255))
                    cv2.putText(orig, Str, (0, self.Vheight-5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                    cv2.putText(orig, Str2, (int(self.Vwidth/3), self.Vheight-5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                    cv2.putText(orig,'frontpaw',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 255, 0))
                    orig = cv2.ellipse(orig, rois[frontIdx][0], (0, 255, 0), 2)
                    cv2.putText(orig,'hindpaw',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 0, 255))
                    orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
                #print 'hind ',
                #dret = decideonAndAddPawPositions(hindpawPos, hcheckPos, rois)
                #hindpawPos.append(dret[1])
                #hcheckPos += dret[0]
                #if not dret[0]:
                #if self.showImages:
                #    cv2.putText(orig,'hindpaw',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 0, 255))
                #    orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
                # pdb.set_trace()
                #print ' '
                #Dchange = abs(np.asarray(cntDistances))
                #Achange = abs(np.asarray(cntArea) / pawPos[checkPos][3] - 1.) * 100.  # in percent
                #DWeight = 0.5
                #print checkPos, Dchange, Achange,
                #Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
                # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
                # print Dindex, Dprojection, pawPos[-2:]
                # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
                # print cntDistances[Aindex]
                #frontpawPos.append([nF,rois[PindMax][0], rois[PindMax][1]])
                #hindpawPos.append([nF, rois[PindMin][0], rois[PindMin][1]])
                #if (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                #    print 'success', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (0,[nF, rois[Pindex][0], rois[Pindex][0][0], cntArea[Pindex]])
                #cntFrontDistances = []
                #cntArea = []
                # pdb.set_trace()
                #for i in range(len(rois)):
                #    cntFrontDistances.append(dist.euclidean(pawPos[checkPos][2], rois[i][0][0]))
                #    cntFrontDistances.append(dist.euclidean(pawPos[checkPos][2], rois[i][0][0]))
                #    cntArea.append(rois[i][1])

                #Dchange = abs(np.asarray(cntDistances))
                #Achange = abs(np.asarray(cntArea) / pawPos[checkPos][3] - 1.) * 100.  # in percent
                #DWeight = 0.9
                # print checkPos, Dchange, Achange,
                #Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
                # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
                # print Dindex, Dprojection, pawPos[-2:]
                # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
                # print cntDistances[Aindex]
                #if (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                #    print 'success', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (0, [nF, rois[Pindex][0], rois[Pindex][0][0], cntArea[Pindex]])
                #    # pawPos.append([nF, rois[Pindex], rois[Pindex][0], cntArea[Pindex]])

                #    # orig2 = cv2.ellipse(orig2, rois[Pindex], (0, 255, 0), 2)
                #    # orig3 = cv2.ellipse(orig3, rois[Pindex], (0, 255, 0), 2)
                #checkPos = -1  # pointLoc = rois[Dindex][0]  # maxStepCurrent = maxStep
                #else:
                #    print 'failure', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (1, [nF, -1, -1, -1])  # pawPos.append([nF, '', -1, -1, -1])  # checkPos -= 1


                #print 'front ',
                #dret = decideonAndAddPawPositions(frontpawPos,fcheckPos,rois)

            else:
                print('failure no rois')
                frontpawPos.append([nF, 'f', [-1,-1], -1, -1])
                hindpawPos.append([nF,'f', [-1,-1], -1, -1])
                fcheckPos -= 1
                hcheckPos -= 1
            # show image with all detected rois, and rois decided to be paws

                
            self.outPaw.write(orig)

            if self.showImages:
                cv2.imshow("Paw-tracking monitor - mouse : %s   rec : %s/%s" % (mouse, date, rec), orig)

            # wait and abort criterion, 'esc' allows to stop
            k = cv2.waitKey(1) & 0xff
            #print k
            if k == 27: break
            elif k ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyAllWindows()
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            elif k == 32:
                pdb.set_trace()
                #cv2.waitKey(100)
            nF += 1

        cv2.destroyAllWindows()

        # save tracked data
        #(test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,'')
        #self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        #self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        #self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])
        if not stopProgram:
            pickle.dump(frontpawPos, open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (mouse, date, rec), 'wb'))
            pickle.dump(hindpawPos, open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (mouse, date, rec), 'wb'))
            pickle.dump(rungs, open( self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb' ) )
        return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo

    ########################################################################################################################
    # frontpawPos,hindpawPos,rungs,fTimes,angularSpeed,linearSpeed,sTimes
    def analyzePawsAndRungs(self,mouse,date,rec,frontpawPos,hindpawPos,rungs,fTimes,angularSpeed,linearSpeed,sTimes,angleTimes):
        # some image streams for verification
        self.showImages = False

        # create video output streams
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungAnalysis.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        # self.outPawRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))



        # # wait and abort criterion, 'esc' allows to stop
        # k = cv2.waitKey(1) & 0xff
        # # print k
        # if k == 27: break

        ##########################################################
        # build array of paw positions
        def calculateDist(a,b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        spacingDegree = 6.81 #360./48.

        rungs = np.asarray(rungs)

        fp = []
        fpAll = []
        hp = []
        lastFrontPos = (frontpawPos[0][2][0],frontpawPos[0][2][1])
        fp.append([0, frontpawPos[0][0],frontpawPos[0][2][0],frontpawPos[0][2][1]])
        fD = []
        hD = []
        threshold = 60.
        skipSteps = 1
        for i in range(1,len(frontpawPos)-1):
            #currPos = (frontpawPos[i][2][0],frontpawPos[i][2][1])
            #nextPos = (frontpawPos[i+1][2][0],frontpawPos[i+1][2][1])
            #lastcurrDist = calculateDist(lastFrontPos,currPos)
            #currNextDist = calculateDist(currPos,nextPos)
            #fD.append([lastcurrDist,currNextDist])
            fp.append([i, frontpawPos[i][0],frontpawPos[i][2][0],frontpawPos[i][2][1]])
            # if (np.abs(lastcurrDist)+np.abs(currNextDist)) < threshold*skipSteps :
            #     fp.append([frontpawPos[i][0],frontpawPos[i][2][0],frontpawPos[i][2][1]])
            #     lastFrontPos = currPos
            #     skipSteps = 1
            # else:
            #     print 'fp', i, lastcurrDist, currNextDist
            #     skipSteps +=1
        lastHindPos = (hindpawPos[0][2][0],hindpawPos[0][2][1])
        hp.append([0,hindpawPos[0][0], hindpawPos[0][2][0], hindpawPos[0][2][1]])
        skipSteps=1
        for i in range(1,len(hindpawPos)):
            #currPos = (hindpawPos[i][2][0],hindpawPos[i][2][1])
            #nextPos = (hindpawPos[i+1][2][0],hindpawPos[i+1][2][1])
            #lastcurrDist = calculateDist(lastHindPos,currPos)
            #currNextDist = calculateDist(currPos, nextPos)
            #hD.append([lastcurrDist,currNextDist])
            hp.append([i, hindpawPos[i][0], hindpawPos[i][2][0], hindpawPos[i][2][1]])
            # if (np.abs(lastcurrDist)+np.abs(currNextDist)) < threshold*skipSteps :
            #     hp.append([hindpawPos[i][0], hindpawPos[i][2][0], hindpawPos[i][2][1]])
            #     lastHindPos = currPos
            #     skipSteps = 1
            # else:
            #     skipSteps +=1

        fp = np.asarray(fp)
        #fpAll = np.asarray(fpAll)
        hp = np.asarray(hp)
        ################################################################################
        # extract stance and swing phase through difference in speed

        def findBeginningAndEndOfStep(speedDiff,speedDiffThresh,minLength,trailingStart,trailingEnd):
            # determine regions during which the speed is different for more than xLength values
            thresholded = speedDiff > speedDiffThresh
            startStop = np.diff(np.arange(len(speedDiff))[thresholded]) > 1
            mmmStart = np.hstack((([True]), startStop))  # np.logical_or(np.hstack((([True]),startStop)),np.hstack((startStop,([True]))))
            mmmStop = np.hstack((startStop, ([True])))
            startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
            stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
            minLengthThres = (stopIdx - startIdx) > minLength
            startStep = startIdx[minLengthThres]-trailingStart
            endStep = stopIdx[minLengthThres]+trailingEnd
            return np.column_stack((startStep,endStep))

        fpTimes = fTimes[np.array(fp[:,1],dtype=int)]
        hpTimes = fTimes[np.array(hp[:,1],dtype=int)]
        speedFP = (np.sqrt((np.diff(fp[:, 2]) / np.diff(fpTimes)) ** 2 + (np.diff(fp[:, 3]) / np.diff(fpTimes)) ** 2)) / 80.
        speedHP = (np.sqrt((np.diff(hp[:,2])/np.diff(hpTimes))**2 + (np.diff(hp[:,3])/np.diff(hpTimes))**2))/80.


        walk_interp = interp1d(sTimes, np.abs(linearSpeed))
        maskF = (fpTimes[1:]>=sTimes[0]) & (fpTimes[1:]<=sTimes[-1])
        newFPWheelSpeed = walk_interp(fpTimes[1:][maskF])
        fpSpeedDiff = speedFP[maskF]-newFPWheelSpeed
        maskH = (hpTimes[1:]>sTimes[0]) & (hpTimes[1:]<sTimes[-1])
        newHPWheelSpeed = walk_interp(hpTimes[1:][maskH])
        hpSpeedDiff = speedHP[maskH]-newHPWheelSpeed


        #pdb.set_trace()
        xLengthFP = 3
        xLengthHP = 4
        speedDiffThreshold = 10.
        startStopFPStep = findBeginningAndEndOfStep(fpSpeedDiff,5.,xLengthFP,2,3)
        startStopHPStep = findBeginningAndEndOfStep(hpSpeedDiff,speedDiffThreshold,xLengthHP,2,4)
        # join times of the respective frames
        startStopFPStep = np.column_stack((startStopFPStep,fpTimes[1:][maskF][startStopFPStep[:,0]],fpTimes[1:][maskF][startStopFPStep[:,1]]))
        startStopHPStep = np.column_stack((startStopHPStep,hpTimes[1:][maskH][startStopHPStep[:,0]],hpTimes[1:][maskH][startStopHPStep[:,1]]))
        #newAngles = walk_interp(fTimes[mask])
        #ax7.plot(sTimes, np.abs(linearSpeed))

        # plt.plot(fpTimes[1:][maskF],speedFP[maskF],c='0.5')
        # plt.plot(sTimes, np.abs(linearSpeed),c='C0')
        # for i in range(len(startStopFPStep)):
        #     #pdb.set_trace()
        #     plt.plot(fpTimes[1:][maskF][int(startStopFPStep[i,0]):int(startStopFPStep[i,1])],speedFP[maskF][int(startStopFPStep[i,0]):int(startStopFPStep[i,1])],c='C1')
        #
        # plt.show()
        # pdb.set_trace()
        ################################################################################
        # fit circle to points of rungscrews

        x = np.r_[rungs[:,1]]
        y = np.r_[rungs[:,2]]

        def calc_R(xc, yc):
            """ calculate the distance of each data points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f_2b(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        def Df_2b(c):
            """ Jacobian of f_2b
            The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
            xc, yc = c
            df2b_dc = np.empty((len(c), x.size))

            Ri = calc_R(xc, yc)
            df2b_dc[0] = (xc - x) / Ri  # dR/dxc
            df2b_dc[1] = (yc - y) / Ri  # dR/dyc
            df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

            return df2b_dc

        center_estimate = [600,600]
        center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

        xc_2b, yc_2b = center_2b
        Ri_2b = calc_R(*center_2b)
        R_2b = Ri_2b.mean()
        pixToCmConversion = 12.5/R_2b
        print('Center, Radius of fitted circle : ', center_2b, R_2b, pixToCmConversion)

        ###########################################################
        # exclude points which are too far away from the circle fit line
        inclP = abs((Ri_2b - R_2b)) < 25
        rungs = rungs[inclP]

        ############################################################
        # fit list of points on a circle to the actual extracted points:
        # determine rotation angle and count rungs
        def angle_between(p1, p2):
            ang1 = np.arctan2(*p1[::-1])
            ang2 = np.arctan2(*p2[::-1])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        def contructCloudOfPointsOnCircle(nPoints,circleCenter,circleRadius,spacingDegree):
            cPoints = np.zeros((nPoints,3))
            for i in range(nPoints):
                cPoints[i,0] = i
                cPoints[i,1] = circleCenter[0] + np.sin(i*spacingDegree*np.pi/180.)*circleRadius
                cPoints[i,2] = circleCenter[1] - np.cos(i*spacingDegree*np.pi/180.)*circleRadius
            return cPoints



        def rotate_around_point(xy, degrees, origin=(0, 0)):
            """Rotate a point around a given point.
            """
            x, y = xy
            offset_x, offset_y = origin
            adjusted_x = (x - offset_x)
            adjusted_y = (y - offset_y)
            cos_rad = math.cos(degrees*np.pi/180.)
            sin_rad = math.sin(degrees*np.pi/180.)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y - sin_rad * adjusted_x + cos_rad * adjusted_y

            return ([qx, qy])


        def fitfunc(d,wheelPoints,genericPoints):
            genericRotated = genericPoints.copy()
            distances = np.zeros(len(genericPoints))
            for i in range(len(genericPoints)):
                genericRotated[i] = rotate_around_point(genericPoints[i],d,origin=center_2b)
                distances[i] = calculateDist(genericRotated[i],wheelPoints[i])
            return np.sum(np.abs(distances))


        rungsList = rungs.tolist()
        rungsList.sort(key=lambda x: x[2])
        rungsList.sort(key=lambda x: x[1])
        rungsList.sort(key=lambda x: x[0])
        rungsSorted = np.asarray(rungsList)

        rungConvergencePoint = ([rungsSorted[0,3],rungsSorted[0,4]])
        # determine first angle in first image
        ppFFirst = rungsSorted[rungsSorted[:,0]==0][:,1:3]
        genericPs = contructCloudOfPointsOnCircle(len(ppFFirst),center_2b,R_2b,spacingDegree)
        d0 = -0.5
        d1, success = optimize.leastsq(fitfunc, d0,args=(ppFFirst,genericPs[:,1:]))

        # loop over all frames - via frameNumbers - not individual points
        frameNumbers = np.arange(len(fTimes)) #np.unique(rungsSorted[:,0])
        dOld = d1
        rungsNumbered = []
        rungCounter = 0
        #videoFileName = self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec)
        #self.video = cv2.VideoCapture(videoFileName)
        #aBtw = []
        frontpawRungDist = []
        hindpawRungDist = []
        pC = np.zeros(8)
        for i in range(len(frameNumbers)):
            #ok, img = self.video.read()
            if self.showImages:
                blank_image = np.zeros((self.Vheight,self.Vwidth,3), np.uint8)

            # determine points per frame
            ppF = rungsSorted[rungsSorted[:,0]==frameNumbers[i]][:,1:]

            # for n in range(1,len(ppF)):
            #     #print angle_between(ppF[n]-center_2b,ppF[n-1]-center_2b),
            #     aBtw.append(angle_between(ppF[n]-center_2b,ppF[n-1]-center_2b))

            genericPs = contructCloudOfPointsOnCircle(len(ppF),center_2b,R_2b,spacingDegree)
            d1, success = optimize.leastsq(fitfunc, d0,args=(ppF[:,:2],genericPs[:,1:]))
            degreeDifference = d1-dOld
            if degreeDifference < -5.:
                rungCounter +=1
                degreeDifference += spacingDegree
            if degreeDifference > 5.:
                rungCounter -=1
                degreeDifference -= spacingDegree
            #dD.append(degreeDifference[0])
            numberedR = np.arange(len(ppF))+rungCounter
            rungsNumbered.append([i,frameNumbers[i],len(ppF),d1,degreeDifference[0],rungCounter,numberedR,ppF[:,:2]])

            # cacluate distance btw paw and rungs
            frontpawPos = fp[fp[:, 1] == frameNumbers[i]]
            hindpawPos = hp[hp[:, 1] == frameNumbers[i]]
            tempFD = []
            tempHD = []
            pC[len(ppF)] += 1
            if len(ppF)==1:
                nAddRungs = 3
            elif len(ppF)==2 or len(ppF)==3:
                nAddRungs = 2
            elif len(ppF)==4 or len(ppF)==5:
                nAddRungs = 1
            elif len(ppF)==6 or len(ppF)==7:
                nAddRungs = 0
            genericPs = contructCloudOfPointsOnCircle(7,center_2b,R_2b,spacingDegree)
            numberedGenericR = np.arange(7) + numberedR[0] - nAddRungs
            genericRotated = genericPs[:,1:].copy()
            #pdb.set_trace()
            for n in range(len(genericPs)):
                genericRotated[n] = rotate_around_point(genericPs[n][1:],d1+nAddRungs*spacingDegree,origin=center_2b)
                #print frameNumbers[i], len(ppF), ppF
                #for n in range(len(ppF)):
                if len(frontpawPos)>0:
                    tempFD.append(np.cross(rungConvergencePoint-genericRotated[n],frontpawPos[0][2:]-genericRotated[n])/norm(rungConvergencePoint-genericRotated[n]))
                if len(hindpawPos)>0:
                    tempHD.append(np.cross(rungConvergencePoint-genericRotated[n],hindpawPos[0][2:]-genericRotated[n])/norm(rungConvergencePoint-genericRotated[n]))
                    #hindpawRungDist.append(tempD[1])
                if self.showImages:
                    cv2.circle(blank_image, (int(genericRotated[n,0]), int(genericRotated[n,1])), 6, (0, 0, 255), 3)
            if len(frontpawPos)>0:
                #tempAFD = np.asarray(tempFD)
                #sortedA = np.argsort(tempAFD)
                #pdb.set_trace()
                frontpawRungDist.append([i,frameNumbers[i]])
                frontpawRungDist[-1].extend(tempFD)
                frontpawRungDist[-1].extend(numberedGenericR.tolist())
            temp = []
            if len(hindpawPos)>0:
                #tempAHD = np.asarray(tempHD)
                #sortedA = np.argsort(tempAHD)
                hindpawRungDist.append([i,frameNumbers[i]])
                hindpawRungDist[-1].extend(tempHD)
                hindpawRungDist[-1].extend(numberedGenericR.tolist())
            if self.showImages:
                for n in range(len(ppF)):
                    cv2.circle(blank_image, (ppF[n,0], ppF[n,1]), 2, (0, 255, 0), 2)
            # draw the center of the circle
            # cv2.circle(blank_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            # self.outRung.write(orig)

            if self.showImages:
                cv2.imshow("Rung rotations : %s   rec : %s/%s" % (mouse, date, rec), blank_image)

            dOld = d1
            #for n in range(len(ppF)):
            # if i in [836,837,838]:
            #     print i
            #     for n in range(len(ppF)):
            #         cv2.putText(img,'%s' % (n+rungCounter),(ppF[n,0],ppF[n,1]),cv2.FONT_HERSHEY_SIMPLEX,2,color=(0, 0, 255))
            #
            #     cv2.imshow("Paw-tracking monitor", img)
            #     # wait and abort criterion, 'esc' allows to stop
            if self.showImages:
                k = cv2.waitKey(50) & 0xff
                if k == 27: break
                elif k == 32: pdb.set_trace()
            #cv2.destroyAllWindows()

            #if degreeDifference[0] > 1:
            #    print i,frameNumbers[i], len(ppF), d1, degreeDifference, np.arange(len(ppF))+rungCounter #, ppF

            #if i == 30 :
            #    pdb.set_trace()
        print('rung number per image : ', pC)
        #pdb.set_trace()
        frontpawRungDist = np.asarray(frontpawRungDist)
        hindpawRungDist = np.asarray(hindpawRungDist)
        ###################################################################
        # substract rotation

        if len(frameNumbers) != len(fTimes):
            print('problem with frame number length')
            pdb.set_trace()

        #pdb.set_trace()
        walk_interp = interp1d(angleTimes[:,0],angleTimes[:,1])

        mask = (fTimes>angleTimes[:,0][0]) & (fTimes<angleTimes[:,0][-1])

        newAngles = walk_interp(fTimes[mask])
        # # sTimes, ttime
        # ax01.plot(angleTimes[:,0],linearSpeed)
        # ax01.plot(ttime[mask],newWalking)
        #
        degreesTurned = 0.
        fpLinear = []
        hpLinear = []
        #pdb.set_trace()
        fInitial = fp[0][2:]
        hInitial = hp[0][2:]
        rotationsHP = 0.
        rotationsFP = 0.
        oldAfp = 0.
        oldAhp = 0.
        for i in range(len(fTimes[mask])):
            fpMask = fp[:,1]==frameNumbers[i]
            hpMask = hp[:,1]==frameNumbers[i]

            degreesTurned += rungsNumbered[i][4]
            #pdb.set_trace()
            rfp = rotate_around_point(fp[fpMask][:,2:][0],-newAngles[i],center_2b)
            rhp = rotate_around_point(hp[hpMask][:,2:][0],-newAngles[i],center_2b)

            afp = angle_between(rfp-center_2b,fInitial-center_2b)
            if oldAfp > 300. and afp < 100. :
                rotationsFP +=1.
            elif oldAfp < 100. and afp > 300. :
                rotationsFP -=1.
            dfp = (rotationsFP + afp/360.)*80.
            ahp = angle_between(rhp-center_2b,hInitial-center_2b)
            if oldAhp > 300. and ahp < 100. :
                rotationsHP +=1.
            elif oldAhp < 100. and ahp > 300. :
                rotationsHP -=1.
            dhp = (rotationsHP + ahp/360.)*80.
            # rotational coordinates to straight motion : distance is y
            fpLinear.append([frameNumbers[i],fTimes[i],dfp,(calculateDist(rfp,center_2b)-R_2b)*pixToCmConversion,newAngles[i],degreesTurned])
            hpLinear.append([frameNumbers[i],fTimes[i],dhp,(calculateDist(rhp,center_2b)-R_2b)*pixToCmConversion,newAngles[i],degreesTurned])
            print(i,degreesTurned,afp,dfp,ahp,dhp, rotationsFP, rotationsHP,newAngles[i])
            oldAfp = afp
            oldAhp = ahp

        #cpdb.set_trace()
        return (fp,hp,rungs,center_2b, R_2b,rungsNumbered,np.asarray(fpLinear),np.asarray(hpLinear),frontpawRungDist,hindpawRungDist,startStopFPStep,startStopHPStep)
