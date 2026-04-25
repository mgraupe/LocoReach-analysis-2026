import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy

import xml.etree.ElementTree as ET

#############################
def parseSVGFile(filename):
    allLines = {}
    nL = 0
    tree = ET.parse(fileName)
    root = tree.getroot()
    coords = []
    for child in root:
        print(child.tag)
        if child.tag == '{http://www.w3.org/2000/svg}g':
            for sub in child:
                if sub.tag == '{http://www.w3.org/2000/svg}path':
                    print(sub.attrib['id'],'coordinates :',sub.attrib['d'])
                    allLines[nL] = {}
                    allLines[nL]['id'] = sub.attrib['id']
                    coordsSt = sub.attrib['d'][2:]
                    encoding = sub.attrib['d'][:1]
                    #print(encoding)
                    coordsSt = coordsSt.replace(' ',',')
                    #print(coordsSt)c
                    exampleList = [float(k) for k in coordsSt.split(',')]
                    if (exampleList[0]<200) and (exampleList[2]>0.) : #encoding == 'M':
                        print('Before : ', exampleList)
                        print('large M encoding')
                        exampleList[2] = exampleList[2]-exampleList[0]
                    print(exampleList)
                    allLines[nL]['coords'] = exampleList #sub.attrib['id']
                    coords.append(exampleList)
                    #allLines[nL]
                    nL+=1

    coords.sort(key=lambda column: column[0])
    return coords
###############################################
def convertCoordsToOldFormat(coords,hh):
    oldCoordFormat = []
    for j in range(int(len(coords)/2)):
        xPosLeftLow = coords[2*j][0]+coords[2*j][2]
        xPosRightLow = coords[2*j+1][0] + coords[2*j+1][2]
        if coords[2*j][3] > 400:
            yPosLow = 600 - (coords[2*j][3])
        else:
            yPosLow = 600 - (coords[2*j][1]+coords[2*j][3])
        #if yPosLow<0:
        #    print(coords)
        #    pdb.set_trace()
        xPosLeftHigh = coords[2*j][0]
        xPosRightHigh = coords[2*j+1][0]
        yPosHigh = 600 - coords[2*j][1]
        oldCoordFormat.append([xPosLeftLow,xPosRightLow,yPosLow,xPosLeftHigh,xPosRightHigh,yPosHigh])
        print(coords[2*j])
        print(oldCoordFormat[-1])

    return np.asarray(oldCoordFormat)

###############################################
# for the batch of simplex animals run in Nov-Dec 2021


fileName = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/201017_m98/201017_m98_2020.12.02_000_locomotion_recording_setup2_000-000_raw_behavior_#2000.svg'

#imgIdx = [0,1000,4000,5000]
images = [['220204_m49','220204_m49_2022.03.25_005_locomotion_recording_setup2_000-003_raw_behavior_#',[4000,6000]],
          ['220204_m49','220204_m49_2022.03.15_006_locomotion_recording_setup2_000-001_raw_behavior_#',[0,2000,]]]

rungs = []
coordinates = []
for n in range(len(images)):
    path = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % images[n][0]
    for i in range(len(images[n][2])):
        fileName = path+'%s%s.svg' % (images[n][1],images[n][2][i])
        coords = parseSVGFile(fileName)
        #pdb.set_trace()
        oldFormat = convertCoordsToOldFormat(coords,600)
        coordinates.append(coords)
        rungs.append(oldFormat)

#pdb.set_trace()
distBelow = []
distAbove = []
posBelow = []
posAbove = []
yAbove = []
yBelow = []

for i in range(len(rungs)):
    print(i,len(distBelow))
    #if i == 0:
    #    above = rungs[i][0,5]
    #    below = rungs[i][0,2]
    yAbove.extend(rungs[i][:,5])
    yBelow.extend(rungs[i][:,2])
    distBelow.extend(np.diff(rungs[i][:,0]))
    distBelow.extend(np.diff(rungs[i][:,1]))
    distAbove.extend(np.diff(rungs[i][:,3]))
    distAbove.extend(np.diff(rungs[i][:,4]))
    posBelow.extend(rungs[i][1:,0])
    posBelow.extend(rungs[i][1:,1])
    posAbove.extend(rungs[i][1:,3])
    posAbove.extend(rungs[i][1:,4])

distBelow = np.asarray(distBelow)
distAbove = np.asarray(distAbove)
posBelow  = np.asarray(posBelow)
posAbove  = np.asarray(posAbove)

# sposBelow = np.copy(dd0[:,0])
# sposBelow  = np.concatenate((sposBelow,dd1[:,0]))
# sposBelow  = np.concatenate((sposBelow,dd2[:,0]))
# angle0 = np.arctan(dd0[:,2]-dd0[:,0])/(dd0[:,3]-dd0[:,1])
# angle1 = np.arctan(dd1[:,2]-dd1[:,0])/(dd1[:,3]-dd1[:,1])
# angle2 = np.arctan(dd2[:,2]-dd2[:,0])/(dd2[:,3]-dd2[:,1])
#
# angle = np.concatenate((angle0,angle1))
# angle = np.concatenate((angle,angle2))


pdb.set_trace()

maskBelow = (distBelow > -100) & (distBelow < 100)
#maskAbove = posAbove > 0
#pdb.set_trace()
polycoeffsBelow = scipy.polyfit(posBelow[maskBelow], distBelow[maskBelow], 3)
print('below %s pix : ' % np.mean(yBelow) ,polycoeffsBelow)
# [ 2.00710807, 1.09204496]
belowData = np.linspace(np.min(posBelow),np.max(posBelow),1000)
yFitBelow = scipy.polyval(polycoeffsBelow, belowData)

polycoeffsAbove = scipy.polyfit(posAbove, distAbove, 3)
print('above %s pix : ' % np.mean(yAbove) ,polycoeffsAbove)
# [ 2.00710807, 1.09204496]
midData = np.linspace(np.min(posAbove),np.max(posAbove),1000)
yFitMid = scipy.polyval(polycoeffsAbove, midData)

fig = plt.figure()
ax0 = fig.add_subplot(1,1,1)
ax0.plot(belowData,yFitBelow,label='fitBelow')
ax0.plot(posBelow[maskBelow],distBelow[maskBelow],'.')
ax0.plot(midData,yFitMid,label='fitAbove')
ax0.plot(posAbove,distAbove,'.')
ax0.set_xlabel('x position (pixel)')
ax0.set_ylabel('x distance to next rung (pixel)')
plt.legend()
#ax1 = fig.add_subplot(2,1,2)
#ax1.plot(sposBelow,angle,'o')

plt.show()
