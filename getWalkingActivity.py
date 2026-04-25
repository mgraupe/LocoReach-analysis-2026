from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.parameters as par
import pdb
import matplotlib.pyplot as plt
import pickle




mouseD = '250121_m01' # id of the mouse to analyze230226_f88

expDateD = 'some'    # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'
recStructure = 'simplexNPX' #'simplexEphy' # specify here 'simplexBehavior', None otherwise
obstacle = False


if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    try:
        expDate = expDateD
    except :
        expDate = 'all'
else:
    expDate = args.date

eSD         = extractSaveData.extractSaveData(mouse,recStruc=recStructure)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
#(foldersRecordings,dataFolder) = eSD.getEphysRecordingsList(expDate=expDate,recordings=recordings)

print(foldersRecordings)
#tracks = []
#print(foldersRecordings)
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle, config) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
        if existence:
            if obstacle:
                (angles, aTimes, timeStamp, monitor, absoluteSignal, obstacle1, obstacle2) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'RotaryEncoder', fileHandle, obstacle=obstacle)
            else:
                (angles, aTimes,timeStamp,monitor) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder',fileHandle)
            (angularSpeed, linearSpeed, sTimes,ASraw,LSraw)  = dataAnalysis.getSpeed(angles,aTimes,par.wheelCircumsphere,par.minSpacing)
            # plt.plot(aTimes,obstacle1)
            # plt.plot(aTimes,obstacle2)
            # plt.plot(aTimes,absoluteSignal)
            # if r==4:
            #plt.plot(sTimes, linearSpeed)
            #plt.plot(sTimes,angularSpeed[:-1])
            #plt.show()
            #pdb.set_trace()
            #wa = [angularSpeed, linearSpeed, sTimes, angles, aTimes, timeStamp,monitor, [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']]
            #pickle.dump(wa, open( 'walkingActivity_%s_rec%s-%s.p' % (foldersRecordings[f][0],foldersRecordings[f][2][r][-7:-4],foldersRecordings[f][2][r][-3:]), 'wb' ) )
            #pdb.set_trace()
            if obstacle:
                eSD.saveObstacleWalkingActivity(angularSpeed, linearSpeed, sTimes, angles, aTimes, timeStamp, monitor, absoluteSignal, obstacle1, obstacle2, [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
            else:
                eSD.saveWalkingActivity(angularSpeed, linearSpeed, sTimes, angles, aTimes, timeStamp, monitor, [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])  # save motion corrected image stack

# plt.plot(aTimes,obstacle2)
# plt.show()
del eSD
