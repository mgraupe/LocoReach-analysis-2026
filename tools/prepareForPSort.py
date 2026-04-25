import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interp1d
import numpy as np
import pdb


path = '/home/mgraupe/psort/'
fileName = 'EphysDataPerSession_2022.04.29_000_locomotionEphys2Motor60sec_002_psort.h5'

f = h5.File(path+fileName, 'r')
fnew = h5.File(path+fileName[:-3]+'_rescaled.h5', 'w')

dd = f['ch_data'].value
tt = f['ch_time'].value
sr = f['sample_rate'].value

interpData = interp1d(tt,dd)
ttNew = np.linspace(tt[0],tt[-1],2*len(tt),endpoint=True)
ddNew = interpData(ttNew)
print(int(1./ttNew[1]))

pdb.set_trace()
fnew.create_dataset('ch_data', data=ddNew)
fnew.create_dataset('ch_time', data=ttNew)
sampleRate = np.array([int(1./ttNew[1])])
fnew.create_dataset('sample_rate',data=sampleRate)

plt.plot(tt,dd)
plt.plot(ttNew,ddNew)
plt.show()