import scipy.io
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
#import ecg_plot
mat = scipy.io.loadmat('Data/noisyecg.mat')
X = mat['ecg1']
Y = mat['ecg2']
Z = mat['ecg3']


ecgData=Y

fs = 500
time = np.arange(ecgData.size) / fs
fig, (ax1,ax2,ax3)=plt.subplots(3,1)
fig.suptitle("3 ECG Data plot")
#plt.plot(time, ecgData.reshape((-1)))

ax1.plot(time, X.reshape((-1)), color='blue',linewidth=0.2)
ax2.plot(time, Y.reshape((-1)), color='blue',linewidth=0.2)
ax3.plot(time, Z.reshape((-1)), color='blue',linewidth=0.2)

ax1.set_ylabel("ECG1")
ax2.set_ylabel("ECG2")
ax3.set_ylabel("ECG3")

fig.set_figheight(9)
fig.set_figwidth(9)
plt.xlabel("time in s")
ax1.grid(color = 'red',linewidth=0.1)
ax2.grid(color = 'red',linewidth=0.1)
ax3.grid(color = 'red',linewidth=0.1)
plt.show()

#ecg_plot.plot(X, sample_rate = 500)
#ecg_plot.show()

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
#fig.colorbar(surf)

#plt.show()