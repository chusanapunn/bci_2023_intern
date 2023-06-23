import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

mat = sio.loadmat("Data/noisyecg.mat")

ecg3 = mat['ecg3']
ecgData=ecg3[0]
DataLength= ecgData.size
freq = np.arange(ecgData.size)
Sampling_rate = 500 # sample Frequency rate
tstep=1/DataLength
N=int(DataLength/Sampling_rate)
fstep=DataLength/N
f=np.linspace(0,(N-1)*fstep,DataLength)

time = DataLength / Sampling_rate # Time recorded
t=np.linspace(0,N,DataLength)
#f=np.linspace(0,fstep,nsample)



#X_mag=np.abs(X)/nsample
frequencies = np.fft.fftfreq(len(ecgData), d=1/Sampling_rate)
fft_X=frequencies[:len(ecgData)]
fft_X=abs(fft_X)

Y =np.fft.fft(ecgData)
fft_Y=np.abs(Y)/N

print("Check :" + str(fft_X))

# Plot ECG
fig, [ax1,ax2] =plt.subplots(2,1)

fig.suptitle("ECG FFT Plot")
ax1.plot(t,ecgData,linewidth=0.2)
ax2.plot(fft_X, fft_Y, color='blue',linewidth=0.2)
fig.set_figheight(9)
fig.set_figwidth(9)
ax1.set_ylabel("ECG raw signal")
ax2.set_ylabel("FFT Processed Signal")
ax1.set_xlabel("Time in Second")
ax2.set_xlabel("Frequency in Hertz")
ax1.grid(color = 'red',linewidth=0.1)
ax2.grid(color = 'red',linewidth=0.1)

plt.show()