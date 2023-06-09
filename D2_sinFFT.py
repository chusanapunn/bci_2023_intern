import numpy as np
from matplotlib import pyplot as plt
Fs=10000
tstep=1/Fs
f0=500

N=int(Fs/f0)

t=np.linspace(0,(N-1)*tstep,N) #timesteps

fstep=Fs/N #frequency interval
f=np.linspace(0,(N-1)*fstep,N)

y=1* np.sin(2*np.pi*f0*t) #Amplitude 1 of 100 hz
X=np.fft.fft(y)


X_mag=np.abs(X)/N

fig, [ax1,ax2]=plt.subplots(nrows=2,ncols=1)

#Graph Plot
ax1.plot(t,y,".-")
ax1.set_ylabel("SineGraph To time")

#FFT plot
ax2.plot(f,X_mag,".-")
ax2.set_ylabel("FFT to frequency")

plt.show()