import numpy as np
import matplotlib.pyplot as plt

fs = 256
second = 2  # 2 second
t = np.linspace(0, second, fs * second, False) 

# Create Sine Wave
sine = np.sin(2 * np.pi * 10 * t)  
plt.plot(t, sine)
plt.title('Sine wave with 10 Hz, fs = 128 Hz')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.show()