from os.path import dirname, join as pjoin

from scipy.fft import fft
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

data_dir = pjoin(dirname(scipy.io.__file__), 'tests', 'data')
wav_fname = pjoin(data_dir, 'FDMAMixedAudio.wav')
samplerate, data = wavfile.read(wav_fname)
print((data.shape))
if len(data.shape) > 1:
   data = data[:, 0]

print(f"number of channels = {data.shape[0]}")
print(samplerate)
N= 2* samplerate

length = data.shape[0] / samplerate
print(f"length = {length}s")
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data, label="Left channel")
plt.plot(time, data, label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

frequency = np.linspace(0.0, 100, int (N/2))

freq_data = fft(data)
y = 2/N * np.abs (freq_data [0:np.int (N/2)])
plt.plot(frequency, y)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

