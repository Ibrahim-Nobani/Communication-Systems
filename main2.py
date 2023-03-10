from os.path import dirname, join as pjoin

from scipy.fft import fft
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import IPython
#Ibrahim Nobani 1190278

def RangeIndex(BWrange,fcenIndex,frq_DS,maxMin):
    RangeIndex = (np.abs(frq_DS - BWrange)).argmin() - fcenIndex
    if (maxMin==0):
        RangeIndexMin = fcenIndex - RangeIndex
        if RangeIndexMin < 0:
            RangeIndexMin = 0
        return RangeIndexMin
    if(maxMin==1):
        RangeIndexMax = fcenIndex + RangeIndex
        if RangeIndexMax > len(frq_DS) - 1:
            RangeIndexMax = len(frq_DS) - 1
        return RangeIndexMax

def plotTimeFreq(y, Fs, BWrange):
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs

    t = np.arange(0, n * Ts, Ts)  # time vector

    frq = k / T  # two sides frequency range
    fcen = frq[int(len(frq) / 2)]
    frq_DS = frq - fcen
    frq_SS = frq[range(int(n / 2))]  # one side frequency range

    Y = np.fft.fft(y)  # fft computing and normalization
    yinv = np.fft.ifft(Y).real  # ifft computing and normalization
    Y_DS = np.roll(Y, int(n / 2))
    Y_SS = Y[range(int(n / 2))]

    fcenIndex = (np.abs(frq_DS)).argmin()
    RangeIndexMin = RangeIndex(BWrange,fcenIndex,frq_DS,0)
    RangeIndexMax = RangeIndex(BWrange, fcenIndex, frq_DS, 1)
    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    ax[1].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Y_DS[RangeIndexMin:RangeIndexMax]),
               'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()
    return yinv
def BBfilter(B,B2,Y_DS,frq_DS,n,Fs,rate):
    fcenIndex = (np.abs(frq_DS)).argmin()
    fBWIndex = (np.abs(frq_DS - B)).argmin()
    fBWIndex2= (np.abs(frq_DS - B2)).argmin()
    B = frq_DS[fBWIndex]
    Mask_DS = np.ones(len(frq_DS))
    Yf_DS = np.copy(Y_DS)
    for cnt in range(len(frq_DS)):
        if ~((-1 * B2 > (frq_DS[cnt]) > -1 * B) or (B2 < (frq_DS[cnt]) < B)):
            Mask_DS[cnt] = 0;
            #print(B,frq_DS[cnt],Yf_DS[cnt])
            Yf_DS[cnt] = Y_DS[cnt] * 0;

    Yf = np.roll(Yf_DS, int(n /2))
    Yinv2 = np.fft.fft(Yf)
    yinv = np.fft.ifft(Yf).real  # ifft computing and normalization
    yinv = np.array(yinv)
    yinv_int = yinv.astype(np.int16)
    RangeIndexMin = RangeIndex(B, fcenIndex, frq_DS, 0)
    RangeIndexMax = RangeIndex(B+50, fcenIndex, frq_DS, 1)
    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Mask_DS[RangeIndexMin:RangeIndexMax]),
               'r')  # plotting the spectrum
    ax[0].set_xlabel('Freq (Hz)')
    ax[0].set_ylabel('|H(freq)|')
   # print(abs(Yf_DS[RangeIndexMin:RangeIndexMax]))
    ax[1].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf_DS[RangeIndexMin:RangeIndexMax]),
               'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()
    return yinv
def demodulation (yinv,t,Fs,rate,n,frq_DS,FC) :
    yinv_int = yinv.astype(np.int16)
    y2 = [float(x) for x in yinv]
    carrier_signal = 0.1 * np.cos(2 * np.pi * FC * t)
    output_signal = y2 * carrier_signal
    plotTimeFreq(output_signal, Fs, BWrange)
    Yf = np.fft.fft(output_signal)
    Yf = np.roll(Yf, int(n / 2))
    LPBW = input("Insert the low pass filter bandwidth: ")
    BW = int(LPBW)
    yinv2=BBfilter(BW,0,Yf,frq_DS,n,Fs,rate)
    #yinv2 = upsampler(yinv2, 30)
    plotTimeFreq(yinv2, Fs, BWrange)
    yinv_int16 = yinv2.astype(np.int16)
    wavfile.write("sound1.wav", rate, yinv_int16)
    IPython.display.Audio(yinv2,rate=rate)
BWrange=10000
Ts=1;

########-------------------
def readMixedandGraph():
    rate1, data1 = wavfile.read('FDMAMixedAudio12.wav')
    length = data1.shape[0] / rate1
    #print(rate1)
    #print(f"length = {length}s")
    time = np.linspace(0., length, data1.shape[0])
    #Fs=rate1*upsamplerate;
    Fs=rate1
    #Fs=1.0/time[1]
    Ts = 1.0/Fs;
    #print(len(time))
    plotTimeFreq(data1, Fs, BWrange)
    t = np.arange(0, len(data1) * Ts, Ts)  # time vector
    y = [float(x) for x in data1]
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    fcen = frq[int(len(frq) / 2)]
    frq_DS = frq - fcen
    frq_SS = frq[range(int(n / 2))]  # one side frequency range

    Y = np.fft.fft(y)  # fft computing and normalization
    yinv = np.fft.ifft(Y).real  # ifft computing and normalization
    Y_DS = np.roll(Y, int(n / 2))
    Y_SS = Y[range(int(n / 2))]

    fcenIndex = (np.abs(frq_DS)).argmin()
    RangeIndex = (np.abs(frq_DS - BWrange)).argmin() - fcenIndex
    LowerFrequency=input("Insert the lower frequency range: ")
    LowerFrequency=int(LowerFrequency)
    HigherFrequency = input("Insert the Higher frequency range: ")
    HigherFrequency = int(HigherFrequency)
    filteredY=BBfilter(HigherFrequency, LowerFrequency, Y_DS,frq_DS,n,Fs,rate1)
    FC = input("Insert the carrier frequency: ")
    FC = int(FC)
    demodulation(filteredY,t,Fs,rate1*35,n,frq_DS,FC)


ans=1
while ans:
    print ("""
    1.Read the FDMA signals and graph in time and frequency domain
    2.Change Bandiwdth Range
    3.Exit/Quit
    """)
    ans=input("What would you like to do? ")
    if ans=="1":
      readMixedandGraph()
    elif ans=="2":
      BWrange=input("Insert the Bandwidth Range: ")
      BWrange = int(BWrange)
    elif ans=="3":
      print("\n Goodbye")
      ans = False
    elif ans !="":
      print("\n Not Valid Choice Try again")