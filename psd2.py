#!/usr/bin/python
# python
""" 
Computes FFT and PSD of a spectra
Author: V. Gofron
Summer 2014 project
"""

from pylab import *
import numpy as np
from scipy.fftpack import fft, ifft

# data is extracted from file sample-array.data

data = np.genfromtxt("sample-array.data", delimiter = " ")
x = np.array(data)
y = fft(x)

print y

yinv = ifft(y)
print yinv

data_len = len(data)
time_int = 0.00005
data_end_pt1 = data_len * time_int
t = arange(0.0, data_end_pt1, time_int)
# signal = np.array(data, dtype = float)
freqs = np.fft.fftfreq(data.size, time_int)

# code for plotting data
plot(t, x, '-r', label = 'Data')
xlabel('Time [s]')
ylabel('Amplitude [arb. units]')
title('Data')
legend(loc = 'upper right')
show ()

# code for plotting FFT
plot(freqs,abs(y), '-g', label = 'FFT')

xlabel('Frequency [s^-1]')
ylabel('Amplitude [arb. units]')
title('FFT')
legend(loc = 'upper right')
show()


# PSD (Power Spectrum Density)
dt = 0.00005
data_end_pt2 = data_len * dt
s = np.array(data)
t = arange(0, data_end_pt2, dt)
nse = randn(len(t))
r = exp(-t/0.0005)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]

psd(s, 512, 1/dt)
title('PSD')
# xlim(0, 200)
show()

# Subplots for displaying all three graphes (data, FFT, PSD)
subplot(311)
plot(t,s, '-r', label = 'Data')
# xlim(0, 2000)
xlabel('                                                    Time [s]')
ylabel('Amplitude [arb. units]')
# title('Data')
legend(loc = 'upper right')

subplot(312)
plot(t, abs(y), '-g', label = 'FFT')
xlabel('Frequency [s^-1]')
ylabel('Amplitude [arb. units]')
# title('FFT')
legend(loc = 'upper right')

subplot(313)
psd(s, 512, 1/dt)
title('PSD')
# legend(loc = 'upper right')
xlim(0, 200)

show()

"""
matplotlib.org/examples/pyplot_examples/psd_demo.html
"""
