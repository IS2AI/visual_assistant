import pyaudio
import numpy as np
import time
import math

RATE = 44100
CHUNK = 32678
chans = 2
volume = 1
f = 440.0
duration = 3
frequencies = [440, 525.3]

def sine_wave(frequency = 440.0, rate = RATE, vol = volume):
    return((np.sin(2 * np.pi * np.arange(rate / frequency) * frequency / rate)).astype(np.float32))

def convert_to_2_channels(left, right):
    stereo = []
    for i in range(len(left)):
        stereo.append(left[i])
        stereo.append(right[i])    
    return(stereo)



sines = [sine_wave(440.0), sine_wave(523.25), sine_wave(659.25), sine_wave(783.99)]
print()