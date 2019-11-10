from controller import *
import numpy as np
from scipy import signal

size = 4000
steps = 4000
N = 4

keck_normalizer = 0.6 * (600e-9 / (2 * np.pi)) *  206265000
truth = np.load('./combined.npy')[:size,0]
_, psd = signal.periodogram(truth + np.random.normal(0, noise, truth.size), fs=f_sampling)
kalman = Controller('kalman', make_kfilter_turb(make_impulse(truth[:size//2], N=N), truth[:N] + np.random.normal(0, noise, (N,))))
vibe = Controller('kalman', make_kfilter_vibe(*vibe_fit_freq(noise_filter(psd), N=2)), calibration_time=200)
stdint = Controller('stdint')
baseline = Controller('baseline')