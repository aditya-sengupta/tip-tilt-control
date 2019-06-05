import numpy as np
import sys
sys.path.append("..")
from dynamics.dynamic import DynamicSystem
from scipy import signal
import matplotlib.pyplot as plt

class TipTilt(DynamicSystem):

    STATE_SIZE = 2
    INPUT_SIZE = 2

    def __init__(self, measurement_noise, dx, dy, sampling_freq=1000):
        self.state = np.array([0, 0])
        self.simend = False
        self.sampling_freq = sampling_freq #Hz
        self.P = np.zeros((2, 2))
        self.dx = dx
        self.dy = dy
        process_noise = np.std(np.concatenate((dx, dy))) #magic number that should come from id_vibe
        self.Q = process_noise * np.identity(2)
        self.H = np.identity(2)
        self.R = measurement_noise * np.identity(2) # noise model to be updated maybe

    def id_vibe(self, applied_freqs=None, applied_amps=None, applied_pa=None):
        # finds the state-transition matrix A from sample dx and dy data
        # applied_ is in there for debug.
        # FFT

        dt = 1/self.sampling_freq
        mode_amps_x = np.fft.fft(self.dx)[:self.dx.size//2] * dt / 2.5
        mode_amps_y = np.fft.fft(self.dy)[:self.dy.size//2] * dt / 2.5

        ind = signal.find_peaks(mode_amps_x)[0][:5]
        maxes = np.abs(mode_amps_x[ind])
        print(maxes / applied_amps)
        freqs = np.fft.fftfreq(self.dx.size, dt)[:self.dx.size//2] * 2 * np.pi

        plt.subplot(2,2,1)
        plt.plot(freqs, np.abs(mode_amps_x))
        plt.xlim(0, 500)
        if applied_freqs is not None:
            plt.scatter(applied_freqs, applied_amps * np.abs(np.sin(applied_pa)), color='g')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (mas)")
        plt.title("Fourier transform of x deviations")

        plt.subplot(2,2,2)

        plt.plot(freqs, np.abs(mode_amps_y))
        if applied_freqs is not None:
            plt.scatter(applied_freqs, applied_amps * np.abs(np.cos(applied_pa)), color='g')
        plt.xlim(0, 500)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (mas)")
        plt.title("Fourier transform of y deviations")

        # now this should match up to dx/dy data

        t = np.arange(0, self.dx.size/self.sampling_freq, 1/self.sampling_freq)
        reconstructed_dx = sum([amp * np.sin(2*np.pi*freq*t) for amp, freq in zip(mode_amps_x, freqs)])
        reconstructed_dy = sum([amp * np.sin(2*np.pi*freq*t) for amp, freq in zip(mode_amps_y, freqs)])

        plt.subplot(2,2,3)
        plt.plot(t, reconstructed_dx, label='reconstructed')
        plt.plot(t, self.dx, label='original')
        plt.xlabel("Time (s)")
        plt.ylabel("Deviation (mas)")
        plt.legend()
        plt.title("Reconstructed and original x deviations")

        #print("SD of x deviations: " + str(np.std(self.dx)))
        #print("SD of x residual: " + str(np.std(self.dx - reconstructed_dx)))

        plt.subplot(2,2,4)
        plt.plot(t, reconstructed_dy, label='reconstructed')
        plt.plot(t, self.dy, label='original')
        plt.xlabel("Time (s)")
        plt.ylabel("Deviation (mas)")
        plt.legend()
        plt.title("Reconstructed and original y deviations")

        plt.show()

        #print("SD of y deviations: " + str(np.std(self.dy)))
        #print("SD of y residual: " + str(np.std(self.dy - reconstructed_dy)))

        return np.identity(2)

    def evolve(self, t, dt):
        A = np.identity(2)
        B = np.identity(2)
        return (A, B)

    def ext_input(self, t):
        k = 1
        return (-k*self.state, "negative state")

    def reset(self):
        self.state = np.array([0, 0])
