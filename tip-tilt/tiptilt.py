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

    def id_vibe(self):
        # finds the state-transition matrix A from sample dx and dy data
        freqs, dx_psd = signal.welch(self.dx, self.sampling_freq)
        _, dy_psd = signal.welch(self.dy, self.sampling_freq)
        freqs = freqs*2*np.pi # dunno why this apparently returns rad/s not Hz
        # I'm pretty sure freqs should be the same for both calls to welch because dx has the same dimensions as dy, and the sampling frequency is the same.
        ind = signal.find_peaks(dx_psd)[0].tolist() # should be the same for dx and dy again
        mode_freqs = freqs[ind]
        mode_amps_x = 4*np.sqrt(dx_psd[ind])
        mode_amps_y = 4*np.sqrt(dy_psd[ind])

        # now this should match up to dx/dy data

        t = np.arange(0, self.dx.size/self.sampling_freq, 1/self.sampling_freq)
        reconstructed_dx = sum([amp * np.sin(2*np.pi*freq*t) for amp, freq in zip(mode_amps_x, mode_freqs)])
        reconstructed_dy = sum([-amp * np.cos(2*np.pi*freq*t) for amp, freq in zip(mode_amps_y, mode_freqs)])

        plt.subplot(2,2,1)
        plt.plot(t, reconstructed_dx, label='reconstructed')
        plt.plot(t, self.dx, label='original')
        plt.xlabel("Time (s)")
        plt.ylabel("Deviation (mas)")
        plt.legend()
        plt.title("Reconstructed and original x deviations")

        #print("SD of x deviations: " + str(np.std(self.dx)))
        #print("SD of x residual: " + str(np.std(self.dx - reconstructed_dx)))

        plt.subplot(2,2,2)
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
