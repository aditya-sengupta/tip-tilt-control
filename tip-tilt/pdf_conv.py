# find PDF of telescope vibrations
# this wasn't very successful.

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from scipy.integrate import quad
from matplotlib import pyplot as plt

def convolve(f1, f2):
    def convolution(t):
        def integrand(tau):
            return f1(tau) * f2(t - tau)
        return quad(integrand, -np.inf, np.inf)[0]
    return convolution

def get_f1(T, lower, upper):
    def f1(x):
        if lower*T <= x and x <= upper*T:
            return (1/((upper - lower)*T))
        else:
            return 0
    return f1

def f2(x):
    if -2*np.pi <= x and x <= 0:
        return 1/(2*np.pi)
    else:
        return 0

def test_conv():
    f1 = get_f1(0.01, 10, 1000)
    test_time = np.arange(-30, 30, 0.01)
    output = np.vectorize(convolve(f1, f2))(test_time)
    print(output)
    plt.show()

def exp_value(N):
    #finds the expected value of vibrations.
    N_vibrations = N

    dx = np.zeros(1000000)
    dy = np.zeros(1000000)
    i = 0

    for t in np.arange(0, 1000, 1e-3):
        vib_freqs    = np.random.uniform(low=10.0, high=1000.0, size=N_vibrations)  # Hz
        vib_amps     = np.random.uniform(low=0.1, high=1, size=N_vibrations) # milliarcseconds
        vib_phase    = np.random.uniform(low=0.0, high=2*np.pi, size=N_vibrations)  # radians
        vib_pa       = np.random.uniform(low=0.0, high=2*np.pi, size=N_vibrations)  # radians
        dx[i] = sum([-vib_amps[j] * np.sin(vib_phase[j]) * np.sin(vib_freqs[j] * t - vib_phase[j]) for j in range(N_vibrations)])
        dy[i] = sum([vib_amps[j] * np.cos(vib_phase[j]) * np.sin(vib_freqs[j] * t - vib_phase[j]) for j in range(N_vibrations)])
        i += 1

    return np.mean(dx**2) + np.mean(dy**2)

exp_value()
