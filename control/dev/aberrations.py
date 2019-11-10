# Make aberrations. To be replaced by real-life data.
# Global variables are set separately in both, to test the effects of not perfectly knowing them.

import sys
sys.path.append("..")
from hcipy.hcipy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, ndimage
from copy import deepcopy

N_vib_app = 10
f_sampling = 1000  # Hz
f_1 = f_sampling / 60  # lowest possible frequency of a vibration mode
f_2 = f_sampling / 3  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
D = 10.95
r0 = 16.5e-2   
pupil_size = 16
focal_samples = 8 # samples per lambda over D
focal_width = 8 # half the number of lambda over Ds
D_magic = 1
wavelength = 500e-9 # meters
focal_size = 2 * focal_samples * focal_width
pupil_grid = make_pupil_grid(pupil_size, D_magic)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, focal_samples, focal_width, wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
aperture = circular_aperture(D_magic)(pupil_grid)
layers = make_standard_atmospheric_layers(pupil_grid)

make_vib_amps = lambda N: np.random.uniform(low=0.1, high=1, size=N)  # milliarcseconds
make_vib_freqs = lambda N:  np.random.uniform(low=f_1, high=f_2, size=N)  # Hz
make_vib_damping = lambda N: np.random.uniform(low=1e-5, high=1e-4, size=N)  # unitless
make_vib_phase = lambda N: np.random.uniform(low=0.0, high=2 * np.pi, size=N)  # radians


def make_vibe_params(N=N_vib_app):
    return [f(N) for f in [make_vib_amps, make_vib_freqs, make_vib_damping, make_vib_phase]]


def make_1D_vibe_data(steps, vib_params=None, N=N_vib_app):
    # adjusted so that each 'pos' mode is the solution to the DE
    # x'' + 2k w0 x' + w0^2 x = 0 with w0 = 2pi*f/sqrt(1-k^2) 
    # (chosen so that vib_freqs matches up with the PSD freq)
    if vib_params is None:
        vib_amps, vib_freqs, vib_damping, vib_phase = make_vibe_params(N)
    else:
        vib_amps, vib_freqs, vib_damping, vib_phase = vib_params
        N = vib_freqs.size

    times = np.linspace(0, steps / f_sampling, steps)
    pos = sum([vib_amps[i] * np.cos(2 * np.pi * vib_freqs[i] * times - vib_phase[i])
               * np.exp(-(vib_damping[i]/(1 - vib_damping[i]**2)) * 2 * np.pi * vib_freqs[i] * times) 
               for i in range(N)])

    return pos

def make_2D_vibe_data(steps, N=N_vib_app):
    # adjusted so that each 'pos' mode is the solution to the DE
    # x'' + 2k w0 x' + w0^2 x = 0 with w0 = 2pi*f/sqrt(1-k^2) 
    # (chosen so that vib_freqs matches up with the PSD freq)
    params_x = make_vibe_params(N)
    params_y = deepcopy(params_x)
    params_y[0] = make_vib_amps(N)
    params_y[3] = make_vib_phase(N)
    return np.vstack((make_1D_vibe_data(steps, params_x, N).T, make_1D_vibe_data(steps, params_y, N).T)).T


def make_noisy_data(pos, noise=measurement_noise):
    return pos + np.random.normal(0, noise, np.shape(pos))


def center_of_mass(f):
    # takes in a Field, returns its CM.
    s = f.grid.shape[0]/2
    normalize = s / (focal_samples * max(f.grid.x))
    return normalize * np.array([(f.grid.x * f).sum(), (f.grid.y * f).sum()])/f.sum()


def make_specific_tt(weights):
    # weights = number of desired lambda-over-Ds the center of the PSF is to be moved. Tuple with (tip_wt, tilt_wt).
    tt = [zernike(*ansi_to_zernike(i), D_magic)(pupil_grid) for i in (1, 2)]
    tt_wf = Wavefront(aperture * np.exp(1j * np.pi * sum([w * z for w, z in zip(weights, tt)])), wavelength)
    return tt_wf

def make_atm_data(steps, wf=None):
    conversion = (wavelength / D) * 206265000 / focal_samples
    if wf is None:
        wf = Wavefront(aperture, wavelength) # can induce a specified TT here if desired

    tt_cms = np.zeros((steps, 2))
    for n in range(steps):
        wf = Wavefront(aperture, wavelength)
        for layer in layers:
            layer.evolve_until(n / f_sampling)
            wf = layer(wf)
        tt_cms[n] = center_of_mass(prop(wf).intensity)

    tt_cms *= conversion # pixels to mas
    return tt_cms
