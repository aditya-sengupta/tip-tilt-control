# Make aberrations. To be replaced by real-life data.
# Global variables are set separately in both, to test the effects of not perfectly knowing them.

import sys
sys.path.append("..")
from hcipy.hcipy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, ndimage

N_vib_app = 10
f_sampling = 1000  # Hz
f_1 = f_sampling / 60  # lowest possible frequency of a vibration mode
f_2 = f_sampling / 3  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
time_id = 1  # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = np.arange(0, time_id, 1 / f_sampling)  # array of times to operate on
num_steps = int(time_id * f_sampling)
D = 10.95
r0 = 16.5e-2   
p = 16
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


def make_vibe_params(N=N_vib_app):
    vib_amps = np.random.uniform(low=0.1, high=1, size=N)  # milliarcseconds
    vib_freqs = np.random.uniform(low=f_1, high=f_2, size=N)  # Hz
    vib_damping = np.random.uniform(low=1e-5, high=1e-2, size=N)  # unitless
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N)  # radians
    return vib_amps, vib_freqs, vib_damping, vib_phase


def make_vibe_data(vib_params=None, N=N_vib_app):
    # adjusted so that each 'pos' mode is the solution to the DE
    # x'' + 2k w0 x' + w0^2 x = 0 with w0 = 2pi*f/sqrt(1-k^2) 
    # (chosen so that vib_freqs matches up with the PSD freq)
    if vib_params is None:
        vib_amps, vib_freqs, vib_damping, vib_phase = make_vibe_params(N)
    else:
        vib_amps, vib_freqs, vib_damping, vib_phase = vib_params
        N = vib_freqs.size

    pos = sum([vib_amps[i] * np.cos(2 * np.pi * vib_freqs[i] * times - vib_phase[i])
               * np.exp(-(vib_damping[i]/(1 - vib_damping[i]**2)) * 2 * np.pi * vib_freqs[i] * times) 
               for i in range(N)])

    return pos


def make_noisy_data(pos, noise=measurement_noise):
    return pos + np.random.normal(0, noise, np.size(times))


def center_of_mass(f):
    # takes in a Field, returns its CM.
    s = f.grid.shape[0]
    x, y = (n.flatten() for n in np.meshgrid(np.linspace(-s/2, s/2-1, s), np.linspace(-s/2, s/2-1, s)))
    return np.round(np.array((sum(f*x), sum(f*y)))/sum(f), 3)


def make_specific_tt(weights):
    # weights = number of desired lambda-over-Ds the center of the PSF is to be moved. Tuple with (tip_wt, tilt_wt).
    tt = [zernike(*ansi_to_zernike(i), D_magic)(pupil_grid) for i in (1, 2)]
    scale = focal_samples / 4.86754191 # magic number to normalize to around one lambda-over-D
    tt_wf = Wavefront(aperture * np.exp(1j * sum([w * z for w, z in zip(tt_weights, tt)]) * scale), wavelength)
    return tt_wf


def make_atm_data(tt_wf=None):
    conversion = (wavelength / D) * 206265000 / focal_samples
    if wf is None:
        tt_wf = Wavefront(aperture) # can induce a specified TT here if desired

    tt_cms = np.zeros((f_sampling * T, 2))
    for n in range(f_sampling * T):
        for layer in layers:
            layer.evolve_until(times[n])
            tt_wf = layer(tt_wf)
        tt_cms[n] = center_of_mass(prop(tt_wf).intensity)

    tt_cms *= conversion # pixels to mas
    return tt_cms.T

pos = make_noisy_data(make_vibe_data() + make_atm_data()[0])
