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
time_id = 1  # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = np.arange(0, time_id, 1 / f_sampling)  # array of times to operate on
num_steps = int(time_id * f_sampling)
D = 10.95
r0 = 16.5e-2   
pupil_size = 40
focal_samples = 20 # samples per lambda over D
focal_width = 20 # the number of lambda over Ds
wavelength = 500e-9 # meters
pupil_grid = make_pupil_grid(pupil_size)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, focal_samples, focal_width, wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
aperture = circular_aperture(1)(pupil_grid)


make_vib_amps = lambda N: np.random.uniform(low=0.1, high=1, size=N)  # milliarcseconds
make_vib_freqs = lambda N:  np.random.uniform(low=f_1, high=f_2, size=N)  # Hz
make_vib_damping = lambda N: np.random.uniform(low=1e-5, high=1e-4, size=N)  # unitless
make_vib_phase = lambda N: np.random.uniform(low=0.0, high=2 * np.pi, size=N)  # radians


def make_vibe_params(N=N_vib_app):
    return [f(N) for f in [make_vib_amps, make_vib_freqs, make_vib_damping, make_vib_phase]]


def make_1D_vibe_data(vib_params=None, times=times, N=N_vib_app):
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

def make_2D_vibe_data(times=times, N=N_vib_app):
    # adjusted so that each 'pos' mode is the solution to the DE
    # x'' + 2k w0 x' + w0^2 x = 0 with w0 = 2pi*f/sqrt(1-k^2) 
    # (chosen so that vib_freqs matches up with the PSD freq)
    params_x = make_vibe_params(N)
    params_y = deepcopy(params_x)
    params_y[0] = make_vib_amps(N)
    params_y[3] = make_vib_phase(N)
    return np.vstack((make_1D_vibe_data(params_x, times, N).T, make_1D_vibe_data(params_y, times, N).T)).T


def make_noisy_data(pos, noise=measurement_noise):
    return pos + np.random.normal(0, noise, np.shape(pos))


def center_of_mass(f):
    # takes in a Field, returns its CM.
    dims = np.array([np.max(f.grid.x) - np.min(f.grid.x), np.max(f.grid.y) - np.min(f.grid.y)])
    average = np.array([(f.grid.x * f).sum(), (f.grid.y * f).sum()]) / (f.sum())
    return average * focal_width * focal_samples / dims


def correction_for(self, wavelength):
    return self.transformation_matrix.dot(self.corrected_coeffs[0] / wavelength)

def correct_until(self, t):
    self.layer.evolve_until(t)

    if len(self.corrected_coeffs) > 0:
        coeffs = self.transformation_matrix_inverse.dot(self.correction_for(1))
    else:
        coeffs = self.transformation_matrix_inverse.dot(self.layer.phase_for(1))
    if len(self.corrected_coeffs) > self.lag:
        self.corrected_coeffs.pop(0)
    self.corrected_coeffs.append(coeffs)
    
ModalAdaptiveOpticsLayer.correction_for = correction_for
ModalAdaptiveOpticsLayer.correct_until = correct_until


def make_specific_tt(weights):
    # weights = number of desired lambda-over-Ds the center of the PSF is to be moved. Tuple with (tip_wt, tilt_wt).
    tt = [zernike(*ansi_to_zernike(i), 1)(pupil_grid) for i in (1, 2)]
    tt_wf = Wavefront(aperture * np.exp(1j * np.pi * sum([w * z for w, z in zip(tt_weights, tt)]) * scale), wavelength)
    return tt_wf


def make_atm_data():
    layers = make_standard_atmospheric_layers(pupil_grid)
    tt = [zernike(*ansi_to_zernike(i), 1)(pupil_grid) for i in (1, 2)] # tip-tilt phase basis
    MOAlayers = [ModalAdaptiveOpticsLayer(layer, controlled_modes=ModeBasis(tt), lag=0) for layer in layers]
    conversion = (wavelength / D) * 206265000 / focal_samples

    tt_cms = np.zeros((f_sampling * time_id, 2))
    for n in range(f_sampling * time_id):
        wf = Wavefront(aperture, wavelength)
        phase = wf.phase
        for layer in MOAlayers:
            layer.correct_until(times[n])
            phase += layer.phase_for(wavelength)
        wf = Wavefront(aperture * np.exp(1j * phase), wavelength)
        total_intensity = prop(wf).intensity
        tt_cms[n] = center_of_mass(total_intensity)

    tt_cms *= conversion # pixels to mas
    return tt_cms
