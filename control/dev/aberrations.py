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
f_2 = f_sampling / 30  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
time_id = 1  # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = np.arange(0, time_id, 1 / f_sampling)  # array of times to operate on
num_steps = int(time_id * f_sampling)
D = 10.95
r0 = 16.5e-2   
p = 24
wavelength = 5e-7
pupil_grid = make_pupil_grid(p, D)
g = make_pupil_grid(p, diameter=D)
mask = circular_aperture(D)(g)


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

    vib_freqs *= 2 * np.pi # conversion to rad/s

    pos = sum([vib_amps[i] * np.cos(vib_freqs[i] * times - vib_phase[i])
               * np.exp(-(vib_damping[i]/(1 - vib_damping[i]**2)) * vib_freqs[i] * times) 
               for i in range(N)])

    return pos


def make_noisy_data(pos, noise=measurement_noise):
    return pos + np.random.normal(0, noise, np.size(times))


def turbulence_phases():
    outer_scale, wind_velocity = 20, 15
    Cn2 = r0**(-5/3) / (0.423 * (2 * np.pi/wavelength)**2)
    # returns a list of Fields representing the turbulence phases over time
    single_layer_atmos = InfiniteAtmosphericLayer(pupil_grid,  Cn_squared=Cn2, L0=outer_scale, 
                                                  velocity=wind_velocity, use_interpolation=True)
    single_layer_turb = [None] * times.size
    for n in range(times.size):
        single_layer_atmos.evolve_until(times[n])
        turb = single_layer_atmos.phase_for(wavelength)
        single_layer_turb[n] = Field(turb, grid=g) * mask
    return single_layer_turb


def make_tt(phases):
    modes = [i for i in range(1,3)] # 1 is tip 2 is tilt
    basis = np.vstack([zernike(*ansi_to_zernike(i), D=D, grid=g) for i in modes]).T
    tt_phases = [None] * times.size
    to_multiply = basis.dot(np.linalg.inv(basis.T.dot(basis)).dot(basis.T))
    for i in range(times.size):
        tt_phases[i] = Field(to_multiply.dot(phases[i]), g)
    return tt_phases


def make_star(cutoff=1e-3, spread=5):
    def star_helper(i, j):
        center = g.shape//2
        intensity = np.exp(-np.sum((np.array([i, j]) - center)**2)/spread)
        if intensity > cutoff:
            return intensity
        return 0.0

    return Field(np.fromfunction(np.vectorize(star_helper), g.shape).flatten(), g)


def convolve(f1, f2):
    assert isinstance(f1, Field)
    assert isinstance(f2, Field)
    assert np.all(f1.grid.shape == f2.grid.shape)
    convolved = signal.convolve2d(np.reshape(f1, f1.grid.shape), np.reshape(f2, f2.grid.shape))[::2, ::2].flatten()
    return Field(convolved, f1.grid)


def make_atm_data():
    propagator = FresnelPropagator(input_grid=g, distance=10, num_oversampling=2, refractive_index=1.5)
    star = make_star()
    tt_phases = make_tt(turbulence_phases())
    conversion = 0.04
    cm = np.zeros((times.size, 2))
    for i in range(times.size):
        convolved_image = convolve(star, tt_phases[i])
        detector_phase = propagator.forward(Wavefront(convolved_image, wavelength)).phase
        cm[i] = ndimage.measurements.center_of_mass(np.array(detector_phase.reshape(g.shape)))

    cm *= conversion    
    cm -= np.tile(np.mean(cm, axis=0), (times.size, 1))
    cm = cm.T
    return cm

pos = make_noisy_data(make_vibe_data() + make_atm_data()[0])
