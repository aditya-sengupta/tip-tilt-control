from observer import *
import numpy as np
from matplotlib import pyplot as plt
import sys
from aberrations import *

def make_filter_help(measurement_noise):
    vib_freqs, vib_amps, vib_phase, vib_damping = make_vibe_params(N_vib_app)
    truth = make_vibe_data((vib_freqs, vib_amps, vib_phase, vib_damping))
    measurements = make_noisy_data(truth, measurement_noise)
    params = np.vstack((vib_amps, vib_freqs, vib_damping, vib_phase)).T
    physics = sum(damped_harmonic(p) for p in params)
    return params, truth, measurements, physics


def make_perfect_filter(process_noise=0, measurement_noise=0.12):
    # test filtering assuming you have perfect physics and you know it
    params, truth, measurements, physics = make_filter_help(measurement_noise)
    state, A, P, Q, H, _ = make_kfilter(params, np.array([process_noise ** 2] * N_vib_app))
    R = np.array([[measurement_noise]])
    return [state, A, P, Q, H, R], truth, measurements, physics


def make_near_perfect_filter(process_noise=0.12, measurement_noise=0.12):
    # test filtering assuming you have (close to) perfect physics and you know it
    params, truth, measurements, physics = make_filter_help(measurement_noise)
    # suppose you have good physics within reason: add white noise by driving Q high,
    # and only get integral frequencies + amplitudes accurate to 0.1.
    params[0] = np.round(params[0], 1)
    params[1] = np.round(params[1])
    state, A, P, Q, H, _ = make_kfilter(params, np.array([process_noise ** 2] * N_vib_app))
    R = np.array([[measurement_noise]])
    return [state, A, P, Q, H, R], truth, measurements, physics


def test_sysid_freq(disp=True, N=10):
    truth = make_atm_data()[0] + make_vibe_data(N)
    measurements = make_noisy_data(truth)
    params, variances = vibe_fit_freq(noise_filter(get_psd(measurements)))
    physics = sum(damped_harmonic(p) for p in params)
    filtered = kfilter(make_kfilter(params, variances), measurements)
    rms = np.sqrt(np.mean((filtered - truth)**2))
    if disp:
        plt.plot(times, truth, label='Truth')
        plt.plot(times, physics, label='Physics')
        plt.plot(times, filtered, label='Filtered')
        plt.legend()
        plt.title("Frequency fit, rms error (mas) = " + str(np.round(rms, 3)))
        plt.show()
    return rms


def test_time_fit(disp=True, N=1):
    truth = make_vibe_data(N)
    measurements = make_noisy_data(truth)
    coeffs = reconstruct_modes(measurements, N)
    plt.plot(times, measurements, label='Measurements')
    plt.plot(times, truth, label='Truth')
    plt.plot(times, sum(np.exp(-coeffs[i] * times).real for i in range(N)), label='Reconstructed')
    plt.legend()
    plt.show()


def test_filter(args, truth, measurements, physics, disp=(True, False, False, True)):
    filtered = kfilter(args, measurements)
    toplot = (truth, measurements, physics, filtered)
    labels = ('Truth', 'Measurements', 'Physics', 'Filtered')
    for t, l, d in zip(toplot, labels, disp):
        if d:
            plt.plot(times, t, label=l)
    plt.legend()
    plt.show()
    return np.sqrt(np.mean((filtered - truth)**2))


test_perfect_filter = lambda: print(test_filter(*make_perfect_filter()))
test_near_perfect_filter = lambda: print(test_filter(*make_near_perfect_filter()))

if __name__ == "__main__":
    test_near_perfect_filter()
