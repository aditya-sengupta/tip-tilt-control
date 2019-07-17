from kalman import *
import numpy as np
from matplotlib import pyplot as plt
import sys
from aberrations import make_atm_data, make_vibe_data, make_noisy_data

def test_quality1(disp=False, process_noise=0, measurement_noise=0.12):
    # test filtering assuming you have (close to) perfect physics and you know it
    vib_freqs = np.random.uniform(low=50, high=350, size=N_vib_app)  # Hz
    vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
    vib_damping = np.random.uniform(low=1e-5, high=1e-4, size=N_vib_app)  # unitless

    truth = sum([vib_amps[i] * np.cos(2 * np.pi * vib_freqs[i] * times - vib_phase[i])
                    * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

    measurements = make_noisy_data(truth, measurement_noise)

    variances = np.array([process_noise ** 2] * N_vib_app)
    params = np.vstack((vib_amps, vib_freqs, vib_damping, vib_phase)).T
    state, A, P, Q, H, R = make_kfilter(params, variances)
    R = np.array([[measurement_noise]])
    filtered = kfilter((state, A, P, Q, H, R), measurements)
    physics = sum(damped_harmonic(p) for p in params)
    #print("Average difference between measurements and physics: ", np.sqrt(np.mean((measurements - physics)**2)))
    if disp:
        plt.plot(times, measurements, label='Measurements')
        #plt.plot(times, physics, label='Physics')
        plt.plot(times, filtered, label='Filtered')
        plt.legend()
        plt.show()
    print(np.sqrt(np.mean((filtered - truth)**2)))

def test_quality2(disp=False, process_noise=0.03, measurement_noise=0.06):
    # test filtering assuming you have (close to) perfect physics and you know it
    vib_freqs = np.random.uniform(low=50, high=350, size=N_vib_app)  # Hz
    vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
    vib_damping = np.random.uniform(low=1e-5, high=1e-4, size=N_vib_app)  # unitless

    truth = sum([vib_amps[i] * np.cos(2 * np.pi * vib_freqs[i] * times - vib_phase[i])
                    * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

    measurements = make_noisy_data(truth, measurement_noise)

    variances = np.array([process_noise ** 2] * N_vib_app)
    # suppose you have good physics within reason: add white noise by driving Q high,
    # and only get integral frequencies + amplitudes accurate to 0.1.
    params = np.vstack((np.round(vib_amps, 1), np.round(vib_freqs), vib_damping, vib_phase)).T
    state, A, P, Q, H, R = make_kfilter(params, variances)
    R = np.array([[measurement_noise]])
    filtered = kfilter((state, A, P, Q, H, R), measurements)
    physics = sum(damped_harmonic(p) for p in params)
    #print("Average difference between measurements and physics: ", np.sqrt(np.mean((measurements - physics)**2)))
    if disp:
        plt.plot(times, measurements, label='Measurements')
        #plt.plot(times, physics, label='Physics')
        plt.plot(times, filtered, label='Filtered')
        plt.legend()
        plt.show()
    print(np.sqrt(np.mean((filtered - truth)**2)))

def test_sysid_freq(disp=True, N=0):
    truth = make_atm_data()[0] + make_vibe_data(N)
    measurements = make_noisy_data(truth)
    params, variances = vibe_fit_freq(get_psd(measurements))
    filtered = kfilter(make_kfilter(params, variances), measurements)
    rms = np.sqrt(np.mean((filtered - truth)**2))
    if disp:
        plt.plot(times, truth, label='Truth')
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

if __name__ == '__main__':
    globals()[sys.argv[1]]()