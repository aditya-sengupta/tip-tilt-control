from ao_observe import *
import numpy as np
from matplotlib import pyplot as plt
import sys

def test_quality0(process_noise=0.12, measurement_noise=0.06):
    N_vib_app = 1

    vib_freqs = np.random.uniform(low=50, high=350, size=N_vib_app)  # Hz
    vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
    vib_damping = np.random.uniform(low=1e-5, high=1e-4, size=N_vib_app)  # unitless

    truth = sum([vib_amps[i] * np.exp(1j * 2 * np.pi * vib_freqs[i] * times - vib_phase[i]).real
                    * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

    measurements = make_noisy_data(truth, 0)

    variances = np.array([process_noise ** 2] * N_vib_app)
    params = np.vstack((vib_amps, vib_freqs, vib_damping, vib_phase)).T
    state, A, P, Q, H, R = make_kfilter(params, variances)
    R = np.array([[measurement_noise]])
    filtered = kfilter((state, A, P, Q, H, R), measurements)
    physics = sum(damped_harmonic(p) for p in params)
    print("Average difference between measurements and physics: ", np.sqrt(np.mean((measurements - physics)**2)))
    plt.plot(times, measurements, label='Measurements')
    plt.plot(times, physics, label='Physics')
    plt.plot(times, filtered, label='Filtered')
    plt.legend()
    plt.show()
    return np.sqrt(np.mean((filtered - truth)**2))

def test_quality1(phy_noise=0.12):
    N_vib_app = 10
 
    vib_freqs = np.random.uniform(low=50, high=350, size=N_vib_app)  # Hz
    vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
    vib_damping = np.random.uniform(low=1e-5, high=1e-2, size=N_vib_app)  # unitless

    truth = sum([vib_amps[i] * np.exp(1j * 2 * np.pi * vib_freqs[i] * times - vib_phase[i]).real
                    * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

    measurements = make_noisy_data(truth)

    # suppose you have good physics within reason: add white noise by driving Q high,
    # and only get integral frequencies
    phy_freqs = np.round(vib_freqs)
    variances = np.array([phy_noise ** 2] * N_vib_app)
    params = np.vstack((vib_amps, phy_freqs, vib_damping, vib_phase)).T
    return np.sqrt(np.mean((kfilter(make_kfilter(params, variances), measurements) - truth)**2))

def test_quality2(phy_noise=0.12):
    while True:
        N_vib_app = 10

        vib_freqs = np.random.uniform(low=50, high=350, size=N_vib_app)  # Hz
        vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
        vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
        vib_damping = np.random.uniform(low=1e-5, high=1e-2, size=N_vib_app)  # unitless

        truth = sum([vib_amps[i] * np.exp(1j * 2 * np.pi * vib_freqs[i] * times - vib_phase[i]).real
                        * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

        measurements = make_noisy_data(truth)
        # suppose you have good physics within reason: add white noise by driving Q high,
        # and only get integral frequencies
        phy_freqs = np.round(vib_freqs)
        variances = np.array([phy_noise ** 2] * N_vib_app)
        params = np.vstack((vib_amps, phy_freqs, vib_damping, vib_phase)).T
        state, A, P, _, H, R = make_kfilter(params, variances)
        Q = np.random.normal(loc=0, scale=phy_noise, size=(2*N_vib_app, 2*N_vib_app))

        error = np.mean(kfilter((state, A, P, Q, H, R), measurements) - truth)
        print(np.mean(np.diag(Q)))
        if error > 1000:
            print(error)
            return

if __name__ == '__main__':
    globals()[sys.argv[1]](float(sys.argv[2]), float(sys.argv[3]))