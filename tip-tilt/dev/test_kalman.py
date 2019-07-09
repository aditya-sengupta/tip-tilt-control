from unittest import TestCase
from ao_observe import *
import numpy as np

class TestKalman(TestCase):
    def test_quality1(self, phy_noise=0.12):
        N_vib_app = 3

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
        return np.mean(kfilter(make_kfilter(params, variances), measurements) - truth)

    def test_quality2(self, phy_noise=0.12):
        while True:
            N_vib_app = 3

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
