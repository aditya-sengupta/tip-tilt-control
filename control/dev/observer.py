# Simulator of an adaptive optics system observer.
# Operates on a single mode at a time.

import numpy as np
from scipy import integrate, optimize, signal, stats
from copy import deepcopy
from matplotlib import pyplot as plt

from aberrations import pos

# global parameter definitions
f_sampling = 1000  # Hz
f_1 = f_sampling / 60  # lowest possible frequency of a vibration mode
f_2 = f_sampling / 3  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
N_vib_max = 10  # number of vibration modes to be detected
energy_cutoff = 1e-8  # proportion of total energy after which PSD curve fit ends
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
time_id = 1  # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = np.arange(0, time_id, 1 / f_sampling)  # array of times to operate on
freqs = np.linspace(0, f_sampling // 2, f_sampling // 2 + 1)  # equivalent to signal.periodogram(...)[0]


def get_psd(pos):
    return signal.periodogram(pos, f_sampling)[1]
    # return np.fft.fftshift(np.fft.fft(pos))[pos.size//2-1:] if you want the FFTR


def noise_filter(psd):
    # takes in a PSD.
    # returns a cleaned PSD with measurement noise hopefully removed.
    ind = np.argmax(freqs > f_w)
    assert ind != 0, "didn't find a high enough frequency"
    avg_measurement_power = np.mean(psd[ind:])
    measurement_noise_recovered = np.sqrt(f_sampling * avg_measurement_power)
    psd -= avg_measurement_power

    # this subtraction is problematic because it goes negative, so quick correction here.
    # Want a better way of doing this.

    for i, p in enumerate(psd):
        if p < 0:
            psd[i] = energy_cutoff

    # squelch: removing noise by applying a smoothing filter (convolution with [0.05, 0.1, 0.7, 0.1, 0.05])
    conv_peak = 0.7
    assert conv_peak < 1, "convolution must have unit gain"
    side = 1 - conv_peak
    kernel = np.array([side / 6, side / 3, conv_peak, side / 3, side / 6])
    c = kernel.size // 2
    psd = np.convolve(psd, kernel)[c:-c]

    # ad hoc low-pass filter
    ind_cutoff = np.argmax(freqs > f_2)
    psd[ind_cutoff:] = energy_cutoff * np.ones(len(psd) - ind_cutoff)  # or all zero?

    # high pass filter removed because of important vibration data

    # bring the peaks back to where they were
    peak_ind = signal.find_peaks(psd, height=energy_cutoff)[0]
    for i in peak_ind:
        psd[i] = psd[i] / conv_peak

    return psd


def atmosphere_fit(psd):
    # takes in a PSD to be fitted to atmospheric data.
    # returns the residual PSD, and a 3x1 np array with fit parameters [k, f, sigma**2]
    # to be changed
    return psd, np.array([0, 0, 0])


def damped_harmonic(pars_model):
    A, f, k, p = pars_model
    return A * np.exp(-k * 2 * np.pi * f * times) * np.cos(2 * np.pi * f * np.sqrt(1 - k**2) * times - p)



def make_psd(pars_model):
    return signal.periodogram(damped_harmonic(pars_model), fs=f_sampling)[1]


def log_likelihood(func, data):
    def get_ll(pars):
        pars_model, sd = pars[:-1], pars[-1]
        data_predicted = func(pars_model)
        LL = -np.sum(stats.norm.logpdf(data, loc=data_predicted, scale=sd))
        return LL

    return get_ll


def psd_f(f):
    def get_psd_f(pars):
        A, k, p = pars
        return make_psd([A, f, k, p])

    return get_psd_f


def reconstruct_modes(signal, N):
    # takes in a time axis for a signal, the signal, and the number of modes to reconstruct.
    # returns an np array of the exponential coefficients.
    # doesn't really work, but is staying in here in case some modification could make it work later
    fact = np.math.factorial
    A = np.zeros((N, N), dtype='complex')
    d = np.zeros(N, dtype='complex')
    v = N + 5

    def permute(n, k):
        # useful because it's the coefficient on the 'k'th derivative of s^n
        assert n >= k, "permute got a bad value"
        return fact(n) / fact(n - k)

    def choose(n, k):
        # useful because product rule works
        assert n >= k, "choose got a bad value"
        return fact(n) / (fact(k) * fact(n - k))

    # set up d

    for i in range(1, N + 1):
        for k in range(N + 2):
            to_integrate = (time_id - times) ** (v - N + k - 2) * times ** (N + i - k) * signal
            integral = integrate.simps(to_integrate, times)
            d[i - 1] += choose(N + i, k) * permute(N + 1, k) * (-1) ** (N + i - k) / fact(v - N + k - 2) * integral

    # set up A
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            for k in range(N + 2 - j):
                to_integrate = (time_id - times) ** (v - N + j + k - 2) * times ** (N + i - k) * signal
                integral = integrate.simps(to_integrate, times)
                A[i - 1][j - 1] += choose(N + i, k) * permute(N + 1 - j, k) * (-1) ** (N + i - k) / fact(
                    v - N + j + k - 2) * integral

    # d = A*theta
    theta = np.linalg.inv(A).dot(d)
    theta = np.hstack((1, -theta))
    return np.roots(theta)  # e^(-2pi k_i f_i + 1j * 2pi f_i)


def vibe_fit_freq(psd, N=N_vib_max):
    # takes in the frequency axis for a PSD, and the PSD.
    # returns a 4xN np array with fit parameters, and a 1xN np array with variances.
    par0 = [0.5, 1e-4, np.pi, 1]
    PARAMS_SIZE = len(par0)  # slightly misleading: std is in par0 and frequency isn't so it matches up.

    # peak detection by correlation
    indshift = int(5 * time_id)  # index shift rescaling a freq shift of 5 Hz due to ID time;
    # number of samples goes up because more time
    reference_peak = psd_f(250)(par0[:-1])
    center = np.argmax(reference_peak)
    reference_peak = reference_peak[center - indshift:center + indshift]
    # any random peak should do; should be independent of the data though.

    peaks = []
    psd_windowed = deepcopy(psd)
    for i in range(N):
        peak = np.argmax(np.correlate(psd_windowed, reference_peak, 'same'))
        if psd_windowed[peak] <= energy_cutoff:  # skip
            continue
        peaks.append(peak)
        psd_windowed[peak - indshift:peak + indshift] = energy_cutoff

    params = np.zeros((len(peaks), PARAMS_SIZE))
    variances = np.zeros(len(peaks))

    # curve-fit to each identified peak and find corresponding parameters
    # currently, the number of identified peaks is always the maximum possible
    # since the idea of an 'applied vibration mode' exists only in simulation, this seems reasonable
    for i, peak_ind in enumerate(peaks):
        l, r = peak_ind - indshift, peak_ind + indshift
        windowed = psd[l:r]
        psd_ll = log_likelihood(lambda pars: psd_f(freqs[peak_ind])(pars)[l:r], windowed)
        A, k, p, sd = optimize.minimize(psd_ll, par0, method='Nelder-Mead').x
        params[i] = [A, freqs[peak_ind], k, p]
        variances[i] = sd ** 2

    return params, variances


def make_state_transition(params):
    STATE_SIZE = 2 * params.shape[0]
    A = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(STATE_SIZE // 2):
        f, k = params[i][1:3]
        w0 = 2 * np.pi * f / np.sqrt(1 - k**2)
        A[2 * i][2 * i] = 2 *  np.exp(-k * w0 / f_sampling) * np.cos(w0 * np.sqrt(1 - k**2) / f_sampling)
        A[2 * i][2 * i + 1] = -np.exp(-2 * k * w0 / f_sampling)
        A[2 * i + 1][2 * i] = 1
    return A


def predict(A, P, Q, state):
    return A.dot(state), A.dot(P.dot(A.T)) + Q


def update(H, P, R, state, measurement):
    error = measurement - H.dot(state)
    K = P.dot(H.T.dot(np.linalg.inv(H.dot(P.dot(H.T)) + R)))
    return state + K.dot(error), P - K.dot(H.dot(P))


def make_kf_state(params):
    dt = 1 / f_sampling
    state = np.zeros(2*params.shape[0])
    for i in range(params.shape[0]):
        a, f, k, p = params[i]
        w_d, w_f = 2*np.pi*f*k/np.sqrt(1-k**2), 2*np.pi*f
        state[2*i] = a * np.cos(p)
        state[2*i + 1] = a * np.cos(-w_f * dt - p) * np.exp(w_d * dt)
    return state


def make_kfilter(params, variances):
    # takes in parameters and variances from which to make a physics simulation
    # and measurements to match it against.
    # returns state, A, P, Q, H, R for kfilter to run.
    A = make_state_transition(params)
    STATE_SIZE = 2 * params.shape[0]
    state = make_kf_state(params)
    H = np.array([[1, 0] * (STATE_SIZE // 2)])
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(variances.size):
        Q[2 * i][2 * i] = variances[i]
    R = measurement_noise * np.identity(1)
    P = np.zeros((STATE_SIZE, STATE_SIZE))
    return state, A, P, Q, H, R


def kfilter(args, measurements):
    state, A, P, Q, H, R = args
    steps = int(f_sampling * time_id)
    pos_r = np.zeros(steps)
    for k in range(steps):
        state, P = update(H, P, R, state, measurements[k])
        pos_r[k] = H.dot(state)
        state, P = predict(A, P, Q, state)
    return pos_r


if __name__ == "__main__":
    psd = get_psd(pos)
    plt.semilogy(freqs, psd)
    plt.ylim(1e-7, 1)
    plt.show()
