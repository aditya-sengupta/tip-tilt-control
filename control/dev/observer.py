# Simulator of an adaptive optics system observer.
# Operates on a single mode at a time.

import numpy as np
from scipy import integrate, optimize, signal, stats
from copy import deepcopy
from matplotlib import pyplot as plt

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
    high_cutoff = np.argmax(freqs > f_2)
    psd[high_cutoff:] = energy_cutoff * np.ones(len(psd) - high_cutoff)  # or all zero?

    # high pass filter
    # low_cutoff = np.argmin(freqs < f_1)
    # psd[:low_cutoff] = energy_cutoff * np.ones(low_cutoff)

    # bring the peaks back to where they were
    peak_ind = signal.find_peaks(psd, height=energy_cutoff)[0]
    for i in peak_ind:
        psd[i] = psd[i] / conv_peak

    return psd


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
        k = pars[0]
        return make_psd([1, f, k, np.pi])

    return get_psd_f


def vibe_fit_freq(psd, N=N_vib_max):
    # takes in the frequency axis for a PSD, and the PSD.
    # returns a 4xN np array with fit parameters, and a 1xN np array with variances.
    par0 = [1e-4, 1]
    PARAMS_SIZE = 2
    width = 1

    peaks = []
    unsorted_peaks = signal.find_peaks(psd)[0]
    freqs_energy = np.flip(np.argsort(psd)) # frequencies ordered by their energy
    for f in freqs_energy:
        if f in unsorted_peaks:
            peaks.append(f)

    params = np.zeros((N, PARAMS_SIZE))
    variances = np.zeros(N)

    i = 0
    for peak_ind in peaks:
        if i >= N:
            break
        if np.any(np.abs(params[:,0] - peak_ind) <= width): #or peak_ind < f_1 + width or peak_ind > f_2 - width:
            continue
        l, r = peak_ind - width, peak_ind + width
        windowed = psd[l:r]
        psd_ll = log_likelihood(lambda pars: psd_f(freqs[peak_ind])(pars)[l:r], windowed)
        k, sd = optimize.minimize(psd_ll, par0, method='Nelder-Mead').x
        params[i] = [freqs[peak_ind], k]
        variances[i] = sd ** 2
        i += 1

    return params, variances


def make_state_transition(params):
    STATE_SIZE = 2 * params.shape[0]
    A = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(STATE_SIZE // 2):
        f, k = params[i]
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

def make_kfilter(params, variances):
    # takes in parameters and variances from which to make a physics simulation
    # and measurements to match it against.
    # returns state, A, P, Q, H, R for kfilter to run.
    A = make_state_transition(params)
    STATE_SIZE = 2 * params.shape[0]
    state = np.zeros(STATE_SIZE)
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
    steady_state = False
    K = np.zeros((1, state.size))
    for k in range(steps):
        if steady_state:
            state = state + K.dot(measurements[k] - H.dot(state))
        else:
            last_P = deepcopy(P)
            state, P = update(H, P, R, state, measurements[k])
        pos_r[k] = H.dot(state)
        state, P = predict(A, P, Q, state)
        if np.allclose(last_P, P):
            steady_state = True
            K = P.dot(H.T.dot(np.linalg.inv(H.dot(P.dot(H.T)) + R)))
    return pos_r

def physics_predict(args, measurements):

    state, A, _, _, H, _ = args
    state = np.ones(state.size) # just to view what dynamics are like, we set a 'unity state'
    steps = measurements.size
    pos_r = np.zeros(steps)
    for k in range(steps):
        pos_r[k] = H.dot(state)
        state = A.dot(state)
    return pos_r

if __name__ == "__main__":
    psd = get_psd(pos)
    plt.semilogy(freqs, psd)
    plt.ylim(1e-7, 1)
    plt.show()
