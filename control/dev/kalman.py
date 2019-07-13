# Simulator of an Adaptive Optics Observer, using Kalman filtering.
# Operates on a single mode at a time.

import numpy as np
from scipy import integrate, optimize, signal, stats
import copy
import itertools

# global parameter definitions
f_sampling = 1000  # Hz
f_1 = f_sampling / 60  # lowest possible frequency of a vibration mode
f_2 = f_sampling / 3  # highest possible frequency of a vibration mode
f_w = f_sampling / 3  # frequency above which measurement noise dominates
N_vib_app = 10  # number of vibration modes being applied
N_vib_max = 10  # number of vibration modes to be detected
energy_cutoff = 1e-8  # proportion of total energy after which PSD curve fit ends
measurement_noise = 0.06  # milliarcseconds; pulled from previous notebook
time_id = 1  # timescale over which sysid runs. Pulled from Meimon 2010's suggested 1 Hz sysid frequency.
times = np.arange(0, time_id, 1 / f_sampling)  # array of times to operate on
freqs = np.linspace(0, f_sampling // 2, f_sampling // 2 + 1)  # equivalent to signal.periodogram(...)[0]


def make_vibe_data(N_vib_app=N_vib_app):
    # takes in nothing (params are globally set in the 'global parameter definitions' cell)
    # returns a 1D np array with the same dimension as times.
    # note that we don't have access to the random parameters, just the output.
    vib_freqs = np.random.uniform(low=f_1, high=f_2, size=N_vib_app)  # Hz
    vib_amps = np.random.uniform(low=0.1, high=1, size=N_vib_app)  # milliarcseconds
    vib_phase = np.random.uniform(low=0.0, high=2 * np.pi, size=N_vib_app)  # radians
    vib_damping = np.random.uniform(low=1e-5, high=1e-2, size=N_vib_app)  # unitless

    pos = sum([vib_amps[i] * np.cos(2 * np.pi * vib_freqs[i] * times - vib_phase[i])
               * np.exp(-2 * np.pi * vib_damping[i] * vib_freqs[i] * times) for i in range(N_vib_app)])

    return pos


def make_noisy_data(pos, noise=measurement_noise):
    return pos + np.random.normal(0, noise, np.size(times))


def make_atm_data():
    # takes in nothing (params are globally set in the 'global parameter definitions' cell)
    # returns a 1D np array with the same dimension as times
    # to be changed, clearly
    return np.zeros(times.size)


def get_psd(pos):
    return signal.periodogram(pos, f_sampling)[1]


def noise_filter(psd):
    # takes in a PSD.
    # returns a cleaned PSD with measurement noise hopefully removed.
    ind = np.argmax(freqs > f_w)
    assert ind != 0, "didn't find a high enough frequency"
    avg_measurement_power = np.mean(psd[ind:])
    # measurement_noise_recovered = np.sqrt(f_sampling * avg_measurement_power)
    # print("Recovered measurement noise: " + str(measurement_noise_recovered))
    # print("Percent error in measurement noise estimate: "
    #      + str(100 * np.abs(measurement_noise_recovered - measurement_noise)/measurement_noise))

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

    # and also a high-pass filter
    ind_cutoff = np.argmin(freqs < f_1)
    psd[:ind_cutoff] = energy_cutoff * np.ones(ind_cutoff)

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
    return A * np.exp(-k * 2 * np.pi * f * times) * np.cos(2 * np.pi * f * times - p)


def damped_derivative(pars_model):
    # utility: returns derivative of damped_harmonic evaluated at time 0
    A, f, k, p = pars_model
    return A * 2 * np.pi * f * (np.sin(p) - k  * np.cos(p))


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


def vibe_fit_freq(psd):
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
    psd_windowed = copy.deepcopy(psd)
    for i in range(N_vib_max):
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


def make_process_noise(variances):
    STATE_SIZE = 2 * len(variances)
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(variances.size):
        Q[2 * i][2 * i] = variances[i]
    return Q


def make_state_transition(params):
    dt = 1 / f_sampling
    STATE_SIZE = 2 * params.shape[0]
    A = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(STATE_SIZE // 2):
        _, f, k, _ = params[i]
        w = 2 * np.pi * f
        a = -w * k
        b = w * np.sqrt(1 - k ** 2)
        c, s = np.cos(b * dt), np.sin(b * dt)
        coeff = np.exp(a * dt) / b
        A[2 * i][2 * i] = coeff * (-a * s + b * c)
        A[2 * i][2 * i + 1] = coeff * s
        A[2 * i + 1][2 * i] = -coeff * w ** 2 * s
        A[2 * i + 1][2 * i + 1] = coeff * (a * s + b * c)
    return A


def make_state_transition_time(alpha):
    dt = 1 / f_sampling
    STATE_SIZE = 2 * alpha.shape[0]
    A = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(STATE_SIZE // 2):
        w = alpha[i].imag
        k = alpha[i].real / alpha[i].imag
        a = -w * k
        b = w * np.sqrt(1 - k ** 2)
        c, s = np.cos(b * dt), np.sin(b * dt)
        coeff = np.exp(a * dt) / b
        A[2 * i][2 * i] = coeff * (-a * s + b * c)
        A[2 * i][2 * i + 1] = coeff * s
        A[2 * i + 1][2 * i] = -coeff * w ** 2 * s
        A[2 * i + 1][2 * i + 1] = coeff * (a * s + b * c)
    return A


def make_measurement_matrix(STATE_SIZE):
    return np.array([[1, 0] * (STATE_SIZE // 2)])


def simulate(params):
    A = make_state_transition(params)
    STATE_SIZE = 2 * params.shape[0]
    state = np.array(list(itertools.chain.from_iterable([[
        damped_harmonic(params[i])[0],
        damped_derivative(params[i])]
        for i in range(STATE_SIZE // 2)])))
    k = 0
    pos_r = np.zeros(int(f_sampling * time_id))
    H = make_measurement_matrix(STATE_SIZE)
    while k < time_id * f_sampling:
        pos_r[k] = H.dot(state)
        state = A.dot(state)
        k += 1
    return pos_r


def simulate_time(alpha, params):
    A = make_state_transition_time(alpha)
    STATE_SIZE = 2 * params.shape[0]
    state = np.array(list(itertools.chain.from_iterable([[
        damped_harmonic(params[i])[0],
        damped_derivative(params[i])]
        for i in range(STATE_SIZE // 2)])))
    k = 0
    pos_r = np.zeros(int(f_sampling * time_id))
    H = make_measurement_matrix(STATE_SIZE)
    while k < time_id * f_sampling:
        pos_r[k] = H.dot(state)
        state = A.dot(state)
        k += 1
    return pos_r


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
    state = np.array(list(itertools.chain.from_iterable([[
        damped_harmonic(params[i])[0],
        damped_derivative(params[i])]
        for i in range(STATE_SIZE // 2)])))
    H = make_measurement_matrix(STATE_SIZE)
    Q = make_process_noise(variances)
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