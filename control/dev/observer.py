# Simulator of an adaptive optics system observer.
# Operates on a single mode at a time.

import numpy as np
from scipy import integrate, optimize, signal, stats
from copy import deepcopy
from matplotlib import pyplot as plt
from fractal_deriv import design_filt

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
a = 1e-6 # the pole location for the f^(-2/3) powerlaw

class KFilter:
    def __init__(self, state, A, P, Q, H, R):
        self.state = state
        self.A = A
        self.P = P
        self.Q = Q
        self.H = H
        self.R = R
        self.K = None

    def predict(self):
        self.state = self.A.dot(self.state)
        self.P = self.A.dot(self.P.dot(self.A.T)) + self.Q

    def set_gain(self):
        self.K = self.P.dot(self.H.T.dot(np.linalg.inv(self.H.dot(self.P.dot(self.H.T)) + self.R)))

    def update(self, measurement):
        error = measurement - self.measure()
        self.set_gain()
        self.state = self.state + self.K.dot(error) 
        self.P = self.P - self.K.dot(self.H.dot(self.P))

    def measure(self, state=None):
        if state is not None:
            return self.H.dot(state)
        return self.H.dot(self.state)

    def run(self, measurements, save_physics=False):
        steps = len(measurements)
        pos_r = np.zeros(steps)
        if save_physics:
            predictions = np.zeros(steps)
        
        for k in range(steps):
            self.update(measurements[k])
            pos_r[k] = self.measure()
            self.predict()
            if save_physics:
                predictions[k] = self.measure() # off by one?
        if save_physics:
            return pos_r, predictions
        return pos_r

    def physics_predict(self, steps):
        #state = np.ones(state.size) # just to view what dynamics are like, we set a 'unity state'
        pos_r = np.zeros(steps)
        for k in range(steps):
            pos_r[k] = self.measure()
            self.predict()
        return pos_r


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

def make_state_transition_vibe(params):
    STATE_SIZE = 2 * params.shape[0]
    A = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(STATE_SIZE // 2):
        f, k = params[i]
        w0 = 2 * np.pi * f / np.sqrt(1 - k**2)
        A[2 * i][2 * i] = 2 *  np.exp(-k * w0 / f_sampling) * np.cos(w0 * np.sqrt(1 - k**2) / f_sampling)
        A[2 * i][2 * i + 1] = -np.exp(-2 * k * w0 / f_sampling)
        A[2 * i + 1][2 * i] = 1
    return A

def make_kfilter_vibe(params, variances):
    # takes in parameters and variances from which to make a physics simulation
    # and measurements to match it against.
    # returns a KFilter object.
    A = make_state_transition(params)
    STATE_SIZE = 2 * params.shape[0]
    state = np.zeros(STATE_SIZE)
    H = np.array([[1, 0] * (STATE_SIZE // 2)])
    Q = np.zeros((STATE_SIZE, STATE_SIZE))
    for i in range(variances.size):
        Q[2 * i][2 * i] = variances[i]
    R = measurement_noise * np.identity(1)
    P = np.zeros((STATE_SIZE, STATE_SIZE))
    return KFilter(state, A, P, Q, H, R)

def get_ft(b, fc, c1, c2):
        def ft(f):
            return b/((1j * f + a)**(np.abs(c1)) * (1j * f + fc)**(np.abs(c2)))
        return np.vectorize(ft)

get_applied_ft = lambda b, fc, c1, c2, freqs: (get_ft(b, fc, c1, c2))(freqs)

def make_impulse(tt, N=20, plot=False):
    # makes an impulse response for a turbulence filter based on time-series tt data.
    freqs, P = signal.periodogram(tt, fs=f_sampling)
    size = int(tt.size/50) # rough heuristic. Just enough to get a good initial guess.
    to_conv = [1/(2*size + 1)] * (2 * size + 1)
    clean_psd = np.convolve(P[1:], to_conv)
    clean_psd = clean_psd[size-1:-size]
    #print(freqs)
    #print(freqs[np.where(freqs < 10)][1:])
    #print(clean_psd[np.where(freqs < 10)][1:])
    c1 = stats.linregress(np.log10(freqs[np.where(freqs < 10)][1:]), np.log10(clean_psd[np.where(freqs < 10)][1:])).slope
    c2 = stats.linregress(np.log10(freqs[np.where(freqs > 10)]), np.log10(clean_psd[np.where(freqs > 10)])).slope

    ft = lambda b, fc, c1, c2: lambda f: b/((1j * freqs + a)**(np.abs(c1)) * (1j * freqs + fc)**(np.abs(c2)))
    def cost(pars):
        b, fc, c1, c2 = pars
        return np.mean((np.log10(np.abs(get_ft(b, fc, c1, c2)(freqs)**2)) - np.log10(P))**2)

    b, fc, c1, c2 = optimize.minimize(cost, [1, 10, c1, c2]).x

    ft = get_ft(b, fc, c1, c2)
    if plot:
        plt.loglog(freqs, P, label='true')
        plt.loglog(freqs, clean_psd, label='cleaned')
        plt.loglog(freqs, np.abs(get_applied_ft(b, fc, c1, c2, freqs))**2, label='fit')
        plt.legend()
        plt.ylim(1e-7)
        plt.show()
    return design_filt(dt=1/f_sampling, N=N, tf = ft, plot=plot)

def make_kfilter_turb(impulse, calibration):
    # takes in an impulse response as generated by make_impulse, and a 'calibration' array with the first N openloop measurements, 
    # and returns a KFilter object.
    n = impulse.size
    state = np.flip(calibration)
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i][i-1] = 1
    A[0] = np.flip(np.real(impulse)/sum(np.real(impulse)))
    # when you start the filter, make sure to start it at time n with the first n measurements identically
    P = np.zeros((n,n))
    Q = np.zeros((n,n))
    Q[0][0] = 1 # arbitrary: I have no idea how to set this yet.
    H = np.zeros((1,n))
    H[:,0] = 1
    R = np.array([measurement_noise**2])
    return KFilter(state, A, P, Q, H, R)
