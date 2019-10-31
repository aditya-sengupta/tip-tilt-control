# Testing different control strategies and showing that Kalman-LQG is better.

import numpy as np
from matplotlib import pyplot as plt
from aberrations import *
from observer import *
from copy import deepcopy

rms = lambda data: np.sqrt(np.mean(data ** 2))
noise = 0.06

class Controller:
    def __init__(self, method, *args, STATE_SIZE=0, calibration_time=None):
        if method == 'stdint':
            self.STATE_SIZE = 3
            self.calibration_time = 3
        else:
            self.STATE_SIZE = STATE_SIZE
            self.calibration_time = calibration_time
        self.state = np.zeros(STATE_SIZE,)
        self.delay = 1
        if method == 'baseline':
            self.strategy = self.strategy_baseline
            self.make_state = self.strategy_baseline
        elif method == 'stdint':
            self.strategy = self.strategy_stdint
            self.make_state = self.make_state_stdint
            self.A = np.array([[-0.6, -0.32, -0.08], [1, 0, 0], [0, 1, 0]])
        elif method == 'kalman': # a controller for turbulence only and without LQG: just Kalman predicting.
            self.strategy = self.strategy_kalman
            self.kfilter = args[0]
            self.calibration_time = self.kfilter.state.size
            self.make_state = self.make_state_AR # this is sort of jank: state and kfilter's state are different.

    def control(self, truth, noise=noise):
        '''
        Simulates the general control problem.

        Parameters:

        truth - ndarray
        The uncorrected truth values of tip or tilt. Conventionally milliarcseconds: one per timestep.

        noise - float
        The noise to add to truth to make open-loop measurements.

        delay - int
        The frame delay of the control setup being simulated, to be passed in as input to strategy. Should only be 1-3.

        Returns:

        residuals - ndarray
        The residual at each step.

        actions - ndarray
        The control actions as a cumulative sum of individual control actions.
        '''
        residuals = np.zeros(truth.size)
        time = self.calibration_time
        if time is None:
            time = self.delay
        calibration = truth[:time] + np.random.normal(0, noise, time)
        self.make_state(calibration)
        residuals[:time] = truth[:time]
        actions = np.zeros(truth.size)

        shifts = np.diff(truth)
        # shifts[i] is the shift from state i to state i + 1
        # i.e. position[i+1] = position[i] + shifts[i] + actions[i]
        # where you were before, plus the change due to environment, plus the control action applied

        print("Starting at timestep", time)
        for i in range(time, truth.size):
            residuals[i] = residuals[i - 1] + shifts[i - 1] - actions[i - 1]
            measurement = residuals[i] + np.random.normal(0, noise)
            if i + self.delay < truth.size:
                actions[i + self.delay] = self.strategy(measurement)
        
        return residuals, np.cumsum(actions)

    def make_state_AR(self, calibration):
        # for autoregressive strategies, makes an initial state that's the first N openloop measurements
        # calibration = the first STATE_SIZE measurements.
        self.state = np.flip(calibration)

    def make_state_stdint(self, calibration):
        self.state = 0.1 * np.flip(calibration)

    def strategy_baseline(self, measurement):
        return 0

    def strategy_stdint(self, measurement):
        self.state = self.A.dot(self.state)
        self.state[0] += 0.1 * measurement
        return self.state[0]

    def strategy_kalman(self, measurement):
        # describes a 'naive Kalman' control scheme, i.e. not LQG
        assert self.kfilter.state.any(), "starting from zero state"
        self.kfilter.update(measurement)
        state = deepcopy(self.kfilter.state)
        self.kfilter.predict()
        state_pred = deepcopy(self.kfilter.state)
        for _ in range(self.delay - 1):
            state_pred = self.kfilter.A.dot(state_pred)
        print("Prediction: ", self.kfilter.measure(state_pred))
        print("Current: ", self.kfilter.measure(state))
        return self.kfilter.measure(state_pred - state)

size = 2000
N = 10
keck_normalizer = 0.6 * (600e-9 / (2 * np.pi)) *  206265000
truth = np.load('./turbulence.npy')[:size,0]# * keck_normalizer
kalman = Controller('kalman', make_kfilter_turb(make_impulse(truth[:size//2], N=N), truth[:N] + np.random.normal(0, noise, (N,))))
stdint = Controller('stdint')
baseline = Controller('baseline')

def show_control(controller_name):
    controller = globals()[controller_name]
    residuals, actions = controller.control(truth)
    plt.figure(figsize=(10,10))
    plt.plot(truth, label='Truth')
    plt.plot(actions, label='Actions')
    plt.plot(residuals, label='Residual')
    plt.legend()
    plt.title("Control action from " + controller_name + ", error = " + str(rms(residuals)))
    plt.show()

def keck_control(size):
    openloop  = np.load('keck_tt/OpenLoop_n0088.npy')[:size,0] * keck_normalizer
    commands  = np.load('keck_tt/Commands_n0088.npy')[:size,0] * keck_normalizer
    centroids = np.load('keck_tt/Centroid_n0088.npy')[:size,0] * keck_normalizer
    return openloop, centroids, commands