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
        self.refresh_time = None
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
            self.is_openloop = False # doesn't matter
        elif method == 'stdint':
            self.strategy = self.strategy_stdint
            self.make_state = self.make_state_stdint
            self.A = np.array([[-0.6, -0.32, -0.08], [1, 0, 0], [0, 1, 0]])
            self.is_openloop = False
        elif method == 'kalman': # a controller for turbulence only and without LQG: just Kalman predicting.
            self.strategy = self.strategy_kalman
            self.kfilter = args[0]
            if calibration_time is None:
                self.calibration_time = self.kfilter.state.size
            else:
                self.calibration_time = calibration_time
            self.make_state = self.make_state_AR # this is sort of jank: state and kfilter's state are different, so this doesn't really matter.
            self.is_openloop = True

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
        cumulative_actions = np.zeros(truth.size)
        time = self.calibration_time
        if time is None:
            time = self.delay
        openloop = np.zeros(truth.size)
        openloop[:time] = truth[:time]
        self.make_state(openloop[:time] + np.random.normal(0, noise, time))
        residuals[:time] = truth[:time]
        actions = np.zeros(truth.size)
        shifts = np.diff(truth)
        # shifts[i] is the shift from state i to state i + 1
        # i.e. position[i+1] = position[i] + shifts[i] + actions[i]
        # where you were before, plus the change due to environment, plus the control action applied

        print("Starting at timestep", time)
        action_applied = 0
        for i in range(time, truth.size):
            residuals[i] = residuals[i - 1] + shifts[i - 1] + actions[i - 1]
            action_applied += actions[i - 1]
            if self.is_openloop:
                # the controller is expecting an open-loop measurement
                openloop[i] = residuals[i] - action_applied
            else:
                # the controller is expecting a closed-loop measurement
                openloop[i] = residuals[i]
            measurement = openloop[i] + np.random.normal(0, noise)
            if i + self.delay < truth.size:
                actions[i + self.delay] = -self.strategy(measurement)
        
        return residuals, np.cumsum(actions), openloop

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
        self.state[0] += 0.3 * measurement
        return self.state[0]

    def strategy_kalman(self, measurement):
        # describes a 'naive Kalman' control scheme, i.e. not LQG
        # assert self.kfilter.state.any(), "starting from zero state"
        # print("Prior: ", self.kfilter.measure())
        self.kfilter.update(measurement)
        # print("Updated with measurement " + str(measurement) + ": " + str(self.kfilter.measure()))
        state = deepcopy(self.kfilter.state)
        self.kfilter.predict()
        state_pred = deepcopy(self.kfilter.state)
        for _ in range(self.delay - 1):
            state_pred = self.kfilter.A.dot(state_pred)
        # print("Current: ", self.kfilter.measure(state))
        # print("Prediction: ", self.kfilter.measure(state_pred))
        return self.kfilter.measure(state) - self.kfilter.measure(state_pred)

    def strategy_LQR(self, measurement):
        # describes LQR control being fed optimal state estimates by a Kalman filter
        pass

def show_control(controller_name, openloop):
    controller = globals()[controller_name]
    residuals, actions, _ = controller.control(openloop)
    plt.figure(figsize=(10,10))
    plt.plot(openloop, label='Truth')
    plt.plot(actions, label='Actions')
    plt.plot(residuals, label='Residual')
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Deviation (mas)")
    plt.title("Control action from " + controller_name + ", error = " + str(rms(residuals)))

def keck_control(size):
    openloop  = np.load('keck_tt/OpenLoop_n0088.npy')[:size,0] * keck_normalizer
    commands  = np.load('keck_tt/Commands_n0088.npy')[:size,0] * keck_normalizer
    centroids = np.load('keck_tt/Centroid_n0088.npy')[:size,0] * keck_normalizer
    return openloop, centroids, commands
