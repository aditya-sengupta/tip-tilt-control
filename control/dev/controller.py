# Testing different control strategies and showing that Kalman-LQG is better.

import numpy as np
from matplotlib import pyplot as plt
from aberrations import *
from observer import *

rms = lambda data: np.sqrt(np.mean(data ** 2))

def control(truth, controller, noise=0.06, delay=1):
    '''
    Simulates the general control problem.

    Parameters:

    truth - ndarray
    The uncorrected truth values of tip or tilt. Conventionally milliarcseconds: one per timestep.

    controller - Control object
    The controller: takes in the open-loop measurements, returns a sequence of control actions in the same units.

    noise - float
    The noise to add to truth to make open-loop measurements.

    delay - int
    The frame delay of the control setup being simulated, to be passed in as input to strategy. Should only be 1-3.

    Returns:

    residual - ndarray
    The residual at each step.
    '''
    residuals = np.zeros(truth.size)
    controller.delay = delay
    time = controller.calibration_time
    if time is None:
        time = delay
    calibration = truth[:time] + np.random.normal(0, noise, time)
    controller.make_state(calibration)
    residuals[:time] = calibration
    actions = np.zeros(truth.size)

    shifts = np.diff(truth)
    # shifts[i] is the shift from state i to state i + 1
    # i.e. position[i+1] = position[i] + shifts[i] + actions[i]
    # where you were before, plus the change due to environment, plus the control action applied

    for i in range(time, truth.size - 1):
        residuals[i] = residuals[i - 1] + shifts[i - 1] + actions[i]
        measurement = residuals[i] + np.random.normal(0, noise)
        if i + delay < truth.size:
            actions[i + delay] = -controller.strategy(measurement)
    
    return residuals, actions


class Controller:
    def __init__(self, method, STATE_SIZE=None, calibration_time=None):
        if method == 'stdint':
            self.STATE_SIZE = 3
        else:
            self.STATE_SIZE = STATE_SIZE
        if calibration_time is None:
            self.calibration_time = self.STATE_SIZE
        self.state = np.zeros(STATE_SIZE,)
        self.delay = 0
        if method == 'baseline':
            self.strategy = self.strategy_baseline
            self.make_state = self.strategy_baseline
        elif method == 'stdint':
            self.strategy = self.strategy_stdint
            self.make_state = self.make_state_stdint
            self.A = np.array([[-0.6, -0.32, -0.08], [1, 0, 0], [0, 1, 0]])
        elif method == 'kalman_turb': # a controller for turbulence only and without LQG: just Kalman predicting.
            self.strategy = self.strategy_kalman_turb
            self.make_state = self.make_state_AR

    def make_state_AR(self, calibration):
        # for autoregressive strategies, makes an initial state that's the first N openloop measurements
        # calibration = the first STATE_SIZE measurements.
        self.state = np.flip(calibration)
        # return STATE_SIZE + delay # + 1?

    def make_state_stdint(self, calibration):
        self.state = 0.1 * np.flip(calibration)

    def strategy_baseline(self, measurement):
        return 0

    def strategy_stdint(self, measurement):
        self.state = self.A.dot(self.state)
        self.state[0] += 0.1 * measurement
        return self.state[0]

    def strategy_kalman_turb(self, measurement):
        pass

def run_stdint(size):
    stdint = Controller('stdint', 3)
    truth = np.load('keck_tt/OpenLoop_n0088.npy')[:size,0]
    residuals, actions = control(truth, stdint)
    return truth, residuals, actions

def run_baseline(size):
    baseline = Controller('baseline')
    truth = np.load('keck_tt/OpenLoop_n0088.npy')[:size,0]
    residuals, actions = control(truth, baseline)
    return truth, residuals, actions

def show_control(truth, residuals, actions):
    plt.plot(truth, label='Truth')
    plt.plot(actions, label='Actions')
    plt.plot(residuals, label='Residual')
    plt.legend()
    plt.title("Truth and actions and residuals, error = " + str(rms(residuals)))
    plt.show()

if __name__ == "__main__":
    show_control(*run_stdint(1000))