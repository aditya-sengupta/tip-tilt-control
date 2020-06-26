import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp
from abc import ABC, abstractmethod
from copy import deepcopy

class DynamicSystem(ABC):
    '''
    An abstract dynamic system with time-dependent state evolution, time-independent state to measurement mapping,
    and initial measurement covariances, constructed to facilitate Kalman filtering.

    Required parameters:
        simend (bool)
        state (ndarray)
        P (ndarray)
        Q (ndarray)
        H (ndarray)
        R (ndarray)
        STATE_SIZE (int)
        INPUT_SIZE (int)
    Required methods:
        evolve
        ext_input
        reset
    '''

    @abstractmethod
    def evolve(self, t, dt):
        # Given t and dt, returns the dynamic system's A and B matrices. Also sets simend based on state.
        pass

    @abstractmethod
    def ext_input(self, t):
        # Given t, returns u(t) and a u_status that may be None.
        pass

    @abstractmethod
    def reset(self):
        # restores all internal variables that may have changed during a simulation run.
        pass

    def predict(self, t, dt):
        # predicts system state at time t+dt based on system state at time t
        A, B = self.evolve(t, dt)
        u, u_status = self.ext_input(t)
        state_predicted = A.dot(self.state) + B.dot(u)
        P_predicted = A.dot(self.P.dot(A.T)) + self.Q
        return (u, u_status, state_predicted, P_predicted)

    def sim_results(self, t, k, states, inputs, terminate):
        self.reset()
        print("Simulation ended at t =", t, "s due to", terminate)
        processed_states = np.zeros([self.STATE_SIZE, k])
        processed_inputs = np.zeros([self.INPUT_SIZE, k])
        states = states.T
        inputs = inputs.T
        for i, state in enumerate(states):
            processed_states[i] = state[:k]
        for i, input in enumerate(inputs):
            processed_inputs[i] = input[:k]
        return (np.linspace(0,t,k+1)[:k], processed_states, processed_inputs)

    def simulate(self, dt=0.01, timeout=30, verbose=False, kalman=None):
        interrupt = False
        t, k = 0, 0
        if kalman is not None:
            m = 0
            measure_times = kalman[0]
        states = np.zeros([int(np.ceil(timeout/dt))+1, self.STATE_SIZE])
        inputs = np.zeros([int(np.ceil(timeout/dt))+1, self.INPUT_SIZE])
        terminate = "error."
        try:
            while t < timeout:
                if verbose and hasattr(self, "compact_status") and k % 100 == 0:
                    self.compact_status(t)
                states[k] = self.state
                inputs[k], input_status, state_predicted, P_predicted = self.predict(t, dt)
                if kalman is not None and m < measure_times.size and np.isclose(t, measure_times[m], atol=dt/2):
                    self.state, self.P = self.update(state_predicted, P_predicted, kalman[1][m])
                    m += 1
                else:
                    self.state = state_predicted
                    self.P = P_predicted
                if verbose and input_status is not None and k % 100 == 0:
                    print(input_status)
                if self.simend:
                    terminate = "end condition."
                    break
                t += dt
                t = np.round(t, -int(np.log10(dt)))
                k += 1
        except KeyboardInterrupt:
            print("\nSteps completed:", k)
            terminate = "interrupt."
        if t >= timeout:
            terminate = "timeout."
        return self.sim_results(t, k, states, inputs, terminate)

    def measure(self, state):
        return self.H.dot(state)

    def update(self, state_predicted, P_predicted, measurement):
        error = measurement - self.measure(state_predicted)
        K = P_predicted.dot(self.H.T.dot(np.linalg.inv(self.H.dot(P_predicted.dot(self.H.T)) + self.R)))
        state_updated = state_predicted + K.dot(error)
        P_updated = P_predicted - K.dot(self.H.dot(P_predicted))
        return (state_updated, P_updated)
