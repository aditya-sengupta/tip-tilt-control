# a replacement for the KFilter observer that just does Kalman and LQG at once, on both dimensions.
# follows the conventions of SPHERE.

import numpy as np
from copy import deepcopy

delay = 2
f_sampling = 1000

class LQGController:
    def __init__(self, N_turb, N_vibe, measurement_error = 0.06):
        # N_turb: number of coefficients in the AR model for turbulence.
        # N_vibe: number of vibration modes.
        # measurement_error: single-axis measurement error in mas.
        self.ss = 2 * N_turb + 4 * N_vibe # state-size
        self.state = np.zeros(ss,)
        crow = [1] + [0] * (N_turb - 1) + [1, 0] * N_vibe
        self.C = np.vstack((crow, crow))
        self.Q = np.zeros((ss,ss))
        self.A = np.zeros((ss,ss))
        self.P = np.zeros((ss,ss)) # P on SPHERE is overall rotation to an orthonormal basis: will put this in later.
        self.R = measurement_error**2 * np.identity(2)

    def identify(self, pol):
        self.identify_physics(pol)
        self.identify_P()

    def identify_physics(self, pol):
        # sets A and Q.
        pass
        # put in or import all the relevant stuff from observer.py here.

    def identify_P(self):
        # solves the DARE for an A and Q that are already set to get P.
        iters = 0
        while True:
            P = self.A.dot(self.P.dot(self.A.T)) + self.Q
            P -= self.A.dot(self.P.dot(self.C.T).dot(np.linalg.inv(self.C.dot(self.P).dot(self.C.T) + self.R))).dot(self.C).dot(self.P).dot(self.A.T)
            if np.allclose(P, self.P):
                return iters
            self.P = P
            iters += 1

    def measure(self):

    def control(self, measurement, action):
        # measurement: a single-timestep measurement in closed loop. 2x1: [tip, tilt].
        # action: the control action at the timestep of the measurement minus 'delay'.
        
