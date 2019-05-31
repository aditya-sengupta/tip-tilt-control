import numpy as np
import sys
sys.path.append("..")
from dynamics.dynamic import DynamicSystem

class TipTilt(DynamicSystem):

    STATE_SIZE = 2
    INPUT_SIZE = 2

    def __init__(self, vibe_noise, measurement_noise):
        self.state = np.array([0, 0])
        self.simend = False
        self.P = np.zeros((2, 2))
        self.Q = vibe_noise * np.identity(2) # noise model to be updated
        self.H = np.identity(2)
        self.R = measurement_noise * np.identity(2) # noise model to be updated

    def evolve(self, t, dt):
        A = np.identity(2)
        B = np.identity(2)
        return (A, B)

    def ext_input(self, t):
        return (-self.state, "negative state")

    def reset(self):
        self.state = np.array([0, 0])
