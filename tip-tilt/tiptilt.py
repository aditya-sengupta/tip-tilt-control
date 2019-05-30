import numpy as np
import sys
sys.path.append("..")
from dynamics.dynamic import DynamicSystem

class TipTilt(DynamicSystem):

    STATE_SIZE = 2
    INPUT_SIZE = 2

    def __init__(self, d=0.01):
        self.state = np.array([0, 0])
        self.simend = False
        self.P = np.zeros((2, 2))
        self.Q = np.array([[d, d**2/2], [d**2/2, d]]) # noise model to be updated
        self.H = np.identity(2)
        self.R = np.array([[d, d**2/2], [d**2/2, d]]) # noise model to be updated

    def evolve(self, t, dt):
        A = np.identity(2)
        B = np.identity(2)
        return (A, B)

    def ext_input(self, t):
        return (-self.state, "negative state")

    def reset(self):
        self.state = np.array([0, 0])
