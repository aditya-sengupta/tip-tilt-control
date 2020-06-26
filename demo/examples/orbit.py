from dynamic import *

class Rotation(DynamicSystem):

    def __init__(self):
        self.state = np.array([1, 0]) #(x, y)
        self.simend = False
        self.H = np.identity(2)
        self.P = np.zeros([2,2])
        self.Q = np.zeros([2,2])
        self.R = np.zeros([2,2])

    def evolve(self, t, dt):
        # B matrix doesn't matter till we add in forces.
        A, B = np.array([[np.cos(dt), -np.sin(dt)], [np.sin(dt), np.cos(dt)]]), np.array([0, 0])
        if self.state[1] < 0 and A.dot(self.state)[1] >= 0:
            self.simend = True
        return A, B

    def ext_input(self, t):
        return 0

class Gravitation(DynamicSystem):
    # this is made to be very specific as it's just a demo, i.e. parameters aren't passed into init. If someone wanted to generalize it though it'd be easy to move initializations to parameters.
    # you could make this an N-body simulation if you changed ext_input to read in the positions of N-1 other masses and set the gravitational force input based on those. But this is a more interesting control example, and simpler.

    def __init__(self):
        self.state = np.array([1, 0, 0, 52]) #(x, dx/dt, y, dy/dt)
        self.simend = False
        self.H = np.identity(4)
        self.P = np.zeros([4,4])
        self.Q = np.zeros([4,4])
        self.R = np.zeros([4,4])
        self.m = 25

    def evolve(self, t, dt):
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        B = np.array([[dt**2/(2*self.m), 0], [dt/self.m, 0], [0, dt**2/(2*self.m)], [0, dt/self.m]])
        return A, B

    def ext_input(self, t):
        G = 16
        M = 144
        r = np.sqrt((self.state[0]**2 + self.state[2]**2))
        return -(G*M*self.m/r**3)*np.array([self.state[0], self.state[2]])

g = Gravitation()
time, states, inputs = g.simulate(dt=1e-6, timeout=100)
