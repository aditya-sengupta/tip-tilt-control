# find PDF of telescope vibrations
# this wasn't very successful.

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from scipy.integrate import quad
from matplotlib import pyplot as plt

def convolve(f1, f2):
    def convolution(t):
        def integrand(tau):
            return f1(tau) * f2(t - tau)
        return quad(integrand, -np.inf, np.inf)[0]
    return convolution

def get_f1(T, lower, upper):
    def f1(x):
        if lower*T <= x and x <= upper*T:
            return (1/((upper - lower)*T))
        else:
            return 0
    return f1

f1 = get_f1(0.01, 10, 1000)

def f2(x):
    if -2*np.pi <= x and x <= 0:
        return 1/(2*np.pi)
    else:
        return 0

test_time = np.arange(-30, 30, 0.01)
output = np.vectorize(convolve(f1, f2))(test_time)
print(output)
plt.show()
