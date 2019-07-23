from aberrations import *
from observer import *
from controller import *
from test_observer import *
import numpy as np

def test_control_naive(args, truth, measurements, physics, disp=(True, False, False, True)):
    controlled = control_naive(args, measurements)
    toplot = (truth, measurements, physics, controlled)
    labels = ('Truth', 'Measurements', 'Physics', 'Controlled')
    for t, l, d in zip(toplot, labels, disp):
        if d:
            plt.plot(times, t, label=l)
    plt.legend()
    plt.show()
    return np.sqrt(np.mean((controlled)**2))

if __name__ == "__main__":
    print(test_control_naive(*make_sysid_freq_filter()))
