from aberrations import *
from observer import *
from controller import *
from test_observer import *
import numpy as np

def test_naive_control(args, truth, measurements, physics, disp=(True, False, False, True)):
    controlled = naive_control(args, measurements)
    toplot = (truth, measurements, physics, controlled)
    labels = ('Truth', 'Measurements', 'Physics', 'Controlled')
    for t, l, d in zip(toplot, labels, disp):
        if d:
            plt.plot(times, t, label=l)
    plt.legend()
    plt.show()
    return np.sqrt(np.mean((controlled)**2))

if __name__ == "__main__":
    print(test_naive_control(*make_perfect_filter()))
