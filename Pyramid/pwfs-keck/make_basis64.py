import numpy as np
from hci import *

def pupil_sin_phase(pupil, wavsx=1, wavsy=0, amplitude=0.1):
    size=int(np.sqrt(pupil.size))
    x=np.arange(size)
    y=np.arange(size)
    sin = np.zeros((size,size))

    if wavsx==0 and wavsy==0:
        return pupil
    elif wavsy==0:
        yfreq=0
        xfreq = 2*np.pi/((size/wavsx))
    elif wavsx==0:
        xfreq=0
        yfreq = 2*np.pi/((size/wavsy))
    else:
        xfreq = 2*np.pi/((size/wavsx))
        yfreq = 2*np.pi/((size/wavsy))

    for i in range(len(x)):
        for j in range(len(y)):
            sin[i,j] = amplitude*np.sin(xfreq*i+yfreq*j)

    return pupil*np.exp(complex(0,1)*sin).ravel()

indices = np.fromfile('indices.dat', dtype=int)
indices.shape = (indices.size//2, 2)
indices = indices.tolist()

modal_ab_basis = []
wf = Wavefront(aperture(pupil_grid))
for x, y in indices:
    modal_ab_basis.append(pupil_sin_phase(wf.electric_field, x, y))

modal_ab_basis = np.asarray(modal_ab_basis)
print(modal_ab_basis.shape)
