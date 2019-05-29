from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

N = 128
D = 9.96
sps = 40
pupil_grid = make_pupil_grid(N, D)
aperture = circular_aperture(D)

def make_pyramid_from_zonal(wf):
    S = np.fromfile('zonal.dat', dtype=complex)
    S.shape = (S.size//(2*sps*sps), 2*sps*sps)
    print(S.shape)
    inversion = np.linalg.inv(S.T.dot(S))
    print("Done inverting")
    least_square = inversion.dot(S.T)
    print(least_square.shape)
    slopes = least_square.dot(WFToNumericVector(wf))
    spsg = make_pupil_grid(sps)
    return Field(slopes[0:sps*sps], spsg), Field(slopes[sps*sps:2*sps*sps], spsg)

def WFToNumericVector(wf):
    numeric = []
    for pixel in wf.electric_field[aperture(pupil_grid) > 0]:
        numeric.append(pixel.real)
        numeric.append(pixel.imag)
    return np.asarray(numeric)

wf = Wavefront(aperture(pupil_grid))
aberrated = wf.copy()
amplitude = 0.3
spatial_frequency = 5
aberrated.electric_field *= np.exp(1j * amplitude * np.sin(2*np.pi * pupil_grid.x / D * spatial_frequency))

x, y = make_pyramid_from_zonal(aberrated)
imshow_field(x.real)
plt.show()
imshow_field(y.real)
plt.show()
