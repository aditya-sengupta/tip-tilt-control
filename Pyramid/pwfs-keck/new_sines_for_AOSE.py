import numpy as np
from hci import *

N = 64
D = 9.96
aperture = circular_aperture(D)
pupil_grid = make_pupil_grid(N, D)

def make_best_sine_approximation(wf):
    S = np.fromfile('orth.dat')
    weights = S.dot(wf.electric_field[aperture(pupil_grid) > 0])
    return S.T.dot(weights)

print("Testing the created method...")
original_electric = Field(pupil_sin_phase(wf.electric_field, 14, 63), wf.grid)
original = Wavefront(original_electric)
imshow_field(original.phase)
plt.show()
as_basis_electric = Field(plot_on_aperture(aperture, make_best_sine_approximation(original).electric_field), wf.grid)
as_basis_c = Wavefront(as_basis_electric)
imshow_field(as_basis_c.phase)
plt.show()
print("Done.")
