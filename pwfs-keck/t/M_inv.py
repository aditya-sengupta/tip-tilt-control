from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

D = 9.96
sps = 20
N = 64
M = np.fromfile('slopes.dat', dtype=float)
M.shape = (3096, 800)
M_list = M.tolist()
M_list = [[el[0:400], el[400:800]] for el in M_list]
M_inv = field_inverse_tikhonov(Field(np.asarray(M_list), make_pupil_grid(sps, D)), 1e-15)
M_inv = M_inv.copy()
M_inv.shape = (3096, 800)
#utility methods copied over from slopes_basis.py; clean up before contribution.

def new_get_sub_images(intensity):
    buffer = 0
    pyramid_grid = make_pupil_grid(N, 3.6e-3)
    images = Field(np.asarray(intensity).ravel(), pyramid_grid)
    images.shape = (54, 54)
    image = images

    sub_images = [image[33:53, 33:53],
                  image[33:53, 0:20],
                  image[0:20, 0:20],
                  image[0:20, 33:53]]
    subimage_grid = make_pupil_grid(sps)
    for count, img in enumerate(sub_images):
        img = img.ravel()
        img.grid = subimage_grid
        sub_images[count] = img
    return sub_images

def estimate(EstimatorObject, images_list):
    I_a = images_list[0]
    I_b = images_list[1]
    I_c = images_list[2]
    I_d = images_list[3]
    norm = I_a + I_b + I_c + I_d
    I_x = (I_a + I_b - I_c - I_d) / norm
    I_y = (I_a - I_b - I_c + I_d) / norm
    I_x = I_x.ravel()
    I_y = I_y.ravel()
    return [I_x, I_y]

aperture = circular_aperture(D)
keck_pyramid_estimator = PyramidWavefrontSensorEstimator(aperture, make_pupil_grid(sps*2, (3.6e-3)*sps*2/N))

pupil_grid = make_pupil_grid(N, D)
aberrated = Wavefront(aperture(pupil_grid))
amplitude = 0.3
spatial_frequency = 5
aberrated.electric_field *= np.exp(1j * amplitude * np.sin(2*np.pi * pupil_grid.x / D * spatial_frequency))
get_pyramid_output = lambda wf: np.asarray(estimate(keck_pyramid_estimator, new_get_sub_images(PyramidWavefrontSensorOptics(pupil_grid, pupil_separation=65/39.3, num_pupil_pixels=sps).forward(wf).intensity))).ravel() #should return a (2*sps*sps,) size NumPy Array

aberrated_images = get_pyramid_output(aberrated)
flat_images = get_pyramid_output(Wavefront(aperture(pupil_grid)))
reconstructed = M_inv.dot(aberrated_images - flat_images).tolist()
project_onto = Wavefront(aperture(pupil_grid)).electric_field
project_onto.shape = (N, N)
project_onto = project_onto.tolist()

count, i, j = 0, 0, 0
while count < 3096:
    if np.real(project_onto[i][j]) > 0:
        project_onto[i][j] = reconstructed[count]
        count += 1
    j += 1
    if j == N - 1:
        j = 0
        i += 1
imshow_field(np.asarray(project_onto).ravel() * aperture(pupil_grid), pupil_grid)
plt.show()
