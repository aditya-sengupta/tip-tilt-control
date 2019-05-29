from hcipy import *
import numpy as np
from math import *
import mpmath
import scipy
import matplotlib.pyplot as plt

def zernike_variance_von_karman(n, m, R, k0, Cn_squared, wavelength):
	'''Calculate the variance of the Zernike mode (`n`,`m`), using a von Karman turbulence spectrum.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	R : scalar
		The radius of the aperture.
	k0 : scalar
		The spatial frequency of the outer scale (1/L0).
	Cn_squared : scalar
		The integrated Cn^2 profile.
	wavelength : scalar
		The wavelength at which to calculate the variance.
	
	Returns
	-------
	scalar
		The variance of the specific Zernike mode.
	'''
	A = 0.00969 * (2*np.pi / wavelength)**2 * Cn_squared
	coeffs_all = (-1)**(n - m) * 2 * (2 * np.pi)**(11./3) * (n + 1) * A * R**(5./3) / (sqrt(np.pi) * np.sin(np.pi * (n + 1./6)))

	term11 = mpmath.hyper([n + (3./2), n + 2, n + 1],[n + (1./6), n + 2, n + 2, 2 * n + 3], (2*np.pi * R * k0)**2)
	term12 = sqrt(np.pi) * (2*np.pi * R * k0)**(2 * n - 5./3) * scipy.special.gamma(n + 1) / (2**(2 * n + 3) * scipy.special.gamma(11./6) * scipy.special.gamma(n + 1./6) * scipy.special.gamma(n + 2)**2)
	term21 = -1 * scipy.special.gamma(7./3) * scipy.special.gamma(17./6) / (2 * scipy.special.gamma(-n + 11./6) * scipy.special.gamma(17./6)**2 * scipy.special.gamma(n + 23./6))
	term22 = mpmath.hyper([11./6, 7./3, 17./6], [-n + 11./6, 17./6, 17./6, n + 23./6], (2*np.pi * R * k0)**2)

	return coeffs_all * (term11 * term12 + term21 * term22)

def test_infinite_atmosphere_zernike_variances():
	wavelength = 0.5e-6 # 500 nm in meters
	D_tel = 1 # meters
	fried_parameter = 0.2 # meters
	outer_scale = 20 # meters
	velocity = 10 # meters/sec
	num_modes = 200

	pupil_grid = make_pupil_grid(512, D_tel)
	aperture = circular_aperture(D_tel)(pupil_grid)
	wf = Wavefront(aperture, wavelength)

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
	layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

	zernike_modes = make_zernike_basis(num_modes + 20, D_tel, pupil_grid, starting_mode=2)
	
	weights = evaluate_supersampled(circular_aperture(1), pupil_grid, 32)
	zernike_modes = ModeBasis([z * np.sqrt(weights) for z in zernike_modes])
	
	transformation_matrix = zernike_modes.transformation_matrix
	projection_matrix = inverse_tikhonov(transformation_matrix, 1e-9)

	num_iterations = 1000
	mode_coeffs = []

	for it in range(num_iterations):
		if it % (num_iterations / 10) == 0:
			print(100 * it / num_iterations, '%')
		layer.reset()
		#layer.t = np.sqrt(2) * D_tel / velocity
		
		phase = layer.phase_for(wavelength)
		coeffs = projection_matrix.dot(phase * np.sqrt(weights))[:num_modes]
		mode_coeffs.append(coeffs)
		
	variances_simulated = np.var(mode_coeffs, axis=0)

	variances_theory = []
	for j in range(num_modes):
		n, m = noll_to_zernike(j + 2)
		variances_theory.append(zernike_variance_von_karman(n, m, D_tel / 2, 1 / outer_scale, layer.Cn_squared, wavelength))
	variances_theory = np.array(variances_theory)
	
	plt.plot(variances_simulated, label='simulated')
	plt.plot(variances_theory, label='theory')
	plt.yscale('log')
	plt.xlabel('Noll index')
	plt.ylabel('Variance (rad^2)')
	plt.legend()
	plt.show()
	
	plt.plot((variances_simulated / variances_theory) - 1)
	plt.yscale('log')
	plt.show()

	#assert np.all(np.abs(variances_simulated / variances_theory - 1) < 1e-1)
	
if __name__ == '__main__':
	test_infinite_atmosphere_zernike_variances()
