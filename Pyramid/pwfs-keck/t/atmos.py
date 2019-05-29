from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
D_tel = 8.0
wavelength = 500e-9  #658e-9

spider_width = 0.4
central_obscuration_ratio = 0.355
aperture = make_obstructed_circular_aperture(D_tel, central_obscuration_ratio, 4, spider_width)
# aperture = circular_aperture(D_tel)

pupil_grid = make_pupil_grid(270, D_tel)
wf = Wavefront(aperture(pupil_grid), wavelength)

r_0 = 0.20
outer_scale = 20
velocity=10

# spectral_noise_factory = SpectralNoiseFactoryFFT(kolmogorov_psd, pupil_grid, 8)
# turbulence_layers = make_standard_multilayer_atmosphere(r_0, wavelength=500e-9)
# atmos = AtmosphericModel(spectral_noise_factory, turbulence_layers)

# layers = make_standard_atmospheric_layers(pupil_grid, outer_scale)
# atmos = MultiLayerAtmosphere(layers, scintilation=False)
# atmos.Cn_squared = Cn_squared_from_fried_parameter(r_0, 500e-9)
# atmos.reset()

Cn_squared = Cn_squared_from_fried_parameter(r_0, 500e-9)
atmos = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

times = np.arange(0, 1.0, 0.01)
mw = GifWriter('adaptive_optics2.gif', 10)

for t in times:
	atmos.t = t
	wf2 = atmos(wf)
	my_phase = wf2.phase * aperture(pupil_grid)
	my_phase[my_phase==0] = np.nan
	# print(  np.sqrt(np.nanmean(my_phase**2)) * (0.5/(2*np.pi)) , ' microns' )
    #print(  np.sqrt(np.nanmean(atmos.phase_for(wavelength)**2)) * (0.5/(2*np.pi)) , ' microns' )
	plt.clf()
	imshow_field(atmos.phase_for(wavelength) * aperture(pupil_grid), cmap='RdBu')
	plt.colorbar()
	plt.draw()
	plt.pause(0.00001)
mw.close()
