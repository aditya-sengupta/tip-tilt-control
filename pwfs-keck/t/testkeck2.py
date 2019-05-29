from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

def make_atmosphere(input_grid):
    #Uses the Guyon/Males 2017 model.
    heights = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
    velocities = np.array([[0.6541, 6.467], [0.005126, 6.55], [-0.6537, 6.568], [-1.326, 6.568],
                           [-21.98, 0.9], [-9.484, -0.5546], [-5.53, -0.8834]])
    outer_scales = np.array([2, 20, 20, 20, 30, 40, 40])
    Cn_squared = np.array([0.672, 0.051, 0.028, 0.106, 0.08, 0.052, 0.012]) * 1e-12
    layers = []
    for h, v, o, cn in zip(heights, velocities, outer_scales, Cn_squared):
        l = InfiniteAtmosphericLayer(input_grid, cn, o, v, h, 2)
        layers.append(l)

    return layers

def timelapse(aperture, input_grid):
    speed = 10
    times = np.arange(0, 1.0 * speed, 0.01 * speed)
    mw = GifWriter('adaptive_optics2.gif', 10)
    # atmosphere = make_atmosphere(input_grid)
    fake_atmosphere = InfiniteAtmosphericLayer(input_grid, 0.1, 2, 0, 5000, 2)
    for t in times:
        # wf = Wavefront(aperture(input_grid))
        wf = lambda time: Wavefront(aperture(input_grid))
        wf_clean = Wavefront(aperture(input_grid))
        assert(wf_clean is not wf(t))
        # for layer in atmosphere:
            # layer.t = t
            # wf = layer(wf)
        fake_atmosphere.t = t
        wf_clean = fake_atmosphere(wf_clean)
        wf = fake_atmosphere(wf(t))
        # my_phase = wf.phase * aperture(input_grid)
        # my_phase[my_phase==0] = np.nan
        # print(  np.sqrt(np.nanmean(my_phase**2)) * (0.5/(2*np.pi)) , ' microns' )
        # print(  np.sqrt(np.nanmean(atmos.phase_for(wavelength)**2)) * (0.5/(2*np.pi)) , ' microns' )
        plt.clf()
        #imshow_field(atmos.phase_for(wavelength)* aperture(pupil_grid),  cmap='RdBu')
        plt.subplot(2,2,1)
        imshow_field(wf.phase * aperture(input_grid), cmap='RdBu')
        plt.subplot(2,2,2)
        imshow_field(keck_pyramid.forward(wf).phase)
        plt.subplot(2,2,3)
        imshow_field(wf_clean.phase * aperture(input_grid), cmap='RdBu')
        plt.subplot(2,2,4)
        imshow_field(keck_pyramid.forward(wf_clean).phase)
        plt.colorbar()
        plt.draw()
        plt.pause(0.00001)
    mw.close()

pupil_grid = make_pupil_grid(270, 9.96)
keck_pyramid = PyramidWavefrontSensorOptics(pupil_grid, pupil_separation=1)

timelapse(make_keck_aperture(False), pupil_grid)
