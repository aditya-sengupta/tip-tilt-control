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

def pyramid_prop(wf, pupsep=1.625, sps=40):
    '''Given a wavefront, returns the result of a pyramid propagation and splitting into sub-images,
    as a list of Field objects.

    Parameters
    ----------
    wf - Wavefront
        The wavefront to propagate through the pyramid.

    Returns
    -------

    '''
    keck_pyramid = PyramidWavefrontSensorOptics(pupil_grid, pupil_separation=pupsep, num_pupil_pixels=sps)
    im = get_sub_images(keck_pyramid.forward(wf).electric_field)
    return im

def estimate(images_list):
    EstimatorObject = PyramidWavefrontSensorEstimator(aperture, make_pupil_grid(sps*2, D_grid*sps*2/N))
    I_b = images_list[0]
    I_a = images_list[1]
    I_c = images_list[2]
    I_d = images_list[3]
    norm = I_a + I_b + I_c + I_d
    I_x = (I_a + I_b - I_c - I_d) / norm
    I_y = (I_a - I_b - I_c + I_d) / norm
    pygrid = make_pupil_grid(sps)
    return Field(I_x.ravel(), pygrid), Field(I_y.ravel(), pygrid)

def make_slopes()
