from hcipy import *
from matplotlib import pyplot as plt
from ast import literal_eval

with open('output.txt', 'r') as input:
    data = [line for line in input.readlines()]
for i in range(len(data)//4):
    N = int(data[i * 4])
    print(N)
    sps = 40 * N//128
    print(data[i * 4 + 1])
    grid = PyramidWavefrontSensorEstimator(circular_aperture(9.96), make_pupil_grid(sps*2, (3.6e-3)*sps*2/N)).pupil_mask.grid
    plt.subplot(2,2,1)
    imshow_field(Field(literal_eval(data[i * 4 + 2]), grid), vmin=-0.005, vmax=0.005)
    plt.subplot(2,2,2)
    imshow_field(Field(literal_eval(data[i * 4 + 3]), grid), vmin=-0.005, vmax=0.005)
    plt.show()
