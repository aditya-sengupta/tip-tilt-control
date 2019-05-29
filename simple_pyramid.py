import numpy as np
import numpy.ma as ma
import math
from termcolor import colored
size = 35
start_input = np.ones((size, size))
x, y = np.meshgrid(range(size), range(size))

def color_print(arr):
    for row in arr:
        for el in row:
            if el == 0:
                print(colored('0', 'red'), end=" ")
            elif el == 1:
                print(colored('1', 'green'), end=" ")
            else:
                if el < 0:
                    print(colored(el, 'yellow'), end="")
                else:
                    print(colored(el, 'blue'), end=" ")
        print()
    print()

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def clean_input_gen(si):
    gen = lambda s: [[0 if (s/2 - x)**2 + (s/2 - y)**2 > (s**2)/4 else 1 for x in range(s)][1:] for y in range(s)][1:]
    clean = gen(si)
    clean.append(clean[0])
    for row in clean[:len(clean)-1]:
        row.append(row[0])
    if check_symmetric(np.asarray(clean)):
        return clean

def binarize(signal):
    #not sure if that's a word but let's go with it
    for row in signal:
        for index, el in enumerate(row):
            if el > 1:
                row[index] = 1
            elif el < 0:
                row[index] = 0
            else:
                row[index] = np.round(el)
    return signal

def adjust(signal, sensitivity):
    #sensitivity is a positive integer; each signal element is rounded to the nearest 1/sensitivity. The higher it is, the more sensitive the sensor. To binarize, set sensitivity to 1.
    for row in signal:
        for index, el in enumerate(row):
            if 0 < el < 1:
                m = el * sensitivity
                low = math.floor(m)/sensitivity
                high = math.ceil(m)/sensitivity
                if abs(el - low) < abs(el - high):
                    row[index] = low
                else:
                    row[index] = high
            if el > 1:
                row[index] = 1
            elif el < 0:
                row[index] = 0
    return signal

def distort(signal, aberration, fr=100):
    distorted = []
    for row in signal:
        distorted.append(row[:])
    for rownum, distrow in enumerate(distorted):
        for colnum in range(len(distrow)):
            distrow[colnum] = aberration(colnum, fr) + signal[rownum][colnum]
    return distorted

def dft(signal):
    wavefront = np.asarray(signal)


sinewave = lambda col, f: np.sin(2*f*np.pi*col/size)
clean_wavefront = clean_input_gen(size)
aberrated_wavefront = distort(clean_wavefront, sinewave, 25)

def main():
    color_print(clean_wavefront)
    color_print(binarize(aberrated_wavefront))
