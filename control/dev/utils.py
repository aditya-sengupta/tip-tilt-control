import numpy as np
from scipy import signal, io
from matplotlib import pyplot as plt
from scipy import signal
from copy import deepcopy

def get_keck_tts(num=128):
    # gets the right keck TTs that have the wrong powerlaw.
    # put in a number 128 through 132
    filename = '../telemetry/n0' + str(num) + '_LGS_trs.sav'
    telemetry = io.readsav(filename)['b']
    commands = deepcopy(telemetry['DTTCOMMANDS'])[0]
    commands = commands - np.mean(commands, axis=0)
    residuals = telemetry['DTTCENTROIDS'][0]
    pol = residuals[1:] + commands[:-1]
    return residuals[1:], commands[:-1], pol

def get_keck_tts_wrong(num=128):
    # gets the wrong keck TTs that have the right powerlaw.
    # put in a number 128 through 132
    filename = '../telemetry/n0' + str(num) + '_LGS_trs.sav'
    telemetry = io.readsav(filename)['a']
    commands = deepcopy(telemetry['TTCOMMANDS'][0])
    commands = commands - np.mean(commands, axis=0)
    residuals = telemetry['RESIDUALWAVEFRONT'][0][:,349:351]
    pol = residuals[1:] + commands[:-1]
    return residuals[1:], commands[:-1], pol

