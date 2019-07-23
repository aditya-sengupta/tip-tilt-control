# Assuming an ideal Kalman filter is providing state estimates,
# make the optimal control law for TT including time delays.

import numpy as np
from aberrations import *
from observer import *

def control_naive(args, measurements, delay=1):
    # naive strategy of Kalman predicting 'delay' times and setting that as the expectation for the next timestep
    state, A, P, Q, H, R = args
    steps = int(f_sampling * time_id)
    pos_r = np.zeros(steps)
    control_action = np.zeros((steps, state.size))
    for k in range(steps):
        measurement = measurements[k]
        state, P = update(H, P, R, state, measurement)
        state += control_action[k]
        pos_r[k] = H.dot(state)
        if k + delay < steps:
            prediction = state
            for _ in range(delay):
                prediction = A.dot(prediction)
            control_action[k + delay] = -prediction
        state, P = predict(A, P, Q, state)
    return pos_r

def control_LQG(args, measurements, delay=1):
    raise NotImplementedError()
