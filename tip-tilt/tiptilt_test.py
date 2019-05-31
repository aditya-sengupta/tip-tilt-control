from tiptilt import TipTilt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_tiptilt(x, y, dt=1e-3):
    fig, ax = plt.subplots(figsize=(5,5))
    limit = max(np.max(np.abs(x)), np.max(np.abs(y)))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))

    pt = ax.plot(x[0], y[0], 'ro-')[0]

    def animate(i):
        pt.set_xdata(x[:i])
        pt.set_ydata(y[:i])

    anim = FuncAnimation(fig, animate, interval=10, frames=len(x)-1)

    plt.draw()
    plt.show()

def make_vibe(N_vibrations, freq, total_time):
    def rotation_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    vib_freqs    = np.random.uniform(low=10.0, high=1000.0, size=N_vibrations)  # Hz
    vib_amps     = np.random.uniform(low=0.0001, high=0.001, size=N_vibrations) # arcseconds
    vib_phase    = np.random.uniform(low=0.0, high=2*np.pi, size=N_vibrations)  # radians
    vib_pa       = np.random.uniform(low=0.0, high=2*np.pi, size=N_vibrations)  # radians

    time_steps = np.arange(0, total_time, 1.0/freq)
    true_positions = np.zeros((len(time_steps),2))

    for i in range(N_vibrations):
        y_init_of_t = vib_amps[i] * np.sin(vib_freqs[i] * time_steps - vib_phase[i])
        x_init_of_t = np.zeros(len(time_steps))
        positions_init = np.vstack((x_init_of_t, y_init_of_t))
        rotated_positions = np.dot(rotation_matrix(vib_pa[i]) , positions_init)
        true_positions = true_positions + np.transpose(rotated_positions)

    return 1000 * true_positions

def filter_tiptilt(N, total_time=0.1):
    zeropoint = 2.82e9 # photons/s/m2, Cedric's H-band zeropoint
    throughput = 0.55 # Cedric's total throughput, not counting QE
    wavelength = 1630e-9 # meters
    diameter = 10.0 # meters

    Hmag = 9.0    # star's H magnitude
    freq = 1000.0 # tip/tilt loop speed, Hz

    true_positions = make_vibe(N, freq, total_time)

    photons_per_measurement = zeropoint * 10**(-0.4*Hmag) * (1.0/freq) * (np.pi * (diameter/2)**2) * throughput
    measurement_noise_single_axis = wavelength/(np.pi * diameter * np.sqrt(photons_per_measurement)) * 206265000 # milliarcseconds

    noisy_positions = true_positions + np.random.normal(loc=0, scale = measurement_noise_single_axis, size = np.shape(true_positions))

    # now let's filter

    vibe_noise = 0.1

    filter = TipTilt(vibe_noise, measurement_noise_single_axis)
    #filter.R = filter.R / 500 # measurement noise model: make this larger and you trust the dynamic model more.

    times, states, inputs = filter.simulate(dt=1.0/freq, timeout=total_time, kalman=(np.arange(0, total_time, 1.0/freq), noisy_positions))

    # fake, just to see what'll happen

    true_positions = np.vstack((np.array([0, 0]), true_positions))[:100]
    noisy_positions = np.vstack((np.array([0, 0]), noisy_positions))[:100]

    res_x = states[0] - true_positions[::,0]
    res_y = states[1] - true_positions[::,1]

    res_x_noise = noisy_positions[::,0] - true_positions[::,0]
    res_y_noise = noisy_positions[::,1] - true_positions[::,1]

    # let's make an animation

    def make_tiptilt_fig():
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,9))
        limit = np.max(np.abs(noisy_positions))
        ax1.set(xlim=(-limit, limit), ylim=(-limit, limit))
        ax2.set(xlim=(-limit, limit), ylim=(-limit, limit))
        plt.xlabel('X position [mas]',fontsize=14)
        plt.ylabel('Y position [mas]',fontsize=14)
        plt.title("Tip-Tilt Filtering, vibration N = " + str(N))
        true_pos = ax1.plot(true_positions[0][0], true_positions[0][1], 'go-', label='True position')[0]
        noisy_pos = ax1.plot(noisy_positions[0][0], noisy_positions[0][0], 'ko-', label='Noisy position')[0]
        filtered_pos = ax1.plot(states[0][0], states[0][1], 'ro-', label='Filtered position')[0]
        res_noise = ax2.plot(res_x_noise[0], res_y_noise[0], 'ko-', label='Noisy residual')[0]
        res_filt = ax2.plot(res_x[0], res_y[0], 'ro-', label='Filtered residual')[0]
        ax1.legend()
        ax2.legend()
        return fig, ax1, ax2, true_pos, noisy_pos, filtered_pos, res_noise, res_filt

    fig, ax1, ax2, true_pos, noisy_pos, filtered_pos, res_noise, res_filt = make_tiptilt_fig()

    def animate(i):
        true_pos.set_xdata(true_positions[:i,0])
        true_pos.set_ydata(true_positions[:i,1])
        noisy_pos.set_xdata(noisy_positions[:i,0])
        noisy_pos.set_ydata(noisy_positions[:i,1])
        filtered_pos.set_xdata(states[0][:i])
        filtered_pos.set_ydata(states[1][:i])
        res_noise.set_xdata(res_x_noise[:i])
        res_noise.set_ydata(res_y_noise[:i])
        res_filt.set_xdata(res_x[:i])
        res_filt.set_ydata(res_y[:i])

    anim = FuncAnimation(fig, animate, interval=200, frames=len(res_x_noise)-1)

    plt.draw()
    plt.show()

    fig, ax1, ax2, true_pos, noisy_pos, filtered_pos, res_noise, res_filt = make_tiptilt_fig()
    animate(len(res_x_noise)-1)
    plt.savefig("tiptilt"+str(N)+".svg", format='svg')

    print("Measurement residual SD, x: " + str(np.std(res_x_noise)))
    print("Measurement residual SD, y: " + str(np.std(res_y_noise)))
    print("Filtered residual SD, x: " + str(np.std(res_x)))
    print("Filtered residual SD, y: " + str(np.std(res_y)))

filter_tiptilt(20) # some vibe
