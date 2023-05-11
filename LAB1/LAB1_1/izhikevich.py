import matplotlib.pyplot as plt
import numpy as np
import yaml
import glob
from animation import animate_plot

def euler_step(f, t, dt, x, *args):
    return x + dt * f(t, x, *args)

def izhikevich(I, a, b, c, d, u0, w0 = None, w_constant = None,
        dt = 0.25, e = 5, f = 140, steps = 100,
        leapfrog = False, plot = True, animate = True, **kwargs):
    f_u = lambda t, u, w: 0.04 * u**2 + e*u + f - w + I[t]

    if w_constant:
        f_w = lambda t, w, u: a * (b * (u - w_constant))
    else:
        f_w = lambda t, w, u: a * (b * u - w)

    us = []
    ws = []
    if w0:
        w0 = -16
    else:
        w0 = b * u0
    u = u0
    w = w0
    ts = np.linspace(0, steps, int(steps/dt))

    for t in range(len(ts)):
        u_new = euler_step(f_u, t, dt, u, w)

        if leapfrog:
            u = u_new
        w_new = euler_step(f_w, t, dt, w, u)

        u = u_new
        w = w_new

        # reset u and w
        if u > 30:
            us.append(30)
            u = c
            w = w + d
        else:
            us.append(u)

        ws.append(w)

    if plot:
        title = kwargs.get('name', 'u-plot')
        fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [1, 0.2, 1]})
        fig.suptitle(title)
        fig.tight_layout()

        axs[0].set_title('potential plot')
        axs[0].set(xlabel='time', ylabel='u')
        if animate:
            potential_animation = animate_plot(fig, axs[0], ts, us)
        else:
            axs[0].plot(ts, us)

        axs[1].set_title('input current')
        axs[1].set(xlabel='time', ylabel='I')

        if animate:
            input_animation = animate_plot(fig, axs[1], ts, I, 'tab:orange')
        else:
            axs[1].plot(ts, I, 'tab:orange')

        axs[2].set_title('phase portrait')
        axs[2].set(xlabel='u', ylabel='w')

        # Plot and annotate starting point in the phase portrait
        axs[2].plot([u0], [w0], 'tab:red', marker='o')
        axs[2].annotate('start', (u0, w0), textcoords='offset points', xytext=(5, 8))
        if animate:
            phase_animation = animate_plot(fig, axs[2], us, ws, 'tab:green', alpha=0.7, dashes=(2,1), max_len=150)
        else:
            axs[2].plot(us, ws, 'tab:green', alpha=0.7, dashes=(2,1))

        plt.show()

def make_step(v, t1, T, dt, v0=0):
    T1 = int(t1 / dt)
    N = int(T / dt)
    return np.array([v0] * T1 + [v] * (N - T1))

def make_linear(step_values, lengths, ts, T, dt, v0=0):
    if not lengths:
        lengths = [None]*len(ts)

    I = np.zeros(int(T / dt))
    for t, v, l in zip(ts, step_values, lengths):
        I_t = make_step(v, t, T, dt, v0=v0)

        t1 = int(t / dt)
        I_t[t1:] = [ v0 + v * ((i+1)*dt) for i,v in enumerate(I_t[t1:]) ]

        if l:
            I_t[(t1+int(l/dt)+1):] = v0

        I = I + I_t

    return I

def make_pulse(step_values, length, ts, T, dt, v0=0):
    N = int(T / dt)
    I = np.zeros(int(T / dt))

    for t, v in zip(ts, step_values):
        t = int(t / dt)
        I_t = np.array([v0] * t + [v] * int(length/dt) + [v0] * (N - t - int(length/dt)))
        I = I + I_t

    return I

def main():
    for filename in glob.glob("features/*.yaml"):
        print(filename)
        with open(filename) as f:
            parameters = yaml.load(f, yaml.Loader)
            dt = parameters['dt']
            steps = parameters['steps']
            input_type = parameters.get('input_type', 'step')
            I0 = parameters.get('I0', 0)

            # if parameters['name'] != 'bistability':
            #     continue

            if input_type == 'linear':
                I = make_linear(parameters['step_values'],
                        parameters.get('lengths', []),
                        parameters['TS'], steps, dt, v0=I0)
            elif input_type == 'pulse':
                I = make_pulse(parameters['step_values'],
                        parameters['pulse_length'], parameters['TS'],
                        steps, dt, v0=I0)
            else: 
                I = make_step(parameters['step_values'][0],
                        parameters['TS'][0], steps, dt, v0=I0)

            izhikevich(I, leapfrog=True, plot=True, **parameters)

if __name__ == '__main__':
    main()