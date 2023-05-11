import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from functools import partial

def init(line):
    line.set_data([], [])
    return line,

def step(line, xs, ys, max_len, i):
    skip_factor = int(5 * len(xs)/500)
    i = (i * skip_factor) % len(xs)
    line.set_data(xs[max(0, i-max_len):i], ys[max(0, i-max_len):i])
    return line,

def animate_plot(fig, ax: Axes, xs: [float], ys: [Number], *args, max_len=np.inf, **kwargs) -> FuncAnimation:
    line, = ax.plot(xs, ys, *args, **kwargs)
    # ax.set_ylim([min(min(ys), -0.5), max(ys) + max(ys)*0.05])
    # ax.set_xlim([min(min(xs) - min(xs)*0.05, -0.5), max(xs) + max(xs)*0.05])

    init_f = partial(init, line)
    step_f = partial(step, line, xs, ys, max_len)
    return animation.FuncAnimation(fig, step_f, init_func=init_f,
                               frames=len(xs), interval=24, blit=True, repeat=True)
