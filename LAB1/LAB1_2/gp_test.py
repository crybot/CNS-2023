import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gp import GaussianProcessRegressor, BayesianOptimizer

# Define the function to be optimized (1D)
def f(x):
    return -np.sin(x) - np.cos(2*x)

# Generate initial data points
initial_x = {'x': 0.5}
initial_y = f(initial_x['x'])

# Create BayesianOptimizer instance
optimizer = BayesianOptimizer(initial_x, initial_y)

limit_x = 4*np.pi
# Define parameter space to sample from
space = [{'x': x} for x in np.linspace(0, limit_x, 100)]

# Perform 10 iterations of optimization
x_obs, y_obs, mu, std = [], [], [], []
SPACE_DIM = 300
epochs = 50
for i in range(epochs):
    # Sample from parameter space
    sample = space
    
    # Select the next parameter to evaluate using the optimizer
    query = optimizer.optimize(sample)
    print('Iteration', i+1, ':', query)
    
    # Evaluate the function at the selected parameter
    y = f(query['x'])
    
    # Update the optimizer with the new data
    optimizer.update(query, y)

    # Save the observation, predictive mean and uncertainty at this iteration
    x_obs.append(query['x'])
    y_obs.append(y)
    x_pred = np.linspace(0, limit_x, SPACE_DIM)
    mu_pred, cov_pred = optimizer.gp.predict(np.array([list(optimizer.encoder.encode_dict({'x': xi}).values()) for xi in x_pred]))
    mu_pred = mu_pred.reshape((len(x_pred),))
    std_pred = np.sqrt(np.diag(cov_pred))
    mu.append(mu_pred)
    std.append(std_pred)

# Generate animation
fig, ax = plt.subplots()
ax.set_xlim([0, limit_x])
ax.set_ylim([-2, 2])
line_obs, = ax.plot([], [], 'ro', markersize=8, label='Observations')
line_pred_mean, = ax.plot([], [], label='Predictive Mean')
uncertainty_area = ax.fill_between([], [], [], alpha=0.2, label='Uncertainty')
ax.plot(np.linspace(0, limit_x, SPACE_DIM), f(np.linspace(0, limit_x, SPACE_DIM)), label='True Function')

def init():
    line_obs.set_data([], [])
    line_pred_mean.set_data([], [])

    # uncertainty_area.remove()
    ax.collections.clear()
    uncertainty_area = ax.fill_between([], [], [], alpha=0.2, label='Uncertainty')

    return [line_obs, line_pred_mean, uncertainty_area]

def update(frame):
    # Update observed points
    line_obs.set_data(x_obs[:frame+1], y_obs[:frame+1])
    # Update predictive mean
    line_pred_mean.set_data(np.linspace(0, limit_x, SPACE_DIM), mu[frame])
    # Update uncertainty
    
    lower_bound = mu[frame] - std[frame]
    upper_bound = mu[frame] + std[frame]
    ax.collections.clear()
    uncertainty_area = ax.fill_between(np.linspace(0, limit_x, SPACE_DIM),  lower_bound, upper_bound, facecolor='tab:blue', alpha=0.2)

    return [line_obs, line_pred_mean, uncertainty_area]

anim = FuncAnimation(fig, update, frames=epochs, init_func=init, blit=True)
anim.save('animation.gif', writer='pillow')
anim.save('animation.mp4', writer='ffmpeg')
plt.legend()
plt.show()

