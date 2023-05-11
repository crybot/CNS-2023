import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import glob
import os.path
from animation import animate_plot
from optimizer import HyperOptimizer
import random

class LSM():
    def __init__(self,
            neurons,
            pe = 0.8,
            win_e = 5, win_i = 2,
            w_e = 0.0, w_i = 0.0,
            threshold = 70,
            a1 = 0.02,
            a2 = 0.02,
            a3 = 0.08,
            b1 = 0.2,
            b2 = 0.25,
            b3 = -0.05,
            recurrent=False,
            k=1,
            **kwargs):
        np.random.seed(42)
        random.seed(42)
        self.neurons = neurons
        self.ne = int(round(neurons * pe))
        self.ni = int(round(neurons * (1-pe)))
        self.win_e = win_e
        self.win_i = win_i
        self.threshold = threshold
        self.recurrent = recurrent
        self.k = k

        self.U = np.append(win_e * np.ones(self.ne), win_i * np.ones(self.ni))
        self.S = np.hstack((w_e * np.random.rand(neurons, self.ne),
                -w_i * np.random.rand(neurons, self.ni)))

        self.noise_e = np.random.rand(self.ne)
        self.noise_i = np.random.rand(self.ni)
        self.a = np.append(a1*np.ones(self.ne), a2 + a3*self.noise_i)
        self.b = np.append(b1*np.ones(self.ne), b2 + b3*self.noise_i)
        self.c = np.append(-65 + 15*self.noise_e**2, -65*np.ones(self.ni))
        self.d = np.append(8 - 6*self.noise_e**2, 2*np.ones(self.ni))
        self.reset()
        self.W = None

    def get_state(self):
        return 1*np.array(self.u >= self.threshold)

    def update(self, x):
        # reset potentials after spike
        fired = self.u >= self.threshold
        self.u[fired] = self.c[fired]
        self.w[fired] = self.w[fired] + self.d[fired]

        # inject recurrent inputs weighted by matrix S
        if self.recurrent:
            x = x + np.sum(self.S[:, fired], axis=1)

        # update variables
        self.u = self.u + 0.5*(0.04 * self.u**2 + 5*self.u + 140 - self.w + x) 
        self.u = np.clip(self.u, a_min=-100000, a_max=100000)
        self.u = self.u + 0.5*(0.04 * self.u**2 + 5*self.u + 140 - self.w + x) 
        self.u = np.clip(self.u, a_min=-10000, a_max=10000)
        self.w = self.w + self.a*(self.b*self.u - self.w)

    def fit(self, x, y, ridge=False):
        states = self.forward(x)
        if ridge:
            states_inv = np.linalg.pinv(np.dot(states.T, states) + self.k * np.eye(states.shape[1]))
            self.W = np.dot(np.dot(states_inv, states.T), y)
        else:
            states_inv = np.linalg.pinv(states)
            self.W = np.dot(states_inv, y)
    
    def reset(self):
        self.u = -65 * np.ones(self.neurons)
        self.w = self.b * self.u

    def forward(self, x):
        self.reset()
        states = []
        for x_t in x:
            x_t = x_t * self.U # input scaling by connectivity matrix
            self.update(x_t)
            states.append(self.get_state())

        return np.array(states)

    def predict(self, x):
        if self.W is not None:
            return np.dot(self.W, self.forward(x).T)
        raise Exception("In order to call LSM.predict() you must first call LSM.fit()")

# best config so far (with recurrent = True): ~19.47 MAE
default_config = {
        'neurons': 250,
        'pe': 0.26575951897247463,
        'win_e': 1.2967051461986072,
        'win_i': 0.6257609186832347,
        'w_e': 0.0001020848612463976,
        'w_i': 0.009975982717974417,
        'threshold': 45,
        'a1': 0.4599611413619474,
        'a2': 0.5699011675311774,
        'a3': 0.4344244242376599,
        'b1': 0.2593002946548699,
        'b2': 0.18139310343647408, 
        'b3': -0.35940152562023175,
        'k': 104
        }



# best config so far (with recurrent = True): ~18.64 MAE
default_config = {'neurons': 100, 'pe': 0.2280878357769644, 'win_e': 0.9365371583305216, 'win_i': 0.3360914883435488, 'w_e': 0.0001586068262191018, 'w_i': 0.0096718589561919305, 'threshold': 20, 'a1': 0.4354700497921135, 'a2': 0.548305169122301, 'a3': 0.6861449407176862, 'b1': 0.3219101966063589, 'b2': 0.29131187582949813, 'b3': -1.3403872501135324, 'k': 142.55064717529}


sweep_config = {
  'method': 'bayes',
  'iterations': 200,
  'metric': {
    'name': 'mae',
    'goal': 'minimize'
  },
  'parameters': {
    'k': {
        'distribution': 'uniform',
        'min': 10.0,
        'max': 200.0
    },
    # 'neurons': {
    #     'values': [100, 200, 300],
    #     'distribution': 'uniform',
    #     'min': 100,
    #     'max': 800
    # },
    # 'pe': {
    #     'values': [0.1, 0.3, 0.6],
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9
    # },
    # 'a1': {
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9,
    # },
    # 'a2': {
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9
    # },
    # 'a3': {
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9
    # },
    # 'b1': {
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9
    # },
    # 'b2': {
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 0.9
    # },
    # 'b3': {
    #     'distribution': 'uniform',
    #     'min': -0.9,
    #     'max': -0.1,
    # },
    # 'win_e': {
    #     'values': [1, 4, 8],
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 10.0
    # },
    # 'win_i': {
    #     'values': [1, 4, 8],
    #     'distribution': 'uniform',
    #     'min': 0.1,
    #     'max': 10.0
    # },
    # 'w_e': {
    #     'values': [0, 0.1, 0.5],
    #     'distribution': 'uniform',
    #     'min': 0.0,
    #     'max': 0.1
    # },
    # 'w_i': {
    #     'values': [0, 0.1, 0.5],
    #     'distribution': 'uniform',
    #     'min': 0.0,
    #     'max': 0.1
    # },
    # 'threshold': {
    #     'values': [20, 30, 70],
    #     'distribution': 'uniform',
    #     'min': 20,
    #     'max': 100
    # }
  }
}

def validation_score(config, train_ds, target_ds, val_ds, target_val_ds):
    p = 0.7
    N = int(round(len(train_ds)*p))

    lsm = LSM(**config, recurrent=True)
    # lsm.fit(train_ds, target_ds, ridge=True)
    # prediction = lsm.predict(val_ds)
    lsm.fit(train_ds[:N], target_ds[:N], ridge=True)
    prediction = lsm.predict(train_ds[N:])

    error = np.mean(np.abs(prediction - target_ds[N:]))
    # error = np.mean(np.abs(prediction - target_val_ds))
    print(error)
    return {
            'mae': error
            }


def main():
    np.random.seed(42)
    random.seed(42)
    DATASET_URL = 'https://drive.google.com/file/d/1GK5fqzuAGoo466PIxhnwxtSP0r3uDFWa/view?usp=sharing'
    DATASET_URL = 'https://drive.google.com/uc?id=' + DATASET_URL.split('/')[-2]

    saved = False
    if os.path.isfile('dataset.csv'):
        DATASET_URL = 'dataset.csv'
        saved = True

    ds = pd.read_csv(DATASET_URL, header=None)

    ts = ds.values[0]
    train_ts = ts[:-500]
    val_ts = ts[len(train_ts):]

    def score_function(config):
        return validation_score(config, train_ts[:-1], train_ts[1:], val_ts[:-1], val_ts[1:])

    best_config = default_config.copy()

    optimizer = HyperOptimizer(default_config, sweep_config, plot_metric=False)
    best_config, error = optimizer.optimize(score_function, return_error=True, window_iterations=1)
    print('Best configuration found: {}\n with error: {}'.format(best_config, error))

    lsm = LSM(**best_config, recurrent=True)
    lsm.fit(train_ts[:-1], train_ts[1:], ridge=True)
    prediction = lsm.predict(val_ts[:-1])
    error = np.mean(np.abs(prediction - val_ts[1:]))

    """ OPTIMIZATION PLOTS """
    # Plot the results
    limit_x = 200.0
    x = np.linspace(0, limit_x, 1000)
    # y = f(x)
    opt = optimizer.optimizer
    mu, cov = opt.gp.predict(np.array([list(opt.encoder.encode_dict({'x': xi}).values()) for xi in x]))
    mu = mu.reshape((len(x),))
    std = np.sqrt(np.diag(cov))

    # plt.plot(x, y, label='True Function')
    plt.plot(opt.x, opt.y, 'ro', markersize=8, label='Observations')
    plt.plot(x, mu, label='Predictive Mean')
    plt.fill_between(x, mu+std, mu-std, alpha=0.2, label='Uncertainty')
    plt.legend()
    plt.show()
    """"""""""""""""""""""""""

    print(error)
    plt.plot(range(len(val_ts[1:])), val_ts[1:])
    plt.plot(range(len(prediction)), prediction)
    plt.show()

    if not saved:
        ds.to_csv('dataset.csv', header=None, index=False)


if __name__ == '__main__':
    main()
