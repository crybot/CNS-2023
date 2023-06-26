#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import glob
import os.path
import random
from firing_rate import FiringRate, HebbFiringRate, OjaFiringRate, SubtractiveNormFiringRate, BCMFiringRate
from operator import itemgetter

def correlation(X):
    return np.dot(X.T, X)

def training(model, x, eta = 1e-3, epochs = 100, eps=1e-5, plot = True, save_file = True, name=None):
    bcm = isinstance(model, BCMFiringRate)
    w_new = model.weights
    w_history = [w_new]
    theta_history = [model.theta] if bcm else []
    
    for epoch in range(epochs):
        np.random.shuffle(x)
        w_old = w_new
        for xi in x:
            y = model(np.array([xi]))
            model.backward(eta)
        w_new = model.weights
        w_history.append(w_new)
        if bcm:
            theta_history.append(model.theta)

        if np.linalg.norm(w_new - w_old) <= eps:
            break;

    corr = correlation(x)
    e_val, e_vec = np.linalg.eig(corr)
    e_vec = e_vec.T
    if plot:
        ws = w_history

        fig, axs = plt.subplots(3, 2)
        fig.tight_layout()

        axs[0, 0].scatter(x[:, 0], x[:, 1])
        axs[0, 0].quiver(*model.weights.T, scale=5, color='tab:red', width=0.009)
        axs[0, 0].quiver(*e_vec[0], scale=5, color='tab:orange', width=0.013)

        axs[0, 1].plot(range(len(ws)), list(map(np.linalg.norm, ws)), color='tab:orange')
        axs[0, 1].set_title('W norm over epochs')
        axs[1, 0].plot(range(len(ws)), list(map(lambda w: w[:, 0], ws)), color='tab:orange')
        axs[1, 0].set_title('W_0 over epochs')
        axs[1, 1].plot(range(len(ws)), list(map(lambda w: w[:, 1], ws)), color='tab:orange')
        axs[1, 1].set_title('W_1 over epochs')
        axs[2, 0].plot(range(len(ws)), list(map(lambda w: w.sum(), ws)), color='tab:orange')
        axs[2, 0].set_title('W sum over epochs')

        if bcm:
            axs[2, 1].plot(range(len(theta_history)), theta_history, color='tab:orange')
            axs[2, 1].set_title('Theta (BCM) over epochs')

        if save_file:
            plt.savefig(f'figures/{name}.png')
        plt.show()

    return w_history, corr, e_val, e_vec

def main():
    np.random.seed(42)
    random.seed(42)

    DATASET_PATH = './lab2_1_data.csv'
    ds = pd.read_csv(DATASET_PATH, header=None)
    ds = ds.transpose()

    model = HebbFiringRate(2, 1)
    ws, _, _, _ = training(model, ds.values, name='hebb')

    model = OjaFiringRate(2, 1)
    ws, _, _, _ = training(model, ds.values, name='oja')

    model = SubtractiveNormFiringRate(2, 1)
    ws, _, _, _ = training(model, ds.values, name='subtractive_norm')

    model = BCMFiringRate(2, 1, theta0 = 0.1, tau = 0.1)
    ws, _, _, _ = training(model, ds.values, epochs = 500, eta = 0.01, name='bcm')

if __name__ == '__main__':
    main()

