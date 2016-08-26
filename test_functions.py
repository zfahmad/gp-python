import numpy as np


def foo(X, noise=0):
    Y = 0.5 * (np.sin(X ** 2) + np.sin(X) + (0.25 * X) - 0.5)
    if noise:
        Y += np.random.normal(0, noise, (np.size(Y, axis=0), 1))
    return Y


def target(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)


def rosenbrock(x):
    return ((1 - x[:, 0]) ** 2) + 100 * ((x[:, 1] - x[:, 0] ** 2) ** 2)
