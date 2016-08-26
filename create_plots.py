import os

import matplotlib.pyplot as plt
import numpy as np

import gp as gauss
import test_functions as tf

iterations = 5

dir = "./plots"

if not os.path.isdir(dir):
    os.makedirs(dir)

# Create initial training data
# Edit the function being called. Using foo() by default

func = tf.foo
init_points = 2

X = 5 * np.random.rand(init_points, 1)
Y = func(0.75 * X)

# Create data to plot objective function

bounds = [0, 5.1]

X_obj = np.arange(bounds[0], bounds[1], 0.1)
Y_obj = func(0.75 * X_obj)

# Create gaussian process instance

gp = gauss.GaussianProcess(X, Y, "sqr_exp")
ac = gauss.AcquisitionFunctions()

# Create data to plot prediction function

X_ = np.arange(0, 5.1, 0.1)

# Create best prediction function

mini_Y = []
mini_sum_mse = np.infty
mini_m = []
mini_mse = []

for j in range(30):
    m, var, mle, sum_mle = gp.noisy_predict(X_, 0)
    Y_ = gp.drawGauss(m, var)
    mse = ((Y_ - m) ** 2) / .5

    if np.sum(mse) < mini_sum_mse:
        mini_Y = Y_
        mini_sum_mse = np.sum(mse)
        mini_mse = mse
        mini_m = m

# Create acquisition function

A = ac.ucb(mini_Y, mini_mse, kappa=1)

# Write files to store plot data for external use i.e. LaTeX

f = open(dir + "/pre_0.txt", "w")
g = open(dir + "/func.txt", "w")
h = open(dir + "/acq_0.txt", "w")
q = open(dir + "/points_0.txt", "w")

for i in range(np.size(X_)):
    f.write(str(X_[i]) + " " + str(mini_Y[i, 0]) + "\r")

for i in range(np.size(X_obj)):
    g.write(str(X_obj[i]) + " " + str(Y_obj[i]) + "\r")

for i in range(np.size(X_)):
    h.write(str(X_[i]) + " " + str(A[i, 0]) + "\r")

for i in range(np.size(X)):
    q.write(str(X[i, 0]) + " " + str(Y[i, 0]) + "\r")

f.close()
g.close()
h.close()
q.close()

# Plot graphs

plt.tick_params(axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off')

plt.tick_params(axis='y',
                which='both',
                left='off',
                right='off',
                labelleft='off')

plt.xlabel(r'$\mathcal{X}$', size=20)
plt.ylabel(r'$f:\mathcal{X}\rightarrow\mathbb{R}$', size=20)

plt.ylim(-2, 3)
plt.xlim(-.1, 5.1)

fig = plt.figure()
ax = fig.gca()

ax.plot(X_, mini_Y, color='green', lw=2, alpha=0.7)  # Prediction
ax.plot(X_obj, Y_obj, color="blue", ls="--", lw=1)  # Objective
ax.plot(X_, A, color="red", lw=2, ls="dotted", alpha=0.7)  # Acquisition
ax.plot(X, Y, "o", markersize=8, color="blue", alpha=1)  # Samples

plt.savefig(dir + "/gpo_0")

# Iterations

for iteration in range(iterations):

    x = np.argmax(A)
    x = X_[x]
    X = np.vstack((X, x))
    Y = func(0.75 * X)

    gp = gauss.GaussianProcess(X, Y, "sqr_exp")

    mini_Y = []
    mini_sum_mse = np.infty
    mini_mse = []

    mini = np.infty
    for j in range(30):
        m, var, mle, sum_mle = gp.noisy_predict(X_, 0)
        Y_ = gp.drawGauss(m, var)
        mse = ((Y_ - m) ** 2) / .5

        if np.sum(mse) < mini_sum_mse:
            mini_Y = Y_
            mini_sum_mse = np.sum(mse)
            mini_mse = mse
            mini_m = m

    A = ac.ucb(mini_Y, mini_mse, kappa=1)

    f = open(dir + "/pre_" + str(iteration + 1) + ".txt", "w")
    h = open(dir + "/acq_" + str(iteration + 1) + ".txt", "w")
    q = open(dir + "/points_" + str(iteration + 1) + ".txt", "w")

    for i in range(np.size(X_)):
        f.write(str(X_[i]) + " " + str(mini_Y[i, 0]) + "\r")

    for i in range(np.size(X_)):
        h.write(str(X_[i]) + " " + str(A[i, 0]) + "\r")

    for i in range(np.size(X)):
        q.write(str(X[i, 0]) + " " + str(Y[i, 0]) + "\r")

    f.close()
    h.close()
    q.close()

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(X_, mini_Y, color='green', lw=2, alpha=0.7)
    ax.plot(X_obj, Y_obj, color="blue", ls="--", lw=1)
    ax.plot(X_, A, color="red", lw=2, ls="dotted", alpha=0.7)
    ax.plot(X, Y, "o", markersize=8, color="blue", alpha=1)

    plt.savefig(dir + "/gpo_" + str(iteration + 1))
