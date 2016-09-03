import matplotlib.cm as cm
import numpy as np
import scipy.interpolate
import scipy.linalg
from matplotlib import pyplot as plt

import cholesky as ch
import kernels
import test_functions as tf


class AcquisitionFunctions():
    def ucb(self, X, var, kappa=0.01):
        return X + kappa * np.sqrt(var)


class GaussianProcess():
    def __init__(self, X, Y, cov):
        self.X = X
        self.Y = Y
        self.func = cov
        self.plotting1d = False
        self.plotting2d = False
        self.lam = 0.00001

    def drawGauss(self, m, K):
        u = np.random.normal(0, 1, (np.size(K, axis=0), 1))
        Y = m + np.dot(K, u)

        return Y

    def predict(self, X_):
        kern = kernels.Kernels()

        K_a = kern.kern_matrix(self.X, self.X, self.func, self.lam)
        K_b = kern.kern_matrix(self.X, X_, self.func)
        K_c = kern.kern_matrix(X_, self.X, self.func)
        K_d = kern.kern_matrix(X_, X_, self.func, self.lam)

        m = np.dot(np.dot(K_c, np.linalg.inv(K_a)), self.Y)
        m = np.reshape(m, (np.size(m), 1))

        C = K_d - np.dot(np.dot(K_c, np.linalg.inv(K_a)), K_b)
        L = ch.cholesky(C)

        return m, L

    def noisy_predict(self, X_, noise):
        kern = kernels.Kernels()

        K_a = kern.kern_matrix(self.X, self.X, self.func, self.lam, noise)
        K_b = kern.kern_matrix(self.X, X_, self.func)
        K_d = kern.kern_matrix(X_, X_, self.func, self.lam)

        L = np.linalg.cholesky(K_a)
        alpha = np.linalg.lstsq(L.T, np.linalg.lstsq(L, self.Y)[0])[0]
        m = np.dot(K_b.T, alpha)

        v = np.linalg.lstsq(L, K_b)[0]
        var = K_d - np.dot(v.T, v)

        sum_diag = 0
        for i in range(np.size(L, axis=0)):
            sum_diag += L[i, i]

        sum_mle = -(0.5) * (np.dot(self.Y.T, alpha)) - sum_diag - \
                  (np.size(self.X, axis=0) / 2) * np.log(2 * np.pi)

        mle = -(0.5) * self.Y * alpha - np.reshape(L.diagonal(),
                                                   (np.size(L.diagonal()), 1)) - \
              0.5 * np.log(2 * np.pi)

        return m, var, mle, sum_mle

    def plot1d(self, X_, Y_):
        if not self.plotting2d:
            self.plotting1d = True
            plt.plot(X_, Y_, lw=2, alpha=0.6)
            plt.plot(self.X, self.Y, "+", markersize=10, mew=2, color="black")

    def plot2d(self, X_, Y_):
        x = np.reshape(X_[:, 0], (1, np.size(X_[:, 0])))
        y = np.reshape(X_[:, 1], (1, np.size(X_[:, 1])))
        z = np.reshape(Y_, (1, np.size(Y_)))
        xi = np.linspace(0, 1.5, 100)
        xi, yi = np.meshgrid(xi, xi)

        rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)
        plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()], cmap=cm.afmhot)
        plt.colorbar()

    def showPlots(self):
        plt.grid()
        plt.show()
        self.plotting1d = False
        self.plotting2d = False


# X = np.sort(5 * np.random.rand(3, 1), axis=0)
X = np.array([[.6], [2.1], [3.2], [4.4]])
Y = tf.foo(0.75 * X)

gp = GaussianProcess(X, Y, "sqr_exp")

X_ = np.arange(0, 5.1, 0.1)

m = np.array([])
var = np.array([])
for x in X_:
    m_, var_, mle, sum_mle = gp.noisy_predict(np.array([x]), noise=0)
    m = np.append(m, m_)
    var = np.append(var, var_)

m_ = np.reshape(m, (np.size(m), 1))

ub = m + 1.966 * np.sqrt(var)
lb = m - 1.966 * np.sqrt(var)

kern = kernels.Kernels()
K = kern.kern_matrix(X_, X_, "sqr_exp", lam=0.00001)
m = np.zeros((np.size(K, axis=0), 1))
#
ub_ = np.zeros(np.size(X_)) + 2
lb_ = np.zeros(np.size(X_)) - 2

X_obj = np.arange(0, 5.05, 0.05)
Y_obj = tf.foo(0.75 * X_obj)

plt.xlabel(r"$x$", fontsize=32)
plt.ylabel(r"$f(x)$", fontsize=32)
plt.xticks([])

plt.plot(X_obj, Y_obj, color="black", ls="--", lw=1)
#
#
for i in range(3):
    Y_ = gp.drawGauss(m, np.linalg.cholesky(K))
    plt.plot(X_, Y_, lw=2, alpha=.6)
# plt.grid()
plt.fill_between(X_, ub_, lb_, color="lightgray", alpha=0.3)
plt.ylim(-3, 3)
plt.show()
#
for i in range(3):
    m, L, mse, sum_mse = gp.noisy_predict(X_, noise=0)
    Y_ = gp.drawGauss(m_, L)
    plt.plot(X_, Y_, lw=2, alpha=.6)

plt.plot(X, Y, "+", markersize=10, mew=2, color="black", alpha=1)
plt.ylim(-3, 3)
plt.fill_between(X_, ub, lb, color="lightgray", alpha=0.3)
#
# plt.plot(X_, m, color='black')
plt.plot(X_obj, Y_obj, color="black", ls="--", lw=1)
#
# # plt.grid()
plt.xlabel(r"$x$", fontsize=32)
plt.ylabel(r"$f(x)$", fontsize=32)
plt.xticks([])
plt.show()
