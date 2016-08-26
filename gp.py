import matplotlib.cm as cm
import numpy as np
import scipy.interpolate
import scipy.linalg
from matplotlib import pyplot as plt
import cholesky as ch

import kernels


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
