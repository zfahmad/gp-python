import numpy as np
import scipy as sc
import scipy.linalg
import kernels
from matplotlib import pyplot as plt
from cholesky import cholesky
from matplotlib.mlab import griddata

def foo(X):
    Y = 0.5 * (np.sin(X**2) + np.sin(X) + (0.25 * X) - 0.5)
    return Y

def rosenbrock(x):
    return (((1 - x[:, 0])**2) + 100 * ((x[:, 1] - x[:, 0]**2)**2))

class GaussianProcess():
    def __init__(self, X, Y, cov):
        self.X = X
        self.Y = Y
        self.func = cov
        self.plotting1d = False
        self.plotting2d = False


    def drawGauss(self, X):
        kern = kernels.Kernels()
        K = kern.kern_matrix(X, X, self.func, 0.00001)
        m = np.zeros((np.size(X), 1))
        u = np.random.normal(0, 1, (np.size(X), 1))
        L = cholesky(K)
        
        Y = m + np.dot(L, u)
        
        return Y

    def predict(self, X_):
        kern = kernels.Kernels()
        
        K_a = kern.kern_matrix(self.X, self.X, self.func, 0.1)
        K_b = kern.kern_matrix(self.X, X_, self.func)
        K_c = kern.kern_matrix(X_, self.X, self.func)
        K_d = kern.kern_matrix(X_, X_, self.func, 0.1)
        
        m = np.dot(np.dot(K_c, np.linalg.inv(K_a)), self.Y)
        m = np.reshape(m, (np.size(m), 1))
        #print(m)
        
        C = K_d - np.dot(np.dot(K_c, np.linalg.inv(K_a)), K_b)

        u = np.random.normal(0, 1, (np.size(X_, axis=0), 1))
        L = np.linalg.cholesky(C)

        Y_ = m + np.dot(L, u)

        return Y_

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
        print(x[0])
        zi = griddata(x[0], y[0], z[0], xi, xi, interp='linear')
        CS = plt.contour(xi, xi, zi)
        plt.show()


    def showPlots():
        plt.grid()
        plt.show()
        self.plotting1d, self.plotting2d = False



def main():
    
    X = np.sort(5 * np.random.rand(5, 1), axis=0)
    Y = foo(0.75 * X)
    
    gp = GaussianProcess(X, Y, "sqr_exp")

    
    #X_ = np.sort(5 * np.random.rand(75, 1), axis=0)
#    X_ = np.arange(0, 5, 0.05)
#    
#    Y_1 = gp.predict(X_)
#    Y_2 = gp.predict(X_)
#    Y_3 = gp.predict(X_)
#    
#    X_a = np.arange(0, 5, 0.01)
#    Y_a = foo(0.75 * X_a)

    #plt.plot(X_a, Y_a, ls="--", lw=2, alpha=0.5)
    
#    plt.plot(X_, Y_1, lw=2, alpha=0.6)
#    plt.plot(X_, Y_2, lw=2, alpha=0.6)
#    plt.plot(X_, Y_3, lw=2, alpha=0.6)
#    plt.plot(X, Y, "+", markersize=10, mew=2, color="black", alpha=1)
#    plt.grid()
#    plt.show()

    X = np.sort(1.5 * np.random.rand(10, 2), axis=0)
    #print(X)
    Y = rosenbrock(X)

    gp2 = GaussianProcess(X, Y, "sqr_exp")

    X_ = np.vstack((np.arange(0, 1.5, 0.05), np.arange(0, 1.5, 0.05)))
    X_ = X_.T
    
    #print(X_)

    Y_ = gp2.predict(X_)
    print(Y_)
    gp.plot2d(X_, Y_)

    
main()
