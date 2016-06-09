import numpy as np
import scipy as sc
import scipy.linalg
import kernels
from matplotlib import pyplot as plt
from cholesky import cholesky


def foo(X):
    Y = 0.5 * (np.sin(X**2) + np.sin(X) + (0.25 * X) - 0.5)
    return Y

    
def drawGauss(X):
    kern = kernels.Kernels()
    K = kern.kern_matrix(X, X, "sqr_exp", 0.00001)
    m = np.zeros((np.size(X), 1))
    u = np.random.normal(0, 1, (np.size(X), 1))
    L = cholesky(K)
    
    Y = m + np.dot(L, u)
    
    return Y
    
def normalize(X, Y):
    X = X / np.sum(X)
    Y = Y / np.sum(Y)
    return X, Y

def GP(X, Y):
    pass


def main():
    X = np.arange(0, 2, 0.01)
    Y = foo(X)
    #print(X)
    #print(Y)
    
    Y_ = drawGauss(X)
    
    plt.plot(X, Y)
    plt.plot(X, Y_)
    plt.show()
    
 
    
main()
