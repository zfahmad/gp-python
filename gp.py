import numpy as np
import scipy as sc
import scipy.linalg
import kernels
from matplotlib import pyplot as plt
from cholesky import cholesky


def foo(X):
    Y = 0.5 * (np.sin(X**2) + np.sin(X) + (0.25 * X) - 0.5)
    return Y
    
def SE(x, x_, l):
    return np.exp(-0.5 * (l**2) * ((x - x_)**2))

def cov(X, lam):
    kern = kernels.Kernels()
    K = np.array([])
    
    for i in range(np.size(X)):
        k = np.array([])
        
        for j in range(np.size(X)):
            k = np.append(k, kern.sqr_exp(X[i], X[j]))
        
        if not np.size(K):
            K = k
        else:
            K = np.vstack((K, k))
    return K + lam * np.identity(np.size(K, axis=0))
    
def drawGauss(X):
    K = cov(X, 0.00001)
    #print(K)
    v, w = sc.linalg.eig(K)
    #print(v)
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
    X = np.arange(0, 5, 0.01)
    Y = foo(X)
    #print(X)
    #print(Y)
    
    Y_ = drawGauss(X)
    
    plt.plot(X, Y)
    plt.plot(X, Y_)
    plt.show()
    
 
    
main()
