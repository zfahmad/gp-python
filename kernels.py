try:
    import numpy as np
except Exception as e:
    print("Warning: numpy not loaded: ", e)

class Kernels():
    def sqr_exp(self, x, x_, sig_var=1, noise_var=0, l=1):
        sq_dist = np.sqrt(np.sum((x - x_)**2))
        sq_dist = sq_dist**2
        k = sig_var * np.exp((-1 / (2 * l**2)) * sq_dist) + \
            noise_var * self.kronecker_delta(x, x_)
        return k
    
    def kronecker_delta(self, x, x_):
        if np.array_equal(x, x_): return 1
        else: return 0

    def kern_matrix(self, X, Y, method, lam=[]):
        func = getattr(self, method)
        K = []
        for x in X:
            k = []
            for y in Y:
                k.append(func(x, y))
            K.append(k)
        
        K = np.asarray(K)
        if lam:
            K += lam * np.identity(np.size(K, axis=0))

        return K
        
        
def main():
    kern = Kernels()
    A = np.array([1])
    B = np.array([2])
    
    C = np.array([[1, 4]])
    D = np.array([[3, 8]])
    
    print(kern.sqr_exp(A, B))
    print(kern.sqr_exp(C, D))

    X = [1, 2, 6, 7, 8]
    Y = [3, 4, 5]

    print(kern.kern_matrix(X, Y, "sqr_exp"))
    
    

#main()
