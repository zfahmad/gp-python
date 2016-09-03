import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcess

import test_functions as tf

init_points = 2
func = tf.foo

dir = "plots"

X_obj = np.arange(0, 5.1, 0.1)
Y_obj = func(0.75 * X_obj)

# X = 5 * np.random.rand(init_points, 1)
X = np.array([[.6], [2.1], [3.2], [4.4]])
Y = func(0.75 * X)

X_ = np.arange(0, 5.1, 0.1)
X_ = np.reshape(X_, (np.size(X_), 1))

gp = GaussianProcess(corr='squared_exponential')
gp.fit(X, Y)
Y_, mse = gp.predict(X_, eval_MSE=True)

mse = np.reshape(mse, (np.size(mse), 1))

ub = Y_ + 1.966 * mse
lb = Y_ - 1.966 * mse

f = open(dir + "/ub.txt", "w")
g = open(dir + "/lb.txt", "w")
h = open(dir + "/pred.txt", "w")

for i in range(np.size(X_)):
    f.write(str(X_[i, 0]) + " " + str(ub[i, 0]) + "\r")

for i in range(np.size(X_obj)):
    g.write(str(X_[i, 0]) + " " + str(lb[i, 0]) + "\r")

for i in range(np.size(X_)):
    h.write(str(X_[i, 0]) + " " + str(Y_[i, 0]) + "\r")

f.close()
g.close()
h.close()

plt.fill_between(X_[:, 0], ub[:, 0], lb[:, 0], alpha=0.2)
# plt.plot(X_, ub, color='purple')
# plt.plot(X_, lb, color='purple')

plt.plot(X_, Y_)
plt.plot(X, Y, "o", markersize=8, color="blue", alpha=1)
plt.plot(X_obj, Y_obj, color="blue", ls="--", lw=1)
plt.show()
