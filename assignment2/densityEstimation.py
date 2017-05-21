import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
import math


def gen_cb(N, a, alpha):
    """
    N: number of points on the checkerboard
    a: width of the checker board (0<a<1)
    alpha: rotation of the checkerboard in radians
    """
    d = np.random.rand(N, 2).T
    d_transformed = np.array([d[0] * np.cos(alpha) - d[1] * np.sin(alpha),
                              d[0] * np.sin(alpha) + d[1] * np.cos(alpha)]).T
    s = np.ceil(d_transformed[:, 0] / a) + np.floor(d_transformed[:, 1] / a)
    lab = 2 - (s % 2)
    data = d.T
    return data, lab


X, y = gen_cb(5000, .25, 3.14159 / 4)
X_test, y_test = gen_cb(5000, .25, 3.14159 / 4)
plt.figure()
plt.title('Initial checker board data plot')
plt.plot(X[np.where(y == 1)[0], 0], X[np.where(y == 1)[0], 1], 'o')
plt.plot(X[np.where(y == 2)[0], 0], X[np.where(y == 2)[0], 1], 's', c='r')
# plt.show()

X1 = X[np.where(y == 1)[0], :]
X2 = X[np.where(y == 2)[0], :]

# Kernel density functions
kdfX1 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X1)
kdfX2 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X2)

score1 = kdfX1.score_samples(X_test)
score2 = kdfX2.score_samples(X_test)

score1Exp = math.e**(score1)
score2Exp = math.e**(score2)


Y = []
i = 0
for i in range(len(X_test)):
    if score1Exp[i] > score2Exp[i]:
        Y.append(1)
    else:
        Y.append(2)

Y = np.array(Y)

plt.figure()
plt.plot(X_test[np.where(Y == 1)[0], 0], X_test[
    np.where(Y == 1)[0], 1], 'o', c='y')
plt.plot(X_test[np.where(Y == 2)[0], 0], X_test[
    np.where(Y == 2)[0], 1], 's', c='r')
plt.title('Plot after applying Kernel Density Estimation')
plt.show()
