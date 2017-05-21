import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pylab as plt
from sklearn.preprocessing import PolynomialFeatures


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
poly = PolynomialFeatures(degree=3)
X_new = poly.fit_transform(X)
lr = LogisticRegression()
lr.fit(X_new, y)
##y_new = lr.predict(X_new)

plt.figure()
plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 's', c = 'b')
plt.plot(X[np.where(y==0)[0], 0], X[np.where(y==0)[0], 1], 'o', c = 'r')
plt.show()

Xte, yte = make_moons(n_samples=1000, noise=0.3, random_state=0)
X_new = poly.fit_transform(Xte)
yhat = lr.predict(X_new)

plt.figure()
plt.plot(Xte[np.where(yhat==1)[0], 0], Xte[np.where(yhat==1)[0], 1], 'o', c = 'r')
plt.plot(Xte[np.where(yhat==0)[0], 0], Xte[np.where(yhat==0)[0], 1], 's', c = 'g')
plt.show()
