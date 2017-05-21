import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mean1 = [-1.2, -1.2]
cov1 = [[1.2, 0], [0, 1.2]]
X1 = np.random.multivariate_normal(mean1, cov1, 5000)
Y1 = np.empty(len(X1))
Y1.fill(-1)
plt.title('Initial Data')
plt.plot(X1[:, 0], X1[:, 1], 's', c='b', label='Negative Class')

mean2 = [4, 4]
cov2 = [[1.2, 0], [0, 1.2]]
X2 = np.random.multivariate_normal(mean2, cov2, 5000)
Y2 = np.empty(len(X2))
Y2.fill(1)
plt.plot(X2[:, 0], X2[:, 1], 'o', c='r', label='Positive Class')
plt.show()

X = np.concatenate((X1, X2), axis=0)
Y = np.concatenate((Y1, Y2), axis=0)
Xtr, Xte, Ytr, Yte = cross_validation.train_test_split(X, Y, test_size=0.25)
plt.title('Test data plot')
plt.plot(Xtr[Ytr == -1, 0], Xtr[Ytr == -1, 1],
         's', c='b', label='Negative Class')
plt.plot(Xtr[Ytr == 1, 0], Xtr[Ytr == 1, 1], 'o', c='r', label='Postive Class')
plt.show()

linear_svm = svm.SVC(kernel='linear').fit(Xtr, Ytr)
lpr = linear_svm.predict(Xte)
plt.title('Plotting the testing data for the Linear Kernel Function')
plt.plot(Xte[lpr == -1, 0], Xte[lpr == -1, 1],
         's', c='b', label='Negative Class')
plt.plot(Xte[lpr == 1, 0], Xte[lpr == 1, 1],
         'o', c='r', label='Positive Class')
plt.show()
score = accuracy_score(Yte, lpr)
print("Linear Kernel Function accuracy score: ", score * 100)

linear_svm = svm.SVC(kernel='poly').fit(Xtr, Ytr)
lpr = linear_svm.predict(Xte)
plt.title('Plotting the testing data for the Polynomial Kernel Function')
plt.plot(Xte[lpr == -1, 0], Xte[lpr == -1, 1],
         's', c='b', label='Negative Class')
plt.plot(Xte[lpr == 1, 0], Xte[lpr == 1, 1],
         'o', c='r', label='Positive Class')
plt.show()
score = accuracy_score(Yte, lpr)
print("Polynomial Kernel Function accuracy score: ", score * 100)
