######################################################################
# Self Learning on synthetically generated 2D gaussian data
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.figure()

mean = [1, 2]
cov = [[1, 0], [-3, 3]]  # diagonal covariance
x1, y1 = np.random.multivariate_normal(mean, cov, 1000).T
#plt.plot(x1, y1, 'o', color='r', markeredgecolor='black')

data1 = np.array([x1, y1]).T
data1 = np.insert(data1, 2, values=0, axis=1)
# print(data1, "\n", data1.shape)

mean = [-1, -1]
cov = [[1, 0], [-3, 3]]  # diagonal covariance
x2, y2 = np.random.multivariate_normal(mean, cov, 1000).T
#plt.plot(x2, y2, 's', color='b', markeredgecolor='black')
# plt.show()
data2 = np.array([x2, y2]).T
data2 = np.insert(data2, 2, values=1, axis=1)
# print(data2, "\n", data2.shape)

data = np.concatenate((data1, data2), axis=0)
# print(data, "\n", data.shape)
data = shuffle(data, random_state=0)
trainData, testData = train_test_split(data, test_size=0.5)
X = trainData[:, 0:2]
y = trainData[:, -1:]
plt.title("Testing Data for Self Learning")
plt.plot(X[np.where(y == 1)[0], 0], X[np.where(y == 1)[0], 1],
         's', color='b', markeredgecolor='black')
plt.plot(X[np.where(y == 0)[0], 0], X[np.where(y == 0)[0], 1],
         'o', color='r', markeredgecolor='black')
plt.show()

Xvalue = 100
for k in range(2):
    labeledData = trainData[:Xvalue, :]
    unlabeledData = trainData[Xvalue:, :]
    # print(unlabeledData.shape)
    # print(labeledData[:, :2], "\n", labeledData[:, -1:])

    for j in range(5):
        knnClassifier = KNeighborsClassifier(n_neighbors=3)
        knnClassifier.fit(labeledData[:, :2], labeledData[:, -1:].ravel())
        probabilities = knnClassifier.predict_proba(unlabeledData[:, :2])
        index = np.array(np.where((probabilities[:, 0] > 0.9) |
                                  (probabilities[:, 1] > 0.9))).tolist()
        index = index[0]
        # print(len(index))

        for i in range(len(index)):
            if (probabilities[index[i]][0] > 0.9 and probabilities[index[i]][1] < 0.1):
                labeledData = np.vstack((
                    labeledData, [unlabeledData[index[i]][0], unlabeledData[index[i]][1], 0]))
            elif(probabilities[index[i]][1] > 0.9 and probabilities[index[i]][0] < 0.1):
                labeledData = np.vstack((
                    labeledData, [unlabeledData[index[i]][0], unlabeledData[index[i]][1], 1]))

        unlabeledData = np.delete(unlabeledData, index, axis=0)
        # print(unlabeledData.shape)

    # print(unlabeledData.shape)
    # print(labeledData.shape)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(labeledData[:, :2], labeledData[:, -1:].ravel())
    # print(testData[:, :2])
    # print(testData[:, -1:])
    prediction = knn.predict(testData[:, :2])
    print("Prediction accuracy for ", Xvalue / 10, "%" " labeled data is ",
          100 * accuracy_score(testData[:, -1:], prediction))
    Xvalue = 250
