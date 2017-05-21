import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


trainData = pd.read_csv('ionosphere_train.csv', header=None)
testData = pd.read_csv('ionosphere_test.csv', header=None)
trainData = np.array(trainData)
testData = np.array(testData)

x = trainData[:, :-1]
y = trainData[:, -1]
xt = testData[:, :-1]
yt = testData[:, -1]

baggingError = []
adaboostError = []

for i in range(2, 30):

    bagging = BaggingClassifier(n_estimators=i)
    bagging.fit(x, y)
    y_predict = bagging.predict(xt)
    accuracy = accuracy_score(yt, y_predict)
    baggingError.append(1 - accuracy)

    boosting = AdaBoostClassifier(n_estimators=i)
    boosting.fit(x, y)
    yboosting_predict = boosting.predict(xt)
    accuracy = accuracy_score(yt, yboosting_predict)
    adaboostError.append(1 - accuracy)

''''
print(baggingError)
print(adaboostError)
val = len(baggingError)
'''
plt.plot(baggingError, '-b', label='Bagging Error')
plt.plot(adaboostError, '-r', label='Adaboost Error')
plt.legend(loc='upper right')
plt.show()
