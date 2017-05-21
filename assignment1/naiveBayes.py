import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

data = np.genfromtxt('/home/bhanu92/Desktop/ml/hw/hw1/spambase.csv', delimiter = ',')
x = data[:, 0:-1]
y = data[:, -1]

gnb = GaussianNB()

error = cross_val_score(gnb, x, y, cv=5)

print(error)
print("Mean score after 5 fold CV is " + str(error.mean()))

