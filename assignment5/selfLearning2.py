######################################################################
# Self Learning on real world data sets
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from statistics import mean

path = '/home/bhanu92/UA-ECE-523-Sp2017/data'

dataSet = [path + '/acute-inflammation.csv',
           path + '/bank.csv',
           path + '/pima.csv',
           path + '/ilpd-indian-liver.csv',
           path + '/breast-cancer-wisc.csv',
           path + '/breast-cancer-wisc-diag.csv',
           path + '/breast-cancer-wisc-prog.csv',
           path + '/blood.csv',
           path + '/titanic.csv',
           path + '/haberman-survival.csv']

print("Datasets are read successfully")

length = len(dataSet)

for k in range(length):
    print("DataSet name is: ", dataSet[k])
    data = np.genfromtxt(dataSet[k], delimiter=',')
    data = shuffle(data, random_state=2)
    kf = KFold(n_splits=5)
    accuracyMean = []
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        partition = int(0.15 * data_train[:, :-1].shape[0])
        labeledData = data_train[:partition, :]
        unlabeledData = data_train[partition:, :]

        for j in range(5):
            knnClassifier = KNeighborsClassifier(n_neighbors=3)
            if(unlabeledData.shape[0] is 0):
                break
            knnClassifier.fit(labeledData[:, :-1], labeledData[:, -1:].ravel())
            probabilities = knnClassifier.predict_proba(unlabeledData[:, :-1])
            index = np.array(np.where((probabilities[:, 0] > 0.9) |
                                      (probabilities[:, 1] > 0.9))).tolist()
            index = index[0]
            # print(len(index))

            for i in range(len(index)):
                if (probabilities[index[i]][0] > 0.9 and probabilities[index[i]][1] < 0.1):
                    # row = np.array(np.append(unlabeledData[index[i]], 0))
                    unlabeledData[index[i]][-1] = 0
                    # print(np.array(unlabeledData[index[i]]).shape)
                    labeledData = np.vstack(
                        (labeledData, unlabeledData[index[i]]))
                elif(probabilities[index[i]][1] > 0.9 and probabilities[index[i]][0] < 0.1):
                    # row = np.append(unlabeledData[index[i]], 1)
                    labeledData = np.vstack(
                        (labeledData, unlabeledData[index[i]]))

            unlabeledData = np.delete(unlabeledData, index, axis=0)
            # print(unlabeledData.shape)

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(labeledData[:, :-1], labeledData[:, -1:].ravel())
            prediction = knn.predict(data_test[:, :-1])
            accuracyMean.append(accuracy_score(data_test[:, -1:], prediction))
            # print(accuracy_score(data_test[:, -1:], prediction))

    print("Accuracy Mean: ", round((100 * mean(accuracyMean)), 2))
    accuracyMean.clear()
