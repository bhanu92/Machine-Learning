import numpy
import pandas
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

inputData = ['data/blood.csv',
             'data/wine.csv',
             'data/yeast.csv',
             'data/seeds.csv',
             'data/pima.csv',
             'data/planning.csv',
             'data/flags.csv',
             'data/glass.csv',
             'data/breast-cancer.csv',
             'data/hepatitis.csv']

data = []
i = 0
dataset = []
svm = []
lr = []
for i in range(len(inputData)):
    dataset.append(str(inputData[i])[5:])
    data = pandas.read_csv(inputData[i])
    data = numpy.array(data)
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data_x = data[:, :-1]
    data_x = minmax.fit_transform(data_x)
    data_y = data[:, -1]
    # if i == 0:
    #    print(data_y)

    # SVM Classifier
    classify = SVC()
    classify.fit(data_x, data_y)
    predict = classify.predict(data_x)
    #print(accuracy_score(data_y, predict))
    svm.append(accuracy_score(data_y, predict))

    # Logistic Regression Classifier
    classify = LogisticRegression()
    classify.fit(data_x, data_y)
    predict = classify.predict(data_x)
    #print(accuracy_score(data_y, predict))
    lr.append(accuracy_score(data_y, predict))

    i += 1


print("******************************************************************************************")
print("Classifiers used are SVM and Logistic Regression")
print("Please look into generated dimensionalityReduction.txt file for the comparision of results")
print("******************************************************************************************")

file = open("dimensionalityReduction.txt", "w")
title = 'Dataset'.ljust(20) + 'SVM'.ljust(20) + 'LogisticRegression' + '\n'
file.write(title)
for i in range(len(dataset)):
    string = (str(dataset[i])).ljust(20) + (
        str(svm[i])).ljust(20) + (str(lr[i])) + ("\n")
    file.write(string)
file.close()
