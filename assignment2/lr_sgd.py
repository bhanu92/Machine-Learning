import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
import math

file = 'data/blood.csv'
data = pd.read_csv(file)
#data = datasets.load_iris()
#test_x = data.data[:100, :2]
#test_y = data.target[:100]
# print(data)
minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))

data = numpy.array(data)
train_data, test_data = train_test_split(data, test_size=0.4)

x_train = train_data[:, :-1]
x_train = minmax.fit_transform(x_train)
y_train = train_data[:, -1]

z = numpy.ones((x_train.shape[0], 1))
x_train = numpy.concatenate((z, x_train), axis=1)
# print(x_train)
# print(y_train)

x_test = test_data[:, :-1]
y_test = test_data[:, -1]
# print(y_test)
'''
count1 = 0
count0 = 0
for i in range(len(y_test)):
    if y_test[i] == 0:
        count0 += 1
    elif y_test[i] == 1:
        count1 += 1
print("Count1 is ", count1)
print("Count0 is ", count0)
'''


def hypothesis(theta, X):
    z = 0
    # print(theta[0])
    for n in range(len(theta)):
        z += X[n] * theta[n]
    sigmoid = 1.0 / float((1.0 + math.exp(-1.0 * z)))
    return sigmoid


def hypothesis_test(theta, X):
    z = theta[0]
    for n in range(len(X)):
        z += X[n] * theta[n + 1]
    sig = sigmoid(z)
    return sig


def costFuncDerivative(X, Y, theta, rate, j):
    # l is number of samples
    '''
    sum = 0
    for k in range(l):
        for m in range(j):
            hypo = hypothesis(theta, X[k])
            error = (hypo - Y[k]) * X[k][m]
            sum += error
    jTheta = float((rate) * sum)
    return jTheta
    '''
    sum = 0
    for i in range(len(Y)):
        hypo = hypothesis(theta, X[i])
        error = (hypo - Y[i]) * X[i][j]
        sum += error

    jTheta = float(rate) * sum
    return jTheta


def sgd(X, Y, theta, rate, l):
    tempTheta = []
    for j in range(len(theta)):
        derivative = costFuncDerivative(X, Y, theta, rate, j)
        temp = theta[j] - derivative
        tempTheta.append(temp)
    return tempTheta


def costFunction(X, Y, theta):
    sum = 0
    for i in range(len(Y)):
        hypo = hypothesis(theta, X[i])
        sum += ((Y[i] * math.log(hypo)) + ((1 - Y[i]) * math.log(1 - hypo)))
    cost = float((-1.0 / float(len(Y))) * sum)
    #cost = sum
    # print(cost)
    return cost


def logisticRegression(X, Y, theta, rate, iteration):
    i = 0
    finalCost = costFunction(X, Y, theta)
    print("initial cost value is ", finalCost)
    while i <= iteration:
        # print(i)
        theta = sgd(X, Y, theta, rate, len(Y))
        cost = costFunction(X, Y, theta)
        #finalCost -= cost
        if i % 100 == 0:
            print("Cost after ", i, "th iteration is", cost)
            '''
        if finalCost < 0.001:
            break
            '''
        i += 1
    return theta


def main():
    learnRate = 0.001
    iterations = 1000
    numTheta = len(x_train[0])
    print("Number of features in the data set are ", numTheta - 1)
    print("Learning rate is ", learnRate)
    print("Number of iterations are ", iterations)
    initialTheta = [0] * (numTheta)
    # print(initialTheta)
    finalTheta = logisticRegression(
        x_train, y_train, initialTheta, learnRate, iterations)
    print("Final Weight values are: ", finalTheta)

    y_test = []
    for i in range(len(x_train)):
        Yi = hypothesis(finalTheta, x_train[i])
        y_test.append(round(Yi))
    # print(y_test)
    # print(len(y_test))
    # print(len(y_train))
    count1 = 0
    count0 = 0
    for i in range(len(y_test)):
        if y_test[i] == y_train[i]:
            count1 += 1
        else:
            count0 += 1
    #print("Count1 is ", count1)
    #print("Count0 is ", count0)
    ''''
    # step size in the mesh
    h = .01

    # Get the minimum and maximum values for each feature
    x_min, x_max = X[:100, 0].min() - 0.5, X[:100, 0].max() + 0.5
    y_min, y_max = X[:100, 1].min() - 0.5, X[:100, 1].max() + 0.5

    # Create a mesh grid using max min calculated
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    z = np.ones((X_mesh.shape[0], 1))
    X_mesh = np.concatenate((z, X_mesh), axis=1)

    # Predict the class for creating a mesh
    Z = pred_values(fitted_values, X_mesh, hard=True)
    Z = Z.reshape(xx.shape)

    # Plot the graph using color mesh and scatter the data points on the graph
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #plt.contourf(xx, yy, Z, 50, cmap="gray")
    plt.scatter(X[:100, 0], X[:100, 1], c=y[:100], s=50,
                cmap=cmap_bold, vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)
    plt.show()
    '''
if __name__ == '__main__':
    main()
