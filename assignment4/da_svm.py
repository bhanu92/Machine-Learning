import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import solvers, matrix
from numpy import rank, matrix
from sklearn.metrics import accuracy_score


class svm():

    def __init__(self):
        pass

    def polynomialKernel(self, x1, x2):
        return (1 + np.dot(x1, x2))**3

    def linearKernel(self, x1, x2):
        return np.dot(x1, x2)

    def alphas(self, *arg):

        xtr = arg[0]
        ytr = arg[1]
        da = arg[2]
        if(len(arg) == 4):
            ws = arg[3]
        print(da)
        rows = xtr.shape[0]
        print(rows)
        kVal = np.zeros(shape=[rows, rows]).astype(np.double)
        H = np.zeros(shape=[rows, rows]).astype(np.double)

        if da is False:
            f = -(np.ones(shape=[rows, 1])).astype(np.double)
            A = np.diag(np.ones(rows) * -1).astype(np.double)
            c = np.zeros(rows).astype(np.double)
        else:
            f = np.zeros(shape=[rows, 1])
            for i in range(rows):
                f[i] = (ytr[i] * (np.dot(xtr[i], ws))) - 1
            f = f.astype(np.double)
            A1 = np.diag(np.ones(rows)).astype(np.double)
            A2 = np.diag(np.ones(rows) * -1).astype(np.double)
            A = np.concatenate((A1, A2), axis=0)
            c1 = np.ones(shape=[rows, 1]).astype(np.double)
            c2 = np.zeros(shape=[rows, 1]).astype(np.double)
            c = np.concatenate((c1, c2), axis=0)

        Aeq = np.array([ytr]).astype(np.double)

        # Aeq = Aeq.T
        ceq = np.zeros(shape=[1, 1]).astype(np.double)

        # Kernel Values
        for i in range(rows):
            for j in range(rows):
                kVal[i][j] = self.linearKernel(xtr[i], xtr[j])

        # H matrix Values
        for i in range(rows):
            for j in range(rows):
                H[i][j] = ytr[i] * ytr[j] * kVal[i][j]

        # print(f, "\n", A, "\n", Aeq, "\n")
        # print(np.linalg.matrix_rank(Aeq))

        H = cvxopt.matrix(H)
        f = cvxopt.matrix(f)
        A = cvxopt.matrix(A)
        Aeq = cvxopt.matrix(Aeq)
        c = cvxopt.matrix(c)
        ceq = cvxopt.matrix(ceq)

        qpSolve = cvxopt.solvers.qp(H, f, A, c, Aeq, ceq)
        alphaVal = np.array(qpSolve['x'])
        # print(alphaVal)
        return alphaVal

    def weights(self, alpha, xtr, ytr):

        # non alpha indices
        I = []
        for i in range(alpha.size):
            if alpha[i] != 0:
                I.append(i)
        I = np.array(I)

        # Support Vectors
        svAlpha = []
        svX = []
        svY = []
        for i in range(I.size):
            svAlpha.append(alpha[I[i]])
            svX.append(xtr[I[i]])
            svY.append(ytr[I[i]])

        # Weight Vectors
        svW = np.zeros(len(svX[0]))
        for i in range(len(svAlpha)):
            svW += svAlpha[i] * svX[i] * svY[i]

        svW = np.sum(svW * ytr[:, None] * xtr, axis=0)
        norm = np.linalg.norm(svW)
        svW = svW / norm
        return svW


if __name__ == "__main__":
    sourceData = pd.read_csv('source_train.csv', header=None)
    sourceData = np.array(sourceData)
    xsTrain = sourceData[:, :-1]
    ysTrain = sourceData[:, -1]
    '''
    plt.figure()
    plt.title('Source Training data')
    plt.plot(sourceData[np.where(sourceData[:, -1] == 1), 0],
             sourceData[np.where(sourceData[:, -1] == 1), 1], 'o', c='b')
    plt.plot(sourceData[np.where(sourceData[:, -1] == -1), 0],
             sourceData[np.where(sourceData[:, -1] == -1), 1], 's', c='r')
    '''
    # plt.show()

    bhanu = svm()
    # Without domain adaption
    lagrangeMultipliers = bhanu.alphas(xsTrain, ysTrain, False)
    # Source Weights
    ws = bhanu.weights(lagrangeMultipliers, xsTrain, ysTrain)

    targetTrain = pd.read_csv('target_train.csv', header=None)
    targetTrain = np.array(targetTrain)
    xtTrain = targetTrain[:, :-1]
    ytTrain = targetTrain[:, -1]
    # With domain adaption
    lagrangeMultipliers = bhanu.alphas(xtTrain, ytTrain, True, ws)
    # Target weights
    wt = bhanu.weights(lagrangeMultipliers, xtTrain, ytTrain)

    print("Source Data Weights: ", ws)
    print("Target Data Weights: ", wt)

    sourceTest = pd.read_csv('source_test.csv', header=None)
    sourceTest = np.array(sourceTest)
    xsTest = sourceTest[:, :-1]
    ysTest = sourceTest[:, -1]
    predictST = []
    for i in range(ysTest.size):
        predictionVal = (xsTest[i][0] * wt[0]) + (xsTest[i][1] * wt[1])
        if (predictionVal < 0):
            predictST.append(-1)
        elif (predictionVal > 0):
            predictST.append(1)
    predictST = np.array(predictST)
    finalPredict = accuracy_score(ysTest, predictST)
    print("Accuracy score for source test data: ", finalPredict)

    targetTest = pd.read_csv('target_test.csv', header=None)
    targetTest = np.array(targetTest)
    xtTest = targetTest[:, :-1]
    ytTest = targetTest[:, -1]
    predictWT = []
    for i in range(ytTest.size):
        predictionVal = (xtTest[i][0] * wt[0]) + (xtTest[i][1] * wt[1])
        if (predictionVal < 0):
            predictWT.append(-1)
        elif (predictionVal > 0):
            predictWT.append(1)
    predictWT = np.array(predictWT)
    finalPredict = accuracy_score(ytTest, predictWT)
    print("Accuracy score for target test data`: ", finalPredict)

    plt.figure()
    plt.title('Target test data')
    plt.plot(targetTest[np.where(targetTest[:, -1] == 1), 0],
             targetTest[np.where(targetTest[:, -1] == 1), 1], 'o', c='b')
    plt.plot(targetTest[np.where(targetTest[:, -1] == -1), 0],
             targetTest[np.where(targetTest[:, -1] == -1), 1], 's', c='r')

    plt.figure()
    plt.title('Target test data with prediction values')
    plt.plot(targetTest[np.where(predictWT[:] == 1), 0],
             targetTest[np.where(predictWT[:] == 1), 1], 'o', c='b')
    plt.plot(targetTest[np.where(predictWT[:] == -1), 0],
             targetTest[np.where(predictWT[:] == -1), 1], 's', c='r')
    plt.show()
