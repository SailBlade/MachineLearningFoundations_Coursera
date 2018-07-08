#!/usr/bin/env python
#coding=utf8

"""
Train DATA: https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat
Test DATA:  https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat
Question 18:
    As the test set for "verifying" the g returned by your algorithm (see lecture 4 about verifying).
    The sets are of the same format as the previous one.
    Run the pocket algorithm with a total of 50 updates on D, and verify the performance of w using the test set.
    Please repeat your experiment for 2000 times, each with a different random seed.
    What is the average error rate on the test set?
Question 19:
    Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of
    Wg (the pocket vector) after 50 updates. Run the modified algorithm on D, and verify the performance
    using the test set. Please repeat your experiment for 2000 times, each with a different random seed.
    What is the average error rate on the test set?
Question 20:
    Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance
    of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed.
    What is the average error rate on the test set?
"""

import random
from numpy import array, inner, zeros

TRAIN_FILE = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTrainData.txt'
TEST_FILE = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTestData.txt'


def sign(x):
    if x <= 0:
        return -1
    return 1


def load_data(infile):
    X = []
    Y = []
    with open(infile) as f:
        for line in f:
            recs = line.split()
            x = [1] + [float(v) for v in recs[:-1]]
            X.append(tuple(x))
            Y.append(int(recs[-1]))
    return array(X), array(Y)


def test(X, Y, W):
    n = len(Y)
    ne = sum([1 for i in range(n) if sign(inner(X[i], W)) != Y[i]])
    error = ne / float(n)
    return error


def train(X, Y, updates=50, pocket=True):
    n = len(Y)
    d = len(X[0])
    W = zeros(d)
    Wg = W

    error = test(X, Y, Wg)
    for j in range(updates):
        idx = random.sample(range(n), n)
        for i in idx:
            if sign(inner(X[i], W)) != Y[i]:
                W = W + Y[i] * X[i]
                e = test(X, Y, W)
                if e < error:
                    error = e
                    Wg = W
                break

    if pocket:
        return Wg
    else:
        return W


def main():
    X, Y = load_data(TRAIN_FILE)
    TX, TY = load_data(TEST_FILE)
    error = 0
    n = 200

    for i in range(n):
        # question 18, output: 0.1377
        #W = train(X, Y, updates=50)
        # question 19, output: 0.3559
        W = train(X, Y, updates=50, pocket=False)
        # question 20, output: 0.1141
        #W = train(X, Y, updates=100, pocket=True)
        error += test(TX, TY, W)
    print (error / n)


if __name__ == '__main__':
    main()