#!/usr/bin/python3

# AUTHOR:  Ningyuan Xiong
# NetID:   nxiong
# csugID:  nxiong
from matplotlib import pyplot as plt
import numpy as np
import math

# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points

# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys

# Do learning with k number of iterations.
def perceptron(train_xs, train_ys, iterations):
    # Find R
    R = 0
    for x in train_xs:
        temp = np.linalg.norm(x)
        if temp > R:
            R = temp
    print(" R  = ", R)

    train_size = np.shape(train_xs)[0];
    parameter_size = np.shape(train_xs)[1];
    weights = np.zeros(parameter_size);

    countUpdates = 0;
    data = []
    accuracyMax = 0
    fig = plt.figure()
    for i in range(0, iterations):
        for j in range(0, train_size):
            if (train_ys[j] * np.dot(train_xs[j], weights)) <= 0:
                weights = weights + train_xs[j] * train_ys[j];
                countUpdates = countUpdates + 1;

        accuracyVal = accuracy(weights, train_xs, train_ys)
        data.append(accuracyVal)
        if accuracyVal > accuracyMax:
            accuracyMax = accuracyVal

    if(accuracyVal == 1):
        upper = math.sqrt((R * R) / countUpdates);
        print('Upper bound of delta: ', upper)

    print('Updates:', countUpdates)
    print('Accuracy:', accuracyVal)
    print('Maximum accuracy:', accuracyMax)
    print('Iteration:', iterations)
    plt.plot(data)
    fig.savefig('plot.png')

    return weights;

# Do learning without knowing iterations
def perceptronStop(train_xs, train_ys):
    # Find R
    R = 0
    for x in train_xs:
        temp = np.linalg.norm(x)
        if temp > R:
            R = temp
    print(" R  = ", R)

    train_size = np.shape(train_xs)[0];
    parameter_size = np.shape(train_xs)[1];
    weights = np.zeros(parameter_size);

    countUpdates = 0;
    iteration = 1
    data = []
    accuracyMax = 0
    fig = plt.figure()
    while True:
        for j in range(0, train_size):
            if (train_ys[j] * np.dot(train_xs[j], weights)) <= 0:
                weights = weights + train_xs[j] * train_ys[j];
                countUpdates = countUpdates + 1;

        accuracyVal = accuracy(weights, train_xs, train_ys)
        data.append(accuracyVal)
        if accuracyVal > accuracyMax:
            accuracyMax = accuracyVal
        if accuracyVal == 1:
            print('Updates:', countUpdates)
            print('Accuracy:', accuracyVal)
            print('Maximum accuracy:', accuracyMax)
            print('Iteration (stop when convergent):', iteration)
            upper = math.sqrt((R * R)/countUpdates);
            print('Upper bound of delta: ', upper)
            plt.plot(data)
            fig.savefig('plot.png')
            break
        if iteration % 10000 == 0:
            print('Accuracy:', accuracyVal)
            print('Maximum accuracy:', accuracyMax)
            print('Iteration:', iteration)
            plt.plot(data)
            fig.savefig('plot.png')
        iteration = iteration + 1
    return weights

# Return the accuracy over the data using current weights.
def accuracy(weights, test_xs, test_ys):
    test_size = np.shape(test_ys)[0];
    accuracyCount = 0;
    for j in range(0, test_size):
        if (test_ys[j] * np.dot(test_xs[j], weights)) > 0:
            accuracyCount += 1;

    return accuracyCount / test_size;
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')
    parser.add_argument('--convergence', type=int, default=0, help='Iterate until convergence')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """

    train_xs, train_ys = parse_data(args.train_file)
    if args.convergence:
        perceptronStop(train_xs, train_ys)
    else:
        weights = perceptron(train_xs, train_ys, args.iterations)
        acc = accuracy(weights, train_xs, train_ys)
        print('Feature weights (bias last): {}'.format(' '.join(map(str, weights))))


if __name__ == '__main__':
    main()

