import argparse
import numpy as np
import random
import analysis
import dataproc
import mlp
from matplotlib import pyplot as plt


def main():
    # parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    # parser.add_argument('--train_file', default=None, help='Path to the training data.')
    # parser.add_argument('--dev_file', default=None, help='Path to the development data.')
    # parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train. (default 10)')
    # parser.add_argument('--learn_rate', type=float, default=1e-1,
    #                     help='The learning rate to use for SGD (default 1e-1).')
    # parser.add_argument('--hidden_units', type=int, default=0, help='The number of hidden units to use. (default 0)')
    # parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for SGD. (default 1)')
    # parser.add_argument('--name', type=str, default='filename', help='The result file name')
    # args = parser.parse_args()

    # Load training and development data and convert labels to 1-hot representation.

    xtrain, ytrain = dataproc.load_data("data/linearSmoke")
    ytrain = dataproc.to_one_hot(ytrain, int(1 + np.max(ytrain[0, :])))
    if (args.dev_file is not None):
        xdev, ydev = dataproc.load_data(args.dev_file)
        ydev = dataproc.to_one_hot(ydev, int(1 + np.max(ytrain[0, :])))

    # Record dimensions and size of dataset.
    N = xtrain.shape[1]
    din = xtrain.shape[0]
    dout = ytrain.shape[0]

    batch_size = args.batch_size
    if (batch_size == 0):
        batch_size = N

    fscore_train = []
    fscore_dev = []
    acc_train = []
    acc_dev = []

    # Create an MLP object for training.
    nn = mlp.MLP(din, dout, args.hidden_units)

    # Evaluate MLP after initialization; yhat is matrix of dim (Dout x N).
    yhat = nn.eval(xtrain)

    best_train = (analysis.mse(ytrain, yhat),
                  analysis.mce(ytrain, yhat),
                  analysis.accuracy(ytrain, yhat) * 100,
                  analysis.fScore(ytrain, yhat))
    print('Initial conditions~~~~~~~~~~~~~')
    print('mse(train):  %f' % (best_train[0]))
    print('mce(train):  %f' % (best_train[1]))
    print('acc(train):  %f' % (best_train[2]))
    print('fscore(train):  %f' % (best_train[3]))
    print('')

    if (args.dev_file is not None):
        best_dev = (analysis.mse(ydev, yhat),
                    analysis.mce(ydev, yhat),
                    analysis.accuracy(ydev, yhat) * 100,
                    analysis.fScore(ydev, yhat))
        print('mse(dev):  %f' % (best_dev[0]))
        print('mce(dev):  %f' % (best_dev[1]))
        print('acc(dev):  %f' % (best_dev[2]))
        print('fscore(dev):  %f' % (best_dev[3]))

    fscore_train.append(best_train[3])
    fscore_dev.append(best_dev[3])
    acc_train.append(best_train[2])
    acc_dev.append(best_dev[2])


    for epoch in range(args.epochs):
        for batch in range(int(N / batch_size)):
            ids = random.choices(list(range(N)), k=batch_size)
            xbatch = np.array([xtrain[:, n] for n in ids]).transpose()
            ybatch = np.array([ytrain[:, n] for n in ids]).transpose()
            nn.sgd_step(xbatch, ybatch, args.learn_rate)

        yhat = nn.eval(xtrain)
        train_ss = analysis.mse(ytrain, yhat)
        train_ce = analysis.mce(ytrain, yhat)
        train_acc = analysis.accuracy(ytrain, yhat) * 100
        train_fs = analysis.fScore(ytrain, yhat)
        best_train = (min(best_train[0], train_ss), min(best_train[1], train_ce), max(best_train[2], train_acc), max(best_train[3], train_fs))

        print('After %d epochs ~~~~~~~~~~~~~' % (epoch + 1))
        print('mse(train):  %f  (best= %f)' % (train_ss, best_train[0]))
        print('mce(train):  %f  (best= %f)' % (train_ce, best_train[1]))
        print('acc(train):  %f  (best= %f)' % (train_acc, best_train[2]))
        print('fscore(train):  %f   (best= %f)' % (train_fs, best_train[3]))

        if (args.dev_file is not None):
            yhat = nn.eval(xdev)
            dev_ss = analysis.mse(ydev, yhat)
            dev_ce = analysis.mce(ydev, yhat)
            dev_acc = analysis.accuracy(ydev, yhat) * 100
            dev_fs = analysis.fScore(ydev, yhat)
            best_dev = (min(best_dev[0], dev_ss), min(best_dev[1], dev_ce), max(best_dev[2], dev_acc), max(best_dev[3], dev_fs))
            print('mse(dev):  %f  (best= %f)' % (dev_ss, best_dev[0]))
            print('mce(dev):  %f  (best= %f)' % (dev_ce, best_dev[1]))
            print('acc(dev):  %f  (best= %f)' % (dev_acc, best_dev[2]))
            print('fscore(dev):  %f  (best= %f)' % (dev_fs, best_dev[3]))

        print('')
        nn.save('modelFile')

        fscore_train.append(train_fs)
        fscore_dev.append(dev_fs)
        acc_train.append(train_acc)
        acc_dev.append(dev_acc)

    fig1 = plt.figure()
    plt.plot(fscore_train)
    plt.xlabel("number of epochs")
    plt.ylabel("f-score")
    plt.title("F-score of the Training Dataset")
    fig1.savefig('plot_train_fscore.png')

    fig2 = plt.figure()
    plt.plot(acc_train)
    plt.xlabel("number of epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of the Training Dataset")
    fig2.savefig('plot_train_acc.png')

    fig3 = plt.figure()
    plt.plot(fscore_dev)
    plt.xlabel("number of epochs")
    plt.ylabel("f-score")
    plt.title("F-score of the Dev Dataset")
    fig3.savefig('plot_dev_fscore.png')

    fig4 = plt.figure()
    plt.plot(acc_dev)
    plt.xlabel("number of epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of the Dev Dataset")
    fig4.savefig('plot_dev_acc.png')

if __name__ == '__main__':
    main()
