# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.

import argparse

from numpy import longdouble
from itertools import chain

from nlputil import *   # utility methods for working with text
from math import log

# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.
class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, vocab_size):
        self.num_states = num_states
        self.vocab_size = vocab_size

        # initialize the parameters randomly
        random = np.random.rand(num_states)
        self.pi = random/sum(random)

        self.transitions = np.ones((num_states, num_states), dtype=longdouble)
        self.transitions = np.random.rand(num_states, num_states)
        self.transitions = self.transitions / self.transitions.sum(axis=0, keepdims=1)

        self.emissions = np.ones((num_states, vocab_size), dtype=longdouble)
        self.emissions = np.ones((num_states, vocab_size), dtype=longdouble)
        random = np.random.rand(num_states, vocab_size)
        self.emissions = random / random.sum(axis=1, keepdims=1)

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):
        loglikelihood = 0.0
        for i in range(len(dataset)):
            alphas, temp = self.loglikelihoodHelper(dataset[i])
            loglikelihood += temp
        return loglikelihood/len(dataset)

    # this function takes a sequence of observation data as input
    def loglikelihoodHelper(self, dataset):
        alphas = np.zeros((len(dataset), self.num_states), dtype=longdouble)
        log_likelihood = 0.0
        for t in range(0, len(dataset)):
            if t == 0:
                for c in range(self.num_states):
                    alphas[0, c] = self.pi[c] * self.emissions[c, int(dataset[t]-1)]
            else:
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        alphas[t, i] += alphas[t - 1, j] * self.transitions[j, i] * self.emissions[i, int(dataset[t]-1)]

            sum = np.sum(alphas[t, :])
            if(sum != 0):
                log_likelihood += log(sum)
            alphas[t, :] /= np.sum(alphas[t, :])

        return alphas, log_likelihood

    # same as the loglikelihoodHelper function, but this function takes the whole dataset as input
    def _forward(self, dataset):
        alphas = np.zeros((len(dataset), self.num_states), dtype=longdouble)
        log_likelihood = 0.0
        for t in range(0, len(dataset)):
            if t == 0:
                for c in range(self.num_states):
                    alphas[0, c] = self.pi[c] * self.emissions[c, int(dataset[t]-1)]
            else:
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        alphas[t, i] += alphas[t - 1, j] * self.transitions[j, i] * self.emissions[i, int(dataset[t]-1)]

            sum = np.sum(alphas[t, :])
            if(sum != 0):
                log_likelihood += log(sum)
            alphas[t, :] /= np.sum(alphas[t, :])

        return alphas, log_likelihood

    def _backward(self, dataset):
        betas = np.zeros((len(dataset), self.num_states))

        betas[len(dataset) - 1, :] = 1
        betas[len(dataset) - 1, :] /= np.sum(betas[len(dataset) - 1, :])

        for t in range(len(dataset) - 2, -1, -1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    betas[t, i] += self.transitions[i, j] * betas[t + 1, j] * self.emissions[j, int(dataset[t+1]-1)]
            betas[t, :] /= np.sum(betas[t, :])

        return betas

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        xi, gamma = self.e_step(dataset)
        self.m_step(xi, gamma, dataset)
        self.normalize() # normalize the pi, transition, and emission

    def e_step(self, dataset):
        alpha, ll = self._forward(dataset)
        beta = self._backward(dataset)

        # initialize the xi and gamma
        xi = np.zeros((len(dataset), self.num_states, self.num_states), dtype=longdouble)
        gamma = np.zeros((len(dataset), self.num_states), dtype=longdouble)

        # calculate xi
        for t in range(len(dataset)-1):
            denominator = np.sum(np.dot(alpha[t, :], self.transitions) * self.emissions[:, int(dataset[t + 1] - 1)] * beta[t + 1, :])
            for i in range(self.num_states):
                molecular = alpha[t, i] * self.transitions[i, :] * self.emissions[:, int(dataset[t + 1] - 1)] * beta[
                                                                                                           t + 1, :]
                xi[t, i, :] = molecular / denominator

        # calculate gamma
        for t in range(len(dataset)):
            for c in range(self.num_states):
                gamma[t][c] = np.sum(xi[t][c])

        return xi, gamma

    def m_step(self, xi, gamma, dataset):
        self.pi = gamma[0, :]

        # using xi and gamma to update transition
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.transitions[i][j] = np.sum(xi[:, i, j]) / np.sum(gamma[:, i])

        # update emission
        data = list(dict.fromkeys(dataset))
        datasetNew = np.array(dataset)
        for i in range(self.num_states):
            for j in data:
                mask = datasetNew == int(j)
                self.emissions[int(i), int(j - 1)] = np.sum(gamma[mask, i], axis=0) / np.sum(gamma[:, i], axis=0)

    def normalize(self):
        for state in range(self.num_states):
            self.emissions[state, :] /= sum(self.emissions[state, :])

        for state in range(self.num_states):
            self.transitions[state, :] /= sum(self.transitions[state, :])

        self.pi[:] /= sum(self.pi)


def main():
    np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None, help='Path to the development data directory.')
    parser.add_argument('--max_iters', type=int, default=30, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()

    # load training data:
    print("Loading the training data.");
    paths = [args.train_path + "/pos", args.train_path + "/neg"]
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    begin = time()
    vocabTrain, count, totalVocab = build_vocab_words(paths)
    end = time()
    print('done in', end-begin, 'seconds.  Found', len(vocabTrain), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    dataTrain = load_and_convert_data_words_to_ints(paths, vocabTrain)
    end = time()
    print('done in', end-begin, 'seconds.')

    flatten_list = list(chain.from_iterable(dataTrain)) # 2d list --> 1d list
    hmm = HMM(args.hidden_states, len(vocabTrain)) # initialize a hmm model
    log = hmm.loglikelihood(dataTrain) # calculate the intial log likelihood
    print("Initialize log-likelihood: ", log)

    # start em-algorithm:
    iter = 0
    loglist = np.zeros(args.max_iters+1) # store the log likelihood after each iteration
    loglist[0] = log
    for i in range(args.max_iters):
        iter += 1
        hmm.em_step(flatten_list)
        log = hmm.loglikelihood(dataTrain)
        loglist[i+1] = log
        print("EM iteration: ", iter, " log likelihood: ", log)
        if((loglist[i+1] - loglist[i]) <= 1): # we set the epsilon to be 1
            break

    # load testing data:
    print("Loading the testing data.")
    paths = [args.dev_path + "/pos", args.dev_path + "/neg"]
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    begin = time()
    vocabTest, count, totalVocab = build_vocab_words(paths)
    end = time()
    print('done in', end-begin, 'seconds.  Found', len(vocabTest), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    dataTest = load_and_convert_data_words_to_ints(paths, vocabTest)
    end = time()
    print('done in', end-begin, 'seconds.')

    # test the model
    log = hmm.loglikelihood(dataTest)
    print("Test data log likelihood: ", log)

if __name__ == '__main__':
    main()
