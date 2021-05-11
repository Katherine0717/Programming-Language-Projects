import numpy as np
import pickle
import math

''' An implementation of an MLP with a single layer of hidden units. '''


class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'a2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.

        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.

        Note: a1 and z1 can be used for caching during backprop/evaluation.

        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2 * (np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2 * (np.random.random((self.dout, self.hidden_units)) - 0.5)

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def eval(self, xdata):
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''
        self.a1 = np.dot(self.W1, xdata) + np.array(self.b1);
        self.z1 = np.tanh(self.a1);
        self.a2 = np.dot(self.W2, self.z1) + np.array(self.b2);

        sum = np.zeros(xdata.shape[1])
        for m in range(0, self.dout):
            sum = sum + np.exp(self.a2[m]);

        # softmax = np.exp(sum);
        yhat = np.array([np.exp(self.a2)[k]/sum for k in range(self.dout)]);
        return yhat;

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. '''
        dE_dW2, dE_db2, dE_dW1, dE_db1 = self.grad(xdata, ydata);
        self.W1 = self.W1 - learn_rate * dE_dW1
        self.b1 = self.b1 - learn_rate * dE_db1
        self.W2 = self.W2 - learn_rate * dE_dW2
        self.b2 = self.b2 - learn_rate * dE_db2

    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter.

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''

        deltaE_a2 = self.eval(xdata) - ydata;
        dE_dW2 = np.dot(deltaE_a2, self.z1.transpose());
        dE_db2 = np.dot(deltaE_a2, np.ones((xdata.shape[1], 1)));

        deltaE_a1 = np.dot(deltaE_a2.transpose(), self.W2) * (1-self.z1*self.z1).transpose();
        dE_dW1 = np.dot(deltaE_a1.transpose(), xdata.transpose());
        dE_db1 = np.dot(deltaE_a1.transpose(), np.ones((xdata.shape[1], 1)));

        return dE_dW2/xdata.shape[1], dE_db2/xdata.shape[1], dE_dW1/xdata.shape[1], dE_db1/xdata.shape[1]

