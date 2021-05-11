import numpy as np

def load_data(filename):
    ''' Parse a dataset into two design matrices for observations and labels.'''
    xdata = []
    ydata = []
    with open(filename) as fh:
        for line in fh:
            if (line.find(';') >= 0):
                x,y = line.split(';')
                x = [float(_) for _ in x.strip().split()]
                y = [float(_) for _ in y.strip().split()]
            else:
                x = [float(_) for _ in line.strip().split()[:-1]]
                y = [float(_) for _ in line.strip().split()[-1]]
            xdata.append(x)
            ydata.append(y)
    return np.array(xdata).transpose(), np.array(ydata).transpose()


def to_one_hot(labels, k):
    ''' Return a one-hot encoded matrix representation of labels.
    
    Labels is assumed to be a row-vector of integers.
    k is the total number of classes.
    '''
    ans = np.zeros((k,labels.shape[1]))
    for n in range(labels.shape[1]):
        ans[int(labels[0][n])][n] = 1
    return ans
