import argparse
import numpy as np
import analysis
import dataproc
import mlp


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a test file.')
    parser.add_argument('--model', help='Path to the model file.')
    parser.add_argument('--test_file', help='Path to the test file.')
    args = parser.parse_args()

    # Load the test data.
    xtest, ytest = dataproc.load_data(args.test_file)
    ytest = dataproc.to_one_hot(ytest,int(1+np.max(ytest[0,:])))

    # Load the mlp.
    nn = mlp.MLP.load_mlp(args.model)

    # Apply the model.
    yhat = nn.eval(xtest)

    # Print the stats.
    print('mse:  %f'%(analysis.mse(ytest, yhat)))
    print('mce:  %f'%(analysis.mce(ytest, yhat)))
    print('acc:  %f'%(analysis.accuracy(ytest, yhat)*100))

if __name__ == '__main__':
    main()
