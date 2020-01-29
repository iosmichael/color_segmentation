import numpy as np
import os

class GaussianNaiveBayes(object):

    def __init__(self, weight_file='model.npy'):
        self.is_fitted = False

        self.weight_file = weight_file
        if os.path.isfile(weight_file):
            conf = np.load(weight_file, allow_pickle=True).item()
            self.prior_stop, self.prior_nstop = conf['prior_stop'], conf['prior_nstop']
            self.fmean_stop, self.fmean_nstop = conf['fmean_stop'], conf['fmean_nstop']
            self.fvar_stop, self.fvar_nstop = conf['fvar_stop'], conf['fvar_nstop']

            self.is_fitted = True
            print('successfully loaded previous weight file from {}'.format(self.weight_file))

    '''
    implementation of gaussian naive bayes discriminative model
    '''
    def fit(self, X, y):
        '''
        input: X to be: n x d (number of examples x number of dimensions)
        input: y to be: n x 1 (number of examples x label)
        '''
        # sanity check on the training data
        n, d = X.shape
        print(n, d)
        assert n == y.shape[0]
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)
        # calculate the prior of two labels: 1 and 0
        num_stop, num_nstop = np.sum(y), n-np.sum(y)
        print('num_l(stop): {}, num_l(nstop): {}, total_num:{}'.format(num_stop, num_nstop, n))
        self.prior_stop, self.prior_nstop = num_stop / n, num_nstop / n
        # calculate the average of each feature vector
        self.fmean_stop = np.sum(X * y, axis=0) / num_stop
        self.fmean_nstop = np.sum(X * (1-y), axis=0) / num_nstop
        self.fmean_stop, self.fmean_nstop = self.fmean_stop.reshape(1,-1), self.fmean_nstop.reshape(1,-1)
        print(self.fmean_stop)
        assert self.fmean_stop.shape == (1, d)
        self.fvar_stop = np.sum(((X - self.fmean_stop) * y) ** 2, axis=0) / num_stop
        self.fvar_nstop = np.sum(((X - self.fmean_nstop) * (1-y)) ** 2, axis=0) / num_nstop
        self.fvar_stop, self.fvar_nstop = self.fvar_stop.reshape(1,-1), self.fvar_nstop.reshape(1,-1)
        print(self.fvar_stop)
        assert self.fvar_stop.shape == (1, d)
        conf = {
                  'prior_stop': self.prior_stop,
                  'prior_nstop': self.prior_nstop,
                  'fmean_stop': self.fmean_stop,
                  'fmean_nstop': self.fmean_nstop,
                  'fvar_stop': self.fvar_stop,
                  'fvar_nstop': self.fvar_nstop
                }
        np.save(self.weight_file, conf)
        print('saved model to {}'.format(self.weight_file))
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise Exception('Gaussian Naive Bayes model has not been trained')
        n, d = X.shape
        y = np.zeros((n, 2))
        y[:, 0] = np.log(self.prior_nstop) - .5 * np.sum(np.log(2 * np.pi * np.sqrt(self.fvar_nstop))) - .5 * np.sum((X - self.fmean_nstop) ** 2 / self.fvar_nstop, axis=1)

        y[:, 1] = np.log(self.prior_stop) - .5 * np.sum(np.log(2* np.pi * np.sqrt(self.fvar_stop))) - .5 * np.sum((X - self.fmean_stop) ** 2 / self.fvar_stop, axis=1)
        print(y[:20, 0])
        print(y[:20, 1])
        return np.argmax(y, axis=1)

    def get_accuracy(self, valX, valY):
        if not self.is_fitted:
            raise Exception('Gaussian Naive Bayes model has not been trained')
        y = self.predict(valX)
        n = y.shape[0]
        accuracy = (n-np.sum(np.logical_xor(y, valY))) / n
        return accuracy
