
import numpy as np
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N*D where each row is an example. y is 1-dimension of size N"""
        self.Xtr = X
        self.ytr = y


    def predict(self, X, k = 1):
        """ X is N * D where each row is an example we wish to predict label for """
        num_test  = X.shape[0]

        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)

            Ypred[i] = self.ytr[min_index]

        return Ypred