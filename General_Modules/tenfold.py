__author__ = 'kesav'
import numpy as np
from sklearn import cross_validation

class Tenfold:
    def __init__(self, data, delegate):
        self.fold_count = 0
        self.data = data
        self.delegate = delegate

    def tenfold_train(self):
        while self.fold_count < 10:
            start = 0
            array_length = len(self.data)
            chunk_size = array_length/10
            iterator = 0
            test_data = None
            train_data = np.zeros_like(self.data[:1, :])

            while iterator < 10:
                end = start + chunk_size
                if iterator != self.fold_count:
                    train_data = np.concatenate((train_data, self.data[start:end, :]), axis=0)
                else:
                    test_data = self.data[start:end, :]
                iterator += 1
                start = start + chunk_size + 1

            self.delegate.train(train_data[1:, :], test_data, self.fold_count)
            self.fold_count += 1
        return

    def tenfold_IthRow_for_test(self):
        while self.fold_count < 10:
            iterator = 0
            train_data = np.zeros_like(self.data[:1, :])
            test_data = np.zeros_like(self.data[:1, :])

            while iterator < len(self.data):
                if iterator % 10 != self.fold_count:
                    train_data = np.concatenate((train_data, self.data[iterator, :].reshape(1, 58)), axis=0)
                else:
                    test_data = np.concatenate((test_data, self.data[iterator, :].reshape(1, 58)), axis=0)
                iterator += 1
            self.delegate.train(train_data[1:, :], test_data[1:, :], self.fold_count)
            self.fold_count += 1
        return

    def inbuilt_tenfold_train(self):
        for i in range(0, 10):
            x_train, x_test, y_train, y_test = \
                cross_validation.train_test_split(self.data[:, :-1], self.data[:, -1:], test_size = 0.1, random_state=i)
            train_data = np.concatenate((x_train, y_train), axis=1)
            test_data = np.concatenate((x_test, y_test), axis=1)
            self.delegate.train(train_data, test_data, i)
        return
