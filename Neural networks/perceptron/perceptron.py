__author__ = 'kesav'
import numpy as np
import pylab as pl
from sklearn import preprocessing

class Perceptron:
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesav/MachineLearning/dataset/perceptronData.txt", 'rb'),
                         delimiter= None,
                         dtype=float)
        self.x_data = self.xy_data[:, :-1]
        self.bias = np.ones(len(self.x_data)*1).reshape(len(self.x_data), 1)
        self.y_data = self.xy_data[:, -1:]
        self.x_data = np.concatenate((self.bias, self.x_data), axis=1)
        self.w = np.zeros_like(self.x_data[0]).T
        self.learning_rate = 1

    def check_side(self, row):
        result = np.dot(row, self.w)
        if result > 0:
            return 1
        else:
            return -1

    def move_hyperplane(self,  misclassified_row):
        self.w += self.learning_rate * misclassified_row.T

    def train(self):
        # flip the negative points
        for i in range(0, len(self.x_data)):
            if self.y_data[i] == -1:
                self.x_data[i] = - self.x_data[i]
                self.y_data[i] = - self.y_data[i]
        M = 0
        iteration = 0
        canIterate = True
        while canIterate and iteration < 10000:
            for i in range(0, len(self.xy_data)):
                side = self.check_side(self.x_data[i])
                if side == -1:
                    self.move_hyperplane(self.x_data[i])
                    M += 1
            print "Mistakes in iteration ", iteration+1, " is: ", M
            iteration += 1
            if M is 0:
                canIterate = False
            M = 0
        min_max_scaler = preprocessing.MinMaxScaler()
        print "W Vector: ", self.w[:-1]
        #print "Normalized W vector:", min_max_scaler.fit_transform(self.w[:-1])

    def plot(self):
        n = pl.norm(self.w)
        ww = self.w/n
        pl.plot(ww, '--k')
        pl.show()

