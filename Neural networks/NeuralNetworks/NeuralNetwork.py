__author__ = 'kesav'
# Algorithm implementation referred from
# http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm

import numpy as np
import pylab as pl


class NeuralNetwork():
    def __init__(self, feature, label, structure):
        # training samples
        self.feature = feature
        self.label = label
        # number of samples
        self.m_size = len(self.feature)
        # layers of networks
        self.nl = len(structure)
        # nodes at layers
        self.topology = structure
        # parameters of networks
        self.w = list()
        self.bias = list()
        self.d_w = list()
        self.d_bias = list()
        self.activation = list()
        self.h_x = list()
        self.error = list()
        for iLayer in range(self.nl - 1):
            self.w.append(np.random.rand(structure[iLayer]*structure[iLayer+1]).reshape(structure[iLayer],structure[iLayer+1]))
            self.bias.append(np.random.rand(structure[iLayer+1]))
            self.d_w.append(np.zeros([structure[iLayer], structure[iLayer+1]]))
            self.d_bias.append(np.zeros(structure[iLayer+1]))
            self.activation.append(np.zeros(structure[iLayer+1]))
            self.h_x.append(np.zeros(structure[iLayer+1]))
            self.error.append(np.zeros(structure[iLayer+1]))

        # value of cost function
        self.j = 0.0
        # active function (logistic function)
        self.sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))
        # learning rate
        self.alpha = .8
        # steps of iteration
        self.steps = 30000

    def backprop(self):
        self.j -= self.j
        for iLayer in range(self.nl-1):
            self.d_w[iLayer] -= self.d_w[iLayer]
            self.d_bias[iLayer] -= self.d_bias[iLayer]
        # propagation (iteration over M samples)
        for i in range(self.m_size):
            for iLayer in range(self.nl - 1):
                if iLayer == 0:
                    self.h_x[iLayer] = np.dot(self.feature[i], self.w[iLayer])
                else:
                    self.h_x[iLayer] = np.dot(self.activation[iLayer-1], self.w[iLayer])
                self.activation[iLayer] = self.sigmod(self.h_x[iLayer] + self.bias[iLayer])
            for iLayer in range(self.nl - 1)[::-1]:
                if iLayer == self.nl-2:
                    self.error[iLayer] = -(self.feature[i] - self.activation[iLayer]) * (self.activation[iLayer]*(1-self.activation[iLayer]))
                    self.j += np.dot(self.label[i] - self.activation[iLayer], self.label[i] - self.activation[iLayer])/self.m_size
                else:
                    self.error[iLayer] = np.dot(self.w[iLayer].T, self.error[iLayer+1]) * (self.activation[iLayer]*(1-self.activation[iLayer]))
                # calculate dW and dB
                if iLayer == 0:
                    self.d_w[iLayer] += self.feature[i][:, np.newaxis] * self.error[iLayer][:, np.newaxis].T
                else:
                    self.d_w[iLayer] += self.activation[iLayer-1][:, np.newaxis] * self.error[iLayer][:, np.newaxis].T
                self.d_bias[iLayer] += self.error[iLayer]
        # update
        for iLayer in range(self.nl-1):
            self.w[iLayer] -= (self.alpha/self.m_size)*self.d_w[iLayer]
            self.bias[iLayer] -= (self.alpha/self.m_size)*self.d_bias[iLayer]

    def encode(self):
        plot_j = {}
        for i in range(self.steps):
            self.backprop()
            plot_j[i] = self.j
        print "Plotting"
        pl.plot(np.arange(i+1), plot_j.values())
        pl.xlabel('Iterations')
        pl.ylabel('Cost Function')
        pl.show()

    def decode(self):
        np.set_printoptions(suppress=True)
        for i in range(self.m_size):
            print "===================================================================="
            print "For Input: ", self.feature[i]
            for iLayer in range(self.nl - 1):
                if iLayer is 0: # input layer
                    self.h_x[iLayer] = np.dot(self.feature[i], self.w[iLayer])
                else:
                    self.h_x[iLayer] = np.dot(self.activation[iLayer-1], self.w[iLayer])
                self.activation[iLayer] = self.sigmod(self.h_x[iLayer] + self.bias[iLayer])
                print "a(", iLayer, "):", self.activation[iLayer]

