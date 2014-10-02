__author__ = 'kesav'
import numpy as np


class GaussianFeatureModel:
    def __init__(self, feature, label):
        self.m = len(feature)
        self.feature = np.matrix(feature).reshape(self.m, 1)
        self.label = label
        self.sets = list()
        self.n_sets = len(self.sets)
        self.mean = list()
        self.std = list()
        self.first_term = list()

    def init_model(self):
        self.divide_set()
        for i in range(0, self.n_sets):
            if len(self.sets[i]) is not 0:
                self.mean.append(np.mean(self.sets[i], axis=0)[0, 0])
                self.std.append(np.std(self.sets[i], axis=0)[0, 0])
            else:
                self.mean.append(0.001)
                self.std.append(0.001)
            if self.std[i] == 0:
                self.std[i] = .00000000000000001
            self.first_term.append(1 / (pow((2 * np.pi), (1 / 2)) * pow(self.std[i], (1/2))))
        return

    def p_of_x_given_y(self, x):
        x = x.reshape(len(x), 1)
        conditional_prob = list()
        for i in range(0, self.n_sets):
            z = (-1 / 2) * ((x-self.mean[i]) / self.std[i])
            z = np.exp(z)
            prob = self.first_term[i] * z
            conditional_prob.append(np.array(prob))
        return conditional_prob

    def divide_set(self):
        set1 = list()
        set2 = list()
        for i in range(0, self.m):
            if self.label[i] == 0:
                set1.append(self.feature[i, 0])
            else:
                set2.append(self.feature[i, 0])
        set1 = np.matrix(set1).reshape(len(set1), 1)
        set2 = np.matrix(set2).reshape(len(set2), 1)
        self.sets.append(set1)
        self.sets.append(set2)
        self.n_sets = len(self.sets)
        return