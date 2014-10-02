__author__ = 'kesav'
import numpy as np


class BernoulliFeatureModel:
    def __init__(self, feature, label):
        self.m = len(feature)
        self.feature = np.matrix(feature).reshape(self.m, 1)
        self.label = label
        self.sets = list()
        self.n_sets = len(self.sets)
        self.mean = np.mean(feature, axis=0)
        self.p_of_x_below_mean = list()
        self.p_of_x_above_mean = list()

    def init_model(self):
        self.divide_set()
        epsilon = 100
        for i in range(0, self.n_sets):
            count_below_mean = 0
            count_above_mean = 0
            feature_set = self.sets[i]
            m_feature = len(feature_set)
            p_below_mean = 0
            p_above_mean = 0
            for j in range(0, m_feature):
                if feature_set[j] <= self.mean:
                    count_below_mean += 1
                else:
                    count_above_mean += 1
            p_below_mean += float(count_below_mean) / (m_feature + 2*epsilon)
            p_above_mean += float(count_above_mean) / (m_feature + 2*epsilon)
            self.p_of_x_below_mean.append(p_below_mean)
            self.p_of_x_above_mean.append(p_above_mean)
        return

    def p_of_x_given_y(self, x):
        conditional_prob = list()
        for i in range(0, self.n_sets):
            probability = list()
            for j in range(0, len(x)):
                if x[j] <= self.mean:
                    probability.append(self.p_of_x_below_mean[i])
                else:
                    probability.append(self.p_of_x_above_mean[i])
            conditional_prob.append(np.array(probability))
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