__author__ = 'kesavsundar'
import numpy as np


class HistogramFeatureModel:
    def __init__(self, feature, label):
        self.m = len(feature)
        self.feature = np.matrix(feature).reshape(self.m, 1)
        self.label = label
        self.mean_overall = np.mean(self.feature, axis=0)
        self.min_overall = np.min(self.feature)
        self.max_overall = np.max(self.feature)
        self.set1_overall, self.set2_overall = \
                self.partition_sets_with_values(self.feature, self.min_overall, self.mean_overall, self.max_overall)
        self.low_mean_overall = np.mean(self.set1_overall, axis=0)
        self.high_mean_overall = np.mean(self.set2_overall, axis=0)

        self.sets = list()
        self.n_sets = len(self.sets)
        self.n_buckets = 4
        self.models_per_class = list()

    def init_model(self):
        # Model has
        # 1. Prob based on bucket
        # 2. min, low mean, mean, high mean, max
        epsilon = 50
        self.divide_set()

        for i in range(0, self.n_sets):
            set_models = list()
            prob_of_bucket = list()
            bucket_list = list()

            # min_feature = min(self.sets[i])[0][0]
            # max_feature = max(self.sets[i])[0][0]
            # mean = np.mean(self.sets[i], axis=0)
            less_than_mean, greater_than_mean = \
                self.partition_sets_with_values(self.sets[i], self.min_overall, self.mean_overall, self.max_overall)
            set1, set2 = self.partition_sets_with_values(less_than_mean, self.min_overall,
                                                         self.low_mean_overall, self.mean_overall)
            set3, set4 = self.partition_sets_with_values(greater_than_mean, self.mean_overall,
                                                         self.high_mean_overall, self.max_overall)
            m_feature = len(self.sets[i])

            bucket_list.append(set1)
            bucket_list.append(set2)
            bucket_list.append(set3)
            bucket_list.append(set4)

            for k in range(0, self.n_buckets):
                prob = (float(len(bucket_list[k]) + 1)) / (m_feature + 4)
                prob_of_bucket.append(prob)
            set_models.append(prob_of_bucket)
            # set_models.append(min_feature)
            # set_models.append(low_mean)
            # set_models.append(mean)
            # set_models.append(high_mean)
            # set_models.append(max_feature)
            self.models_per_class.append(set_models)
        return

    def p_of_x_given_y(self, x):
        conditional_prob = list()
        for i in range(0, self.n_sets):
            probability = list()
            for j in range(0, len(x)):
                #if x[j] < self.models_per_class[i][1]:
                if x[j] < self.min_overall:
                    # print "Below range: ", x[j]
                    probability.append(.0000000001)
                #elif self.models_per_class[i][1] <= x[j] <= self.models_per_class[i][2]:
                elif self.min_overall <= x[j] <= self.low_mean_overall:
                    probability.append(self.models_per_class[i][0][0])
                #elif self.models_per_class[i][2] < x[j] <= self.models_per_class[i][3]:
                elif self.low_mean_overall < x[j] <= self.mean_overall:
                    probability.append(self.models_per_class[i][0][1])
                #elif self.models_per_class[i][3] < x[j] <= self.models_per_class[i][4]:
                elif self.mean_overall < x[j] <= self.high_mean_overall:
                    probability.append(self.models_per_class[i][0][2])
                #elif self.models_per_class[i][4] < x[j] <= self.models_per_class[i][5]:
                elif self.high_mean_overall < x[j] <= self.max_overall:
                    probability.append(self.models_per_class[i][0][3])
                else:
                    # print "Out of range: ", x[j]
                    probability.append(.0000001)
            conditional_prob.append(np.array(probability))
        return conditional_prob

    def partition_sets_with_values(self, param_set, min_val, middle_val, max_val):
        less_set = list()
        more_set = list()
        for i in range(0, len(param_set)):
            if min_val <= param_set[i] <= middle_val:
                less_set.append(param_set[i])
            elif middle_val < param_set[i] <= max_val:
                more_set.append(param_set[i])
        return less_set, more_set

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