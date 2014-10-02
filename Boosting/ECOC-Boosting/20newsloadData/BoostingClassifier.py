__author__ = 'kesavsundar'
import numpy as np
import math

import BoostingStump as Dt


class BoostingClassifier:
    def __init__(self, train_data, test_data, one_labels, iter_count):
        self.train_data = train_data
        self.n_features = len(self.train_data[0]) - 1
        self.len_data = len(self.train_data)
        self.test_data = test_data
        self.one_labels = one_labels
        self.dp_weights = None
        self.decision_stumps = list()
        self.belief = list()
        self.weak_learners = list()
        self.strFile_name = 'output' + str(iter_count)
        self.f = open(self.strFile_name, 'w')

    def generate_weak_learners(self):

        print self.train_data.shape, self.n_features
        for col in range(0, self.n_features):
            stump = Dt.DecisionTreeClassifier(self.train_data[:, col], self.train_data[:, self.n_features], col,
                                              self.dp_weights, self.one_labels)
            list_of_node = stump.learn()
            self.decision_stumps.append(list_of_node)
        return

    def get_best_learner(self):
        best_stump = None
        best_error = 2.0
        for i in range(0, len(self.decision_stumps)):
            col_stump = self.decision_stumps[i]
            for j in range(0, len(col_stump)):
                error = col_stump[j].get_error(self.dp_weights)
                if self.is_new_error_best(best_error, error):
                    best_error = error
                    best_stump = col_stump[j]
        str_to_write = "Col: " + str(best_stump.col) + "    | val: " + \
                       str(best_stump.value) + "    | error: " + str(best_error)
        self.f.write(str_to_write)
        self.f.flush()
        return best_stump, best_error

    def is_new_error_best(self, best, current):
        if best > 0.5:
            best = 1 - 0.5
        if current > 0.5:
            current = 1 - .5
        if current < best:
            return True
        else:
            return False
        return

    def calc_belif(self, error):
        belif = 0.5 * math.log1p(float(1 - error) / error)
        return belif

    ## Not sure what to do with zero frequency counts
    def update_weights(self, best_node, best_error):
        trust = self.calc_belif(best_error)
        self.belief.append(trust)
        # print "Belief: ", trust
        correctly_classified_points = set(np.arange(self.len_data)).difference(best_node.misclassified_set)
        for point in best_node.misclassified_set:
            self.dp_weights[point] *= float(np.exp(trust))
        for point in correctly_classified_points:
            self.dp_weights[point] *= float(np.exp(-trust))

        total_weight = sum(self.dp_weights)
        self.dp_weights = self.dp_weights / total_weight
        return

    def get_label(self, param_label):
        if param_label in self.one_labels:
            return 1
        else:
            return 0

    def get_prediction_labels(self):
        prediction = list()
        len_test_data = len(self.test_data)
        for row in range(0, len_test_data):
            f_x_0 = 0.0
            f_x_1 = 0.0
            for i in range(0, len(self.weak_learners)):
                node = self.weak_learners[i]
                if node.predict(self.test_data[row, node.col]) == 0:
                    f_x_0 += self.belief[i]
                else:
                    f_x_1 += self.belief[i]

            if f_x_0 >= f_x_1:
                prediction.append(0)
            else:
                prediction.append(1)
        return prediction
        acc = accuracy.Accuracy(self.test_data[label_col], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate: ", accuracy_rate
        return

    def train(self):
        self.dp_weights = np.array([1.0/self.len_data] * self.len_data)
        print "Generating all the weak learners..."
        self.generate_weak_learners()
        print "over Generating all the weak learners..."

        for i in range(0, 100):
            weak_train_node, best_error = self.get_best_learner()
            self.weak_learners.append(weak_train_node)
            self.update_weights(weak_train_node, best_error)
            # self.calc_accuracy(i+1)
        return

