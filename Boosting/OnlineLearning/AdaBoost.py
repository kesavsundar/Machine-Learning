from numpy.core.multiarray import dtype

__author__ = 'kesavsundar'
import numpy as np
import DecisionTree as Dt
import math
import sys
import random
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import accuracy


class OnlineAdaBoost():
    def __init__(self):
        self.xy_data = None
        self.train_data = None
        self.test_data = None
        self.config_data = None
        self.len_data = None
        self.dp_weights = None
        self.belief = list()
        self.weak_learners = list()
        self.predictions = list()
        self.n = 0
        self.labels = list()

    def init_boosting(self):
        with open("/home/kesavsundar/machinelearning/data/crx/crx.config", 'rb') as conf:
            self.config_data = [line.strip() for line in conf]

        list_config = list()
        self.n = len(self.config_data) - 2
        self.len_data = int(self.config_data[0].split("\t")[0])
        for row in range(1, len(self.config_data)):
            data = self.config_data[row].split("\t")
            list_config.append(int(data[0]))
            if row == self.n:
                self.labels.append(data[1])
                self.labels.append(data[2])
        self.config_data = list_config

        with open("/home/kesavsundar/machinelearning/data/crx/crx.data", 'rb') as dat:
            self.xy_data = [line.strip() for line in dat]

        data_list = list()
        for i in range(0, self.n):
            data_list.append(list())

        for i in range(0, len(self.xy_data)):
            temp_row = self.xy_data[i].split("\t")
            for j in range(0, len(temp_row)):
                if self.config_data[j] == -1000:
                    if temp_row[j] == '?':
                        x = -1
                    else:
                        x = float(temp_row[j])
                    data_list[j].append(x)
                else:
                    data_list[j].append(temp_row[j])
        self.xy_data = data_list

        return

    def weak_train(self):
        best_node = None
        label_col = self.n - 1
        for col in range(0, label_col):
            if self.config_data[col] == -1000:
                isfloat = True
            else:
                isfloat = False
            stump = Dt.DecisionTreeClassifier(self.train_data[col], self.train_data[label_col], self.dp_weights,
                                              isfloat, self.labels, col)
            node = stump.learn()
            # print "|", node.col, " |", node.value, "|", node.error
            if best_node is None:
                best_node = node
            elif node.best_error <= best_node.best_error:
                best_node = node
        # print ">>>>>>>>|", best_node.col, " |", best_node.best_value, "|", best_node.best_error
        return best_node

    def calc_belif(self, node):
        belif = 0.5 * math.log1p(float(1 - node.best_error) / node.best_error)
        return belif

    def update_weights(self, best_node):
        trust = self.calc_belif(best_node)
        self.belief.append(trust)
        # print "Belief: ", trust
        label_col = self.n - 1

        weights = list()
        for row in range(0, self.len_data):
            if best_node.predict(self.train_data[best_node.col][row]) == self.train_data[label_col][row]:
                weights.append(float(np.exp(-trust)) * self.dp_weights[row])
            else:
                weights.append(float(np.exp(trust)) * self.dp_weights[row])
        total_weight = sum(weights)
        weights = np.array(weights)
        weights = weights / total_weight
        # print "Updated Weights", weights
        return weights

    def calc_accuracy(self, t):
        prediction = list()
        label_col = self.n - 1
        # self.test_data = self.xy_data
        len_test_data = len(self.test_data[0])
        for row in range(0, len_test_data):
            f_x_0 = 0.0
            f_x_1 = 0.0
            for i in range(0, t):
                node = self.weak_learners[i]
                if node.predict(self.test_data[node.col][row]) == self.labels[0]:
                    f_x_0 += self.belief[i]
                else:
                    f_x_1 += self.belief[i]

            if f_x_0 >= f_x_1:
                prediction.append(self.labels[0])
            else:
                prediction.append(self.labels[1])
        acc = accuracy.Accuracy(self.test_data[label_col], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        return accuracy_rate

    def train(self,  train, test):
        f = open('random_output_file', 'w')
        self.xy_data = train
        self.test_data = test
        # Select 5% of training set
        # do the below step
        # first again randomly choose the training set
        # check how the accuracy increases
        # predict the rest of the points and take 5% of min confidence
        total_percent = 0
        previously_selected_indices = set()
        isRandom = True
        self.len_data = len(self.xy_data[0])
        c = int(.05 * self.len_data)
        min_percent = int(.05 * self.len_data)
        all_indices_set = set(np.arange(self.len_data))
        selected_indices = self.get_random_x_percent(all_indices_set.
                                                                 difference(previously_selected_indices), c)
        while total_percent < 0.5:
            self.generate_train_data_randomly(selected_indices)
            self.len_data = len(self.train_data[0])
            self.dp_weights = np.array([1.0/self.len_data] * self.len_data)
            self.belief = list()
            self.weak_learners = list()
            self.predictions = list()

            for i in range(0, 350):
                weak_train_node = self.weak_train()
                # if i > 1 and weak_train_node.best_error > self.weak_learners[i-1].best_error:
                #     exit(0)

                self.weak_learners.append(weak_train_node)
                updated_weights = self.update_weights(weak_train_node)
                self.dp_weights = updated_weights
            acc_rate = self.calc_accuracy(i)

            print "Accuracy for percent ", total_percent, " is", acc_rate
            str_op = str(total_percent) + "::" + str(acc_rate) + "\n"
            f.write(str_op)
            total_percent += .05
            previously_selected_indices = set(selected_indices).union(previously_selected_indices)
            if isRandom:
                selected_indices = self.get_random_x_percent(all_indices_set.
                                                                 difference(previously_selected_indices), c)
            else:
                selected_indices = self.get_min_confidence_data(all_indices_set.
                                                                difference(previously_selected_indices), min_percent)
        return

    def get_random_x_percent(self, indices_set, percent_to_select):
        indices = random.sample(indices_set, percent_to_select)
        return indices

    def get_min_confidence_data(self, indices_set, percent_to_select):
        indices_list = list(indices_set)
        f_x_list = list()
        min_confidence_indices = list()
        for index in indices_list:
            f_x = 0.0
            for i in range(0, len(self.weak_learners)):
                node = self.weak_learners[i]
                if node.predict(self.xy_data[node.col][index]) == self.labels[0]:
                    f_x += self.belief[i]
                else:
                    f_x -= self.belief[i]
            f_x_list.append(abs(f_x))
        f_x_list = np.array(f_x_list)
        indices_sorted = f_x_list.argsort()

        for i in range(0, percent_to_select):
            min_confidence_indices.append(indices_sorted[i])
        return min_confidence_indices

    def generate_train_data_randomly(self, indices):
        train = list()
        for i in range(0, self.n):
            train.append(list())
            for j in range(0, len(indices)):
                train[i].append(self.xy_data[i][indices[j]])
        self.train_data = train

    def do_tenfold(self):
        total_data = self.xy_data
        total_len = self.len_data
        fold = 0
        total_fold = 10
        accuracy_rate = list()
        train = list()
        test = list()
        for i in range(0, self.n):
            train.append(list())
            test.append(list())

        while fold < 10:
            for i in range(0, total_len):
                if i % total_fold == fold:
                    for j in range(0, self.n):
                        test[j].append(total_data[j][i])
                else:
                    for j in range(0, self.n):
                        train[j].append(total_data[j][i])
            accuracy_rate.append(self.train(train, test))
            exit(0)
            fold += 1
        print "Average accuracy is ", sum(accuracy_rate) / 10
        return

if __name__ == '__main__':
    ada_boost = OnlineAdaBoost()
    ada_boost.init_boosting()
    ada_boost.do_tenfold()
