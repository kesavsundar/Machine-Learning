__author__ = 'kesavsundar'
import numpy as np
import DecisionStump as Ds
import math
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import accuracy
import binarizer


class Boosting:
    def __init__(self):
        self.xy_data = None
        self.test_data = None
        self.config_data = None
        self.len_data = None
        self.dp_weights = None
        self.belief = list()
        self.weak_learners = list()
        self.predictions = list()

    def init_boosting_model(self):
        with open("/home/kesavsundar/machinelearning/data/crx/crx.config", 'rb') as f:
            self.config_data = f.readlines()
        list_config = list()
        for row in range(1, len(self.config_data)):
            data = self.config_data[row].split("\t")
            list_config.append(int(data[0]))
        self.config_data = list_config
        dt = list()
        for i in range(0, len(self.config_data)):
            col_name = 'c' + str(i)
            if self.config_data[i] == -1000:
                dt.append((col_name, np.float64, (1, )))
            else:
                dt.append((col_name, np.str, 1))
        self.xy_data = np.loadtxt(open("/home/kesavsundar/machinelearning/data/crx/crx.data", 'rb'),
                                  delimiter="\t", dtype=np.str)
        return

    def weak_train(self):
        best_node = None
        n_features = len(self.xy_data[0, :]) - 1
        label_col = len(self.xy_data[0, :])-1
        for col in range(0, n_features):
            stump = Ds.DecisionTree(self.xy_data[:, col], self.xy_data[:, label_col],
                                    self.dp_weights, self.config_data[col], col)
            node = stump.weak_learner_train()
            # print "|", node.col, " |", node.value, "|", node.error
            if best_node is None:
                best_node = node
            elif node.error <= best_node.error:
                best_node = node
        print ">>>>>>>>|", best_node.col, " |", best_node.value, "|", best_node.error
        return best_node

    def calc_belif(self, node):
        belif = 0.5 * math.log1p(float(1 - node.error) / node.error)
        return belif

    def predict_set(self, value, col):
        prediction = list()
        for i in range(0, len(self.xy_data)):
            if self.config_data[col] == -1000:
                if self.xy_data[i, col] == '?' and value[0, 0] == '?':
                    prediction.append('+')
                elif value[0, 0] == '?' and self.xy_data[i, col] != '?':
                    prediction.append('-')
                elif value[0, 0] != '?' and self.xy_data[i, col] == '?':
                    prediction.append('+')
                elif float(self.xy_data[i, col]) <= float(value[0, 0]):
                    prediction.append('+')
                else:
                    prediction.append('-')
            else:
                if self.xy_data[i, col] == value:
                    prediction.append('+')
                else:
                    prediction.append('-')
        return prediction

    def update_weights(self, node):
        trust = self.calc_belif(node)
        self.belief.append(trust)
        # print "Belief: ", trust
        label_col = len(self.xy_data[0, :])-1
        predicted = self.predict_set(node.value, node.col)
        self.predictions.append(predicted)

        weights = list()
        for row in range(0, self.len_data):
            if predicted[row] == self.xy_data[row, label_col]:
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

        for row in self.test_data:
            f_x = 0.0
            for i in range(0, t):
                node = self.weak_learners[i]
                if self.config_data[node.col] == -1000:
                    if row[node.col] == '?':
                        f_x + self.belief[i]
                    elif float(row[node.col]) <= float(node.value[0, 0]):
                        f_x += self.belief[i]
                    else:
                        f_x -= self.belief[i]
                else:
                    if row[node.col] == node.value:
                        f_x += self.belief[i]
                    else:
                        f_x -= self.belief[i]
            if f_x < 0:
                prediction.append('+')
            else:
                prediction.append('-')

        acc = accuracy.Accuracy(self.test_data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate: ", accuracy_rate
        # print "AUC:",

    def train(self, train, test, fold_count):
        self.xy_data = train
        self.test_data = test
        self.len_data = len(self.xy_data)
        self.dp_weights = np.array([1.0/self.len_data] * self.len_data)

        for i in range(0, 100):
            weak_train_node = self.weak_train()
            self.weak_learners.append(weak_train_node)
            updated_weights = self.update_weights(weak_train_node)
            self.dp_weights = updated_weights
            self.calc_accuracy(i)
        print "I'm done!"


if __name__ == '__main__':
    ada_boost = Boosting()
    ada_boost.init_boosting_model()
    tf = tenfold.Tenfold(ada_boost.xy_data, ada_boost)
    tf.inbuilt_tenfold_train()



