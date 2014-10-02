__author__ = 'kesavsundar'

import numpy as np
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import normalize
import accuracy

class KNearestNeighbour():
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.normalized_data = None
        self.train_data = None
        self.test_data = None
        self.accuracy_rates_test = list()
        self.window_size = np.zeros(len(self.xy_data[0, :-1])).reshape(1, len(self.xy_data[0, :-1]))
        self.window_size += .001
        return


    def startup(self):
        print "Setting Up"
        print "Normalizing data..."
        self.do_nomalization()
        print "Starting tenfold validation ..."
        self.do_tenfold_validation()
        return

    def do_tenfold_validation(self):
        tf = tenfold.Tenfold(self.xy_data, self)
        tf.inbuilt_tenfold_train()
        print "Average Accuracy rates are"
        print "Test: ", sum(self.accuracy_rates_test) / len(self.accuracy_rates_test)
        return

    def do_nomalization(self):
        norm = normalize.Normalize(self.xy_data)
        self.normalized_data = norm.min_max_normalize_data()

    def train(self, train, test, fold_count):
        self.train_data = train
        self.test_data = test
        prediction = list()
        for row in self.test_data:
            predicted_label = self.get_k_nearest_neighbours(row)
            prediction.append(predicted_label)
        acc = accuracy.Accuracy(self.test_data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate for fold ", fold_count, " is: ", accuracy_rate
        self.accuracy_rates_test.append(accuracy_rate)
        return

    def get_k_nearest_neighbours(self, test_data_point):
        k = 9
        test_data_point = np.matrix(test_data_point).reshape(1, len(test_data_point))
        dist = list()
        nearest_labels = dict()
        nearest_labels[0] = 0
        nearest_labels[1] = 1

        for row in self.train_data:
            row = np.matrix(row).reshape(1, len(row))
            ecludian_distnce = abs(np.linalg.norm(row[:, :-1] - test_data_point[:, :-1]))
            dist.append(ecludian_distnce)
        indices = np.array(dist).argsort()

        for i in range(0, k):
            label = self.train_data[indices[i], -1:]
            nearest_labels[label[0]] += 1

        if nearest_labels[0] > nearest_labels[1]:
            return 0
        else:
            return 1

if __name__ == '__main__':
    parzen_window = KNearestNeighbour()
    parzen_window.startup()
    print "Average Accuracy is ", sum(parzen_window.accuracy_rates_test) / 10