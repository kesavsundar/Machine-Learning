__author__ = 'kesavsundar'

__author__ = 'kesavsundar'
import numpy as np
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import normalize
import accuracy

class ParzenWindowKernalDensity():
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.normalized_data = None
        self.train_data = list()
        self.test_data = None
        self.accuracy_rates_test = list()
        self.window_size = .07

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

    def divide_set(self, train):
        set1 = list()
        set2 = list()
        for i in range(0, len(train)):
            if train[i, -1:] == 0:
                set1.append(train[i, :])
            else:
                set2.append(train[i, :])

        # Allocate the given
        # print len(set1), len(set2)
        set1 = np.matrix(set1)
        set2 = np.matrix(set2)
        self.train_data.append(set1)
        self.train_data.append(set2)
        return

    def train(self, train, test, fold_count):
        self.test_data = test
        prediction = list()
        self.divide_set(train)
        total_length = len(train)

        p_0 = float(len(self.train_data[0])) / total_length
        p_1 = float(len(self.train_data[1])) / total_length
        for row in self.test_data:
            prob_for_0_given_x = p_0 * self.get_window_majority(row, self.train_data[0])
            prob_for_1_given_x = p_1 * self.get_window_majority(row, self.train_data[1])

            if prob_for_0_given_x > prob_for_1_given_x:
                prediction.append(0)
            else:
                prediction.append(1)
        acc = accuracy.Accuracy(self.test_data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate for fold ", fold_count,  " is: ", accuracy_rate
        self.accuracy_rates_test.append(accuracy_rate)
        return

    def get_window_majority(self, test_data_point, train_datas):
        k = 0
        test_data_point = np.matrix(test_data_point).reshape(1, len(test_data_point))
        for row in train_datas:
            train_pt = np.matrix(row[:, :-1])
            test_pt = np.matrix(test_data_point[:, :-1])
            dist = np.linalg.norm(train_pt - test_pt)
            dist = dist ** 2
            t1 = (((2 * np.pi) * (self.window_size ** 2)) ** (57 / 2))
            t2 = np.exp(-float((1 / (2 * (self.window_size ** 2)))) * dist)
            k += float(t2) * t1
        p = (k / len(train_datas))
        return p

if __name__ == '__main__':
    parzen_window = ParzenWindowKernalDensity()
    parzen_window.startup()
    print "Average Accuracy is ", sum(parzen_window.accuracy_rates_test) / 10