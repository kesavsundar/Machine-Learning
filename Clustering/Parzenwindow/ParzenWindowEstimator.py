__author__ = 'kesavsundar'
import numpy as np
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import normalize
import accuracy

class ParzenWindowMajorityLabel():
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.normalized_data = None
        self.train_data = None
        self.test_data = None
        self.accuracy_rates_test = list()
        self.window_size = .25
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
            predicted_label = self.get_window_majority(row)
            prediction.append(predicted_label)
        acc = accuracy.Accuracy(self.test_data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate for fold ", fold_count,  " is: ", accuracy_rate
        self.accuracy_rates_test.append(accuracy_rate)
        return

    def get_window_majority(self, test_data_point):

        inside_labels = dict()
        inside_labels[0] = 0
        inside_labels[1] = 1
        test_data_point = np.matrix(test_data_point).reshape(1, len(test_data_point))
        for row in self.train_data:
            is_inside = 1
            row = np.matrix(row).reshape(1, len(row))
            train_pt = row[:, :-1].tolist()[0]
            test_pt = test_data_point[:, :-1].tolist()[0]
            for axis, center_point in zip(train_pt, test_pt):
                if np.abs(axis-center_point) > (self.window_size/2):
                    is_inside = 0
            if is_inside == 1:
                label = row[:, -1:]
                inside_labels[label[0, 0]] += 1
        if inside_labels[0] > inside_labels[1]:
            return 0
        else:
            return 1

if __name__ == '__main__':
    parzen_window = ParzenWindowMajorityLabel()
    parzen_window.startup()
    print "Average Accuracy is ", sum(parzen_window.accuracy_rates_test) / 10
