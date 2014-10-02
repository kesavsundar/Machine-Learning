__author__ = 'kesavsundar'
import numpy as np

from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer

import BoostingClassifier as BC


class ECOCBoost():
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.k_weak_learners = None
        self.n_features = None
        self.len_data = None
        self.len_test_data = None
        self.classifier_labels = None
        self.classifier_error_codes = None
        self.label_index = None
        self.trained_classifiers = list()
        self.classifier_outputs = list()

    def get_codes(self):
        X, Y = make_multilabel_classification(n_samples=15, n_labels=8, n_classes=8, random_state=0)
        self.classifier_labels = Y
        self.classifier_error_codes = LabelBinarizer().fit_transform(Y)
        print self.classifier_labels
        print self.classifier_error_codes

        f = open('ecoc_classifiers', 'w')

        for row in self.classifier_labels:
            str_op = '['
            for label in row:
                str_op += str(label) + ','
            str_op += ']'
            f.write(str_op)
        f.write('\n')

        for row in self.classifier_error_codes:
            str_op = '['
            for label in row:
                str_op += str(label) + ','
            str_op += ']'
            f.write(str_op)
        f.flush()
        return

    def load_train_data(self):
        with open("feature_names.txt", 'rb') as features_file:
            features_data = [line.strip() for line in features_file]
        self.n_features = len(features_data)
        self.label_index = self.n_features + 1
        features_file = None
        features_data = None

        with open("train.txt", 'rb') as dat_file:
            temp_data = [line.strip() for line in dat_file]
        self.len_data = len(temp_data)
        self.train_data = np.zeros(self.len_data*self.label_index).reshape(self.len_data, self.label_index)
        dat_file = None

        for i in range(0, self.len_data):
            sparse_data = temp_data[i].split(' ')
            self.train_data[i, self.label_index-1] += int(sparse_data[0])
            for j in range(1, len(sparse_data)-3):
                feature_freq = sparse_data[j].split(':')
                self.train_data[i, int(feature_freq[0])] += int(feature_freq[1])
        temp_data = None
        print "Train data loaded!"
        return

    def load_test_data(self):
        with open("test.txt", 'rb') as test_dat_file:
            temp_test_data = [line.strip() for line in test_dat_file]
        self.len_test_data = len(temp_test_data)
        self.test_data = np.zeros(self.len_test_data*self.label_index).reshape(self.len_test_data, self.label_index)
        test_dat_file = None

        for i in range(0, self.len_test_data):
            sparse_data = temp_test_data[i].split(' ')
            self.test_data[i, self.label_index-1] += int(sparse_data[0])
            for j in range(1, len(sparse_data)-3):
                feature_freq = sparse_data[j].split(':')
                self.test_data[i, int(feature_freq[0])] += int(feature_freq[1])
        temp_test_data = None

        # x_train, x_test, y_train, y_test = \
        #         cross_validation.train_test_split(self.train_data[:, :-1], self.train_data[:, -1:], test_size = 0.1, random_state=i)
        # self.train_data = np.concatenate((x_train, y_train), axis=1)
        # self.test_data = np.concatenate((x_test, y_test), axis=1)
        print "Test data loaded!"
        return

    def start_ecoc(self):
        i = 0
        for label in self.classifier_labels:
            print "==========================================================================="
            print "Training for classifier ", label
            classifier = BC.BoostingClassifier(self.train_data, self.test_data, label, i)
            classifier.train()
            # self.trained_classifiers.append(classifier)
            op = classifier.get_prediction_labels()
            self.classifier_outputs.append(op)
            str_file = 'classified' + str(i)
            f = open(str_file, 'w')
            str_op = '['
            for r in op:
                str_op += str(r)
            f.write(str_op)
            f.flush()
            op = None
            classifier = None
            i += 1
        self.get_predictions_based_on_spamming_distance()
        return

    def get_predictions_based_on_spamming_distance(self):
        # first re frame the data
        data_points_op = list()
        for j in range(0, len(self.test_data)):
            each_classifier_op = list()
            for i in range(0, len(self.classifier_outputs)):
                each_classifier_op.append(self.classifier_outputs[i][j])
            data_points_op.append(each_classifier_op)

        # Now Iterate through the data point
        prediction = list()
        for i in range(0, len(data_points_op)):
            min_dist = 27
            decision = 30
            for j in range(0, len(self.classifier_error_codes[0])):
                hamming_dist = self.get_spanning_distance(data_points_op[i], self.classifier_error_codes[:, j])
                if hamming_dist < min_dist:
                    min_dist = hamming_dist
                    decision = j
            prediction.append(decision)
        correct = 0
        for pred, actual in zip(prediction, self.train_data[:, -1:].tolist()):
            if pred == actual:
                correct += 1

        accuracy = correct / len(prediction)
        print accuracy
        f = open('Accuracy.txt', 'w')
        f.write(str(accuracy))
        f.flush()
        print "Accuracy Rate: ", accuracy
        return

    def get_spanning_distance(self, output, code):
        diffs = 0
        for ch1, ch2 in zip(output, code):
                if ch1 != ch2:
                        diffs += 1
        return diffs

if __name__ == '__main__':
    ecoc = ECOCBoost()
    ecoc.get_codes()
    ecoc.load_train_data()
    ecoc.load_test_data()
    ecoc.start_ecoc()