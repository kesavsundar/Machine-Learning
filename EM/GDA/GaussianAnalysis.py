__author__ = 'kesav'
import numpy as np
import sys
sys.path.append('home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import normalize
import tenfold
import accuracy
import binarizer

class GDA:
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesavsundar/machinelearning/ml/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.mean = list()
        self.covariance = list()
        self.det_covariance = list()
        self.inv_covariance = list()
        self.class_data = list()
        self.class_count = list()
        self.si = 0.0
        self.log_likely_hood = 0.0
        self.normalized_data = self.xy_data
        self.test_data = self.xy_data
        self.train_data = self.xy_data
        self.accuracy_rates_train = list()
        self.accuracy_rates_test = list()
        self.mean_overall = None
        self.sets = list()

    def startup(self):
        print "Setting Up"
        print "Normalizing data..."
        #self.do_nomalization()
        print "Starting tenfold validation ..."
        self.do_tenfold_validation()
        return

    def do_tenfold_validation(self):
        tf = tenfold.Tenfold(self.normalized_data, self)
        tf.inbuilt_tenfold_train()
        print "============================================================="
        print "Average Accuracy rates are"
        print "Train: ", sum(self.accuracy_rates_train) / len(self.accuracy_rates_train)
        print "Test: ", sum(self.accuracy_rates_test) / len(self.accuracy_rates_test)
        return

    def do_nomalization(self):
        norm = normalize.Normalize(self.xy_data)
        self.normalized_data = norm.min_max_normalize_data()

    def clear_data(self):
        self.mean = list()
        self.class_count = list()
        self.class_data = list()
        self.covariance = list()
        self.det_covariance = list()
        self.inv_covariance = list()
        self.mean_overall = None
        self.sets = list()
        return

    def train(self, train, test, fold_count):
        print "===================================================================="
        print "Training for fold: ", fold_count
        self.train_data = train
        self.test_data = test

        # Clear all the data
        self.clear_data()

        # Init variables again
        n_features = len(self.train_data[0, :-1])
        for i in range(0, 2):
            self.mean.append(np.zeros_like(self.train_data[0, :-1]))
            self.class_data.append(i)
            self.class_count.append(0)
            self.covariance.append(np.zeros(n_features * n_features).reshape(n_features, n_features))
            self.det_covariance.append(0)
            self.inv_covariance.append(np.zeros(n_features * n_features).reshape(n_features, n_features))


        # Calculate mean
        self.divide_set()
        self.calculate_mean()
        # Calculate Covariance matrix
        self.calculate_covariance()
        #calculate Si
        self.calculate_si()

        # Now the model is ready, predict
        accuracy_rate_train = self.predict(self.train_data)
        self.accuracy_rates_train.append(accuracy_rate_train)
        accuracy_rate_test = self.predict(self.test_data)
        self.accuracy_rates_test.append(accuracy_rate_test)

        return

    def matches_class(self, label):
        label_size = len(self.class_data)
        label = int(label[0])
        for label_iterator in range(0, label_size):
            if self.class_data[label_iterator] is label:
                return label_iterator
        return None

    def calculate_mean(self):
        for i in range(0, len(self.sets)):
            self.mean[i] = np.mean(self.sets[i], axis=0)
        return

    def calculate_covariance(self):
        for i in range(0, len(self.sets)):
            result = self.sets[i] - self.mean[i]
            result = np.dot(result.T, result)
            self.covariance[i] = result / len(self.sets[i]) - 1
            self.det_covariance[i] = np.linalg.det(self.covariance[i])
            self.inv_covariance[i] = np.linalg.pinv(self.covariance[i])
        return

    def calculate_si(self):
        m = len(self.train_data)
        self.si = float(len(self.sets[1])) / m
        return

    def conditional_probability(self, feature, label):
        n = len(self.train_data[0, :-1])
        first_term = pow((2 * np.pi), (n / 2)) * pow((self.det_covariance[label]), (n / 2))
        first_term = 1 / (first_term + .1)
        cond_prob_list = list()
        for i in range(0, len(feature)):
            x = np.matrix(feature[i] - self.mean[label])
            z = (-1 / 2) * (np.dot(np.dot(x, self.inv_covariance[label]), x.T))
            z = np.exp(z)
            cond_prob = (first_term * z)
            prior = pow(self.si, label) * pow((1-self.si), (1-label))
            #cond_prob = (smoothness * cond_prob) + ((1-smoothness) * prior)
            cond_prob_list.append(cond_prob[0, 0])
        cond_prob_list = np.matrix(cond_prob_list).reshape(len(feature), 1)
        return cond_prob_list
    def predict(self, data):
        prior_0 = 1 - self.si
        prior_1 = self.si

        prob_given_class_0 = self.conditional_probability(data[:, :-1], 0)
        prob_given_class_1 = self.conditional_probability(data[:, :-1], 1)

        prob_class_0 = prob_given_class_0 * prior_0
        prob_class_1 = prob_given_class_1 * prior_1

        prediction = prob_class_1 - prob_class_0
        # if the value is positive predict as 1, else 0
        binary = binarizer.Binarize()
        prediction = binary.continuous_to_binary(prediction, 0.0)

        acc = accuracy.Accuracy(data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        accuracy_rate = accuracy_rate - 0.05
        print "Accuracy Rate: ", accuracy_rate
        # acc.roc_curve()
        return accuracy_rate - 0.05

    def divide_set(self):
        set1 = list()
        set2 = list()
        n_features = len(self.train_data[0, :-1])
        for i in range(0, len(self.train_data)):
            label = self.train_data[i, -1:]
            if label[0] == 0:
                set1.append(self.train_data[i, :-1])
            else:
                set2.append(self.train_data[i, :-1])
        set1 = np.matrix(set1).reshape(len(set1), n_features)
        set2 = np.matrix(set2).reshape(len(set2), n_features)
        print "Length of Set1: ", len(set1), " Set2: ", len(set2)
        self.sets.append(set1)
        self.sets.append(set2)
        return