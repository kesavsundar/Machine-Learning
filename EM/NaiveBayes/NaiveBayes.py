__author__ = 'kesav'
import numpy as np
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import normalize
import accuracy
import binarizer
import BernoulliFeatureModel as bm
import GaussianFeatureModel as gm
import HistogramFeatureModel as hm

class NaiveBayes:
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesavsundar/machinelearning/ml/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.normalized_data = self.xy_data
        self.feature_models = list()
        self.class_data = list()
        self.class_count = list()
        self.train_data = None
        self.test_data = None
        self.m = 0
        self.n = 0
        self.accuracy_rates_train = list()
        self.accuracy_rates_test = list()
        self.si = 0

    def init_bernoulli_models(self, label):
        for i in range(0, self.n):
            bernoulli = bm.BernoulliFeatureModel(self.train_data[:, i], label)
            bernoulli.init_model()
            self.feature_models.append(bernoulli)
        return

    def init_gaussian_models(self, label):
        for i in range(0, self.n):
            gaussian = gm.GaussianFeatureModel(self.train_data[:, i], label)
            gaussian.init_model()
            self.feature_models.append(gaussian)
        return

    def init_histogram_models(self, label):
        for i in range(0, self.n):
            histogram = hm.HistogramFeatureModel(self.train_data[:, i], label)
            histogram.init_model()
            self.feature_models.append(histogram)
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
        print "Train: ", sum(self.accuracy_rates_train) / len(self.accuracy_rates_train)
        print "Test: ", sum(self.accuracy_rates_test) / len(self.accuracy_rates_test)
        return

    def do_nomalization(self):
        norm = normalize.Normalize(self.xy_data)
        self.normalized_data = norm.min_max_normalize_data()

    def clear_data(self):
        self.feature_models = list()
        self.class_data = list()
        self.class_count = list()
        self.train_data = None
        self.test_data = None
        self.m = 0
        self.n = 0
        return

    def train(self, train, test, fold_count):
        print "===================================================================="
        print "Training for fold: ", fold_count
        # Clear all the data
        self.clear_data()
        # Init variables again for this fold

        self.train_data = train
        self.test_data = test
        self.m = len(self.train_data)
        self.n = len(self.train_data[0, :-1])
        for i in range(0, 2):
            self.class_data.append(i)
            self.class_count.append(0)

        # Init bernoulli Model
        # self.init_bernoulli_models(self.train_data[:, -1:])
        # Init gaussian Model
        # self.init_gaussian_models(self.train_data[:, -1:])
        #Init Histogram model
        self.init_histogram_models(self.train_data[:, -1:])

        # Calculate Prior
        self.calculate_class_count()
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
            if self.class_data[label_iterator] == label:
                return label_iterator
        return None

    def calculate_class_count(self):
        for i in range(0, self.m):
            label_class = self.matches_class(self.train_data[i, -1:])
            self.class_count[label_class] += 1
        return

    def calculate_si(self):
        self.si = float(self.class_count[1]) / self.m
        return

    def predict(self, data):
        # Calculate Prior
        prior = list()
        prior.append(1 - self.si)
        prior.append(self.si)
        #Calculate Probability of features given class from model
        prob_given_class = self.conditional_probability(data)
        #Calculate probability of class given features
        prob_given_features = list()
        for i in range(0, len(self.class_data)):
            prob_given_features.append(prob_given_class[i] * prior[i])
        prediction = prob_given_features[1] - prob_given_features[0]

        # if the value is positive predict as 1, else 0
        binary = binarizer.Binarize()
        prediction = binary.continuous_to_binary(prediction, 0.0)

        acc = accuracy.Accuracy(data[:, -1:], prediction)
        tpr, fpr, accuracy_rate = acc.compute_accuracy()
        print "Accuracy Rate: ", accuracy_rate
        # acc.roc_curve()
        return accuracy_rate

    def conditional_probability(self, data):
        conditional_probability = list()
        m_test = len(data)
        p_x_given_0 = None
        p_x_given_1 = None
        # For each rows in test set
        for j in range(0, self.n):
            cond_prob = self.feature_models[j].p_of_x_given_y(data[:, j])
            if p_x_given_0 is None:
                p_x_given_0 = cond_prob[0]
            else:
                p_x_given_0 *= cond_prob[0]

            if p_x_given_1 is None:
                p_x_given_1 = cond_prob[1]
            else:
                p_x_given_1 *= cond_prob[1]

        conditional_probability.append(p_x_given_0.reshape(m_test, 1))
        conditional_probability.append(p_x_given_1.reshape(m_test, 1))
        return conditional_probability