__author__ = 'kesav'
import numpy as np
from copy import deepcopy
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import tenfold
import normalize

class EMAlgorithm:
    def __init__(self):
        self.xy_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                                  delimiter=",", dtype=float)
        self.normalized_data = self.xy_data

        self.mean_per_class = list()
        self.sigma_per_class = list()
        self.cluster_per_class = list()
        self.w_per_class = list()
        self.log_likelyhood_per_class = list()
        self.scores_per_class = list()
        self.scores_normalized_per_class = list()
        self.k = 9
        self.n_features = 57

        self.train_data = list()
        self.test_data = None
        self.m = len(self.xy_data[0, :-1])
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
        self.mean_per_class = list()
        self.sigma_per_class = list()
        self.cluster_per_class = list()
        self.w_per_class = list()
        self.log_likelyhood_per_class = list()
        self.scores_per_class = list()
        self.scores_normalized_per_class = list()
        self.train_data = list()
        self.test_data = None
        self.m = len(self.xy_data[0, :-1])
        return

    def init_arrays(self):
        for n_class in range(0, len(self.train_data)):
            self.mean_per_class.append(list())
            self.sigma_per_class.append(list())
            self.cluster_per_class.append(list())
            self.w_per_class.append(list())
            self.log_likelyhood_per_class.append(list())
            self.scores_per_class.append(list())
            self.scores_normalized_per_class.append(list())
        return

    def init_w_component(self):
        for class_data in range(0, len(self.train_data)):
            w_of_class = list()
            for column in range(0, self.n_features):
                w_of_column = list()
                for component in range(0, self.k):
                    # w_of_column.append(float(len(self.cluster_per_class[class_data][column][component]))
                    #                    / len(self.train_data[class_data]))
                    w_of_column.append(float(1)/self.k)

                t = np.array([float(0.2), float(0.1), float(0.15), float(0.12),
                              float(0.14), float(0.11), float(0.04), float(0.1), float(0.04)])
                w_of_class.append(t)
                #w_of_class.append(w_of_column)
            self.w_per_class[class_data] = w_of_class
        return

    def randomly_distribute_datapoints_among_components(self):
        for class_data in range(0, len(self.train_data)):
            data = self.train_data[class_data]
            mean_of_class = list()
            sigma_of_class = list()
            cluster_of_class = list()

            # for column in range(0, self.n_features):
            #     mus = rand.sample(data[:, column].tolist(), self.k)
            #     print "&&&&&&&&&&&& mus", mus
            #     mean_of_class.append(mus)
            #     clusters = [[] for dummy in range(0, self.k)]
            #     # For each row in a feature
            #     for row in data[:, column]:
            #         r = np.argmin([np.linalg.norm(row - m) for m in mus])
            #         clusters[r].append(row[0, 0])
            #     # For each hidden component
            #     sigmas = list()
            #     for component in range(0, self.k):
            #         cluster_matrix = np.matrix(clusters[component])
            #         sigmas.append(np.std(cluster_matrix))
            #     sigma_of_class.append(sigmas)
            #     cluster_of_class.append(clusters)
            # l = [0.6396106216741642, 0.43806547889903125, 0.35617984102266065, -0.7240859191231772, -0.18568982238624088, 0.6881091625545144, 0.7017489733577338, -0.31429575283772904, -0.1809288328637324]
            # mean_of_class.append(l)

            for column in range(1, self.n_features):
                rand_means = list()
                rand_sigmas = list()
                for i in range(0, self.k):
                    rand_means.append(np.random.uniform(-1, 1))
                    rand_sigmas.append(1.0)
                mean_of_class.append(rand_means)
                sigma_of_class.append(rand_sigmas)
            self.mean_per_class[class_data] = mean_of_class
            self.sigma_per_class[class_data] = sigma_of_class
            #self.cluster_per_class[class_data] = cluster_of_class
        return

    def divide_set(self, train):
        set1 = list()
        set2 = list()
        for i in range(0, len(train)):
            if train[i, -1:] == 0:
                set1.append(train[i, :-1])
            else:
                set2.append(train[i, :-1])

        # Allocate the given
        print len(set1), len(set2)
        set1 = np.matrix(set1)
        set2 = np.matrix(set2)
        print set1.shape, set2.shape
        self.train_data.append(set1)
        self.train_data.append(set2)
        return


    def train(self, train, test, fold_count):
        print "===================================================================="
        print "Training for fold: ", fold_count

        # Clear all the data
        self.clear_data()

        # Init variables again for this fold
        self.test_data = test
        self.divide_set(train)
        self.init_arrays()

        # Create clusters
        print "Distributing data points among clusters..."
        self.randomly_distribute_datapoints_among_components()
        print "Init weight vectors"
        self.init_w_component()

        for n_class in range(0, len(self.train_data)):
            ll_per_class = list()
            score_per_class = list()
            score_normalized_per_class = list()
            for x in range(0, self.n_features):
                ll_per_class.append(0.00)
                score_per_class.append(list())
                score_normalized_per_class.append(list())

            for n_column in range(4, self.n_features):
                converged = False
                print "Processing feature ", n_column
                while not converged:
                    # E Step
                    score, norm_score, ll = self.calculate_ll(n_class, n_column)
                    score_per_class[n_column] = score
                    score_normalized_per_class[n_column] = norm_score

                    print "ll is", ll
                    if abs(ll_per_class[n_column] - ll) <= 0.000001:
                        print "******************************************Converged at ", ll
                        print "*W Vector", self.w_per_class[n_class][n_column]
                        # print "Mean:", self.mean_per_class[n_class][n_column]
                        break

                    ll_per_class[n_column] = ll
                    self.scores_per_class[n_class] = score_per_class
                    self.scores_normalized_per_class[n_class] = score_normalized_per_class
                    self.log_likelyhood_per_class[n_class] = ll_per_class

                    #M Step
                    self.update_parameters(n_class, n_column)
        return

    def pdf(self, n_class, n_feature, n_component):
        feature = self.train_data[n_class][:, n_feature]
        feature = np.matrix(feature).reshape(len(feature), 1)
        mean = self.mean_per_class[n_class][n_feature][n_component]
        covariance = self.sigma_per_class[n_class][n_feature][n_component]
        if covariance == 0:
            covariance = 0.00001
        den = float(pow((2.0 * np.pi), .5) * pow(abs(covariance), 0.5))
        x_minus_mean = feature - mean
        inv_covariance = float(1)/covariance
        z = (-0.5 * (x_minus_mean * inv_covariance))
        exp_term = np.exp(z)
        matrix = float(1)/den * exp_term
        return matrix

    def calculate_ll(self, class_data, column):
        scores_of_component = list()
        scores_normalized_of_component = list()
        ll_shape = len(self.train_data[class_data][:, 0])
        ll = np.zeros(ll_shape).reshape(ll_shape, 1)

        for component in range(0, self.k):
            pdf_matrix = self.pdf(class_data, column, component)

            score = self.w_per_class[class_data][column][component] * pdf_matrix
            ll += score
            scores_of_component.append(score)
            scores_normalized_of_component.append(np.zeros(ll_shape).reshape(ll_shape, 1))
        #score_per_class.append(scores_of_component)
        ll_sum = 0.0
        for row in range(0, ll_shape):
            for comp in range(0, self.k):
                div = ll[row]
                if div != 0:
                    scores_normalized_of_component[comp][row, 0] = float(scores_of_component[comp][row, 0]) / div
                else:
                    ll[row] = .0000001
                    scores_normalized_of_component[comp][row, 0] = 0
            ll_sum += np.log(ll[row])
        ll_of_class = float(ll_sum) / len(self.train_data[class_data])

        #ll_per_class.append(ll_of_class)
        return scores_of_component, scores_normalized_of_component, ll_of_class

    def update_parameters(self, class_data, column):
        n_len = len(self.train_data[class_data])
        nj_list = [0]*self.k
        for component in range(0, self.k):
            nj = float(np.sum(self.scores_normalized_per_class[class_data][column][component], axis=0))
            nj_list[component] = nj
            self.w_per_class[class_data][column][component] = float(nj) / n_len

        temp_mean = list()

        # if column == 3:
        #     print "&&&&&:", self.scores_normalized_per_class[class_data][column]

        for n_comp in range(0, self.k):
            x = 0.0
            for row in range(0, n_len):
                x += (self.scores_normalized_per_class[class_data][column][n_comp][row, 0] *
                                      self.train_data[class_data][row, column])
            if nj_list[n_comp] == 0:
                temp_mean.append(0.0)
            else:
                temp_mean.append(float(x) / nj_list[n_comp])

        # print "***", temp_mean
        self.mean_per_class[class_data][column] = temp_mean

        temp_sd = list()
        for n_comp in range(0, self.k):
            y = 0.0
            for row in range(0, n_len):
                x_minus_mean = self.train_data[class_data][row, column] - \
                               self.mean_per_class[class_data][column][n_comp]
                y += self.scores_normalized_per_class[class_data][column][n_comp][row, 0]\
                                   * (x_minus_mean ** 2)
            if y <= 0.000001 or nj_list[n_comp] == 0:
                temp_sd.append(.0001)
            else:
                temp_sd.append(float(y) / nj_list[n_comp])
        self.sigma_per_class[class_data][column] = temp_sd
        return