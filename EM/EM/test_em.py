__author__ = 'kesavsundar'

import numpy as np
import random as rand
class nGaussianRecover:
    def __init__(self, data_set, test_data):
        self.data = data_set
        self.testdata = test_data
        self.w = list()
        self.mean = list()
        self.sd = list()
        self.n_cluster = list()
        self.score = None
        self.gamma = None
        self.ll = 0.0
        self.k = 9
        self.n_features = 1

    def random_assignment(self):
        # mus = rand.sample(self.data[:, :].tolist(), self.k)
        # a = list()
        # for i in range(0, self.k):
        #     a.append(mus[i][0])
        # self.mean = a
        #
        # # clusters = [np.ones_like(self.data[0, :]) for dummy in range(0, self.k)]
        # self.n_cluster = [0.0 for dummy in range(0, self.k)]
        # # For each row in a feature
        # for row in self.data:
        #     r = np.argmin([np.linalg.norm(row - m) for m in mus])
        #     # clusters[r] = np.concatenate((clusters[r], row), axis=0)
        #     self.n_cluster[r] += 1
        # # For each hidden component
        #
        # for component in range(0, self.k):
        #     # cluster_matrix = np.matrix(clusters[component][1:, :]).\
        #     #     reshape(int(self.n_cluster[component]), self.n_features)
        #     # x_m_mu = cluster_matrix - self.mean[component]
        #     # self.sd.append(np.dot(x_m_mu.T, x_m_mu) / self.n_cluster[component])
        #     self.sd.append(1.0)
        #     self.w.append(float(self.n_cluster[component]) / len(self.data))
        rand_means = list()
        for i in range(0, self.k):
            rand_means.append(rand.uniform(-1, 1))
            self.sd.append(1.0)
        self.mean = rand_means
        self.w = ([float(0.2), float(0.1), float(0.15), float(0.12), float(0.14), float(0.11), float(0.04), float(0.1), float(0.04)])
        return

    def pdf(self, comp):
        covariance = self.sd[comp]

        inv_covariance = 1 / covariance
        den = float(pow((2.0 * np.pi), .5) * pow(covariance, 0.5))
        pdf_list = list()
        for row in range(0, len(self.data)):
            x_minus_mean = self.data[row, 0] - self.mean[comp]
            temp = x_minus_mean / covariance
            z = (-0.5 * temp)
            exp_term = np.exp(z)
            temp3 = float(1)/den * exp_term
            pdf_list.append(temp3)
        pdf_matrix = np.matrix(pdf_list).reshape(len(self.data), 1)
        return pdf_matrix

    def calc_log_likelihood(self):
        n = len(self.data)
        self.score = np.ones(len(self.data)).reshape(len(self.data), 1)
        self.gamma = np.ones(len(self.data)*self.k).reshape(len(self.data), self.k)
        for n_comp in range(0, self.k):
            likelihood = self.w[n_comp] * self.pdf(n_comp)
            for row in range(0, len(self.data)):
                if likelihood[row, 0] == float('inf'):
                    likelihood[row, 0] = .000001
            self.score = np.concatenate((self.score, likelihood), axis=1)
        self.score = self.score[:, 1:]
        sum_likelihood = np.zeros_like(self.score[:, 0])
        for n_comp in range(0, self.k):
            sum_likelihood += self.score[:, n_comp]

        log_likelihood = 0.0
        for row in range(0, n):
            for n_comp in range(0, self.k):
                div = sum_likelihood[row]
                if div == 0:
                    self.gamma[row, n_comp] = 0
                    sum_likelihood[row] = 1
                else:
                    self.gamma[row, n_comp] = self.score[row, n_comp] / div
            log_likelihood += np.log(sum_likelihood[row])
        # print "log likely hood", log_likelihood / n
        return (log_likelihood / n), sum_likelihood

    def update_parameters(self):
        n = len(self.data)
        nj_list = [0] * self.k

        for row in range(0, len(self.gamma)):
            for n_comp in range(0, self.k):
                nj_list[n_comp] += self.gamma[row, n_comp]

        for n_comp in range(0, self.k):
            self.w[n_comp] = float(nj_list[n_comp]) / n

        init_means = [0.0]* self.k
        #[np.matrix([[0], [0]]), np.matrix([[0], [0]])]
        # initCovariances = [np.matrix([[0, 0],[0, 0]]), np.matrix([[0, 0],[0, 0]]),np.matrix([[0, 0],[0, 0]])]

        for row in range(0, n):
            for n_comp in range(0, self.k):
                old_mean = init_means[n_comp]
                init_means[n_comp] = old_mean + (self.gamma[row, n_comp] * self.data[row, 0])

        for n_comp in range(0, self.k):
            # if init_means[n_comp] == 0:
                # print nj_list
                # print "Here"
            if nj_list[n_comp] != 0:
                init_means[n_comp] /= nj_list[n_comp]
            else:
                init_means[n_comp] = 0.0
        self.mean = init_means
        # print "Mean:", self.mean

        init_covariance = list()
        for n_comp in range(0, self.k):
            init_covariance.append(0.0)

        for row in range(0, n):
            t = np.zeros_like(self.sd[0])
            for n_comp in range(0, self.k):
                x_minus_mean = self.data[row] - self.mean[n_comp]
                temp = self.gamma[row, n_comp] * x_minus_mean
                t = temp * x_minus_mean
                t = t[0, 0]
                c = init_covariance[n_comp]
                init_covariance[n_comp] = c + t

        for n_comp in range(0, self.k):
            if nj_list[n_comp] != 0:
                init_covariance[n_comp] /= nj_list[n_comp]
                if init_covariance[n_comp] < 0.0001:
                    init_covariance[n_comp] = 0.001
            else:
                init_covariance[n_comp] = 0.0001
        self.sd = init_covariance
        # print "SD: ", self.sd
        return

    def train(self, column):
        self.random_assignment()
        i = 0
        while True:
            i += 1
            # print "========================================================="
            # print "Round: ", i

            log_likelihood, sum_ll = self.calc_log_likelihood()

            if i > 10 or (i > 5 and abs(self.ll - log_likelihood) < 0.0001):
                self.ll = log_likelihood
                self.data = self.testdata

                test_ll, sum_ll = self.calc_log_likelihood()
                print "***************Converging feature: ", column, " after round: ", i, "****************"
                print "log likelyhood: ", self.ll
                return sum_ll
            self.ll = log_likelihood
            print "ll:", self.ll
            # print "Mean:", self.mean
            # print "SD:", self.sd
            # print "Before Round ", i, "ll:", self.ll
            self.update_parameters()
            # print "Mean: ", self.mean
            # print "Sigma: ", self.sd
            # print "Weights:", self.w


        return