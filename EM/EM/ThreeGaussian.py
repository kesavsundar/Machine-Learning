__author__ = 'kesavsundar'
__author__ = 'kesav'
import numpy as np
import random as rand
class nGaussian:
    def __init__(self):
        self.data = np.loadtxt(open("/home/kesavsundar/machinelearning/ml/EM/3gaussian.txt", 'rb'),
                                  delimiter=None, dtype=float)
        self.w = list()
        self.mean = list()
        self.sd = list()
        self.n_cluster = list()
        self.score = None
        self.gamma = None
        self.ll = 0.0
        self.k = 3
        self.n_features = 2

    def random_assignment(self):
        # mus = rand.sample(self.data[:, :].tolist(), self.k)
        # self.mean = np.matrix(mus)
        # clusters = [[] for dummy in range(0, self.k)]
        # self.n_cluster = [0.0 for dummy in range(0, self.k)]
        # # For each row in a feature
        # for row in self.data:
        #     r = np.argmin([np.linalg.norm(row - m) for m in mus])
        #     clusters[r].append(row.tolist())
        #     self.n_cluster[r] += 1
        # # For each hidden component
        #
        # for component in range(0, self.k):
        #     cluster_matrix = np.matrix(clusters[component]).reshape(self.n_cluster[component], self.k)
        #     x_m_mu = cluster_matrix - self.mean[component]
        #     self.sd.append(np.dot(x_m_mu.T, x_m_mu) / self.n_cluster[component])
        #     self.w.append(float(self.n_cluster[component]) / len(self.data))

        temp = np.ones(self.k*2).reshape(self.k, 2)
        temp[0, 0] = 1
        temp[0, 1] = 4
        temp[1, 0] = 2
        temp[1, 1] = 5
        temp[2, 0] = 4
        temp[2, 1] = 1

        self.mean = temp

        self.sd = [np.matrix([[0.5, 1], [0.25, 3]]), np.matrix([[0.7, 0.2], [1, 1]]), np.matrix([[0.4, 0.2], [0.2, 0.5]])]
        self.w = [float(0.4), float(0.4), float(0.2)]
        return

    def pdf(self, comp):
        covariance = np.linalg.det(self.sd[comp])
        inv_covariance = np.linalg.pinv(self.sd[comp])
        den = float(pow((2.0 * np.pi), .5) * pow(abs(covariance), 0.5))
        pdf_list = list()
        for row in range(0, len(self.data)):
            x_minus_mean = np.matrix(self.data[row, :] - self.mean[comp, :]).reshape(1, self.n_features)
            temp = np.dot(x_minus_mean, inv_covariance)
            temp2 = np.dot(temp, x_minus_mean.T)
            z = (-0.5 * temp2)
            exp_term = np.exp(z)
            # print "x_minus_mean", x_minus_mean
            # print "inv covar", inv_covariance
            # print den, exp_term
            temp3 = float(1)/den * exp_term
            pdf_list.append(temp3[0, 0])
        pdf_matrix = np.matrix(pdf_list).reshape(len(self.data), 1)
        return pdf_matrix

    def calc_log_likelihood(self):
        n = len(self.data)
        self.score = np.ones(len(self.data)).reshape(len(self.data), 1)
        self.gamma = np.ones(len(self.data)*self.k).reshape(len(self.data), self.k)
        for n_comp in range(0, self.k):
            likelihood = self.w[n_comp] * self.pdf(n_comp)
            self.score = np.concatenate((self.score, likelihood), axis=1)
        print self.score.shape
        self.score = self.score[:, 1:]
        sum_likelihood = self.score[:, 0] + self.score[:, 1] + self.score[:, 2]

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
        print "log likely hood", log_likelihood / n
        return log_likelihood / n

    def update_parameters(self):
        n = len(self.data)
        n0 = 0.0
        n1 = 0.0
        n2 = 0.0

        for row in range(0, len(self.gamma)):
            n0 += self.gamma[row, 0]
            n1 += self.gamma[row, 1]
            n2 += self.gamma[row, 2]

        print "--------", n0, n1, n2
        self.w[0] = float(n0) / n
        self.w[1] = float(n1) / n
        self.w[2] = float(n2) / n

        initMeans = np.zeros(self.k*2).reshape(self.k, 2)
        #[np.matrix([[0], [0]]), np.matrix([[0], [0]])]
        initCovariances = [np.matrix([[0, 0],[0, 0]]), np.matrix([[0, 0],[0, 0]]),np.matrix([[0, 0],[0, 0]])]


        for row in range(0, n):
            for n_comp in range(0, self.k):
                initMeans[n_comp, :] += (self.gamma[row, n_comp] * self.data[row, :])
        initMeans[0, :] /= n0
        initMeans[1, :] /= n1
        initMeans[2, :] /= n2
        self.mean = initMeans

        for row in range(0, n):
            t = np.zeros_like(self.sd[0])
            for n_comp in range(0, self.k):
                x_minus_mean = np.matrix(self.data[row] - self.mean[n_comp, :]).reshape(1, self.n_features)
                temp = np.matrix(self.gamma[row, n_comp] * x_minus_mean).reshape(1, self.n_features)
                t = np.dot(temp.T, x_minus_mean)
                c = initCovariances[n_comp]
                initCovariances[n_comp] = c + t
        new_sd = [(initCovariances[0]/n0), (initCovariances[1]/n1), (initCovariances[2]/n2)]
        self.sd = new_sd
        return

    def train(self):
        self.random_assignment()
        i = 0
        while True:
            i += 1
            print "========================================================="
            print "Round: ", i

            log_likelihood = self.calc_log_likelihood()

            if abs(self.ll - log_likelihood) < 0.00001:
                print "*******************************Converged"
                print "Mean: ", self.mean
                print "Standard Deviation: ", self.sd
                print "W Vector: ", self.w
                print "*****************************************"
                break
            self.ll = log_likelihood
            print "Before Round ", i, "ll:", self.ll
            self.update_parameters()
            # print "Mean: ", self.mean
            # print "Sigma: ", self.sd
            # print "Weights:", self.w


        return