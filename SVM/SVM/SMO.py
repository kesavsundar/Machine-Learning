__author__ = 'kesavsundar'
import numpy as np
import random

class SMO():
    def __init__(self):
        self.train_img = None
        self.train_label = None
        self.test_img = None
        self.test_label = None
        self.c = 2.3
        self.gamma = 0.001
        self.m = None
        self.alphas = None
        self.b = 0
        self.error_cache = None
        self.k = None

    def kernelTrans(self, x, a):
        k = x * a.T
        return k

    def load_train_data(self, positive_label):
        with open('/home/kesavsundar/Dropbox/Books/SVM/train.txt', 'rb') as train_img:
            temp_data = [line.strip() for line in train_img]
        r = len(temp_data)
        c = len(temp_data[0].split(','))

        self.train_img = np.mat(np.zeros((r, c)))
        for i in range(0, r):
            row_data = temp_data[i].split(',')
            splitted_data = [float(row_data[k]) for k in range(0, c)]
            for j in range(0, c):
                self.train_img[i, j] += splitted_data[j]
        train_img.close()

        with open('/home/kesavsundar/Dropbox/Books/SVM/trainLabels.txt', 'rb') as train_label:
            temp_data = [line.strip() for line in train_label]
        r = len(temp_data)
        self.train_label = np.zeros(r).reshape(r, 1)
        for i in range(0, r):
            lbl = float(temp_data[i])
            if positive_label == lbl:
                self.train_label[i, 0] = 1.0
            else:
                self.train_label[i, 0] = -1.0

        train_label.close()
        self.train_label = self.train_label.T

        self.m = np.shape(self.train_img)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.error_cache = np.mat(np.zeros((self.m, 2)))
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = self.kernelTrans(self.train_img, self.train_img[i])

        return

    def load_test_data(self):
        with open('/home/kesavsundar/Dropbox/Books/SVM/test.txt', 'rb') as test_img:
            temp_data = [line.strip() for line in test_img]
        r = len(temp_data)
        c = len(temp_data[0].split(','))

        self.test_img = np.mat(np.zeros((r, c)))
        for i in range(0, r):
            row_data = temp_data[i].split(',')
            splitted_data = [float(row_data[k]) for k in range(0, c)]
            for j in range(0, c):
                self.test_img[i, j] += splitted_data[j]
        test_img.close()

        with open('/home/kesavsundar/Dropbox/Books/SVM/testLabels.txt', 'rb') as test_label:
            temp_data = [line.strip() for line in test_label]
        r = len(temp_data)
        self.test_label = np.zeros(r).reshape(r, 1)
        for i in range(0, r):
            self.test_label[i, 0] = float(temp_data[i])
        test_label.close()
        self.test_label = self.test_label.T

        # print self.train_img, self.train_label
        # print self.test_img, self.test_label

        return

    def calculate_error(self, k):
        score_k = float(np.multiply(self.alphas, self.train_label[0, k]).T * self.k[:, k]) + self.b
        error_k = score_k - self.train_label[0, k]
        return error_k

    def select_random_j(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.m))
        return j

    def select_j_index(self, i, error_i):
        max_k = -1
        max_error_change = 0
        error_j = 0
        valid_error_cached_list = np.nonzero(self.error_cache[:, 0].A)[0]
        if len(valid_error_cached_list) > 1:
            for k in valid_error_cached_list:
                if k == i:
                    continue
                error_k = self.calculate_error(k)
                error_change = abs(error_i - error_k)
                if max_error_change < error_change:
                    max_k = k
                    max_error_change = error_change
                    error_j = error_k
            return max_k, error_j
        else:
            j = self.select_random_j(i)
            error_j = self.calculate_error(j)
        return j, error_j

    def update_error_k(self, k):
        error_k = self.calculate_error(k)
        self.error_cache[k] = [1, error_k]
        return

    def change_alpha(self, alpha_j, H, L):
        if alpha_j > H:
            alpha_j = H
        if L > alpha_j:
            alpha_j = L
        return alpha_j

    def inner_l(self, inner_i):
        error_i = self.calculate_error(inner_i)
        if (((self.train_label[0, inner_i] * error_i) < -self.gamma)
            and (self.alphas[inner_i] < self.c)) \
        or (((self.train_label[0, inner_i] * error_i) > self.gamma) \
            and (self.alphas[inner_i] > 0)):
            inner_j, error_j = self.select_j_index(inner_i, error_i)
            i_old_alphas = self.alphas[inner_i].copy()
            j_old_alphas = self.alphas[inner_j].copy()
            if self.train_label[0, inner_i] != self.train_label[0, inner_j]:
                L = max(0, self.alphas[inner_j] - self.alphas[inner_i])
                H = min(self.c, self.c + self.alphas[inner_j] - self.alphas[inner_i])
            else:
                L = max(0, self.alphas[inner_j] + self.alphas[inner_i] - self.c)
                H = min(self.c, self.alphas[inner_j] + self.alphas[inner_i])
            if L == H:
                # print "L and H are equal"
                return 0
            eta = 2.0 * self.k[inner_i, inner_j] - self.k[inner_i, inner_i] - self.k[inner_j, inner_j]
            if eta > 0:
                # print "ETA is greater than 0"
                return 0
            self.alphas[inner_j] -= self.train_label[0, inner_j] * (error_i - error_j) / eta
            self.alphas[inner_j] = self.change_alpha(self.alphas[inner_j], H, L)
            self.update_error_k(inner_j)

            if abs(self.alphas[inner_j] - j_old_alphas) < 0.00001:
                # print "J is not moving"
                return 0
            self.alphas[inner_i] += self.train_label[0, inner_j] * self.train_label[0, inner_i] * (j_old_alphas - self.alphas[inner_j])
            self.update_error_k(inner_i)

            b1 = self.b - error_i - self.train_label[0, inner_i] * (self.alphas[inner_i] - i_old_alphas) * \
                                    self.k[inner_i, inner_i] - self.train_label[0, inner_j] * \
                                                   (self.alphas[inner_j] - j_old_alphas) * self.k[inner_i, inner_j]


            b2 = self.b - error_j - self.train_label[0, inner_i] * (self.alphas[inner_i] - i_old_alphas) * \
                                    self.k[inner_i, inner_j] - self.train_label[0, inner_j] * \
                                                   (self.alphas[inner_j] - j_old_alphas) * self.k[inner_j, inner_j]
            if (0 < self.alphas[inner_i]) and (self.c > self.alphas[inner_i]):
                self.b = b1
            elif (0 < self.alphas[inner_j]) and (self.c > self.alphas[inner_j]):
                self.b = b2
            else:
                self.b = float(b1 + b2) / 2.0
            return 1

        else:
            return 0

    # def calcWs(self):
    #     m, n = np.shape(self.train_img)
    #     w = np.mat(np.zeros((n, 1)))
    #     for i in range(m):
    #         w += np.multiply(self.alphas[i]*self.train_label[0, i], self.train_img[i, :].T)
    #     return w

    def smo_full(self):
        iteration = 0
        max_iteration = 100
        entire_set = True
        alpha_pairs_changed = 0
        while (iteration < max_iteration) and ((alpha_pairs_changed > 0) or entire_set):
        # while (alpha_pairs_changed > 0) or entire_set:
            alpha_pairs_changed = 0
            if entire_set:
                for outer_i in range(0, self.m):
                    alpha_pairs_changed += self.inner_l(outer_i)
                # print "fullSet, iter: %d i:%d, pairs changed %d" % (iteration, outer_i, alpha_pairs_changed)
                iteration += 1
            else:
                non_bound_id = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.c))[0]
                for non_bound_i in non_bound_id:
                    alpha_pairs_changed += self.inner_l(non_bound_i)
                    # print "non-bound, iter: %d i:%d, pairs changed %d" % (iteration, non_bound_i, alpha_pairs_changed)
                iteration += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print "Iteration number: %d" % iteration
        return self.b, self.alphas

    def make_predictions(self):
        active_alphas_indices = np.nonzero(self.alphas.A > 0)[0]
        # print active_alphas_indices
        support_vectors = np.mat(self.train_img[active_alphas_indices])
        label_support_vectors = np.mat(self.train_label[0, active_alphas_indices])
        print np.shape(self.test_img)
        m_test, n_test = np.shape(self.test_img)
        predictions = list()
        prediction_score = list()
        for point in range(m_test):
            test_k = self.kernelTrans(support_vectors, self.test_img[point, :])
            predict = test_k.T * np.multiply(label_support_vectors, self.alphas[active_alphas_indices])
            predict = float((np.sum(predict)) + self.b)
            predictions.append(np.sign(predict))
            prediction_score.append(predict)
        return predictions, prediction_score

if __name__ == '__main__':
    sm = SMO()
    labels = list()
    scores = list()
    sm.load_test_data()
    for itera in range(0, 10):
        print itera, " vs Many Running"
        sm.load_train_data(itera)
        sm.smo_full()
        label, score = sm.make_predictions()
        labels.append(label)
        scores.append(score)
    # print labels, scores
    correct = 0.0
    m = np.shape(sm.test_label)[1]
    for i in range(0, m):
        true_label = sm.test_label[0, i]
        for j in range(0, len(labels)):
            best_score = None
            best_label = None
            if labels[j][i] == 1 and best_score is None:
                best_score = scores[j][i]
                best_label = j
            elif labels[j][i] == 1 and best_score is not None:
                if best_score < scores[j][i]:
                    best_score = scores[j][i]
                    best_label = j
            else:
                continue
        if true_label == best_label:
            correct += 1.0
    accuracy_test = float(correct) / m
    print "Accuracy is ", accuracy_test

