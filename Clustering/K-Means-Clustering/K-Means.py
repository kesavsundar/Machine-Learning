__author__ = 'kesavsundar'
import numpy as np
import random as rand
import sys
sys.path.append('/home/kesavsundar/Dropbox/CS6140_K_Gopal/General_Modules')
import accuracy
import scipy

class KMeans():
    def __init__(self):
        self.len_data = 0
        self.len_test_data = 0
        self.n_features = 0
        self.label_index = 0
        self.train_data = None
        self.test_data = None
        self.k = 8
        self.centroids = None
        self.clusters = None
        self.cluster_labels = None
        self.label_kmeans = None

    def train_data_load(self):
        with open("/home/kesavsundar/machinelearning/data/20newsgroup/feature_names.txt", 'rb') as features_file:
            features_data = [line.strip() for line in features_file]
        self.n_features = len(features_data)
        self.label_index = self.n_features + 1
        features_file = None
        features_data = None

        with open("/home/kesavsundar/machinelearning/data/20newsgroup/train.txt", 'rb') as dat_file:
            temp_data = [line.strip() for line in dat_file]
        self.len_data = len(temp_data)
        self.train_data = np.zeros(self.len_data*self.label_index).reshape(self.len_data, self.label_index)
        dat_file = None

        dict_each_label_data = dict()

        for i in range(0, self.len_data):
            sparse_data = temp_data[i].split(' ')
            self.train_data[i, self.label_index-1] += int(sparse_data[0])

            for j in range(1, len(sparse_data)-3):
                feature_freq = sparse_data[j].split(':')
                self.train_data[i, int(feature_freq[0])] += int(feature_freq[1])
            if int(sparse_data[0]) in dict_each_label_data.keys():
                pass
            else:
                dict_each_label_data[int(sparse_data[0])] = self.train_data[i, :-1].tolist()

        temp_data = None
        print "Train Data Loaded!!!"
        self.centroids = dict_each_label_data.values()
        return

    def test_data_load(self):
        with open("/home/kesavsundar/machinelearning/data/20newsgroup/test.txt", 'rb') as test_dat_file:
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
        print "Test Data Loaded!!!"
        return

    def initial_setup(self):
       # self.centroids = rand.sample(self.train_data[:, :-1].tolist(), self.k)
       iteration = 0
       while True:
            clusters = [[] for dummy in range(0, self.k)]
            cluster_labels = [[] for dummy in range(0, self.k)]

            for i in range(0, self.len_data):
                r = np.argmin([scipy.spatial.distance.cosine(self.train_data[i, :-1], m) for m in self.centroids])
                clusters[r].append(self.train_data[i, :-1])
                cluster_labels[r].append(self.train_data[i, -1:][0])
            # find new center for clusters

            mean_list = list()
            for i in range(0, self.k):
                mean_list.append(np.mean(clusters[i], axis=0))
            if iteration != 0:
                if self.has_converged(mean_list, self.centroids):
                    break
            self.clusters = clusters
            self.centroids = mean_list
            self.cluster_labels = cluster_labels
            print self.find_max_occour_label()
            iteration += 1

    def has_converged(self, mu, oldmu):
        final = True
        for new, old in zip(mu, oldmu):
            for x, y in zip(new, old):
                if (x - y) > .00001:
                    final = False
                    break
        return final

    def find_max_occour_label(self):
        from collections import Counter
        labels_kmeans = list()
        for each_cluster in self.cluster_labels:
            most_common = -1
            if len(each_cluster) > 0:
                most_common, num_most_common = Counter(each_cluster).most_common(1)[0]
            labels_kmeans.append(most_common)
        self.label_kmeans = labels_kmeans
        return labels_kmeans

    def predict_test_data_points(self):
        prediction = list()
        for i in range(0, self.len_test_data):
            r = np.argmin([scipy.spatial.distance.cosine(self.train_data[i, :-1], m) for m in self.centroids])
            label = self.label_kmeans[r]
            prediction.append(label)
        correct = 0.0
        for pred, actual in zip(prediction, self.train_data[:, -1:].tolist()):
            if pred == actual:
                correct += 1

        accuracy = correct / len(prediction)
        print accuracy
        # acc = accuracy.Accuracy(self.test_data[:, -1:], prediction)
        # tpr, fpr, accuracy_rate = acc.compute_accuracy()
        # print "Accuracy Rate: ", accuracy_rate

    def find_majority_label(self, r):
        labels_dict = list()
        for i in range(0, 7):
            labels_dict.append(0)
        for row in self.clusters[r]:
            label = row[:, -1:]
            labels_dict[label[0, 0]] += 1
        return np.argmax(labels_dict)

if __name__ == '__main__':
    clustering = KMeans()
    clustering.train_data_load()
    clustering.initial_setup()
    clustering.test_data_load()
    clustering.predict_test_data_points()
