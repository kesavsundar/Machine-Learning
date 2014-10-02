__author__ = 'kesavsundar'
import numpy as np
import ClusterNode as cn
from scipy import spatial

# take all the points as unique cluster.
# start merging until you get 8 clusters

class HierarchialClustering():
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
        self.proximity_matrix = None
        self.cluster_objects = list()

    def train_data_load(self):
        with open("/home/kesavsundar/machinelearning/data/20newsgroup/feature_names.txt", 'rb') as features_file:
            features_data = [line.strip() for line in features_file]
        self.n_features = len(features_data)
        self.label_index = self.n_features + 1
        features_file = None
        features_data = None

        with open("/home/kesavsundar/machinelearning/data/20newsgroup/processed_data/train.txt", 'rb') as dat_file:
            temp_data = [line.strip() for line in dat_file]
        self.len_data = len(temp_data)
        self.train_data = np.zeros(self.len_data*self.label_index).reshape(self.len_data, self.label_index)
        dat_file = None
        print "Length of train data: ", self.len_data
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
        self.proximity_matrix = np.zeros(self.len_data * self.len_data).reshape(self.len_data, self.len_data)
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
        f = open('output', 'w')
        # Init object for each cluster!
        for row in self.train_data:
            row = np.matrix(row)
            node = cn.ClusterNode(row[:, :-1], row[:, -1:])
            self.cluster_objects.append(node)

        # Update proximity matrix
        while len(self.cluster_objects) >= 9:
            for i in range(0, len(self.cluster_objects)):
                self.proximity_matrix[i, i] = float('NaN')
                for j in range(i+1, len(self.cluster_objects)):
                    average_dist_seperation = spatial.distance.cosine(self.cluster_objects[i].centroid,
                                                                            self.cluster_objects[j].centroid)
                    self.proximity_matrix[i, j] = average_dist_seperation
                    self.proximity_matrix[j, i] = average_dist_seperation

            min_tuple, min_avg = self.find_min_tuple()
            str1 = "===============================================================" + '\n'
            str2 = "Merging Tuple" + str(min_tuple) + "with AVG link distance: " + str(min_avg) + '\n'
            str3 = "Now the size of cluster is: " + str(len(self.cluster_objects)) + '\n'
            str4 = "Shape of proximity matrix is " + str(self.proximity_matrix.shape) + '\n'
            f.writelines([str1, str2, str3, str4])
            f.flush()
            self.merge_tuple(min_tuple[0], min_tuple[1])
        return

    def merge_tuple(self, i, j):
        # Update clusters
        self.cluster_objects[i].merge_cluster(self.cluster_objects[j])
        self.cluster_objects.pop(j)
        # Update Proximity matrix
        self.proximity_matrix = np.delete(self.proximity_matrix, j, axis=0)
        self.proximity_matrix = np.delete(self.proximity_matrix, j, axis=1)

        for row in range(0, len(self.cluster_objects)):
            average_dist_seperation = spatial.distance.cosine(self.cluster_objects[i].centroid,
                                                                    self.cluster_objects[row].centroid)
            self.proximity_matrix[i, row] = average_dist_seperation
            self.proximity_matrix[row, i] = average_dist_seperation
        return

    def find_min_tuple(self):
        min_avg_dist = 100000
        for i in range(0, len(self.cluster_objects)):
            for j in range(i+1, len(self.cluster_objects)):
                if self.proximity_matrix[i, j] < min_avg_dist:
                    min_avg_dist = self.proximity_matrix[i, j]
                    min_tuple = (i, j)
        return min_tuple, min_avg_dist

    def predict_test_data_points(self):
        prediction = list()
        for i in range(0, self.len_test_data):
            r = np.argmin([np.linalg.norm(self.train_data[i, :-1] - m.centroid) for m in self.cluster_objects])
            label = self.label_kmeans[r]
            prediction.append(label)
        correct = 0.0
        for pred, actual in zip(prediction, self.train_data[:, -1:].tolist()):
            if pred == actual:
                correct += 1
        accuracy = correct / len(prediction)
        print accuracy

if __name__ == '__main__':
    clustering = HierarchialClustering()
    clustering.train_data_load()
    clustering.initial_setup()
    clustering.test_data_load()
    clustering.predict_test_data_points()
