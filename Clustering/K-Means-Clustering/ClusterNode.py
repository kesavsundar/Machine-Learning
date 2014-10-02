__author__ = 'kesavsundar'
import numpy as np

class ClusterNode:
    def __init__(self, point=None, label=None):
        self.cluster_points = [point]
        self.centroid = point
        self.label = [label]
        return

    def merge_cluster(self, another_cluster):
        for point, label in zip(another_cluster.cluster_points, another_cluster.label):
            self.cluster_points.append(point)
            self.label.append(label)
        # Revaluate centroid
        self.centroid = np.mean(self.cluster_points, axis=0)[0].tolist()

    def get_label(self):
        from collections import Counter
        most_common, num_most_common = Counter(self.label).most_common(1)[0]
        return most_common