__author__ = 'kesav'
from sklearn import preprocessing as norm


class Normalize:
    def __init__(self, data):
        self.xy_data = data
        self.x_data = self.xy_data[:, :-1]

    def min_max_normalize_data(self):
        min_max_scalar = norm.MinMaxScaler()
        min_max_normalized_data = min_max_scalar.fit_transform(self.x_data)
        self.xy_data[:, :-1] = min_max_normalized_data
        return self.xy_data

    def mean_normalize_data(self):
        return self.xy_data
