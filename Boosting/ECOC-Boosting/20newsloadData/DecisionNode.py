__author__ = 'kesavsundar'
import numpy as np

class DecisionNode():
    def __init__(self, col, value, misclassified_set, l, r):
        self.col = col
        self.value = value
        self.misclassified_set = misclassified_set
        self.lesser_branch = l
        self.greater_branch = r
        return

    def get_error(self, weights):
        error = 0
        for index in self.misclassified_set:
            error += weights[index]
        # if error > 0.5:
        #     error = 1 - error
        #     self.lesser_branch = 0
        #     self.greater_branch = 1
        #     all_indices_set = set(np.arange(len(weights)))
        #     self.misclassified_set = all_indices_set.difference(self.misclassified_set)
        return error

    def predict(self, test_val):
        if test_val == self.value:
            return self.lesser_branch
        else:
            return self.greater_branch
