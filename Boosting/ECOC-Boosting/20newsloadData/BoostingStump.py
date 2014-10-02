__author__ = 'kesavsundar'
import numpy as np

import DecisionNode as dn


class DecisionTreeClassifier():
    def __init__(self, x, y, col, weights, one_labels):
        self.n = len(x)
        self.x = x
        self.y = y
        self.weight = weights
        self.data = None
        self.one_labels = one_labels
        self.col = col
        return

    def init_classifier(self):
        # sort all the data
        # self.data = np.concatenate((self.index, self.x, self.y), axis=1)
        self.data = np.zeros(self.n * 3).reshape(self.n, 3)
        new_list = np.array([self.x, self.y])
        indices = new_list[0].argsort()
        for i in range(0, len(indices)):
            self.data[i, 0] += indices[i]
            self.data[i, 1] += self.x[indices[i]]
            self.data[i, 2] += self.y[indices[i]]
        return

    # def start_thresholding(self):
    #     # first get the misclassified points for initial value
    #     initial_value = self.data[0, 1]
    #     decision_stumps = list()
    #     above_misclassified, below_misclassified,\
    #     next_val_index = self.get_initial_misclassified_set(initial_value)
    #     # Stump for initial value
    #     # a_label, b_label = self.get_majority(a_dict, b_dict)
    #     # stump = self.generate_decision_node(initial_value,
    #     #                                     above_misclassified.union(below_misclassified), a_label, b_label)
    #     stump = self.generate_decision_node(initial_value, above_misclassified.union(below_misclassified))
    #
    #     decision_stumps.append(stump)
    #     if next_val_index < len(self.data):
    #         next_val = self.data[next_val_index, 1]
    #         i = next_val_index
    #         while i < self.n:
    #             if self.data[i, 1] == next_val:
    #                 if self.data[i, 0] not in below_misclassified:
    #                     # a_dict[b_label] += 1
    #                     # b_dict[b_label] -= 1
    #                     above_misclassified.add(self.data[i, 0])
    #                 else:
    #                     # a_dict[a_label] += 1
    #                     # b_dict[a_label] -= 1
    #                     below_misclassified.remove(self.data[i, 0])
    #                 # a_label, b_label = self.get_majority(a_dict, b_dict)
    #                 i += 1
    #             else:
    #                 # a_label, b_label = self.get_majority(a_dict, b_dict)
    #                 # stump = self.generate_decision_node(next_val,
    #                 #                                     above_misclassified.union(below_misclassified), a_label, b_label)
    #                 stump = self.generate_decision_node(next_val, above_misclassified.union(below_misclassified))
    #                 decision_stumps.append(stump)
    #                 next_val = self.data[i, 1]
    #                 break
    #     return decision_stumps

    def generate_decision_node(self, value, wrong_set, l, r):
        node = dn.DecisionNode(self.col, value, wrong_set, l, r)
        return node

    def get_initial_misclassified_set(self, initial_val):
        above_misclassified = set()
        below_misclassified = set()
        i = 0

        # above_dict = dict()
        # above_dict[0] = 0
        # above_dict[1] = 1
        # below_dict = dict()
        # below_dict[0] = 0
        # below_dict[1] = 1
        #
        # while i < self.n and self.data[i, 1] == initial_val:
        #     above_dict[self.get_label(self.data[i, 2])] += 1
        #     i += 1
        #
        # while i < self.n:
        #     below_dict[self.get_label(self.data[i, 2])] += 1
        #     i += 1

        # above_branch_label, below_branch_label = self.get_majority(above_dict, below_dict)

        i = 0
        while i < self.n and self.data[i, 1] == initial_val:
            if self.get_label(self.data[i, 2]) != 0:
                above_misclassified.add(self.data[i, 0])
            i += 1
        next_val_index = i
        while i < self.n:
            if self.get_label(self.data[i, 2]) != 1:
                below_misclassified.add(self.data[i, 0])
            i += 1
        return above_misclassified, below_misclassified, 0, 1

    def start_thresholding(self):
        initial_value = self.data[0, 1]
        decision_stumps = list()
        above_misclassified, below_misclassified, a_label, b_label =\
            self.get_initial_misclassified_set(initial_value)
        stump = self.generate_decision_node(initial_value, above_misclassified.union(below_misclassified), a_label, b_label)
        decision_stumps.append(stump)
        return decision_stumps

    def get_majority(self, above, below):
        if above[0] >= above[1]:
            above_branch_label = 0
        else:
            above_branch_label = 1

        if below[0] >= below[1]:
            below_branch_label = 0
        else:
            below_branch_label = 1
        return above_branch_label, below_branch_label

    def get_label(self, param_label):
        if param_label in self.one_labels:
            return 1
        else:
            return 0

    def learn(self):
        self.init_classifier()
        decision_stumps = self.start_thresholding()
        self.data = None
        self.x = None
        self.y = None
        self.weight = None

        return decision_stumps

if __name__ == '__main__':
    x = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0]
    y = [1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2]
    one_label = [1]
    weights = [1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2]
    test = DecisionTreeClassifier(x, y, 1, weights, one_label)
    test.init_classifier()
    test.start_thresholding()