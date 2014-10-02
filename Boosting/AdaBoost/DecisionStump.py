__author__ = 'kesavsundar'
import DecisionNode as Dn
import numpy as np


class DecisionTree:

    def __init__(self, x, y, weight, config, col_number):
        x = np.matrix(x).reshape(len(x), 1)
        y = np.matrix(y).reshape(len(y), 1)
        weight = np.matrix(weight).reshape(len(weight), 1)

        self.xy = np.concatenate((x, y, weight), axis=1)

        self.config = config
        self.col_number = col_number

    def sort_based_on_feature(self):
        sorted_index = np.lexsort(self.xy[:, 0].T)
        sorted_data = np.zeros_like(self.xy)
        for row in range(0, len(self.xy)):
            sorted_data[row, :] = self.xy[sorted_index[0, row], :]
        self.xy = sorted_data
        # if self.config == -1:
        #     self.xy.dtype = 'float32,str,float32'
        # else:
        #     self.xy.dtype = 'str,str,float32'
        return

    def classify(self):
    def calc_error(self, value, prev_index, prev_error, start):
        error = prev_error
        next_val_index = prev_index + 1

        if self.config != -1000 and prev_index >= 0:
            for i in range(0, prev_index):
                if self.xy[i, 1] != '-':
                    error[i] = float(self.xy[i, 2])

        while next_val_index < len(self.xy) and self.xy[next_val_index, 0] == value:
            if self.xy[next_val_index, 1] != '+':
                error[next_val_index] = float(self.xy[next_val_index, 2])
            next_val_index += 1

        if start:
            for i in range(next_val_index, len(self.xy)):
                if self.xy[i, 1] != '-':
                    error[i] = float(self.xy[i, 2])
        return error, (next_val_index - 1), False

    def weak_learner_train(self):
        self.sort_based_on_feature()
        best_criteria = None
        best_error = None
        column_values = {}
        for row in self.xy[:, 0]:
            column_values[row[0, :]] = 1
        keys = column_values.keys()
        prev_index = -1
        prev_error = [0.0] * len(self.xy)
        start = True
        for val in range(0, len(keys)):
            error, current_index, start = self.calc_error(keys[val], prev_index, prev_error, start)
            if best_error is None:
                best_error = sum(error)
                best_criteria = keys[val]
            elif error < best_error:
                best_error = sum(error)
                best_criteria = keys[val]
            prev_index = current_index
            prev_error = error
        decision_node = Dn.DecisionNode(col=self.col_number, value=best_criteria, error=best_error)
        return decision_node