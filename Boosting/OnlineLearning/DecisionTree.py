__author__ = 'kesavsundar'
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, x, y, weight, isfloat, labels, col):
        self.n = len(x)
        self.x = x
        self.y = y
        self.weight = weight
        self.labels = labels
        self.best_value = None
        self.true_branch = None
        self.false_branch = None
        self.isfloat = isfloat
        self.best_error = None
        self.col = col
        return

    def predict(self, value, best_value=None, label_tb=None, label_fb=None):
        if label_fb is None and label_fb is None and best_value is None:
            label_tb = self.true_branch
            label_fb = self.false_branch
            best_value = self.best_value
        if self.isfloat:
            if value <= best_value:
                return label_tb
            else:
                return label_fb
        else:
            if value == best_value:
                return label_tb
            else:
                return label_fb

    def learn(self):
        column_values = {}
        for row in self.x:
            column_values[row] = 1
        for val in column_values.keys():
            tb, fb = self.separate_set(val)
            tbl, fbl = self.assign_label_for_branch(tb, fb)
            error = self.calc_error(tb, fb, tbl, fbl, val)
            # print "$$$ |", self.col," |", val ,"|", error
            if self.best_error is None:
                self.best_error = error
                self.true_branch = tbl
                self.false_branch = fbl
                self.best_value = val

            elif error < self.best_error:
                self.best_error = error
                self.true_branch = tbl
                self.false_branch = fbl
                self.best_value = val

        return self

    def separate_set(self, value):
        tb = list()
        fb = list()

        if self.isfloat:
            for row in range(0, self.n):
                if self.x[row] <= value:
                    tb.append((self.x[row], self.y[row], self.weight[row]))
                else:
                    fb.append((self.x[row], self.y[row], self.weight[row]))
        else:
            for row in range(0, self.n):
                if self.x[row] == value:
                    tb.append((self.x[row], self.y[row], self.weight[row]))
                else:
                    fb.append((self.x[row], self.y[row], self.weight[row]))
        return tb, fb

    def assign_label_for_branch(self, tb, fb):
        tb_count = {}
        fb_count = {}
        for label in self.labels:
            tb_count[label] = 0
            fb_count[label] = 0
        for row in tb:
            tb_count[row[1]] += 1
        for row in fb:
            fb_count[row[1]] += 1

        if tb_count[self.labels[0]] > tb_count[self.labels[1]]:
            tbl = self.labels[0]
            fbl = self.labels[1]
        else:
            tbl = self.labels[1]
            fbl = self.labels[0]
        return tbl, fbl

    def calc_error(self, tb, fb, label_tb, label_fb, best_value):
        error = 0.0
        for row in tb:
            if self.predict(row[0], best_value, label_tb, label_fb) != row[1]:
                error += row[2]
        for row in fb:
            if self.predict(row[0], best_value, label_tb, label_fb) != row[1]:
                error += row[2]
        return error
