__author__='kesav'
import numpy as np


class RegressionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None, mse=0):
        self.col = col
        self.value = value
        self.results = results
        self.true_branch = tb
        self.false_branch = fb
        self.mse = mse


def labelcount(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results:results[r] = 0
        results[r] += 1
    return results


def partitionset(rows, column, value):
    part_function = lambda row:row[column] >= value
    tb = [row for row in rows if part_function(row)]
    fb = [row for row in rows if not part_function(row)]
    return (tb, fb)


def mse(rows):
    if len(rows) is not 0:
        label = {}
        iterator = 0
        for row in rows:
            r = row[len(row) - 1]
            label[iterator] = r
            iterator += 1
        #print label
        label = label.values()
        mean_label = sum(label) / len(label)
        mean_errors = {}
        # Calculate the mean squared error
        iterator = 0
        for row in label:
            mean_errors[iterator] = (row - mean_label)**2
            iterator += 1
        #return the sum of mean erros
        mean_errors = mean_errors.values()
        return sum(mean_errors)
    return 0


def train(train_data, level, level_mse, leve_of_tree):
    if len(train_data) is 0:
        return RegressionNode()
    current_mse = level_mse / len(train_data)
    if level < leve_of_tree:
        min_error = level_mse
        best_criteria = None
        best_sets = None
        best_mse_set1 = 0
        best_mse_set2 = 0
        column_count = len(train_data[0]) - 1
        for col in range(0, column_count):
            column_values = {}
            for row in train_data:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = partitionset(train_data, col, value)
                mse_set1 = mse(set1)
                mse_set2 = mse(set2)
                mean_sq_error = mse_set1 + mse_set2
                if mean_sq_error < min_error and len(set1) > 0 and len(set2) > 0:
                    min_error = mean_sq_error
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
                    best_mse_set1 = mse_set1
                    best_mse_set2 = mse_set2
        if min_error > 0:
            truebranch = train(best_sets[0], level + 1, best_mse_set1, leve_of_tree)
            falsebranch = train(best_sets[1], level + 1, best_mse_set2, leve_of_tree)
            return RegressionNode(col=best_criteria[0], value=best_criteria[1],
                                   tb=truebranch, fb=falsebranch, mse=current_mse)
    return RegressionNode(mse=current_mse, results=labelcount(train_data))


def predictor(test_data, tree=None):
    if tree.results is not None:
        label = tree.results.keys()
        mean_label = sum(label) / len(label)
        return mean_label
    else:
        v = test_data[tree.col]
        if v >= tree.value:
            branch = tree.true_branch
        else:
            if v == tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        return predictor(test_data, branch)


def printtree(tree, level, mse_level):
    if tree is None:
        return
    print_level = level
    tab_string = ""
    while print_level > 0:
        tab_string += "\t"
        print_level -= 1
    print tab_string + "Level:" + str(level) + " (MSE = " + str(tree.mse) \
          + " value:" + str(tree.value) + " feature:" + str(tree.col) + " count:" + ")"
    mse_level[level] += tree.mse
    if tree.true_branch is not None:
        printtree(tree.true_branch, level + 1, mse_level)
    if tree.false_branch is not None:
        printtree(tree.false_branch, level + 1, mse_level)
    return mse_level


########## MAIN ##########

if __name__ == '__main__':
    training_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/housing_train.txt", 'rb'),
                               delimiter=None)
    testing_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/housing_test.txt", 'rb'),
                              delimiter=None,
                              dtype=float)
    leve_of_tree = 3
    tree = train(training_data, 0, mse(training_data), leve_of_tree)
    iterator = 0
    mse_level_keys ={}
    while iterator < leve_of_tree+1:
        mse_level_keys[iterator] = 0
        iterator += 1
    mse_per_level = printtree(tree, 0, mse_level_keys)
    print "Training Error by level is:"

    for key in mse_per_level.keys():
        print "Level:" + str(key) + " Mse is:" + str(mse_per_level[key] / (2**key))

    mse_rate = {}
    iterator = 0
    for row in testing_data:
        mse_rate[iterator] = predictor(row, tree)
        iterator += 1

    mse_rate = mse_rate.values()
    print "Test Mse is" + str(sum(mse_rate) / len(mse_rate))
