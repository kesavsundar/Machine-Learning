__author__ = 'kesav'
import numpy as np

######### CLASSES ##########


class DecisionNode:
    def __init__(self,col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.true_branch = tb
        self.false_branch = fb

def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    from math import log
    log2 = lambda x:log(x) / log(2)
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent


def divideset(rows, column, value):
    split_function = lambda row:row[column] >= value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def buildtree(rows, level):
    if len(rows) == 0:
        return DecisionNode()
    current_score = entropy(rows)
    print "Entropy for the above rows: " + str(current_score)
    if level <= 2:
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(rows[0]) - 1
        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = divideset(rows, col, value)
                p = float(len(set1)) / len(rows)
                gain = current_score - p * entropy(set1) - (1 - p) * entropy(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        if best_gain > 0:
            truebranch = buildtree(best_sets[0], level + 1)
            falsebranch = buildtree(best_sets[1], level + 1)
            # And build out new nodes
            return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                                tb=truebranch, fb=falsebranch)
    return DecisionNode(results=uniquecounts(rows))


def classify(observation, tree=None):
    import operator
    if tree.results is not None:
        return max(tree.results.iteritems(), key=operator.itemgetter(1))[0]
    else:
        v = observation[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        else:
            if v == tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        return classify(observation, branch)


def normalize(data):
    data_min = data - data.min(axis=0)
    data = data_min / data_min.max(axis=0)
    return data


def train(train_data, test_data):
    tree = buildtree(train_data, 0)
    predicted_values = {}
    iterator = 0
    count_1s = 0
    print count_1s
    for row in test_data:
        predicted_values[iterator] = classify(row, tree)
        value = predicted_values[iterator]
        if value == row[len(row) - 1]:
            count_1s += 1
        iterator += 1

    print count_1s
    return float(count_1s) / len(test_data)


def tenfoldvalidation(data):
    fold_count = 0
    mean_accuracy = {}
    print "In ten fold validation"
    while fold_count < 10:
        start = 0
        array_length = len(data)
        chunk_size = array_length / 10
        iterator = 0
        test_data = None
        train_data = np.zeros_like(data[:1, :])
        while iterator < 10:
            end = start+chunk_size
            if iterator != fold_count:
                train_data = np.concatenate((train_data, data[start:end, :]), axis=0)
            else:
                test_data = data[start:end, :]
            iterator += 1
            start = start+chunk_size+1

        mean_accuracy[fold_count] = train(train_data[1:, :], test_data)
        print "accuracy for the fold:", fold_count, "  is:", mean_accuracy[fold_count]
        fold_count += 1
    return mean_accuracy


if __name__ == '__main__':
    training_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                               delimiter=",",
                               dtype=float)
    print "Finished reading data!"
    np.set_printoptions(suppress=True)
    # Build the tree
    error_rate = tenfoldvalidation(training_data)
    print error_rate
    print float (sum(error_rate.values()) / len(error_rate))