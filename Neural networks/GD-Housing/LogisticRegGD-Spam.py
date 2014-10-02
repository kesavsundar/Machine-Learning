__author__ = 'kesav'
__author__ = 'kesav'
import numpy as np
from sklearn import preprocessing as norm
import pylab as pl
import math


def tenfoldvalidation(data):
    fold_count = 0
    error = {}

    while fold_count < 10:
        print "================================================================"
        print "Fold ", fold_count+1
        start = 0
        array_length = len(data)
        chunk_size = array_length/10
        iterator = 0
        test_data = None
        train_data = np.zeros_like(data[:1, :])

        while iterator < 10:
            end = start + chunk_size
            if iterator != fold_count:
                train_data = np.concatenate((train_data, data[start:end, :]), axis=0)
            else:
                print "test data started at:", start, "ended at:", end
                test_data = data[start:end, :]
            iterator += 1
            start = start + chunk_size + 1
        error[fold_count] = train(train_data[1:, :], test_data, fold_count)
        fold_count += 1
    return error


def mean_normalize_data():

    xy_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'),
                            delimiter=",",
                            dtype = float)
    # Normalize data
    x_data = xy_data[:, :-1]
    min_max_scaler = norm.MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(x_data)
    xy_data[:, :-1] = norm_data

    return xy_data


def compute_error(predictions, y):
    sqerrors = (predictions - y)
    J = (1.0 / len(y)) * sqerrors.T.dot(sqerrors)
    return J[0, 0]


def compute_accuracy(h_x, y):
    p = 0.0
    n = 0.0
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    tpr = 0.0
    fpr = 0.0
    for i in range(0, len(h_x)):
        if h_x[i][0] >= 0.5:
            h_x[i][0] = 1
        else:
            h_x[i][0] = 0
    for i in range(0, len(h_x)):
        if h_x[i][0] == y[i][0]:
            if y[i][0] == 0:
                p += 1
                tp += 1
            else:
                n += 1
                tn += 1
        else:
            if y[i][0] == 0:
                p += 1
                fn += 1
            else:
                n += 1
                fp += 1
    if p != 0:
        tpr = tp / p
    if n != 0:
        fpr = fp / n
    accuracy = (tp + tn) / (p + n)
    return tpr, fpr, accuracy


def roc_curve(y, scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)

    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


def train(train_data, test_data, fold):
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1:]

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1:]
    theta_vector = np.zeros(shape=(len(x_train[0]), 1))
    theta_vector = np.matrix(theta_vector)

    # update gradient descent each time

    m = len(x_train[0])
    learning_curve = {}
    learning_errors = {}
    learning_accuracy = {}
    ll_count = 1
    sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))

    while ll_count < 2:
        learning_rate = 10**(-ll_count)
        alpha_by_m = learning_rate / m
        old_theta = theta_vector
        j = {}

        for i in range(0, 1000):
            h_x = np.dot(x_train, old_theta)
            diff_theta = np.dot(x_train.T, (sigmod(h_x) - y_train))
            n_theta = learning_rate * diff_theta
            old_theta = old_theta - n_theta
            #j[iterator] = compute_error(h_x, y_train, y_train, y_train)
        solution_vector = old_theta

        # #calculate mean squared error
        train_error = compute_error(sigmod(h_x), y_train)
        test_error = compute_error((sigmod(x_test.dot(solution_vector))),
                                   y_test)

        tpr, fpr, train_accuracy = compute_accuracy((sigmod(x_train.dot(solution_vector))),
                                          y_train)
        tpr, fpr, test_accuracy = compute_accuracy((sigmod(x_test.dot(solution_vector))),
                                         y_test)

        learning_accuracy[(train_accuracy, test_accuracy)] = ll_count
        learning_errors[(train_error, test_error)] = ll_count
        learning_curve[ll_count] = j

        ll_count += 1
    if fold is 4:
        roc_curve(y_train, (sigmod(x_train.dot(solution_vector))))
    # pl.plot(np.arange(iterator),
    #         learning_curve[learning_erros.get(min(learning_erros.keys()))].values())
    # pl.xlabel('Iterations')
    # pl.ylabel('Cost Function')
    #pl.show()

    print "Best Learning rate is ", 10**(- learning_errors.get(min(learning_errors.keys()))), \
         "with mse values:", min(learning_errors.keys())
    print "Accuracy is ", max(learning_accuracy.keys())
    return min(learning_errors.keys()), max(learning_accuracy.keys())

########## MAIN ##########
if __name__ == '__main__':
    xy_data = mean_normalize_data()
    result_tenfold = tenfoldvalidation(xy_data)
    sum_train_error = 0
    sum_test_error = 0
    sum_train_accuracy = 0
    sum_test_accuracy = 0

    for i in range(0, 9):
        (e1, e2), (a1, a2) = result_tenfold[i]
        sum_train_error += e1
        sum_test_error += e2
        sum_train_accuracy += a1
        sum_test_accuracy += a2

    print "\n\nTraining Error is ", sum_train_error / 9
    print "Training Accuracy is ", sum_train_accuracy / 9
    print "Testing Error is ", sum_test_error / 9
    print "Testing Accuracy is ", sum_test_accuracy / 9