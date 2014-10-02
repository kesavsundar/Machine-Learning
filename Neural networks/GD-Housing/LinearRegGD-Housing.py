__author__ = 'kesav'

import numpy as np
from sklearn import preprocessing as norm
import pylab as pl


def mean_normalize_data():

    train_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/housing_train.txt", 'rb'),
                               delimiter=None,
                               dtype=float)
    test_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/housing_test.txt", 'rb'),
                              delimiter=None,
                              dtype=float)
    # Normalize data
    xy_data = np.concatenate((train_data, test_data), axis=0)
    x_data = xy_data[:, :-1]
    #norm_data = norm.scale(x_data)
    min_max_scaler = norm.MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(x_data)
    xy_data[:, :-1] = norm_data

    # Separate training Data and testing Data

    feature_train = xy_data[:len(train_data), :-1]
    label_train = xy_data[:len(train_data), -1:]

    feature_test = xy_data[(len(train_data) + 1):, :-1]
    label_test = xy_data[(len(train_data) + 1):, -1:]

    return feature_train, label_train, feature_test, label_test


def compute_error(x, y, theta):
    predictions = x.dot(theta)
    sqerrors = (predictions - y)
    J = (1.0 / (2 * len(y))) * sqerrors.T.dot(sqerrors)
    return J[0, 0]


########## MAIN ##########
if __name__ == '__main__':

    x_train, y_train, x_test, y_test = mean_normalize_data()
    x_train_transpose = x_train.T

    theta_vector = np.zeros(shape=(len(x_train[0]), 1))
    theta_vector = np.matrix(theta_vector)
    print theta_vector.shape, x_train.shape

    # update gradient descent each time

    m = len(x_train[0])
    learning_curve = {}
    learning_erros = {}
    ll_count = 0

    while ll_count < 5:
        learning_rate = 10**(-ll_count)
        print learning_rate
        alpha_by_m = learning_rate / m
        old_theta = theta_vector
        iterator = 0
        j = {}

        while iterator < 5000:
            h_x = np.dot(x_train, old_theta)
            hx_theta = h_x - y_train

            diff_theta = np.dot(x_train_transpose, hx_theta)
            n_theta = alpha_by_m * diff_theta
            new_theta = old_theta - n_theta
            old_theta = new_theta
            j[iterator] = compute_error(x_train, y_train, old_theta)
            iterator += 1

        solution_vector = old_theta

        print "Plotting"
        pl.plot(np.arange(iterator), j.values())
        pl.xlabel('Iterations')
        pl.ylabel('Cost Function')
        pl.show()

        predicted_matrix_train = np.dot(x_train, solution_vector)
        predicted_matrix_test = np.dot(x_test, solution_vector)

        # #calculate mean squared error
        train_error = (predicted_matrix_train - y_train)
        test_error = (predicted_matrix_test - y_test)

        # #print train_error
        avg_error_train = np.dot(train_error.T, train_error)
        avg_error_test = np.dot(test_error.T, test_error)
        #
        print "Errors", avg_error_train / len(train_error), avg_error_test / len(test_error)
        learning_erros[(avg_error_train / len(train_error), avg_error_test / len(test_error))] = learning_rate
        ll_count += 1

