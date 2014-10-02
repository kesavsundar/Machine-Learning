__author__ = 'kesav'
import numpy as np
from sklearn import preprocessing as norm
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
    #xy_data[:, :-1] = norm_data
    xy_data[:, :-1] = x_data
    # Separate training Data and testing Data

    feature_train = xy_data[:len(train_data), :-1]
    label_train = xy_data[:len(train_data), -1:]

    feature_test = xy_data[(len(train_data) + 1):, :-1]
    label_test = xy_data[(len(train_data) + 1):, -1:]

    return feature_train, label_train, feature_test, label_test


def train(train_data, test_data):
    #calculate theta matrix
    feature_matrix = np.matrix(train_data[:, :-1])
    label_matrix = np.matrix(train_data[:, -1:])
    xt_x = np.dot(feature_matrix.T, feature_matrix)
    xt_y = np.dot(feature_matrix.T, label_matrix)
    theta_matrix = np.dot(xt_x.I, xt_y)
    #print theta_matrix

    #predict the values for testdata
    test_feature = np.matrix(test_data[:, :-1])
    test_label = np.matrix(test_data[:, -1:])
    predicted_matrix_train = np.dot(feature_matrix, theta_matrix)
    predicted_matrix_test = np.dot(test_feature, theta_matrix)

    #calculate mean squared error
    train_error = (predicted_matrix_train - label_matrix)
    test_error = (predicted_matrix_test - test_label)

    print train_error
    avg_error_train = np.dot(train_error.T, train_error)
    avg_error_test = np.dot(test_error.T, test_error)
    #print avg_error_train

    return avg_error_train/ len(train_error), avg_error_test / len(test_error)


########## MAIN ##########
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = mean_normalize_data()
    np.concatenate((x_train, y_train), axis=1)
    np.concatenate((x_test, y_test), axis=1)
    train, test = train(x_train, x_test)
    print "Training error is:", train
    print "Testing error is:", test


