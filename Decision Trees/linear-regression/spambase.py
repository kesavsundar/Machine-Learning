__author__ = 'kesav'
from sklearn import preprocessing


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


def train(train_data, test_data):
    #calculate theta matrix
    feature_matrix = np.matrix(train_data[:, :-1])
    label_matrix = np.matrix(train_data[:, -1:])
    xt_x = np.dot(feature_matrix.T, feature_matrix)
    xt_y = np.dot(feature_matrix.T, label_matrix)
    inverse_martrix = np.linalg.pinv(xt_x)
    theta_matrix = np.dot(inverse_martrix, xt_y)

    #PREDICT FOR TRAIN DATA
    train_predict = np.dot(feature_matrix, theta_matrix)
    error_train = train_predict - label_matrix
    train_mse = np.dot(error_train.T, error_train) / len(label_matrix)

    #predict the values for testdata
    test_feature = np.matrix(test_data[:, :-1])
    test_label = np.matrix(test_data[:, -1:])
    predicted_matrix = np.dot(test_feature, theta_matrix)
    error = predicted_matrix - test_label
    test_mse = np.dot(error.T, error) / len(test_label)
    # predictedmatrix = []
    # rowid = 0
    # for row in error:
    #     if row >= 0 and test_label[rowid] is not 1:
    #         predictedmatrix.append(0)
    #     else:
    #         predictedmatrix.append(1)
    #     rowid += 1
    # thresmatrix = np.matrix(predictedmatrix)
    # threserror = test_label - thresmatrix.T
    # mse = (threserror.T * threserror) / len(threserror)
    return train_mse, test_mse


def tenfoldvalidation(data):
    fold_count = 0
    error = {}

    while fold_count < 10:
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

        print train_data.shape
        print test_data.shape
        error[fold_count] = train(train_data[1:, :], test_data)
        fold_count += 1
    return error

import numpy as np
########## MAIN ##########

if __name__ == '__main__':
    training_data = np.loadtxt(open("/home/kesav/MachineLearning/HW1/dataset/spambase.data", 'rb'), delimiter=",", dtype = float)
    np.set_printoptions(suppress=True)

    #Normalize the whole matrix
    data_for_normalization = training_data[:, :-1]
    training_sub_min = data_for_normalization - data_for_normalization.min(axis=0)
    training_sub_min = training_sub_min / training_sub_min.max(axis=0)
    training_data_normailized = training_data
    training_data_normailized[:, :-1] = training_sub_min

    #call tenfold validation method - getting singular matrix error
    mean_square_errors = tenfoldvalidation(training_data_normailized)
    train_error = 0.0
    test_error = 0.0
    for i in range(0, 10):
        mse1, mse2 = mean_square_errors.values()[i]
        train_error += mse1
        test_error += mse2
    print "Training error: ", train_error / 10
    print "Testing error: ", test_error / 10


