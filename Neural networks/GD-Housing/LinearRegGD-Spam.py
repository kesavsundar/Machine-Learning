__author__ = 'kesav'
import numpy as np
from sklearn import preprocessing as norm
import pylab as pl


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

def roc_curve(y, scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve (y, scores, pos_label=1)
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

    theta_vector = np.random.rand(len(x_train[0])).reshape(1,len(x_train[0])).T
    print theta_vector.shape
    theta_vector = np.matrix(theta_vector)

    # update gradient descent each time

    m = len(x_train[0])
    learning_curve = {}
    learning_erros = {}
    ll_count = 2

    while ll_count < 3:
        learning_rate = 10**(-ll_count)
        old_theta = theta_vector
        j = {}

        for i in range(0, 4000):
            h_x = np.dot(x_train, old_theta)
            diff_theta = np.dot(x_train.T, (h_x - y_train))
            n_theta = learning_rate * diff_theta
            old_theta = old_theta - n_theta

        solution_vector = old_theta

        predicted_matrix_train = np.dot(x_train, solution_vector)
        predicted_matrix_test = np.dot(x_test, solution_vector)
        # #calculate mean squared error
        train_error = (predicted_matrix_train - y_train)
        test_error = (predicted_matrix_test - y_test)

        # #print train_error
        avg_error_train = np.dot(train_error.T, train_error)
        avg_error_test = np.dot(test_error.T, test_error)

        #
        train_error = (avg_error_train / len(train_error))[0][0]
        test_error = (avg_error_test / len(test_error))[0][0]
        learning_erros[(train_error, test_error)] = ll_count
        learning_curve[ll_count] = j
        ll_count += 1
    # pl.plot(np.arange(iterator),
    #         learning_curve[learning_erros.get(min(learning_erros.keys()))].values())
    # pl.xlabel('Iterations')
    # pl.ylabel('Cost Function')
    # pl.show()
    if fold is 4:
        roc_curve(y_train, (x_train.dot(solution_vector)))

    print "Best Learning rate is ", 10**(- learning_erros.get(min(learning_erros.keys()))), \
        "with values: ", min(learning_erros.keys())
    return min(learning_erros.keys())

########## MAIN ##########
if __name__ == '__main__':
    xy_data = mean_normalize_data()

    errors_tenfold = tenfoldvalidation(xy_data)
    sum_train_error = 0
    sum_test_error = 0
    for i in range(0, 10):
        e1, e2 = errors_tenfold[i]
        sum_train_error += e1
        sum_test_error += e2
    print "Training Error is ", sum_train_error / len(errors_tenfold)
    print "Testing Error is ", sum_test_error / len(errors_tenfold)
