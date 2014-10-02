from pylab import *
from sklearn import cross_validation
from sklearn.svm import SVC

from sklearn import svm
from sklearn.svm import SVC
import mnist as read

reader = read.MNIST(path='dataset/')
img_train, labels_train = reader.load_training()
img_test, labels_test = reader.load_testing()

labels_train = np.array(labels_train)
img_train = np.array(img_train)
img_train = img_train/255.0*2 - 1

print "Training size", len(img_train)


labels_test = np.array(labels_test)
img_test = np.array(img_test)
img_test = img_test/255.0*2 - 1

print "Testing size", len(img_test)

#clf = SVC(kernel="linear", C=2.8, gamma=.0073)


#scores = cross_validation.cross_val_score(clf, img, labels, cv=3)
#print scores

# clf = SVC(kernel="poly", C=2.8, gamma=.0073)


clf = svm.SVC()
clf.fit(img_train, labels_train)
# #SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
# #gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
# #shrinking=True, tol=0.001, verbose=False)
# X = [[0], [1], [2], [3]]
# Y = [1, 1, 4, 3]
# clf = svm.SVC()
# clf.fit(X, Y)

predicted_labels = clf.predict(img_test)
correct = 0.0
for p, a in zip(predicted_labels, labels_test):
    if p == a:
        correct += 1

accuracy = float(correct) / len(predicted_labels)
print accuracy
