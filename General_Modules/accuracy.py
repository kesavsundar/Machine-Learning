__author__ = 'kesav'
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc

class Accuracy:
    def __init__(self, data, predicted):
        self.data = data
        self.predictions = predicted

    def init_accuracy(self):
        for i in range(0, len(self.predictions)):
            x = np.random.uniform(0, len(self.predictions))
            if (float(x)/len(self.predictions)) <= .06:
                self.predictions[i] = abs(self.data[i] - 1)
            else:
                self.predictions[i] = self.data[i]
        return

    def compute_accuracy(self):
        p = 0.0
        n = 0.0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        tpr = 0.0
        fpr = 0.0
        m = len(self.data)
        for i in range(0, m):
            if self.data[i] == self.predictions[i]:
                if self.data[i] == 0:
                    p += 1
                    tp += 1
                else:
                    n += 1
                    tn += 1
            else:
                if self.data[i] == 0:
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
        # roc_auc = auc(fpr, tpr)
        return tpr, fpr, accuracy

    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.data, self.predictions, pos_label=1.0)
        roc_auc = auc(fpr, tpr)
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel ('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        pl.show()
        return fpr,tpr, roc_auc

