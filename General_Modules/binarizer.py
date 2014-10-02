__author__ = 'kesav'
from sklearn.preprocessing import binarize
import numpy as np
class Binarize:
    def continuous_to_binary(self, predictions, threshold):
        return binarize(predictions, threshold, False)
