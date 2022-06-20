import os
import sys
sys.path.append('../')
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import imblearn

from utils import make_ndarray_from_csv, get_int_label
from collections import Counter, OrderedDict
from imblearn.over_sampling import SMOTE, RandomOverSampler

cfg = config.SMOTE_config_dict

def SMOTE_to_array(fold, mode):
    features, labels = make_ndarray_from_csv(fold, mode)
    
    labels_count = Counter(labels)
    sorted_labels_count = OrderedDict(labels_count.most_common())
    
    X = features
    y = np.array([get_int_label(label) for label in labels])

    minor_classes = list(sorted_labels_count.keys())[cfg['TAKE_MINOR_AT']:]
    
    minor_indexes = []
    i = 0
    for label in labels:
        if label in minor_classes:
            minor_indexes.append(i)
        i += 1
    minor_fetures = np.array([features[i] for i in minor_indexes])
    minor_labels = np.array([labels[i] for i in minor_indexes])
    major_features = np.array([features[i] for i in range(len(features)) if i not in minor_indexes])
    major_labels = np.array([labels[i] for i in range(len(features)) if i not in minor_indexes])
    
    X_minor = minor_fetures
    y_minor = minor_labels
    X_major = major_features
    y_major = major_labels
    
    smote = SMOTE(sampling_strategy = "auto", random_state = 42)
    new_X_minor, new_y_minor = smote.fit_resample(X_minor, y_minor)
    
    new_X = np.append(X_major, new_X_minor, axis = 0)
    new_y = np.append(y_major, new_y_minor)

    return new_X, new_Y
    
def save_SMOTE_array(fold, mode):
    balanced_X, balanced_Y = SMOTE_to_array(fold, mode)
    

if __name__ == "__main__":
    print('SMOTE.py running...')