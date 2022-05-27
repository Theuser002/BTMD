import os
import numpy as np
import pandas as pd
import config

from torch.nn.functional import softmax, one_hot

def brier_score_tensor(logits, categorical_labels):
    class_probs = softmax(logits, dim = 1)
    one_hot_labels =  one_hot(categorical_labels.long(), num_classes = class_probs.shape[1])
    class_probs = class_probs.detach().cpu().numpy()
    one_hot_labels = one_hot_labels.detach().cpu().numpy()
    return np.mean(np.sum((class_probs - one_hot_labels)**2, axis=1))

def make_ndarray_from_csv(fold, mode = 'None'):
    cfg = config.config_dict
    
    if mode.lower() == 'train':
        train_csv_path = os.path.join(cfg['TRAIN_CSV_DIR'], f'{fold}_train.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)    
        train_features = np.array(df_train.iloc[:,:-1])
        train_labels = np.array(df_train.iloc[:,-1])
    
        return train_features, train_labels    
    
    elif mode.lower() == 'test':
        test_csv_path = os.path.join(cfg['TEST_CSV_DIR'], f'{fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        test_features = np.array(df_test.iloc[:,:-1])
        test_labels = np.array(df_test.iloc[:,-1])

        return test_features, test_labels
    
    else:        
        train_csv_path = os.path.join(cfg['TRAIN_CSV_DIR'], f'{fold}_train.csv')
        test_csv_path = os.path.join(cfg['TEST_CSV_DIR'], f'{fold}_test.csv')
        df_train = pd.read_csv(train_csv_path, index_col = 0).fillna(0)
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        
        train_features = np.array(df_train.iloc[:,:-1])
        train_labels = np.array(df_train.iloc[:,-1])
        test_features = np.array(df_test.iloc[:,:-1])
        test_labels = np.array(df_test.iloc[:,-1])
        
        return train_features, train_labels, test_features, test_labels