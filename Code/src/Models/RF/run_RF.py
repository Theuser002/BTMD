import sys
sys.path.append('/media/data/hungnt/work/Datasets/BTMD/Code/src')

import os
import config
import utils
import sklearn
import argparse
import numpy as np
import joblib

from utils import make_ndarray_from_csv, get_int_label
from sklearn.ensemble import RandomForestClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    print("Running random forest classifiers")
    args = parse_args()
    save = args.save
    cfg = config.config_dict
    
    print(f'root: {config.root_dir}')
    print(f"device: {cfg['device']}")
    print(f'save mode: {save}')
    
    # All folds
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
    inner_folds = [x for x in folds if x not in outer_folds]
    # Remove some folds that are not going to be trained
    trained_folds = folds
    
    for fold in trained_folds:
        print(f'Fold {fold}: ')
        train_features, train_labels = make_ndarray_from_csv(fold, mode = 'train')
        val_features, val_labels = make_ndarray_from_csv(fold, mode = 'test')
        train_labels_int = np.array([get_int_label(label) for label in train_labels])
        val_labels_int = np.array([get_int_label(label) for label in val_labels])
        
        clf = RandomForestClassifier(max_depth=None, criterion='log_loss', oob_score=True, random_state=42, verbose=1)
        clf.fit(train_features, train_labels_int)
        acc = clf.score(val_features, val_labels_int)
        print(f'Accuracy: {acc}')
        if save == 'save':
            print('=> Saving model...')
            # dump model to pickle/cpickle/joblib(?) file
            MODEL_PATH = os.path.join(cfg['class_RF_paths']['MODELS_DIR'], f'{fold}_model.joblib')
            joblib.dump(clf, MODEL_PATH)
          
        
    
    
    