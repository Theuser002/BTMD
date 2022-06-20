import os
import torch

from pathlib import Path

root_dir = '/media/data/hungnt/work/Datasets/BTMD'
config_dict = {
# PATHS:
    # Data paths:
    'R_LABELS_PATH': os.path.join(root_dir, 'Dataset', 'Labels', 'y.RData'),
    'CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv'),
    'TRAIN_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'train'),
    'TEST_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'test'),
    'CALIBRATE_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'calibrate'),
    
    # Class_MLP paths:
    'MLP_BEST_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'MLP', 'Best_weights'),
    'MLP_BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'MLP', 'Best_models'),
    
    # Class_RF paths:
    'class_RF_paths': {
        'MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'RF', 'Models'),
    },
    
    # Cal_MLP paths:
    'MLP_CAL_BEST_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'MLP', 'Best_calibration_weights'),
    'MLP_CAL_BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'MLP', 'Best_calibration_models'),
    'PROBS_PICKLES_DIR': os.path.join(root_dir, 'Probs', 'pickles', 'MLP'),
    
    # Cal_RF paths:
    'cal_RF_paths':{
        'RF_CAL_BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'RF', 'Calibration_models'),
        'RF_PROBS_PICKLES_DIR': os.path.join(root_dir, 'Probs', 'pickles', 'RF'),
    },
    
# HYPERPARAMS:
    'device':('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_features': 10000,
    'n_classes': 14,
    
    # Class_MLP:
    'n_epochs': 60,
    'lr': 1e-4,
    'weight_decay': 0,
    'train_batch_size': 8,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'FIRST_TIME': True,
    'diff_threshold': 1e-4,
    'max_patience': 3,
    
    # Class_RF:
    'class_RF_params': {
        'n_epochs': 40,
        'FIRST_TIME': True
    },
    
    # Cal_MLP:
    'n_models': 5,
    'cal_n_features': 14,
    'cal_n_classes': 14,
    'cal_n_epochs': 40,
    'cal_lr': 1e-3,
    'cal_weight_decay': 0,
    'cal_train_batch_size': 8,
    'cal_val_batch_size': 16,
    'cal_test_batch_size': 16,
    'CAL_FIRST_TIME': True,
    'cal_diff_threshold': 1e-2,
    'cal_max_patience': 3,
    
    # Cal_RF:
    'cal_RF_params': {
      'n_models': 5,  
    },
}

SMOTE_config_dict = {
    'TAKE_MINOR_AT': -6,
    'BALANCED_PICKLE_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'balanced'),
    'BALANCED_TRAIN_PICKLE_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'balanced', 'train'),
    'BALANCED_TEST_PICKLE_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'balanced', 'test'),
}

important_features_config_dict = {
    'MLP_FEATURES_PATH': os.path.join(root_dir, 'Saved_features', 'MLP'),
    'RF_FEATURES_PATH': os.path.join(root_dir, 'Saved_features', 'RF')
}