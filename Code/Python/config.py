import os
import torch

from pathlib import Path

root_dir = '/media/data/hungnt/work/Datasets/BTMD'
config_dict = {
    # Paths:
        # General:
    'R_LABELS_PATH': os.path.join(root_dir, 'Dataset', 'Labels', 'y.RData'),
    'PICKLES_DIR': os.path.join(root_dir, 'Pickles'),
        # Classifier model:
    'CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv'),
    'TRAIN_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'train'),
    'TEST_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'test'),
    'BEST_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'Best_weights'),
    'BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'Best_models'),
        # Calibration model:
    'CALIBRATE_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'calibrate'),
    'CAL_BEST_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'Best_calibration_weights'),
    'CAL_BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'Best_calibration_models'),
    
    # Hyperparams
        # General:
    'device':('cuda' if torch.cuda.is_available() else 'cpu'),
        # Classifier model:
    'n_features': 10000,
    'n_classes': 91,
    'n_epochs': 60,
    'lr': 1e-4,
    'weight_decay': 0,
    'train_batch_size': 8,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'FIRST_TIME': True,
    'diff_threshold': 1e-4,
    'max_patience': 3,
        # Calibration model:
    'n_models': 5,
    'cal_n_features': 91,
    'cal_n_classes': 91,
    'cal_n_epochs': 40,
    'cal_lr': 1e-3,
    'cal_weight_decay': 0,
    'cal_train_batch_size': 8,
    'cal_val_batch_size': 16,
    'cal_test_batch_size': 16,
    'CAL_FIRST_TIME': True,
    'cal_diff_threshold': 1e-2,
    'cal_max_patience': 3
}