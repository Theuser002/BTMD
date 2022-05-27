import os
import torch

from pathlib import Path

# Paths
root_dir = '/media/data/hungnt/work/Datasets/BTMD'
config_dict = {
    'CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv'),
    'TRAIN_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'train'),
    'TEST_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'test'),
    'CALIBRATE_CSV_DIR': os.path.join(root_dir, 'Dataset', 'csv', 'calibrate'),
    'BEST_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'Best_weights'),
    'BEST_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'Best_models'),
    'BEST_CALIBRATION_STATES_DIR': os.path.join(root_dir, 'Saved_models', 'Best_calibration_weights'),
    'BEST_CALIBRATION_MODELS_DIR': os.path.join(root_dir, 'Saved_models', 'Best_calibration_models'),
    'R_LABELS_PATH': os.path.join(root_dir, 'Dataset', 'Labels', 'y.RData'),
    'PICKLES_DIR': os.path.join(root_dir, 'Pickles'),
    
    'device':('cuda' if torch.cuda.is_available() else 'cpu'),
    'n_features': 10000,
    'n_classes': 91,
    'n_epochs': 30,
    'n_calibration_epochs': 30,
    'lr': 1e-4,
    'weight_decay': 0,
    'train_batch_size': 8,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'FIRST_TIME': True,
    'FIRST_CALIBRATION_TIME': True,
    'diff_threshold': 1e-3,
    'max_patience': 3,
    'calibration_diff_threshold': 1e-3,
    'calibration_max_patience': 3
}