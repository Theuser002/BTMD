import os
import pandas as pd
import numpy as np
import torch
import pyreadr
import config
import Dataset
import time
import train
import argparse

from Dataset import CNS
from torch.utils.data import DataLoader
from Model import TestNet
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils import make_ndarray_from_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type = str, default='no_save')
    args, _ = parser.parse_known_args()
    return args

def get_label_code (label, label_codes):
    if label in label_codes.keys():
        return label_codes[label]
    else:
        return -1 

if __name__ == "__main__":
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
    
    df_labels = pyreadr.read_r(cfg['R_LABELS_PATH'])['y']
    all_labels = list(df_labels['y'])
    # print(all_labels)
    label_codes = {label:index for index, label in enumerate(np.unique(all_labels))}
    # print(label_codes)
    
    performances = {}
    
    # Train the inner folds
    for fold in trained_folds:
        # Read from csv to dataframe
        train_features, train_labels, test_features, test_labels = make_ndarray_from_csv(fold)
        
        # Encode the labels
        encoded_train_labels = np.array([get_label_code(label, label_codes) for label in train_labels])
        encoded_test_labels = np.array([get_label_code(label, label_codes) for label in test_labels])
        
        # Create datasets and Dataloader
        train_dataset = CNS(train_features, encoded_train_labels, mode = 'Train')
        test_dataset = CNS(test_features, encoded_test_labels, mode = 'Val')
        train_loader = DataLoader(train_dataset, batch_size = cfg['train_batch_size'], shuffle = True, num_workers = 2)
        test_loader = DataLoader(test_dataset, batch_size = cfg['val_batch_size'], shuffle = False, num_workers = 3)
        
        # Init model object
        in_features = cfg['n_features']
        model = TestNet(in_features, cfg['n_classes'])
        if cfg['FIRST_TIME'] == False:
            # Load model based on fold
            BEST_STATE_PATH = os.path.join(cfg['BEST_STATES_DIR'], f'{fold}_best_state.pth')
            model.load_state_dict(torch.load(BEST_STATE_PATH))
        
        # Define training and validating hyperparams
        criterion = CrossEntropyLoss(weight=None)
        optimizer = Adam(model.parameters(), lr = cfg['lr'], weight_decay = cfg['weight_decay'])
        if save == 'save':
            print('Running with save mode')
            best_accs = train.run(fold, train_loader, test_loader, model, criterion, optimizer, cfg)
        else:
            print('Running in no save mode')
            best_accs = train.run_no_save(fold, train_loader, test_loader, model, criterion, optimizer, cfg)
        performances[f'{fold}'] = best_accs
        
    
    # Summary after all folds
    # print(performances)
    avg = sum(performances.values())/len(performances.values())
    # print(avg)
    # Write the performance values to a text file
    f = open('../evaluation.txt', 'w')
    f.write('Perf: ' + str(performances) + '\nAvg: ' + str(avg))
    f.close()
    print('Avg: ', sum(performances.values())/len(performances.values()))
        

        