import os
import torch
import numpy as np
import pandas as pd
import config
import Dataset
import Model
import pyreadr
import train_calibration
import torch.nn as nn

from Dataset import CNS
from torch.utils.data import DataLoader
from Model import TestNet
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils import softXEnt

# Method will get the features of each sample and the respective classification probabilites (predicted by the respective fold model) and add them to lists. Return those lists
def get_fold_probs(model, test_loader, device):
    saved_features = []
    saved_probs = []
    
    model = model.to(device)
    
    with torch.no_grad():
        for features in tqdm(test_loader):
            # print(features.shape)
            features = features.to(device)
            logits = model(features)
            features = features.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            # iterate through the batch to take single sample
            for feature, logit in zip (features, logits):
                class_prob = nn.Softmax(dim = 0)(torch.tensor(logit)).detach().cpu().numpy()
                # print(class_prob, class_prob.shape, np.sum(class_prob))
                saved_features.append(feature)
                saved_probs.append(class_prob)
    return saved_features, saved_probs

def get_label_code (label, label_codes):
    if label in label_codes.keys():
        return label_codes[label]
    else:
        return -1    
    
if __name__ == "__main__":
    cfg = config.config_dict
    
    # Fold definition
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5']
    outer_folds = ['1.0', '2.0', '3.0', '4.0', '5.0']
    inner_folds = [x for x in folds if x not in outer_folds]
    # Remove some folds that are not going to be trained
    trained_folds = [['1.1', '1.2', '1.3', '1.4', '1.5'],
                     ['2.1', '2.2', '2.3', '2.4', '2.5'],
                     ['3.1', '3.2', '3.3', '3.4', '3.5'],
                     ['4.1', '4.2', '4.3', '4.4', '4.5'],
                     ['5.1', '5.2', '5.3', '5.4', '5.5']]
    
    # Make the label code dictionary
    df_labels = pyreadr.read_r(cfg['R_LABELS_PATH'])['y']
    all_labels = list(df_labels['y'])
    label_codes = {label:index for index, label in enumerate(np.unique(all_labels))}
    
    for i in range(len(outer_folds)):
        outer_fold = outer_folds[i]
        subfolds = trained_folds[i]
        print(f'Calibration process - outer fold {outer_fold}')
        # Make test set (from outer fold's test set)
        test_csv_path = os.path.join(cfg['TEST_CSV_DIR'], f'{outer_fold}_test.csv')
        df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
        test_features = np.array(df_test.iloc[:,:-1])
        test_labels = np.array(df_test.iloc[:,-1])
        encoded_test_labels = np.array([get_label_code(label, label_codes) for label in test_labels])
        test_dataset = CNS(test_features, encoded_test_labels, mode = 'Val')
        test_loader = DataLoader(test_dataset, batch_size = cfg['val_batch_size'], shuffle = True, num_workers = 3)
        
        # Make calibration set
        cal_features = []
        cal_probs = []    
        # Get the calibration dataset from subfolds
        for fold in subfolds:
            print(f'Calibration process - Subfold {fold} of outer fold {outer_fold}')
            sub_cal_csv_path = os.path.join(cfg['TEST_CSV_DIR'], f'{fold}_test.csv')
            df_sub_cal = pd.read_csv(sub_cal_csv_path, index_col = 0).fillna(0)
            sub_cal_features = np.array(df_sub_cal.iloc[:,:-1])
            sub_cal_labels = np.array(df_sub_cal.iloc[:,-1])
            
            encoded_sub_cal_labels = np.array([get_label_code(label, label_codes) for label in sub_cal_labels])
            
            # Make sub_calibration dataset with respect to each subfold
            sub_cal_dataset = CNS(sub_cal_features, encoded_sub_cal_labels, mode = 'Test')
            sub_cal_loader = DataLoader(sub_cal_dataset, batch_size = cfg['test_batch_size'], shuffle = False, num_workers = 3)
            
            # Load the subfold's model
            in_features = cfg['n_features']
            subfold_model = TestNet(in_features, cfg['n_classes'])
            BEST_STATE_PATH = os.path.join(cfg['BEST_STATES_DIR'], f'{fold}_best_state.pth')
            subfold_model.load_state_dict(torch.load(BEST_STATE_PATH))
            
            # save the subfold's result
            fold_features, fold_probs = get_fold_probs(subfold_model, sub_cal_loader, cfg['device'])
            cal_features = cal_features + fold_features
            cal_probs = cal_probs + fold_probs
            
        cal_features, cal_probs = np.array(cal_features), np.array(cal_probs)
        
        cal_dataset = CNS(cal_features, cal_probs, mode='Cal')
        cal_loader = DataLoader(cal_dataset, batch_size = cfg['train_batch_size'], shuffle = True, num_workers = 2)
        
        print(cal_probs.max())
        
        
        # Init model object
        # in_features = cfg['n_features']
        # model = TestNet(in_features, cfg['n_classes'])
        # if cfg['FIRST_CALIBRATION_TIME'] == False:
        #     # Load model based on fold
        #     BEST_CALIBRATION_STATE_PATH = os.path.join(cfg['BEST_CALIBRATION_STATES_DIR'], f'{fold}_best_calibration_state.pth')
        #     model.load_state_dict(torch.load(BEST_CALIBRATION_STATE_PATH))
        
        # # Define training and validating hyperparams
        # criterion = CrossEntropyLoss(weight=None)
        # optimizer = Adam(model.parameters(), lr = cfg['lr'], weight_decay = cfg['weight_decay'])
        
        # best_cal_accs = train_calibration.run(outer_fold, cal_loader, test_loader, model, criterion, optimizer, cfg)
        # print(best_cal_accs)
            
        
        
        