import os
import numpy as np
import sys
import torch
sys.path.append('../')
import config
import pandas as pd
import pickle

from Model import TestNet
from torch.nn.functional import softmax
from tqdm import tqdm

def dump_probs ():
    cfg = config.config_dict
    PICKLES_DIR = cfg['PICKLES_DIR']
    for i in range(1, 6):
        all_probs = []
        all_labels = []
        probs_pickle_filename = f'{i}.0_probs.pickle'
        labels_pickle_filename = f'{i}.0_labels.pickle'
        probs_pickle_filepath = os.path.join(PICKLES_DIR, probs_pickle_filename)
        labels_pickle_filepath = os.path.join(PICKLES_DIR, labels_pickle_filename)
        for j in tqdm(range(1, 6)):
            # Take innerfold
            fold = f'{i}.{j}'
            # print(fold)
            test_csv_path = os.path.join(cfg['TEST_CSV_DIR'], f'{fold}_test.csv')
            df_test = pd.read_csv(test_csv_path, index_col = 0).fillna(0)
            test_features = np.array(df_test.iloc[:,:-1])
            test_labels = np.array(df_test.iloc[:,-1])
            
            # [Not optimized] Getting probs and labels for each sample and append to respective global lists
            in_features = cfg['n_features']
            model = TestNet(in_features, cfg['n_classes'])
            BEST_STATE_PATH = os.path.join(cfg['BEST_STATES_DIR'], f'{fold}_best_state.pth')
            model.load_state_dict(torch.load(BEST_STATE_PATH))
            # print(len(test_labels))
            for feature, label in zip(test_features, test_labels):
                feature = torch.Tensor(feature).float()
                feature.to(cfg['device'])
                prob = softmax(model(feature), dim = 0)
                prob = prob.detach().cpu().numpy()
                all_probs.append(prob)
                all_labels.append(label)
        
        with open (probs_pickle_filepath, 'wb') as handle:
            pickle.dump(all_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open (labels_pickle_filepath, 'wb') as handle:
            pickle.dump(all_labels, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return

def load_probs (outer_fold):
    cfg = config.config_dict
    PICKLES_DIR = cfg['PICKLES_DIR']
    probs_pickle_filename = f'{outer_fold}_probs.pickle'
    labels_pickle_filename = f'{outer_fold}_labels.pickle'
    probs_pickle_filepath = os.path.join(PICKLES_DIR, probs_pickle_filename)
    labels_pickle_filepath = os.path.join(PICKLES_DIR, labels_pickle_filename)
    
    with open(probs_pickle_filepath, 'rb') as handle:
        probs = pickle.load(handle)
    with open(labels_pickle_filepath, 'rb') as handle:
        labels = pickle.load(handle)
    return probs, labels
    

if __name__ == "__main__":
    dump_probs()
    probs, labels = load_probs('1.0')
    print(len(labels))