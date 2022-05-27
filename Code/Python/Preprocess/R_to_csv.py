import os
import numpy as np
import pandas as pd
import pyreadr
import csv
import argparse
import time

from tqdm import tqdm
from pathlib import Path
from pyreadr import read_r

ROOT_PATH = '../../../'
DATASET_FOLDER = os.path.join(ROOT_PATH, 'Dataset')
CSV_FOLDER_PATH = os.path.join(DATASET_FOLDER, 'csv')
ANNO_FOLDER_PATH = os.path.join(DATASET_FOLDER, 'Anno')
FEATURES_FOLDER_PATH = os.path.join(DATASET_FOLDER, 'Features')
LABELS_FOLDER_PATH = os.path.join(DATASET_FOLDER, 'Labels')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = str, required = True)
    args, _ = parser.parse_known_args()
    return args
    
def R_to_csv(R_features_path, R_labels_path, R_anno_path, csv_folder_path, fold):
    train_csv_path = os.path.join(csv_folder_path, 'train', fold + '_train.csv')
    test_csv_path = os.path.join(csv_folder_path, 'test', fold + '_test.csv')
    
    # Read from R files to dataframe
    dict_features = pyreadr.read_r(R_features_path)
    df_train = dict_features['betas.train']
    df_test = dict_features['betas.test']
    df_labels = pyreadr.read_r(R_labels_path)['y']
    df_anno = pyreadr.read_r(R_anno_path)['anno']
    # df_total = pd.concat([df_train, df_test])

    # Create the linking dictionary for titles and labels
    titles_list = list(df_anno.loc[:, 'title'])
    titles_list = [','.join(label.split(',')[:-1]) for label in titles_list]
    df_link = df_anno.loc[:, ['title', 'sentrix']]
    df_link.loc[:, 'title'] = df_link['title'].replace(list(df_link['title']), titles_list)
    dict_link = {sentrix: title for sentrix, title in zip(df_link['sentrix'], df_link['title'])}
    
    # Create the label column for the train set corresponding to each sample with the correct order
    list_labels = [dict_link[id] for id in list(df_train.index)]
    df_train['label'] = list_labels
    
    # Save the csv file
    df_train.to_csv(train_csv_path, index = True)
    # Re-read from csv file to test
    df_train = pd.read_csv(train_csv_path, index_col = 0)
    
    # Create the label column for the train set corresponding to each sample with the correct order
    list_labels = [dict_link[id] for id in list(df_test.index)]
    df_test['label'] = list_labels
    
    # Save the csv file
    df_test.to_csv(test_csv_path, index = True)
    # Re-read from csv file to test
    df_test = pd.read_csv(test_csv_path, index_col = 0)
    return df_train, df_test

if __name__ == "__main__":
    start = time.time()
    # args = parse_args()
    # fold = args.fold
    
    folds = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', 
             '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', 
             '3.0', '3.1', '3.2', '3.3', '3.4', '3.5',
             '4.0', '4.1', '4.2', '4.3', '4.4', '4.5',
             '5.0', '5.1', '5.2', '5.3', '5.4', '5.5',]
    
    for fold in folds:
        # csv_file_name = fold + '.csv'
        R_features_file_name = 'betas.' + fold + '.RData'
        R_label_file_name = 'y.RData'
        R_anno_file_name = 'anno.RData'
        
        R_features_path = os.path.join(FEATURES_FOLDER_PATH, R_features_file_name)
        R_labels_path = os.path.join(LABELS_FOLDER_PATH, R_label_file_name)
        R_anno_path = os.path.join(ANNO_FOLDER_PATH, R_anno_file_name)
        # csv_path = os.path.join(CSV_FOLDER_PATH, csv_file_name)
        
        df_train, df_test = R_to_csv(R_features_path, R_labels_path, R_anno_path, CSV_FOLDER_PATH, fold)
        print(len(df_train), len(df_test))
        print(f'Fold {fold}. Time elapsed: {time.time() - start}')
    

