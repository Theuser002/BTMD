import os
import numpy as np
import sys
import torch
sys.path.append('../')
import config
import pandas as pd
import pickle
import utils

from Model import TestNet
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import make_ndarray_from_csv

def dump_probs ():
    cfg = config.config_dict
    PICKLES_DIR = cfg['PICKLES_DIR']
    device = cfg['device']
    in_features = cfg['n_features']
    n_classes = cfg['n_classes']
    for i in range(1, 6):
        outer_fold = f'{i}.0'
        all_probs = []
        all_labels = []
        probs_pickle_filename = f'{outer_fold}_combined_probs.pickle'
        labels_pickle_filename = f'{outer_fold}_combined_labels.pickle'
        probs_pickle_filepath = os.path.join(PICKLES_DIR, probs_pickle_filename)
        labels_pickle_filepath = os.path.join(PICKLES_DIR, labels_pickle_filename)
        features, labels = make_ndarray_from_csv(outer_fold, mode = 'train')
        for feature, label in zip(tqdm(features), labels):
            feature = torch.Tensor(feature).float()
            feature.to(device)
            combined_prob = []
            for j in range(1, 6):
                # Take innerfold
                inner_fold = f'{i}.{j}'
                model = TestNet(in_features, n_classes)
                BEST_STATE_PATH = os.path.join(cfg['BEST_STATES_DIR'], f'{inner_fold}_best_state.pth')
                model.load_state_dict(torch.load(BEST_STATE_PATH))
                logit = model(feature)
                prob = softmax(logit, dim = 0).detach().cpu().numpy()
                combined_prob.append(prob)
            combined_prob = np.array(combined_prob)
            all_probs.append(combined_prob)
            all_labels.append(label)
            # print('Cheese')
        with open (probs_pickle_filepath, 'wb') as handle:
            pickle.dump(all_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open (labels_pickle_filepath, 'wb') as handle:
            pickle.dump(all_labels, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return

def load_probs (outer_fold):
    print(f'Loading classification probabilities and labels from pickle file for fold {outer_fold}')
    cfg = config.config_dict
    PICKLES_DIR = cfg['PICKLES_DIR']
    probs_pickle_filename = f'{outer_fold}_combined_probs.pickle'
    labels_pickle_filename = f'{outer_fold}_combined_labels.pickle'
    probs_pickle_filepath = os.path.join(PICKLES_DIR, probs_pickle_filename)
    labels_pickle_filepath = os.path.join(PICKLES_DIR, labels_pickle_filename)
    
    with open(probs_pickle_filepath, 'rb') as handle:
        probs = pickle.load(handle)
    with open(labels_pickle_filepath, 'rb') as handle:
        labels = pickle.load(handle)
    return probs, labels



def cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    
    for probs, labels in tqdm(cal_train_loader):
        # Move tensors to device
        probs, labels = probs.to(device), labels.to(device)
        
        # Zero out gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(probs)
        print(logits.detach().cpu().numpy().shape, labels.detach().cpu().numpy().shape)
        loss = criterion(logits, labels)
         
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate batch's loss
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    # Epoch's average loss
    train_loss = total_loss / len(cal_train_loader)
    train_acc = (correct / total) * 100
    
    return train_loss, train_acc

def cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for probs, labels in tqdm(cal_val_loader):
            # Move tensors to device
            probs, labels = probs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(probs)
            loss = criterion(logits, labels)
            
            # Calculate batch's loss
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        val_loss = total_loss / len(cal_val_loader)
        val_acc = (correct / total) * 100
    
    return val_loss, val_acc

def cal_run (fold, cal_train_loader, cal_val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['cal_n_epochs']
    CAL_BEST_STATES_DIR = config['CAL_BEST_STATES_DIR']
    CAL_BEST_MODELS_DIR = config['CAL_BEST_MODELS_DIR']
    CAL_BEST_STATES_PATH = os.path.join(CAL_BEST_STATES_DIR, f'{fold}_best_cal_state.pth')
    CAL_BEST_MODELS_PATH = os.path.join(CAL_BEST_MODELS_DIR, f'{fold}_best_cal_model.pth')
    diff_threshold = config['cal_diff_threshold']
    max_patience = config['cal_max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of the inner folds corresponding to  outer fold {fold}')
        train_loss, train_acc = cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc = cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)

        print('train_loss: %.5f | train_acc: %.3f' %(train_loss, train_acc))
        print('val_loss: %.5f | val_acc: %.3f' %(val_loss, val_acc))
        
        if val_loss == min(history['val_losses']):
            print('Lowest validation loss => saving model weights...')
            torch.save(model.state_dict(), CAL_BEST_STATES_PATH)
        if len(history['val_losses']) > 1:
            if abs(history['val_losses'][-2] - val_loss) < diff_threshold or history['val_losses'][-2] < val_loss:
                patience = patience + 1
                print(f'Patience increased to {patience}')
                if patience == max_patience:
                    print('Early stopping.')
                    break
            else:
                patience = 0
        print('---------------------------------------------')
    return max(history['val_accs'])

def cal_run_no_save (fold, cal_train_loader, cal_val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['cal_n_epochs']
    CAL_BEST_STATES_DIR = config['CAL_BEST_STATES_DIR']
    CAL_BEST_MODELS_DIR = config['CAL_BEST_MODELS_DIR']
    CAL_BEST_STATES_PATH = os.path.join(CAL_BEST_STATES_DIR, f'{fold}_best_cal_state.pth')
    CAL_BEST_MODELS_PATH = os.path.join(CAL_BEST_MODELS_DIR, f'{fold}_best_cal_model.pth')
    diff_threshold = config['cal_diff_threshold']
    max_patience = config['cal_max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of the inner folds corresponding to  outer fold {fold}')
        train_loss, train_acc = cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc = cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)

        print('train_loss: %5.f | train_acc: %.3f' %(train_loss, train_acc))
        print('val_loss: %5.f | val_acc: %.3f' %(val_loss, val_acc))

        if len(history['val_losses']) > 1:
            if abs(history['val_losses'][-2] - val_loss) < diff_threshold or history['val_losses'][-2] < val_loss:
                patience = patience + 1
                print(f'Patience increased to {patience}')
                if patience == max_patience:
                    print('Early stopping.')
                    break
            else:
                patience = 0
        print('---------------------------------------------')
    return max(history['val_accs'])

if __name__ == "__main__":
    # dump_probs()
    probs, labels = load_probs('1.0')
    probs = np.array(probs)
    labels = np.array(labels)
    print(probs.shape, labels.shape)
    print(probs[0], type(probs[0]), probs[0].shape)