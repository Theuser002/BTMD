import sys
sys.path.append('/media/data/hungnt/work/Datasets/BTMD/Code/src')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config
import itertools

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils import brier_score_tensor
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax, one_hot
    

def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    model.to(device)
    model.train()
    
    # For loop through all batches
    all_labels = []
    all_logits = []
    for features, labels in tqdm(train_loader):
        # Move tensors to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Zero out gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)
        # print(logits.shape, labels.shape)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # batch BS
        batch_bs = brier_score_tensor(logits, labels)
        total_bs += batch_bs
        # save logits and labels to calculate AUC
        for logit, label in zip(logits,labels):
            all_labels.append(label.item())
            all_logits.append(np.array(logit.detach().cpu().numpy()))
        
            
    # epoch's avrage LL
    train_loss = total_loss / len(train_loader)
    # epoch's average acc & ME
    train_acc = (correct / total) * 100.
    train_me = 100 - train_acc
    # epoch's average BS
    train_bs = total_bs/len(train_loader)
    # epoch's average AUC
    all_labels_one_hot = one_hot(torch.Tensor(np.array(all_labels)).long())
    all_probs = softmax(torch.Tensor(np.array(all_logits)), dim = 1)
    train_auc = roc_auc_score(all_labels_one_hot, all_probs)
    
    return train_loss, train_acc, train_me, train_bs, train_auc

def val_epoch(epoch, model, val_loader, criterion, device):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    model.to(device)
    # For loop through all batches
    with torch.no_grad():
        # For loop through all batches
        all_labels = []
        all_logits = []
        for features, labels in tqdm(val_loader):
            # Move tensors to device
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Evaluation and batch loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total  += labels.size(0)
            
            # batch BS
            batch_bs = brier_score_tensor(logits, labels)
            total_bs += batch_bs
            
            # save logits and labels to calculate AUC
            for logit, label in zip(logits,labels):
                all_labels.append(label.item())
                all_logits.append(np.array(logit.detach().cpu().numpy()))
        
        # epoch's average LL
        val_loss = total_loss / len(val_loader)
        # epoch's average acc & ME
        val_acc = (correct / total) * 100
        val_me = (100 - val_acc)
        # epoch's average BS
        val_bs = total_bs/len(val_loader)
        # epoch's AUC
        all_labels_one_hot = one_hot(torch.Tensor(np.array(all_labels)).long())
        all_probs = softmax(torch.Tensor(np.array(all_logits)), dim = 1)
        val_auc = roc_auc_score(all_labels_one_hot, all_probs)
        
    return val_loss, val_acc, val_me, val_bs, val_auc
            
    

def run(fold, train_loader, val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    # print(config['device'])
    # model.to(config['device'])
    
    n_epochs = config['n_epochs']
    BEST_STATES_DIR= config['MLP_BEST_STATES_DIR']
    BEST_MODELS_DIR = config['MLP_BEST_MODELS_DIR']
    BEST_STATE_PATH = os.path.join(BEST_STATES_DIR, f'{fold}_best_state.pth')
    BEST_MODEL_PATH = os.path.join(BEST_MODELS_DIR, f'{fold}_best_model.pth')
    diff_threshold = config['diff_threshold']
    max_patience = config['max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of fold {fold}')
        
        train_loss, train_acc, train_me, train_bs, train_auc = train_epoch(epoch, model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_me, val_bs, val_auc = val_epoch(epoch, model, val_loader, criterion, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        
        print('train_loss: %.5f | train_acc: %.3f | train_me: %.3f | train_bs: %.3f | train_auc: %.3f' % (train_loss, train_acc, train_me, train_bs, train_auc))
        print('val_loss: %.5f | val_acc: %.3f | val_me: %3f | val_bs: %.3f | val_auc: %.3f' % (val_loss, val_acc, val_me, val_bs, val_auc))
        
        if val_loss == min(history['val_losses']):
            print('Lowest validation loss => saving model weights...')
            torch.save(model.state_dict(), BEST_STATE_PATH)
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


def run_no_save(fold, train_loader, val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    # print(config['device'])
    model.to(config['device'])
    n_epochs = config['n_epochs']
    BEST_STATES_DIR= config['MLP_BEST_STATES_DIR']
    BEST_MODELS_DIR = config['MLP_BEST_MODELS_DIR']
    BEST_STATE_PATH = os.path.join(BEST_STATES_DIR, f'{fold}_best_state.pth')
    BEST_MODEL_PATH = os.path.join(BEST_MODELS_DIR, f'{fold}_best_model.pth')
    diff_threshold = config['diff_threshold']
    max_patience = config['max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of fold {fold}')
        
        train_loss, train_acc, train_me, train_bs, train_auc = train_epoch(epoch, model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_me, val_bs, val_auc = val_epoch(epoch, model, val_loader, criterion, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        
        print('train_loss: %.5f | train_acc: %.3f | train_me: %.3f | train_bs: %.3f | train_auc: %.3f' % (train_loss, train_acc, train_me, train_bs, train_auc))
        print('val_loss: %.5f | val_acc: %.3f | val_me: %3f | val_bs: %.3f | val_auc: %.3f' % (val_loss, val_acc, val_me, val_bs, val_auc))
        
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
    cfg = config.config_dict
    print(cfg['MLP_BEST_STATES_DIR'])
    print(os.path.isdir(cfg['MLP_BEST_STATES_DIR']))
    print(cfg['device'])