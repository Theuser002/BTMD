import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from imblearn.over_sampling import RandomOverSampler

def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    # For loop through all batches
    for features, labels in tqdm(train_loader):
        # Move tensors to GPU
        features = features.to(device)
        labels = labels.to(device)
        
        # Zero out gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        _, hard_labes = labels.max(1)
        # print(predicted, hard_labes)
        correct += predicted.eq(hard_labes).sum().item()
        total += labels.size(0)
        
    # Averaging the loss for the whole epoch
    train_loss = total_loss / len(train_loader)
    train_acc = (correct / total) * 100.
    
    return train_loss, train_acc

def val_epoch(epoch, model, val_loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    
    # For loop through all batches
    with torch.no_grad():
        # For loop through all batches
        for features, labels in tqdm(val_loader):
            # Move tensors to GPU
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Evaluation
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            print('Val predicted: ', predicted)
            correct += predicted.eq(labels).sum().item()
            total  += labels.size(0)
        # Averaging the loss for the whole epoch
        val_loss = total_loss / len(val_loader)
        val_acc = (correct / total) * 100
         
    return val_loss, val_acc
    

def run(fold, train_loader, val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['n_calibration_epochs']
    BEST_CALIBRATION_STATES_DIR= config['BEST_CALIBRATION_STATES_DIR']
    BEST_CALIBRATION_MODEL_DIR = config['BEST_MODELS_DIR']
    BEST_STATE_PATH = os.path.join(BEST_CALIBRATION_STATES_DIR, f'{fold}_best_calibration_state.pth')
    BEST_MODEL_PATH = os.path.join(BEST_CALIBRATION_MODEL_DIR, f'{fold}_best_calibration_model.pth')
    diff_threshold = config['calibration_diff_threshold']
    max_patience = config['calibration_max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of fold {fold}')
        
        train_loss, train_acc = train_epoch(epoch, model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc = val_epoch(epoch, model, val_loader, criterion, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        
        print('train_loss: %.5f | train_acc: %.3f' % (train_loss, train_acc))
        print('val_loss: %.5f | val_acc: %.3f' % (val_loss, val_acc))
        
        if val_acc == max(history['val_accs']):
            print('Best validation accuracy => saving model weights...')
            torch.save(model.state_dict(), BEST_STATE_PATH)
        if len(history['val_accs']) > 1:
            if abs(history['val_accs'][-2] - val_acc) < diff_threshold or history['val_accs'][-2] > val_acc:
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
    print(cfg['BEST_CALIBRATION_STATES_DIR'])
    print(os.path.isdir(cfg['BEST_CALIBRATION_STATES_DIR']))
    print(cfg['device'])