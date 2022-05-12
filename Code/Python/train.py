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
from torch.nn.functional import softmax, one_hot

def brier_score_tensor(logits, categorical_labels):
    class_probs = softmax(logits, dim = 1)
    one_hot_labels =  one_hot(categorical_labels.long(), num_classes = class_probs.shape[1])
    class_probs = class_probs.detach().cpu().numpy()
    one_hot_labels = one_hot_labels.detach().cpu().numpy()
    # print(class_probs.shape, one_hot_labels.shape)
    # print(class_probs[0], one_hot_labels[0])
    return np.mean(np.sum((class_probs - one_hot_labels)**2, axis=1))
    

def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    total_loss = 0
    correct = 0
    total = 0
    total_bs = 0
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
        # print(outputs.shape, labels.shape)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # batch BS
        batch_bs = brier_score_tensor(outputs, labels)
        # print(batch_bs)
        total_bs += batch_bs
        
        
    # Averaging the loss for the whole epoch
    train_loss = total_loss / len(train_loader)
    train_acc = (correct / total) * 100.
    
    # epoch's average ME
    train_me = 100 - train_acc
    # epoch's average BS
    train_bs = total_bs/len(train_loader)
    
    return train_loss, train_acc, train_me, train_bs

def val_epoch(epoch, model, val_loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    total_bs = 0
    
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
            correct += predicted.eq(labels).sum().item()
            total  += labels.size(0)
            
            # batch BS
            batch_bs = brier_score_tensor(outputs, labels)
            # print(batch_bs)
            total_bs += batch_bs
            
        # Averaging the loss for the whole epoch
        val_loss = total_loss / len(val_loader)
        val_acc = (correct / total) * 100
        
        # epoch's average ME
        val_me = (100 - val_acc)
        # epoch's average BS
        val_bs = total_bs/len(val_loader)
         
    return val_loss, val_acc, val_me, val_bs
            
    

def run(fold, train_loader, val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['n_epochs']
    BEST_STATES_DIR= config['BEST_STATES_DIR']
    BEST_MODEL_DIR = config['BEST_MODELS_DIR']
    BEST_STATE_PATH = os.path.join(BEST_STATES_DIR, f'{fold}_best_state.pth')
    BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, f'{fold}_best_model.pth')
    diff_threshold = config['diff_threshold']
    max_patience = config['max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of fold {fold}')
        
        train_loss, train_acc, train_me, train_bs = train_epoch(epoch, model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_me, val_bs = val_epoch(epoch, model, val_loader, criterion, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        
        print('train_loss: %.5f | train_acc: %.3f | train_me: %.3f | train_bs: %.3f' % (train_loss, train_acc, train_me, train_bs))
        print('val_loss: %.5f | val_acc: %.3f | val_me: %3f | val_bs: %.3f' % (val_loss, val_acc, val_me, val_bs))
        
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
    print(cfg['BEST_STATES_DIR'])
    print(os.path.isdir(cfg['BEST_STATES_DIR']))
    print(cfg['device'])