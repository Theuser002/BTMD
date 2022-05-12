import torch
from torch.utils.data import Dataset

class CNS (Dataset):
    def __init__(self, features, labels, mode = 'Train', split = None):
        self.mode = mode
        self.split = split
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.mode == 'Test' or self.mode == 'test':
            return torch.tensor(feature).float()
        # Calibration mode for training calibration model, with the labels as an tensor of class_probabilities (read the paper for better interpretation)
        elif self.mode == 'Cal' or self.mode == 'cal':
            return torch.tensor(feature).float(), torch.tensor(label).float()
        else:
            return torch.tensor(feature).float(), torch.tensor(label).long()