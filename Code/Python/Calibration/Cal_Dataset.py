import torch
from torch.utils.data import Dataset

class Cal_Dataset(Dataset):
    def __init__(self, probs, labels, mode = 'Train', split = None):
        self.mode = mode
        self.split = split
        self.probs = probs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prob = self.probs[idx]
        label = self.labels[idx]
        if self.mode.lower() == 'test':
            return torch.tensor(prob).float()
        else:
            return torch.tensor(prob).float(), torch.tensor(label).long()