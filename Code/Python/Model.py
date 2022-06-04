import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__ (self, in_features, n_classes):
        super(TestNet, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.densenet = nn.Sequential(
            # nn.Linear(self.in_features, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(self.in_features, n_classes),
            # nn.Dropout(p = 0.3)
            # Softmax: will be added in loss
        )
        
    def forward (self, x):
        x = self.densenet(x)
        return x