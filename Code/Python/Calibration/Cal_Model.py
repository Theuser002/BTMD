import torch
import torch.nn as nn
import sys
sys.path.append('../')
import config

class CalNet(nn.Module):
    def __init__ (self, input_shape, n_classes):
        super(CalNet, self).__init__()
        self.cnn = nn.Sequential(
            # 2D Conv acts as 1D Conv
            # The correct order is: Conv > Normalization > Activation > Dropout > Pooling.
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (input_shape[0], 1), padding = (0, 0)),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Flatten(),
            nn.Linear(input_shape[1], n_classes),
            # Softmax: will be added in loss
        )
        
    def forward (self, x):
        x = self.cnn(x)
        return x