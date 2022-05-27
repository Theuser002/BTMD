import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
import config

from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

from torch.nn.functional import one_hot
import itertools
if __name__ == "__main__":
    cfg = config.config_dict
    print(config.root_dir)
    print(cfg['TRAIN_CSV_DIR'])
    for i in range(1, 6):
        print(i)