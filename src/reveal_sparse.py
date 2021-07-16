import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln
import utils
from scipy import sparse
import numpy as np

#  NOT DONE...

cnn = ln.Lenet300_100_nonlinear()
standard = torch.load("../model/31_pruned.pkl")
compact = torch.load("../model/31_compact.pkl")
ncompact = OrderedDict()
for i in compact:
    ncompact[i] = torch.from_numpy(compact[i].todense())
    print(ncompact[i].shape)
    print(standard[i].shape)



# cnn.load_state_dict(ncompact)

p = cnn.parameters()
    




