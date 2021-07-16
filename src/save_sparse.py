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


cnn = ln.Lenet300_100_nonlinear()
cnn.load_state_dict(torch.load('../model/31_pruned.pkl'))

p = cnn.parameters()
    
compact = OrderedDict()
ncompact = OrderedDict()

for i in cnn.state_dict():
    print(i)

    a = cnn.state_dict()[i].numpy()
    compact[i] = sparse.csr_matrix(a)
    print(a.size)
    print(compact[i].size)

torch.save(compact,"../model/31_compact.pkl")



