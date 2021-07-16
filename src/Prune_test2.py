import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln

import utils


cnn = ln.Lenet300_100_nonlinear()
cnn.load_state_dict(torch.load('../model/31.pkl'))

utils.model_test(cnn)
utils.prune_tools.print_sparse(cnn)

print("test pruning")


cnn.load_state_dict(torch.load("../model/31.pkl"))
cnn = utils.prune_tools.prune(cnn,i)
