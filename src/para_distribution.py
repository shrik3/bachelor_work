import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet5 as ln
import utils

cnn = ln.Lenet5()
cnn.load_state_dict(torch.load('../model/l5.pkl')


plt.hist(cnn.parameters,bins=50,normed=true)

