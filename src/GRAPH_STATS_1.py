import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import Nets.Lenet5 as ln
import Nets.Lenet300_100 as ln31

import utils

## PARA DISTRIBUTION ON ORIGINAL LENET

cnn = ln.Lenet5()
cnn.load_state_dict(torch.load('../model/l5.pkl'))

cnn_31 = ln31.Lenet300_100_nonlinear()
cnn_31.load_state_dict(torch.load('../model/31.pkl'))


para_l5=np.array([])
for i in cnn.parameters():
    t = i.data.view(-1 ).numpy()
    para_l5 = np.concatenate((para_l5,t))


para_31=np.array([])
for i in cnn_31.parameters():
    t = i.data.view(-1 ).numpy()
    para_31 = np.concatenate((para_31,t))


para_l5p=np.array([])
cnn.load_state_dict(torch.load('../model/l5_pruned.pkl'))
for i in cnn.parameters():
    t = i.data.view(-1 ).numpy()
    para_l5p= np.concatenate((para_l5p,t))



fg = plt.figure(figsize=(10000,2))
p1 = plt.subplot(121)
p1.set_title("Parameters Distribution on LeNet-5")
plt.hist(para_l5,bins=500,normed=True)

p2 = plt.subplot(122)
p2.set_title("Parameters Distribution on LeNet-300-100")
plt.hist(para_31,bins=500,normed=True)



plt.show()

# plt.hist(cnn.parameters,bins=50,normed=True)