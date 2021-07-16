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
import re

import utils
import seaborn

## QUANT TEST STATS.

cnn = ln31.Lenet300_100_nonlinear()

cnn.load_state_dict(torch.load('../model/31_q.pkl'))
tags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for name,i in cnn.named_parameters():
    if(re.search('bias',name)):
        continue
    print(name)
    t = i.data.view(-1 ).numpy()
    g = seaborn.countplot(t[t!=0])
    g.set_xticklabels(tags)
    title = "Quant16 on LeNet-300-100, Layer: "+ name 
    g.set_title(title)
    plt.show()





cnn.load_state_dict(torch.load('../model/31_q2.pkl'))
tags = [1,2,3,4]
for name,i in cnn.named_parameters():
    if(re.search('bias',name)):
        continue
    print(name)
    t = i.data.view(-1 ).numpy()
    g = seaborn.countplot(t[t!=0])
    g.set_xticklabels(tags)
    title = "Quant4 on LeNet-300-100, Layer: "+ name 
    g.set_title(title)
    plt.show()








# fg = plt.figure(figsize=(10000,2))
# p1 = plt.subplot(121)
# p1.set_title("Parameters Distribution on LeNet-5 after Pruning")
# plt.hist(para_l5[abs(para_l5)>0],bins=1000)

# p2 = plt.subplot(122)
# p2.set_title("Parameters Distribution on LeNet-300-100 after Pruning")
# plt.hist(para_31[abs(para_31)>0],bins=1000)



# plt.show()
