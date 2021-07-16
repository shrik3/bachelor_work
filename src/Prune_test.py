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


sparse_list = []
threshold_list = [x*0.001+0.001 for x in range(1000)]
acc_list = []
for i in threshold_list:
    cnn.load_state_dict(torch.load("../model/31.pkl"))
    cnn = utils.prune_tools.prune(cnn,i)
    acc_list.append(utils.model_test(cnn))
    sparse_list.append(utils.prune_tools.print_sparse(cnn))
    threshold_list.append


fg = plt.figure(figsize=(10000,3))
s1 = plt.subplot(131)
s1.set_title("threshold - accuracy")
plt.plot(threshold_list,acc_list)
s2 = plt.subplot(132)
s2.set_title("sparse - accuracy")
plt.plot(sparse_list,acc_list)
s3 = plt.subplot(133)
s3.set_title("threshold - sparse")
plt.plot(threshold_list,sparse_list)
plt.show()