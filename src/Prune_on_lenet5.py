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
cnn.load_state_dict(torch.load('../model/l5.pkl'))

print("Original Model :")
utils.model_test_l5(cnn)
print(utils.prune_tools.print_sparse(cnn))

print("++++++++++++++")
print("starting Pruning with threshold 0.2")

cnn = utils.prune_tools.prune(cnn,0.2)
print("Model After Pruning")
utils.model_test_l5(cnn)
print(utils.prune_tools.print_sparse(cnn))

print("fine tuning :")
utils.train_model_fine_tuning(cnn,5)

print("Model After Fine Tuning")
utils.model_test_l5(cnn)
print(utils.prune_tools.print_sparse(cnn))

# torch.save(cnn.state_dict(),'../model/l5_pruned.pkl')

# utils.model_test(cnn)
# utils.prune_tools.print_sparse(cnn)



