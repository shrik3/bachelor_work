import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln

import utils
print("testing file:")

f = open('./data/repeat_prune_dyn.txt','a')
f.write('file test successed \n')
f.close()


cnn = ln.Lenet300_100_nonlinear()
cnn.load_state_dict(torch.load('../model/31.pkl'))

acc_list=[]
sparse_list=[]
th_list = []


acc_list.append(utils.model_test(cnn))
sparse_list.append(utils.prune_tools.print_sparse(cnn))

for iter in range(0,10):
    th = 0.25 - 0.01*iter
    print("starting Pruning with threshold ",th)
    cnn = utils.prune_tools.prune(cnn,th)
    print("starting fine tuning :")
    utils.train_model_fine_tuning(cnn,5)

    acc_list.append(utils.model_test(cnn))
    sparse_list.append(utils.prune_tools.print_sparse(cnn))
    th_list.append(th)

print("complete ! ----dyn thr on l31----- ")
print("acc list: ",acc_list)
print("sparse list: ",sparse_list)
print("th list: ",th_list)

f = open('./data/repeat_prune_dyn.txt','a')
f.write("\nacc list: "+str(acc_list))
f.write("\nsparse list: "+str(sparse_list))
f.write("\nth list: "+str(th_list))
f.write("\n--------------\n")
f.close()



# torch.save(cnn.state_dict(),'../model/31_pruned_repeat.pkl')

# utils.model_test(cnn)
# utils.prune_tools.print_sparse(cnn)



