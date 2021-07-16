import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln

import utils

# 选取固定threshold，进行重复剪枝-训练

cnn = ln.Lenet300_100_nonlinear()
cnn.load_state_dict(torch.load('../model/31.pkl'))

utils.model_test(cnn)
utils.prune_tools.print_sparse(cnn)

print("test pruning")


th_list = [0.05,0.075,0.1,0.125]
sparse_list = []
iter_times_list = [x for x in range(10)]
acc_list = []
flag = 0
for th in th_list:
    cnn.load_state_dict(torch.load('../model/31.pkl'))
    sparse_list1 = []
    acc_list1 = []

    for i in iter_times_list:
        cnn = utils.prune_tools.prune(cnn,th)
        utils.train_model(cnn)
        acc = utils.model_test(cnn)
        sparse = utils.prune_tools.print_sparse(cnn)
        acc_list1.append(acc)
        sparse_list1.append(sparse)
        flag = flag +1
        print('Threshold: %.4f' %th,'| iter: %d' % i, '| acc: %.4f' % acc, '| sparse: %.4f' % sparse)
        acc_list.append(acc_list1)
        sparse_list.append(sparse_list1)


fg = plt.figure(figsize=(10,3))
s1 = plt.subplot(131)
s1.set_title("iter times - accuracy")
plt.plot(iter_times_list,acc_list[0],color='olive')
plt.plot(iter_times_list,acc_list[1],color = 'blue')
plt.plot(iter_times_list,acc_list[2], color='red')
plt.plot(iter_times_list,acc_list[3],color = 'black')

s2 = plt.subplot(132)
s2.set_title("iter times - sparse")
plt.plot(iter_times_list,acc_list[0],color='olive')
plt.plot(iter_times_list,acc_list[1],color='blue')
plt.plot(iter_times_list,acc_list[2],color = 'red')
plt.plot(iter_times_list,acc_list[3],color = 'black')

s3 = plt.subplot(133)
s3.set_title("sparse - accuracy")
plt.plot(sparse_list,acc_list,color = 'olive')
plt.plot(sparse_list,acc_list,color = 'blue')
plt.plot(sparse_list,acc_list,color = 'red')
plt.plot(sparse_list,acc_list, color = 'black')

plt.show()