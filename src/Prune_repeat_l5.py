import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln
import Nets.Lenet5 as l5

import utils
print("testing file:")

f = open('./data/repeat_prune_l5.txt','a')
f.write('file test successed \n')
f.close()




cnn = l5.Lenet5()

for nth in range(0,10):
    th = 0.2+nth*0.01
    cnn.load_state_dict(torch.load('../model/l5.pkl'))

    acc_list=[]
    sparse_list=[]

    acc_list.append(utils.model_test_l5(cnn))
    sparse_list.append(utils.prune_tools.print_sparse(cnn))

    print("starting Pruning with threshold ",th)
    for iter in range(0,20):
        print("++++++++++++++")
        print("iteration: ",iter)

        cnn = utils.prune_tools.prune(cnn,th)
        utils.train_model_fine_tuning(cnn,5)

        acc_list.append(utils.model_test_l5(cnn))
        sparse_list.append(utils.prune_tools.print_sparse(cnn))



    print("complete ! --------- ")
    print("using th: ", th)
    print("acc list: ",acc_list)
    print("sparse list: ",sparse_list)

    f = open('./data/repeat_prune_l5.txt','a')
    f.write("USING TH: " + str(th))
    f.write("\nacc list: "+str(acc_list))
    f.write("\nsparse list: "+str(sparse_list))
    f.write("\n--------------\n")
    f.close()



# torch.save(cnn.state_dict(),'../model/l5_pruned_repeat.pkl')

# utils.model_test(cnn)
# utils.prune_tools.print_sparse(cnn)



