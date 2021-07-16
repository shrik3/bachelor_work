import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import Nets.Lenet300_100 as ln

from sklearn.cluster import KMeans
import numpy as np

import utils


cnn = ln.Lenet300_100_nonlinear().cuda()
cnn.load_state_dict(torch.load('../model/31_pruned.pkl'))

utils.model_test(cnn)

kmean_list = []
bit = 4

for name,i in cnn.named_parameters():
    ## 取出数据，降维
    data = i.data.clone().view(-1).cpu().detach().numpy().reshape(-1)
    data = data[data != 0]
    ## 如果某层的数据数量小于量化位数所能表达的数量，那么这一层不进行聚类
    if data.size < 2 ** bit:
        kmean_list.append(None)
        continue

    ## 线性划分初始化聚类
    init = [x*(np.max(data)+np.min(data))/(2 ** bit) + np.min(data) for x in range(2 ** bit)]
    
    kmn = KMeans(2 ** bit,init=np.array(init).reshape(2 ** bit,1))
    kmn.fit(data.reshape((-1,1)))
    kmean_list.append(kmn)
    # print(kmn)
    # print(name,i.shape)



for i,(name,f) in enumerate(cnn.named_parameters()):
    data = f.data.clone().view(-1).cpu().detach().numpy().reshape(-1)
    data_nozero = data[data != 0].reshape((-1,1))

    # 没有非零元素/规模小于聚类数量/该层没有聚类
    if data_nozero.size == 0 or data.size < 2 ** bit or kmean_list[i] is None:
        f.kmeans_result = None
        f.kmeans_label = None
        continue

    result = data.copy()
    result[result == 0] = -1
    
    
    # tag params according to KMeans
    label = kmean_list[i].predict(data_nozero).reshape(-1)

    # 使用根据标签取出聚类中心
    new_data = np.array([kmean_list[i].cluster_centers_[x] for x in label])

    # 使用聚类中心取代原始数据
    data[data != 0] = new_data.reshape(-1)
    f.data = torch.from_numpy(data).view(f.data.shape).cuda()

    result[result != -1] = label
    f.kmeans_result = torch.from_numpy(result).view(f.data.shape).cuda()
    f.kmeans_label = torch.from_numpy(kmean_list[i].cluster_centers_).cuda()
    print("layer name:",name,"result: ",kmean_list[i].cluster_centers_)
utils.model_test(cnn)
utils.train_model_quant(cnn,bit)
utils.model_test(cnn)


for i,(name,f) in enumerate(cnn.named_parameters()):
    print(name," list:",f.kmeans_result)


torch.save(cnn.state_dict(),'../model/31_q.pkl')
