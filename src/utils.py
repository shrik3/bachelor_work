import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet300_100 as ln


class prune_tools():
    def prune(model,threshold):
        for i in model.parameters():
            i.data[torch.abs(i) < threshold] = 0
        return model
    
    def watch(model):
        print(model.parameters())
        return model


    def grad_prune(model):
        for i in model.parameters():
            mask = i.clone()
            mask[mask != 0] = 1
            i.grad.data.mul_(mask)

    def print_sparse(model):
        result = []
        total_num = 0
        total_sparse = 0
        print("-----------------------------------")
        print("Layer sparse")
        for name,f in model.named_parameters():
            num = f.view(-1).shape[0]
            total_num += num
            sparse = torch.nonzero(f).shape[0]
            total_sparse+= sparse
            print("\t",name,(sparse)/num)
            result.append((sparse)/num)
        total = total_sparse/total_num
        return total

    def cal_sparse_simple(model):
        result = []
        total_num = 0
        total_sparse = 0
        for name,f in model.named_parameters():
            num = f.view(-1).shape[0]
            total_num += num
            sparse = torch.nonzero(f).shape[0]
            total_sparse+= sparse
            result.append((sparse)/num)
        total = total_sparse/total_num
        return total



def model_test(model):
    model.cuda()
    test_data = torchvision.datasets.MNIST(
    root = '../mnist/',
    train = False
    )
    ## CHANGED INTO CUDA
    test_x = torch.unsqueeze(test_data.data, dim=2).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.targets[:2000].cuda()
    test_output = model(test_x.cuda())

            # !!!!!!!! Change in here !!!!!!!!! #
    pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print( '| test accuracy: %f' % accuracy)
    return accuracy

def model_test_l5(model):
    model.cuda()
    test_data = torchvision.datasets.MNIST(
    root = '../mnist/',
    train = False
    )
    ## CHANGED INTO CUDA
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.targets[:2000].cuda()
    test_output = model(test_x.cuda())

            # !!!!!!!! Change in here !!!!!!!!! #
    pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print( '| test accuracy: %f' % accuracy)
    return accuracy

def train_model(model,EPOCH=1):
    torch.manual_seed(1)

    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MINST = False

    train_data = torchvision.datasets.MNIST(
        root = '../mnist/',
        train=True,
        transform = torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MINST

    )
    train_loader = Data.DataLoader(dataset=train_data , batch_size=BATCH_SIZE,shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()


## USING CUDA HERE 
    model.cuda()
    loss_func.cuda()

    for epoch in range(EPOCH):
        for step, (b_x,b_y) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = model(b_x)
            loss = loss_func(output,b_y)
            # print(loss)
            # print("iter: "+ str(step))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if step % 50 == 0:
            #     print('step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy())

def get_acc(outputs,label):
    _,data = torch.max(outputs,dim=1)
    return torch.mean((data.float()==label.float()).float()).item()

def train_model_fine_tuning(model,EPOCH=1):
    torch.manual_seed(1)

    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MINST = False

    train_data = torchvision.datasets.MNIST(
        root = '../mnist/',
        train=True,
        transform = torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MINST

    )
    train_loader = Data.DataLoader(dataset=train_data , batch_size=BATCH_SIZE,shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()


## USING CUDA HERE 
    model.cuda()
    loss_func.cuda()

    for epoch in range(EPOCH):
        for step, (b_x,b_y) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = model(b_x)
            loss = loss_func(output,b_y)
            # print(loss)
            # print("iter: "+ str(step))
            optimizer.zero_grad()
            loss.backward()
            prune_tools.grad_prune(model)
            optimizer.step()

            # if step % 50 == 0:
            #     print('step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy())

def train_model_quant(model,bit):
    train_data = torchvision.datasets.MNIST(
        root = '../mnist/',
        train=True,
        transform = torchvision.transforms.ToTensor(),
        download=False

    )
    train_loader = Data.DataLoader(dataset=train_data , batch_size=50,shuffle=True)
    lossfunc = torch.nn.CrossEntropyLoss().cuda()
    lr = 0.001
    for _ in range(1):
        for a,(data,label) in enumerate(train_loader):
            data,label = data.cuda(),label.cuda()
            model.zero_grad()
            outputs = model(data)
            loss = lossfunc(outputs,label)
            loss.backward()

            for name,i in model.named_parameters():
                if i.kmeans_result is None:
                    continue
                for x in range(2 ** bit):
                    grad = torch.sum(i.grad.detach()[i.kmeans_result == x])
                    i.kmeans_label[x] += -lr * grad.item()
                    i.data[i.kmeans_result == x] = i.kmeans_label[x].item()
            # if a % 100 == 0:
                # print(a,get_acc(outputs,label))

