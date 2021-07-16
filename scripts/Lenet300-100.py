import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict


torch.manual_seed(1)

EPOCH = 50
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINST = False

train_data = torchvision.datasets.MNIST(
    root = '../mnist/',
    train=True,
    transform = torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST

)


test_data = torchvision.datasets.MNIST(
    root = '../mnist/',
    train = False
)

train_loader = Data.DataLoader(dataset=train_data , batch_size=BATCH_SIZE,shuffle=True)


## CHANGED INTO CUDA
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # self.l1 = nn.Linear(28*28,300)
        # self.l2 = nn.Linear(300,100)
        # self.l3 = nn.Linear(100,10)
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(True))
        self.l2 = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True))
        self.l3 = nn.Sequential(
            nn.Linear(100, 10))

    def forward(self,x):
        x = x.view(-1,self.num_flat_features(x))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()


## USING CUDA HERE 
cnn.cuda()
loss_func.cuda()

for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        # print(loss)
        # print("iter: "+ str(step))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show some info 
        iter = step+(epoch*1200)
        if step % 200 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('iter: ', iter, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy)


        


# test_output = cnn(test_x[:100])
# pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
# print(pred_y.cpu(), 'prediction number')
# print(test_y[:100].cpu().numpy(), 'real number')

torch.save(cnn.state_dict(),'../model/300-100.pkl')
    
