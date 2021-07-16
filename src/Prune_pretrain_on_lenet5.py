import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
import Nets.Lenet5 as ln

torch.manual_seed(1)

EPOCH = 10
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


cnn = ln.Lenet5()
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

        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)




test_output = cnn(test_x[:100])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
# print(pred_y.cpu(), 'prediction number')
# print(test_y[:100].cpu().numpy(), 'real number')

torch.save(cnn.state_dict(),'../model/l5.pkl')
    
