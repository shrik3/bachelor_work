import torch
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # LeNet takes 32*32 , Mnist is 28*28 , add padding = 2 
        self.conv1 = nn.Conv2d(1,6,5,padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(16*5*5 , 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = f.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = f.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = x.view(-1,16*5*5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))

        x = self.fc3(x)
        return x

cnn = CNN()
cnn.cuda()
cnn.load_state_dict(torch.load('../model/testmodel.pkl'))

test_data = torchvision.datasets.MNIST(
    root = '../mnist/',
    train = False
)

## CHANGED INTO CUDA
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].cuda()
test_output = cnn(test_x[:100])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y.cpu(), 'prediction number')
print(test_y[:100].cpu().numpy(), 'real number')

