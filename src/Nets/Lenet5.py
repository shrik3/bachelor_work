import torch
import torch.nn as nn 
import torch.nn.functional as f

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
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

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

