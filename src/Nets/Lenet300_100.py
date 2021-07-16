import torch
import torch.nn as nn 
import torch.nn.functional as f

class Lenet300_100_nonlinear(nn.Module):
    def __init__(self):
        super(Lenet300_100_nonlinear,self).__init__()
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


class Lenet300_100(nn.Module):
    def __init__(self):
        super(Lenet300_100,self).__init__()
        self.l1 = nn.Linear(28*28,300)
        self.l2 = nn.Linear(300,100)
        self.l3 = nn.Linear(100,10)
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
