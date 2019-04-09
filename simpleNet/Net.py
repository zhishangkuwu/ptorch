import torch
from torch.autograd import Variable
from torch import nn,optim

in_dim = 784
n_hidden_1 =200
n_hidden_2 = 100
out_dim = 10

class SimpleNet(nn.Module):
    def __init__(self,in_dim=784,n_hidden_1=200,n_hidden_2=100,out_dim=10):
        super(SimpleNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

