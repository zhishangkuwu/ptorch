import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(2,3)

    def forward(self,x):
        x = self.lr(x)
        x = F.log_softmax(x,dim=1)
        return x

model = torch.load('./softmax_model.pth')

with open('./test.txt','r') as f:
    data_list = f.readlines()
    #data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split() for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]
    label = [int(i[3])-1 for i in data_list]

data_max = np.max(data,0)
data_min = np.min(data,0)
data = (data-data_min)/data_max

correct = 0
for i in range(len(label)):
    d = torch.Tensor(np.array(data[i]).reshape(1,-1))
    la = torch.Tensor(label[i])
    out = model(d)
    print(out)
    out = torch.argmax(out,1)
    print(out)
    #print(label[i])
    print(out.data[0]==label[i])
    if(out.data[0]==label[i]):
        correct += 1
    #print(correct)
print(correct/len(label))


