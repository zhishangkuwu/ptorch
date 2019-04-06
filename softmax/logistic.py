import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


with open('./test.txt','r') as f:
    data_list = f.readlines()
    #data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split() for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]
    label = [int(i[3])-1 for i in data_list]

data_max = np.max(data,0)
data_min = np.min(data,0)
data = (data-data_min)/data_max

data = torch.Tensor(data)
label = torch.LongTensor(label)

#transform = transforms.Compose()

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(3,3)

    def forward(self,x):
        x = self.lr(x)
        x = F.log_softmax(x,dim=1)
        return x

model = LogisticRegression()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.01)

if torch.cuda.is_available():
    model.cuda()

epoches = 200000
correct = 0

for i in range(epoches):
    if torch.cuda.is_available():
        x = Variable(data).cuda()
        y = Variable(label).cuda()
    else:
        x = Variable(data)
        y = Variable(label)

    out = model(x)
    loss = criterion(out,y)
    print_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i %10 ==0:
        print("Train Epoch:{},Loss is {}".format(i,print_loss))

torch.save(model,'./softmax_model.pth')