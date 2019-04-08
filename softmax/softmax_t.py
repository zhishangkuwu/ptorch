import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from pytorch_pra.softmax import model

model = model.LogisticRegression()
checkpoint = torch.load('./softmax_model.pth')
model.load_state_dict(checkpoint['net'])

model.eval()
print("hello world")
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
    out = torch.argmax(out,1)
    if(out.data[0]==label[i]):
        correct += 1
print(correct/len(label))


