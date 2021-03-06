import torch
from torch import nn,optim
from torch.autograd import Variable
from pytorch_pra.softmax import model
import numpy as np
import os



with open('./test.txt','r') as f:
    data_list = f.readlines()
    data_list = [i.split() for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]
    label = [int(i[3])-1 for i in data_list]

data_max = np.max(data,0)
data_min = np.min(data,0)
data = (data-data_min)/data_max

data = torch.Tensor(data)
label = torch.LongTensor(label)

#transform = transforms.Compose()



model = model.LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.01)




if torch.cuda.is_available():
    model.cuda()

epoches = 200000
correct = 0

if os.path.getsize('./softmax_model.pth'):
    #print(torch.load('./softmax_model.pth'))
    checkpoint = torch.load('./softmax_model.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    pass

for epoch in range(epoches):
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

    if epoch %10 ==0:
        print("Train Epoch:{},Loss is {}".format(epoch,print_loss))

    if epoch % 100 == 0:
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, './softmax_model.pth')
