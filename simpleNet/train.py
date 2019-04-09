import pytorch_pra.simpleNet.Net as model
from pytorch_pra.simpleNet.data import train_loader
import torch
from torch import nn,optim
from torch.autograd import Variable
import os

epoches = 10


model = model.SimpleNet()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

if os.path.exists('./simpleNet.pth') and os.path.getsize('./simpleNet.pth'):
    checkpoint = torch.load('./simpleNet.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    pass

for i in range(epoches):
    for data,label in train_loader:
        data = data.view(data.size()[0],-1)
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            label = Variable(label).cuda()
        else:
            data = Variable(data)
            label = Variable(label)
        out = model(data)
        loss = criterion(out,label)
        print_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print('epoch is {},the loss is {}'.format(i,print_loss))

    state = {'net':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(state, './simpleNet.pth')



