import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

#线性回归
####需要拟合的方程 Y = 0.9 + 0.5*x + 3*x**2 + 2.4*x**3

def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

w_target = torch.Tensor([[0.5],[3],[2.4]])
b_target = torch.Tensor([0.9])

def f(x):
    return x.mm(w_target)+b_target[0]

def get_batch(batch_size=20):
    data = torch.randn(batch_size)
    x = make_features(data)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(),Variable(y).cuda()
    else:
        return Variable(x),Variable(y)

class MultRegress(nn.Module):
    def __init__(self):
        super(MultRegress,self).__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        out = self.linear(x)
        return out

if torch.cuda.is_available():
    model = MultRegress().cuda()
else:
    model = MultRegress()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=1e-3)

epoch = 0
while(True):
    batch_x,batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output,batch_y)
    print_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    print("epoch:{},loss:{}".format(epoch,print_loss))
    if print_loss < 1e-3:
        break
state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
torch.save(state,'./model.pth')





