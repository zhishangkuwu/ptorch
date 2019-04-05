import torch
from torch import nn

class MultRegress(nn.Module):
    def __init__(self):
        super(MultRegress,self).__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        out = self.linear(x)
        return out

checkpoint = torch.load('./model.pth')
model = MultRegress()
model.load_state_dict(checkpoint['net'])
print(checkpoint['net'])
model.eval()
x = torch.Tensor([2,4,8])
print(model(x).item())