from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(3,3)

    def forward(self,x):
        x = self.lr(x)
        x = F.log_softmax(x,dim=1)
        return x