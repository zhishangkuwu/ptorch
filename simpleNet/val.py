from pytorch_pra.simpleNet import Net
from pytorch_pra.simpleNet.data import test_loader
import torch
from torch import nn
from torch.autograd import Variable

if __name__ == '__main__':
    model = Net.SimpleNet()
    checkpoint = torch.load('./simpleNet.pth')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    val_cor = 0
    val_loss = 0


    for data,label in test_loader:
        data = data.view(data.size()[0],-1)
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            label = Variable(label).cuda()
        else:
            data = Variable(data)
            label = Variable(label)

        output = model(data)
        output = torch.argmax(output,1)
        acc = (output==label).sum()
        #print(acc)
        val_cor += acc.item()

    print(val_cor/(len(test_loader)*20))


