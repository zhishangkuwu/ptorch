import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_tf,download=True)

train_loader = DataLoader(train_dataset,batch_size=20,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=20,shuffle=True)
