import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import batch
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.contracting1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,
                kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU())
        self.contracting2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU())
        self.contracting3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.Conv2d(256,256,3),
            nn.ReLU())
        self.contracting4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3),
            nn.ReLU(),
            nn.Conv2d(512,512,3),
            nn.ReLU())
        self.bottom = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512,1024,3),
            nn.ReLU(),
            nn.Conv2d(1024,1024,3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,
                kernel_size=2,stride=2),)
        self.expanding1 = nn.Sequential(
            nn.Conv2d(1024,512,3),
            nn.ReLU(),
            nn.Conv2d(512,512,3),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,2,stride=2),)
        self.expanding2 = nn.Sequential(
            nn.Conv2d(512,256,3),
            nn.ReLU(),
            nn.Conv2d(256,256,3),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,2,stride=2),)
        self.expanding3 = nn.Sequential(
            nn.Conv2d(256,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,stride=2))
        self.expanding4 = nn.Sequential(
            nn.Conv2d(128,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.Conv2d(64,3,1))

    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x):
        block1 = self.contracting1(x)
        block2 = self.contracting2(block1)
        block3 = self.contracting3(block2)
        block4 = self.contracting4(block3)
        block5 = self.bottom(block4)
        bridge1 = self.crop(block4, block5.size()[2])
        block6 = torch.cat([block5,bridge1],1)
        block7 = self.expanding1(block6)
        bridge2 = self.crop(block3,block7.size()[2])
        block8 = torch.cat([block7,bridge2],1)
        block9 = self.expanding2(block8)
        bridfe3 = self.crop(block2,block9.size()[2])
        block10 = torch.cat([block9,bridfe3],1)
        block11 = self.expanding3(block10)
        bridge4 = self.crop(block1,block11.size()[2])
        block12 = torch.cat([block11,bridge4],1)
        block13 = self.expanding4(block12)
        return F.softmax(block13)


net = torch.load('model/torchmodel')

image, mask = pro.load_data_unet_torch()

image = image.reshape(350,1,572,572).astype(np.float32)
mask = mask.reshape(350,388,388,3)
n = random.randint(0,350)
image = image[n,:,:,:].reshape(1,1,572,572)

out = net(Variable(torch.from_numpy(image).cuda()))

_, pred = torch.max(out,1)
pred = pred.cpu()
pred = pred.data.numpy()
pred = pred.reshape(388,388)

cell = np.zeros((388,388))
nuc = np.zeros((388,388))

for i,x_ in enumerate(pred):
    for j,y_ in enumerate(x_):
        if y_ == 0:
            cell[i,j] = 0
            nuc[i,j] = 0
        elif y_ == 1:
            cell[i,j] = 1
            nuc[i,j] = 0
        else:
            cell[i,j] = 1
            nuc[i,j] = 1

print(len(np.where(pred==0)[0]))
print(len(np.where(pred==1)[0]))
print(len(np.where(pred==2)[0]))



fig = plt.figure(figsize=(8,8))
sub = fig.add_subplot(1,2,1)
sub.imshow(cell)
sub = fig.add_subplot(1,2,2)
sub.imshow(nuc)
plt.show()
