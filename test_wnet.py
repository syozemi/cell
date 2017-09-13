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


class Conv(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)
        self.activation = activation

    def forward(self,x):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        return out

class Up(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(ins,outs,2,stride=2)
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)
        self.activation = activation

    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self,x,bridge):
        up = self.up(x)
        crop = self.crop(bridge,up.size()[2])
        out = torch.cat([up,crop],1)
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.activation = F.relu
        self.conv_3_8 = Conv(3,8)
        self.conv_8_16 = Conv(8,16)
        self.conv_16_32 = Conv(16,32)
        self.conv_32_64 = Conv(32,64)
        self.conv_64_128 = Conv(64,128)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.up1 = Up(128,64)
        self.up2 = Up(64,32)
        self.up3 = Up(32,16)
        self.up4 = Up(16,8)
        self.last = nn.Conv2d(8,3,1)

    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x):
        block1 = self.conv_3_8(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_8_16(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_16_32(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_32_64(pool3)
        pool4 = self.pool4(block4)

        bottom = self.conv_64_128(pool4)

        up1 = self.up1(bottom,block4)

        up2 = self.up2(up1,block3)

        up3 = self.up3(up2,block2)

        up4 = self.up4(up3,block1)

        return F.softmax(self.last(up4))



image,mask,img = pro.load_data_wnet()
print(image.shape)
print(mask.shape)
print(img.shape)

net = torch.load('model/torchmodel_wnet')

net = Net()
net.cuda()

n = random.randint(0,350)

image = image[n,...]
mask = mask[n,...]
img = img[n,...]
print(image.shape)
print(mask.shape)
print(img.shape)

mid_c = image[1,...]
mid_n = image[2,...]

act_c = mask[:,:,1] + mask[:,:,2]
act_n = mask[:,:,2]

out = net(Variable(torch.from_numpy(image.reshape(1,3,572,572)).cuda()))
pred = out.cpu()
pred = pred.data.numpy()
print(pred.shape)
pred = pred.reshape(3,388,388)

'''
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
'''

fig = plt.figure(figsize=(8,8))
#sub = fig.add_subplot(3,3,1)
#sub.imshow(img,cmap='gray')
sub = fig.add_subplot(3,3,2)
sub.imshow(act_c,cmap='gray')
sub = fig.add_subplot(3,3,3)
sub.imshow(act_n,cmap='gray')
sub = fig.add_subplot(3,3,5)
sub.imshow(mid_c,cmap='gray')
sub = fig.add_subplot(3,3,6)
sub.imshow(mid_n,cmap='gray')
sub = fig.add_subplot(3,3,8)
sub.imshow(pred[1],cmap='gray')
sub = fig.add_subplot(3,3,9)
sub.imshow(pred[2],cmap='gray')

plt.show()

