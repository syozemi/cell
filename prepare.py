import os
import matplotlib.pyplot as plt
import process_data as pro
import random
import pickle
import numpy as np
import time
import batch
import cv2 as cv
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.conv_1_8 = Conv(1,8)
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
        block1 = self.conv_1_8(x)
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

net1 = torch.load('model/torchmodel')
net2 = torch.load('model/torchmodel')
net1.cuda()
net2.cuda()

folders = os.listdir('data')

for ii,folder in enumerate(folders):
    print(folder)
    ipath = 'data/%s/image572' % folder
    with open(ipath,'rb') as f:
        image = pickle.load(f)
    image = image.reshape(50,1,572,572).astype(np.float32)
    image1,image2 = image[:25,...],image[25:,...]

    with torch.cuda.device(0):
        out1 = net1(Variable(torch.from_numpy(image1).cuda()))
        _,out1 = torch.max(out1,1)
        out1 = out1.cpu()
        out1 = out1.data.numpy()
    with torch.cuda.device(1):
        out2 = net2(Variable(torch.from_numpy(image2).cuda()))
        _,out2 = torch.max(out2,1)
        out2 = out2.cpu()
        out2 = out2.data.numpy()
    out = np.vstack(out1,out2).reshape(-1,388,388)
    cell,nuc = np.zeros((50,388,388)),np.zeros((50,388,388))
    for i,x in enumerate(out):
        for j,y in enumerate(x):
            for k,z in enumerate(y):
                if z == 0:
                    cell[i,j,k] = 0
                    nuc[i,j,k] = 0
                elif z == 1:
                    cell[i,j,k] = 1
                    nuc[i,j,k] = 0
                else:
                    cell[i,j,k] = 1
                    nuc[i,j,k] = 1

    c = []
    n = []
    for x in cell:
        c.append(cv.resize(x,(572,572)))
    for x in nuc:
        n.append(cv.resize(x,(572,572)))
    c = np.array(c)
    n = np.array(n)
    last = []
    last.append(image.reshape(50,572,572))
    last.append(c)
    last.append(n)
    last = np.array(last)
    last = np.swapaxes(last,0,1)
    pro.save(last, 'data/%s' % folder, 'wnet')
    print('done')

