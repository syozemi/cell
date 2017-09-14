import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import batch
import process_data as pro
import pickle
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
        self.conv_1_8 = Conv(1,8)
        self.conv_8_16 = Conv(8,16)
        self.conv_16_32 = Conv(16,32)
        self.conv_32_64 = Conv(32,64)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.up1 = Up(64,32)
        self.up2 = Up(32,16)
        self.up3 = Up(16,8)
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

        bottom = self.conv_32_64(pool3)

        up1 = self.up1(bottom,block3)

        up2 = self.up2(up1,block2)

        up3 = self.up3(up2,block1)

        return F.softmax(self.last(up3))

image, mask, tmask = pro.load_data_unet_torch2()

image = image.reshape(350,1,284,284).astype(np.float32)
mask = mask.reshape(350,196,196,3).astype(np.float32)
mask = np.swapaxes(mask,1,3)
mask = np.swapaxes(mask,2,3)

net = Net()
net.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

learningtime = 100
for i in range(learningtime):
    r = np.random.choice([True,False],size=(len(image),),p=[0.3,0.7])
    image_tmp = np.compress(r,image,axis=0)
    mask_tmp = np.take(r,mask,axis=0)
    x = Variable(torch.from_numpy(image_tmp).cuda())
    y = Variable(torch.from_numpy(mask_tmp).cuda())
    optimizer.zero_grad()
    out = net(x)
    loss = criterion(out,y)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        _, pred = torch.max(out,1) #(n,388,388)のVariable, 一枚は、0,1,2でできた配列
        pred = pred[1]
        pred = pred.cpu()
        pred = pred.data.numpy()
        correct = len(np.where(pred==tmask)[0])
        acc = correct / tmask.size
        print('======================')
        print(loss.data)
        print('accuracy: %s' % str(acc))
        print(str(i)+'/'+str(learningtime))
        print('======================')

torch.save(net, 'model/torchmodel_unet2')

























