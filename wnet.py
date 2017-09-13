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

image,mask,tmask = pro.load_data_wnet()

image.reshape(350,3,572,572).astype(np.float32)
mask = mask.reshape(350,388,388,3).astype(np.float32)
mask = np.swapaxes(mask,1,3)
mask = np.swapaxes(mask,2,3)

net = Net()
net.cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(net.parameters())
learningtime = 2000
for i in range(learningtime):
    r = random.randint(0,300)
    imagee = image[r:r+30,...]
    maskk = mask[r:r+30,...]
    x = Variable(torch.from_numpy(imagee).cuda())
    y = Variable(torch.from_numpy(maskk).cuda())
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
        tmaskk = tmask[r:r+30,...]
        correct = len(np.where(pred==tmaskk)[0])
        acc = correct / tmaskk.size
        print('======================')
        print(loss.data)
        print('accuracy: %s' % str(acc))
        print(str(i)+'/'+str(learningtime))
        print('======================')

torch.save(net, 'model/torchmodel_wnet')

