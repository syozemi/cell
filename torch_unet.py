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
    def __init__(self, ins, outs):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)

    def forward(self,x):
        out = nn.ReLU(conv1(x))
        out = nn.ReLU(conv2(out))
        return out

class Expand(nn.Module):
    def __init__(self, ins, middles, outs):
        super(Expand,self).__init__()
        self.conv1 = nn.Conv2d(ins,middles,3)
        self.conv2 = nn.Conv2d(middles,middles,3)
        self.transpose = nn.ConvTranspose2d(middles,outs,2,stride=2)

    def forward(self,block,x):
        out = torch.cat([block,x],1)
        out = nn.ReLU(conv1(out))
        out = nn.ReLU(conv2(out))
        out = nn.ReLU(transpose(out))
        return out

class Bottom(nn.Module):
    def __init__(self, ins, outs):
        super(Bottom,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)
        self.transpose = nn.ConvTranspose2d(outs,ins,2,stride=2)

    def forward(self,x):
        out = nn.ReLU(conv1(out))
        out = nn.ReLU(conv2(out))
        out = nn.ReLU(transpose(out))
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1_8 = Conv(1,8)
        self.conv_8_16 = Conv(8,16)
        self.conv_16_32 = Conv(16,32)
        self.conv_32_64 = Conv(32,64)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.bottom = Bottom(64,128)
        self.expand1 = Expand(64,128,64)
        self.expand2 = Expand(128,64,32)
        self.expand3 = Expand(64,32,16)
        self.expand4 = Expand(32,16,8)
        self.conv1 = nn.Conv2d(16,8,3)
        self.conv2 = nn.Conv2d(8,8,3)
        self.last = nn.Conv2d(8,3,1)


    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x):
        b1 = conv_1_8(x)
        b2 = conv_8_16(pool1(b1))
        b3 = conv_16_32(pool2(b2))
        b4 = conv_32_64(pool3(b3))
        out = bottom(pool4(b4))
        block1 = crop(b4,out.size()[2])
        out = expand2(block,out)
        block2 = crop(b3,out.size()[2])
        out = expand3(block,out)
        block3 = crop(b2,out.size()[2])
        out = expand4(block,out)
        block4 = crop(b1,out.size()[2])
        out = F.ReLU(conv1(out))
        out = F.ReLU(conv2(out))
        out = last(out)
        return F.softmax(out)

image, mask = pro.load_data_unet_torch()

image = image.reshape(350,1,572,572).astype(np.float32)
mask = mask.reshape(350,388,388,3).astype(np.float32)
mask = np.swapaxes(mask,1,3)
mask = np.swapaxes(mask,2,3)

net = Net()
net.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
learningtime = 1000
for i in range(learningtime):
    r = random.randint(0,348)
    imagee = image[r:r+2,:,:,:]
    maskk = mask[r:r+2,:,:,:]
    x = Variable(torch.from_numpy(imagee).cuda())
    y = Variable(torch.from_numpy(maskk).cuda())
    optimizer.zero_grad()
    out = net(x)
    loss = criterion(out,y)
    loss.backward()
    optimizer.step()
    print(loss)
    print(str(i)+'/'+str(learningtime))

torch.save(net, 'model/torchmodel')
