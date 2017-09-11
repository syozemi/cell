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

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Contract,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
            kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size),

    def forward(self,x):
        x = nn.ReLU(conv1(x))
        x = nn.ReLU(conv2(x))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1_8 = Convolution(1,8,3,0)
        self.conv_8_16 = Convolution(8,16,3,0)
        self.conv_16_32 = Convolution(16,32,3,0)
        self.conv_32_64 = Convolution(32,64,3,0)
        self.conv_64_128 = Convolution(64,128,3,0)
        self.conv_128_64 = Convolution(128,64,3,0)
        self.conv_64_32 = Convolution(64,32,3,0)
        self.conv_32_16 = Convolution(32,16,3,0)
        self.conv_16_8 = Convolution(16,8,3,0)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.transpose1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.transpose2 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.transpose3 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.transpose4 = nn.ConvTranspose2d(16,8,2,stride=2)
        self.last = nn.Conv2d(8,3,3)


    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x):
        b1 = conv_1_8(x)
        b2 = conv_8_16(pool1(b1))
        b3 = conv_16_32(pool2(b2))
        b4 = conv_32_64(pool3(b3))
        y = conv_64_128(pool4(b4))
        
        return F.softmax(block13)

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
    r = random.randint(0,320)
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
