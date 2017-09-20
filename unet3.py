import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import process_data as pro
import pickle
import random

class Conv(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)
        self.activation = activation
        self.norm = nn.BatchNorm2d(outs)

    def forward(self,x):
        out = self.activation(self.norm(self.conv1(x)))
        out = self.activation(self.norm(self.conv2(out)))
        return out

class Up(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(ins,outs,2,stride=2)
        self.conv1 = nn.Conv2d(ins,outs,3)
        self.conv2 = nn.Conv2d(outs,outs,3)
        self.activation = activation
        self.norm = nn.BatchNorm2d(outs)

    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self,x,bridge):
        up = self.up(x)
        crop = self.crop(bridge,up.size()[2])
        out = torch.cat([up,crop],1)
        out = self.activation(self.norm(self.conv1(out)))
        out = self.activation(self.norm(self.conv2(out)))
        return out

class MulWeight(nn.Module):
    def __init__(self, weight):
        super(MulWeight, self).__init__()
        self.w0 = weight[0]
        self.w1 = weight[1]
        self.w2 = weight[2]

    def forward(self,x):
        s0 = x[:,0,:,:]
        s1 = x[:,1,:,:]
        s2 = x[:,2,:,:]
        s0 = torch.mul(s0,self.w0)
        s1 = torch.mul(s1,self.w1)
        s2 = torch.mul(s2,self.w2)
        out = torch.stack([s0,s1,s2],1)
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
        self.weight = MulWeight([0.8,1,1])
        self.mask_loss = nn.MSELoss()

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

        raw_score = self.last(up4)

        score = self.weight(raw_score)

        return F.softmax(score)


class Criterion(nn.Module):
    def __init__(self, ratio):
        super(Criterion,self).__init__()
        self.mask_coefficient = ratio[0]
        self.ncr_coefficient = ratio[1]
        self.mask_criterion = nn.MSELoss()
        self.ncr_criterion = nn.MSELoss()

    def forward(self,x,mask,ncratio):
        batch_size,features,height,width = x.size()
        mask_loss = self.mask_criterion(x,mask)
        _,pred = torch.max(x,1)
        ones = torch.ones((batch_size,1,height,width)).long()
        ones = Variable(ones.cuda())
        print(ones.size())
        c = torch.ge(pred,ones)
        n = torch.gt(pred,ones)
        print(c.size())
        c = torch.sum(torch.sum(c,2),2)
        n = torch.sum(torch.sum(n,2),2)
        print(n.size())
        ncr = torch.div(n,c)
        print(ncr.size())
        ncr_loss = self.ncr_criterion(ncr,ncratio)
        return ratio[0]*mask_loss + ratio[1]*ncr_loss

def ncr_calculator(x):
    batch_size,features,height,width = x.size()
    _,pred = torch.max(x,1)
    ones = torch.ones((batch_size,1,height,width))
    c = torch.ge(pred,ones)
    n = torch.gt(pred,ones)
    c = torch.sum(torch.sum(c,2),3)
    n = torch.sum(torch.sum(n,2),3)
    ncr = torch.div(torch.div(n,c),0.01)
    return ncr

def train():
    image, mask, ncratio = pro.load_unet3_data()

    image = image.reshape(350,1,572,572).astype(np.float32)
    mask = mask.reshape(350,3,388,388).astype(np.float32)
    ncratio = ncratio.reshape(350,1,1,1).astype(np.float32)

    net = Net()
    net.cuda()

    criterion = Criterion((1.0,0.5))
    criterion.cuda()

    optimizer = optim.Adam(net.parameters())

    learningtime = 100
    for i in range(learningtime):
        r = random.randint(0,339)
        x = Variable(torch.from_numpy(image[r:r+10,...]).cuda())
        y = Variable(torch.from_numpy(mask[r:r+10,...]).cuda())
        y_ = Variable(torch.from_numpy(ncratio[r:r+10,...]).cuda())
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out,y,y_)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pred_ncr = ncr_calculator(out)
            pred_ncr = pred_ncr.cpu()
            pred_ncr = pred_ncr.data.numpy()
            pred_ncr = pred_ncr.reshape(n)
            correct = 0
            for p,a in zip(pred_ncr, ncratio):
                if np.absolute(p-a) <= 3:
                    correct += 1
                else:
                    pass
            acc = correct / len(ncratio)

            print('===========================')
            print('%s/%s = %s' % (str(correct),str(len(ncratio)),str(acc)))
            print(loss)
            print('===========================')


if __name__ == '__main__':
    train()























