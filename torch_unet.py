import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import process_data as pro
import pickle
import random
import os
from collections import defaultdict



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


def train():
    image, mask, nmask = pro.load_data_unet_torch()
    image = image.reshape(350,1,572,572).astype(np.float32)
    mask = mask.reshape(350,3,388,388).astype(np.float32)
    net = Net()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters())
    learningtime = 10
    for i in range(learningtime):
        r = random.randint(0,329)
        tmp_image = image[r:r+20,...]
        tmp_mask = mask[r:r+20,...]
        x = Variable(torch.from_numpy(tmp_image).cuda())
        y = Variable(torch.from_numpy(tmp_mask).cuda())
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            _, pred = torch.max(out,1) #(n,388,388)のVariable, 一枚は、0,1,2でできた配列
            pred = pred.cpu()
            pred = pred.data.numpy()
            pred.reshape(20,388,388)
            tmp_nmask = nmask[r:r+20,...]
            correct = len(np.where(pred==tmp_nmask)[0])
            acc = correct / tmp_nmask.size
            print('======================')
            print(loss)
            print(acc)
            print(str(i)+'/'+str(learningtime))
            print('======================')

    files = os.listdir('model')

    for i in range(len(files)):
        if os.path.exists('model/torchmodel_unet%s' % str(i)):
            pass
        else:
            filename = 'torchmodel_unet%s' % str(i)
            torch.save(net, 'model/%s' % filename)
            print('saved model as %s' % filename)
            break

def eval(model_path,test_data,answers):
    #test_data_setは(n,1,572,572)の配列
    #answersは(1,n)の配列
    #prediction
    net = torch.load(model_path)
    net.cuda()
    out = net(Variable(torch.from_numpy(test_data).cuda()))
    _, pred = torch.max(out,1) #(n,388,388)で要素は0,1,2の配列
    pred = pred.cpu()
    pred = pred.data.numpy()
    ncpred = []
    for x in pred:
        c = len(np.where(x>=1)[0])
        n = len(np.where(x==2)[0])
        ncr = (c / n) // 0.01
        ncpred.append(int(ncr))
    
    print(type(ncpred))
    print(type(answers))
    print(len(ncpred))
    print(len(answers))

    #check
    correct = 0
    diff_dict = defaultdict(int)
    for p,a in zip(ncpred, answers):
        diff = np.absolute(p-a)
        print(type(diff))
        if diff <= 5:
            correct += 1
        else:
            pass
        diff_dict[diff] += 1
    data_num = len(answers)
    accuracy = correct / data_num
    print('%s / %s = %s' % (str(correct),str(data_num),str(accuracy)))
    print(diff_dict)


if __name__ == '__main__':
    train()


