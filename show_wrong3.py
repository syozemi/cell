import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import process_data as pro
from tqdm import tqdm


class Conv(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3,padding=1)
        self.conv2 = nn.Conv2d(outs,outs,3,padding=1)
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
        self.conv1 = nn.Conv2d(ins,outs,3,padding=1)
        self.conv2 = nn.Conv2d(outs,outs,3,padding=1)
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
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.up1 = Up(64,32)
        self.up2 = Up(32,16)
        self.up3 = Up(16,8)
        self.last = nn.Conv2d(8,3,1)
        self.weight = MulWeight([0.1,1,1])

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

        raw_score = self.last(up3)

        score = self.weight(raw_score)

        return F.softmax(score)



def eval(seed):
    #imageは(n,1,360,360)
    #answersは(1,n)

    image, answers, num_mask = pro.load_unet2_data(seed,mode=1)

    image = image.reshape(-1,1,360,360).astype(np.float32)

    net = torch.load('model/unet3/%d' % seed)
    net.cuda()

    ncpred = []
    mask_pred = np.array([]).reshape(0,360,360)

    for i in tqdm(range(10)):
        start = i * 20
        img = image[start:start+20]
        out = net(Variable(torch.from_numpy(img).cuda()))
        _, pred = torch.max(out,1) #(n,360,360)で要素は0,1,2の配列
        pred = pred.cpu()
        pred = pred.data.numpy()
        pred = pred.reshape(-1,360,360)
        mask_pred = np.vstack((mask_pred,pred))
        for x in pred:
            c = len(np.where(x>=1)[0])
            n = len(np.where(x==2)[0])
            ncr = n / c
            ncpred.append(ncr)

    num_of_ans, num_of_correct,prob, diff_dict = pro.validate_ncr(answers, ncpred)

    print(num_of_ans)
    print(num_of_correct)
    print(prob)
    print(diff_dict)

    wrong_list = pro.wrong(answers, ncpred)
    print(wrong_list)

    for i,x in enumerate(wrong_list):
        print(answers[x], ncpred[x])
        img = image[x]
        pred = mask_pred[x]
        msk = num_mask[x]
        img = img.reshape(360,360)
        pred = pred.reshape(360,360)
        msk = msk.reshape(360,360)
        plt.imsave(fname='wrong/unet3/%d/%dimage.png' % (seed, i), arr=img, cmap='gray', format='png')
        plt.imsave(fname='wrong/unet3/%d/%dmask.png' % (seed, i), arr = msk, cmap='gray', format='png')
        plt.imsave(fname='wrong/unet3/%d/%dprediction.png' % (seed, i), arr = pred, cmap='gray', format='png')
        fig = plt.figure(figsize=(8,8))
        sub = fig.add_subplot(1,3,1)
        sub.imshow(img,cmap='gray')
        sub = fig.add_subplot(1,3,2)
        sub.imshow(msk,cmap='gray')        
        sub = fig.add_subplot(1,3,3)
        sub.imshow(pred,cmap='gray')
    plt.show()


if __name__ == '__main__':
    pro.make_dir('wrong')
    pro.make_dir('wrong/unet3')
    print('input seed')
    seed = int(input())
    pro.make_dir('wrong/unet3/%d' % seed)
    eval(seed)