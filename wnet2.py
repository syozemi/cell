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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time


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
        self.weight = MulWeight([0.8,1,1])

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


class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.activation = F.relu
        self.conv_3_8 = Conv(3,8)
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
        self.weight = MulWeight([0.8,1,1])

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


def train(seed):
    #imageは入力データ
    #maskは教師用データ
    #num_maskはマスクの数字版で、validationに使う

    start_time = time.time()

    net = torch.load('model/unet2/%s' % str(seed))
    net.cpu()
    image, mask, num_mask = pro.load_unet2_data(seed,mode=0)
    image = image.reshape(850,1,360,360).astype(np.float32)
    mask = mask.reshape(850,3,360,360).astype(np.float32)

    validation_log = []

    tmp_x = Variable(torch.from_numpy(image))
    print('calculating first unet')
    tmp_out = net(tmp_x) #(850,3,360,360)
    print('done')
    tmp_out = tmp_out.data.numpy()

    tmp_out = tmp_out[:,1:,:,:] #(850,2,360,360)
    images = np.hstack(image,tmp_out) #(850,3,360,360)

    train_images = images[:830] #(830,3,360,360)
    train_mask = mask[:830] #(830,3,360,360)

    validation_images = images[830:] #(20,3,360,360)
    validation_num_mask = num_mask[830:].reshape(20,360,360) #(20,360,360)

    net2 = Net2()
    net2.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(net2.parameters())

    #validation用に作っておく
    val_x = Variable(torch.from_numpy(validation_images).cuda())

    learning_times = 20000
    for i in range(learning_times):
        r = random.randint(0,829)
        tmp_images = train_images[r:r+20,...]
        tmp_mask = mask[r:r+20,...]

        x = Variable(torch.from_numpy(tmp_images).cuda())
        y = Variable(torch.from_numpy(tmp_mask).cuda())

        optimizer.zero_grad()
        out = net2(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        #10回に一回validateする
        #ピクセル単位でどれだけ正しく予測できているか
        if i % 10 == 0:
            out_val = net2(val_x)
            _, pred = torch.max(out_val,1) #(n,388,388)のVariable, 一枚は、0,1,2でできた配列
            pred = pred.cpu()
            pred = pred.data.numpy()
            pred.reshape(20,360,360)
            correct = len(np.where(pred == validation_num_mask)[0])
            acc = correct / validation_num_mask.size
            validation_log.append(acc)
            print('======================')
            print(loss)
            print(acc)
            print(str(i)+'/'+str(learning_times))
            print('======================')

    torch.save(net, 'model/wnet2/%s' % str(seed))

    if os.path.exists('log'):
        pass
    else:
        os.mkdir('log')

    if os.path.exists('log/wnet2'):
        pass
    else:
        os.mkdir('log/wnet2')

    with open('log/wnet2/%s' % str(seed), 'wb') as f:
        pickle.dump(validation_log, f)

    print('saved model as model/wnet2/%s' % str(seed))

    took_time = (end_time - start_time) / 60

    print('took %s minutes' % str(took_time))


def eval(seed):
    #imageは(n,1,360,360)
    #answersは(1,n)
    #prediction
    image, answers = pro.load_unet2_data(seed,mode=1)

    image = image.reshape(-1,1,360,360).astype(np.float32)

    net = torch.load('model/unet2/%s' % str(seed))
    net2 = torch.load('model/wnet2/%s' % str(seed))

    print('calculating first unet')
    tmp_out = net(Variable(torch.from_numpy(image)))
    print('done')
    tmp_out = tmp_out.data.numpy()
    tmp_out = np.hstack(image,tmp_out)

    print('calculating second unet')
    out = net2(Variable(torch.from_numpy(tmp_out)))
    print('done')

    _,pred = torch.max(out,1)
    pred = pred.data.numpy()

    ncpred = []

    for x in pred:
        c = len(np.where(x>=1)[0])
        n = len(np.where(x==2)[0])
        ncr = (n / c) // 0.01
        ncpred.append(int(ncr))

    num_of_ans, num_of_correct, diff_dict = pro.validate(answers, ncpred)

    print(num_of_ans)
    print(num_of_correct)
    print(diff_dict)


def view(seed):
    image, mask = pro.load_unet2_data(seed,mode=2)
    image = image.reshape(-1,1,360,360).astype(np.float32)
    n = int(len(image) // 4)
    for i in range(n):
        start = i * 4
        img = image[start:start+4]
        msk = mask[start:start+4]
        net = torch.load('model/unet2/%s' % str(seed))
        net2 = torch.load('model/wnet2/%s' % str(seed))
        net.cuda()
        net2.cuda()
        x = Variable(torch.from_numpy(img).cuda())
        print('calculating first unet')
        out = net(x)
        print('done')
        out.cpu()
        out = out.data.numpy()
        x = np.hstack(img,out)
        print('calculating second unet')
        out = net2(Variable(torch.from_numpy(out).cuda()))
        print('done')
        _, pred = torch.max(out,1) #(n,388,388)で要素は0,1,2の配列
        pred = pred.cpu()
        pred = pred.data.numpy()
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(4,3,1)
        sub.imshow(img[0].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,2)
        sub.imshow(msk[0,1],cmap='gray')
        sub = fig.add_subplot(4,3,3)
        sub.imshow(pred[0],cmap='gray')
        sub = fig.add_subplot(4,3,4)
        sub.imshow(img[1].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,5)
        sub.imshow(msk[1,1],cmap='gray')
        sub = fig.add_subplot(4,3,6)
        sub.imshow(pred[1],cmap='gray')
        sub = fig.add_subplot(4,3,7)
        sub.imshow(img[2].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,8)
        sub.imshow(msk[2,1],cmap='gray')
        sub = fig.add_subplot(4,3,9)
        sub.imshow(pred[2],cmap='gray')
        sub = fig.add_subplot(4,3,10)
        sub.imshow(img[3].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,11)
        sub.imshow(msk[3,1],cmap='gray')
        sub = fig.add_subplot(4,3,12)
        sub.imshow(pred[3],cmap='gray')
    plt.show()



if __name__ == '__main__':
    if os.path.exists('model/wnet2'):
        pass
    else:
        os.mkdir('model/wnet2')
    files = os.listdir('model/wnet2')
    seed = 1
    train(seed)
    eval(seed)
    view(seed)



