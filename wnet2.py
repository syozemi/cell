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
import unet2
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
        self.weight = MulWeight([0.5,1,1])

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
        super(Net2, self).__init__()
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
        self.weight = MulWeight([0.5,1,1])

    def forward(self, x):
        block1 = self.conv_3_8(x)
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

    #まず、unet2を使って、入力用のデータを作る
    net = torch.load('model/unet2/%d' % seed)
    net.cuda()
    image, mask, num_mask = pro.load_unet2_data(seed,mode=0)

    tmp_image = np.array([]).reshape(0,2,360,360)

    print('start calculating first unet')

    for i in tqdm(range(50)):
        start = i * 17
        tmp_x = Variable(torch.from_numpy(image[start:start+17]).cuda())
        tmp_out = net(tmp_x)
        tmp_out = tmp_out.cpu()
        tmp_out = tmp_out.data.numpy()
        tmp_out = tmp_out[:,1:,:,:]
        tmp_image = np.vstack((tmp_image,tmp_out))
        print(i)

    print('done')

    images = np.hstack((image,tmp_image)) #(850,3,360,360)

    #train用のとvalidation用に分ける
    #validationはとりあえずgpuのメモリに乗る範囲で20にした。loopを回したらもっと多くできるけど、とりあえず。
    train_images = images[:830].astype(np.float32) #(830,3,360,360)
    train_mask = mask[:830].astype(np.float32) #(830,3,360,360)

    validation_images = images[830:].astype(np.float32) #(20,3,360,360)
    validation_num_mask = num_mask[830:].reshape(20,360,360) #(20,360,360)

    net2 = Net2()
    net2.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(net2.parameters())

    #validation用に作っておく
    val_x = Variable(torch.from_numpy(validation_images).cuda())

    learning_times = 1000
    log_frequency = 10

    log = pro.Log(seed, learning_times, log_frequency)

    #学習のループ
    for i in range(learning_times):
        r = random.randint(0,809)
        tmp_images = train_images[r:r+20]
        tmp_mask = mask[r:r+20]

        x = Variable(torch.from_numpy(tmp_images).cuda())
        y = Variable(torch.from_numpy(tmp_mask).cuda())

        optimizer.zero_grad()
        out = net2(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        #10回に一回validateする
        #ピクセル単位でどれだけ正しく予測できているか
        if i % log_frequency == 0:
            tmp_num_mask = num_mask[r:r+20]
            out_val = net2(val_x)
            training_accuracy = pro.calculate_accuracy(out, tmp_num_mask)
            validation_accuracy = pro.calculate_accuracy(out_val, validation_num_mask)

            log.loss.append(loss.data[0])
            log.training_accuracy.append(training_accuracy)
            log.validation_accuracy.append(validation_accuracy)

            print('=========================================================')
            print('training times:      %d/%d' % (i, learning_times))
            print('training accuracy:   %d' % training_accuracy)
            print('validation accuracy: %d' % validation_accuracy)
            print('loss:                %d' % loss.data[0])
            print('estimated time:      %d' % est_time)
            print('=========================================================')

    torch.save(net2, 'model/wnet2/%d' % seed)

    #lossとvalidation(ピクセル単位の正解率)のログを保存しておく。
    pro.make_dir('log')
    pro.make_dir('log/wnet2')

    pro.save(log, 'log/wnet2', str(seed))  

    print('saved model as model/wnet2/%d' % seed)

    end_time = time.time()

    #かかった時間を出力する
    time_taken = (end_time - start_time) / 60

    print('took %d minutes' % time_taken)


def eval(seed):
    #imageは(n,1,360,360)
    #answersは(1,n)

    #prediction
    #データをロード
    image, answers, num_mask = pro.load_unet2_data(seed,mode=1)

    tmp = unet2.make_data_for_wnet2(seed)

    tmp = tmp.astype(np.float32)

    images = np.hstack((image,tmp))
    images = images.astype(np.float32)

    net2 = torch.load('model/wnet2/%d' % seed)

    print('calculating wnet')

    ncpred = []
    mask_pred = np.array([]).reshape(0,360,360)

    for i in tqdm(range(10)):
        start = i * 20
        tmp_image = images[start:start+20]
        tmp_x = Variable(torch.from_numpy(tmp_image).cuda())
        out = net2(tmp_x)
        _,pred = torch.max(out,1)
        pred = pred.cpu()
        pred = pred.data.numpy()
        pred = pred.reshape(-1,360,360)
        mask_pred = np.vstack((mask_pred,pred))
        for x in pred:
            c = len(np.where(x>=1)[0])
            n = len(np.where(x==2)[0])
            ncr = n / c
            ncpred.append(ncr)          

    print('done')

    num_of_ans, num_of_correct,prob, diff_dict = pro.validate_ncr(answers, ncpred)
    f_measure = pro.validate_mask(num_mask, mask_pred)

    print(num_of_ans)
    print(num_of_correct)
    print(prob)
    print(diff_dict)
    print(f_measure)


def view(seed):
    image, mask = pro.load_unet2_data(seed,mode=2)
    image = image.reshape(-1,1,360,360).astype(np.float32)
    n = int(len(image) // 4)
    for i in tqdm(range(n)):
        start = i * 4
        img = image[start:start+4]
        msk = mask[start:start+4].reshape(4,360,360)
        net = torch.load('model/unet2/%d' % seed)
        net2 = torch.load('model/wnet2/%d' % seed)
        net.cuda()
        net2.cuda()
        x = Variable(torch.from_numpy(img).cuda())
        print('calculating first unet')
        out = net(x)
        print('done')
        out = out.cpu()
        out = out.data.numpy()
        out = out[:,1:,:,:]
        x = np.hstack((img,out))
        print('calculating second unet')
        out = net2(Variable(torch.from_numpy(x).cuda()))
        print('done')
        _, pred = torch.max(out,1) #(n,388,388)で要素は0,1,2の配列
        pred = pred.cpu()
        pred = pred.data.numpy()
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(4,3,1)
        sub.imshow(img[0].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,2)
        sub.imshow(msk[0],cmap='gray')
        sub = fig.add_subplot(4,3,3)
        sub.imshow(pred[0],cmap='gray')
        sub = fig.add_subplot(4,3,4)
        sub.imshow(img[1].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,5)
        sub.imshow(msk[1],cmap='gray')
        sub = fig.add_subplot(4,3,6)
        sub.imshow(pred[1],cmap='gray')
        sub = fig.add_subplot(4,3,7)
        sub.imshow(img[2].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,8)
        sub.imshow(msk[2],cmap='gray')
        sub = fig.add_subplot(4,3,9)
        sub.imshow(pred[2],cmap='gray')
        sub = fig.add_subplot(4,3,10)
        sub.imshow(img[3].reshape(360,360),cmap='gray')
        sub = fig.add_subplot(4,3,11)
        sub.imshow(msk[3],cmap='gray')
        sub = fig.add_subplot(4,3,12)
        sub.imshow(pred[3],cmap='gray')
    plt.show()


if __name__ == '__main__':
    pro.make_dir('model/wnet2')
    files = os.listdir('model/wnet2')
    seed = len(files)
    train(seed)
    eval(seed)
    view(seed)



