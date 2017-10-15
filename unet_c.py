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
        self.last = nn.Conv2d(8,2,1)

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

        return F.softmax(raw_score)


def train(seed):
    #imageは入力データ
    #maskは教師用データ
    #num_maskはマスクの数字版で、validationに使う

    start_time = time.time()
  
    image, mask, num_mask = pro.load_unet_c_data(seed,mode=0)

    train_image = image[:830] #(230,1,360,360)
    train_mask = mask[:830] #(230,3,360,360)

    validation_image = image[830:] #(20,1,360,360)
    validation_num_mask = num_mask[830:] #(20,360,360)

    #validation用に作っておく
    val_x = Variable(torch.from_numpy(validation_image).cuda())

    net = Net()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters())

    learning_times = 20000
    log_frequency = 20

    log = pro.Log(seed, learning_times, log_frequency)

    for i in range(learning_times):
        r = random.randint(0,809)
        tmp_image = image[r:r+20,...]
        tmp_mask = mask[r:r+20,...]
        #もっといいバッチの作り方はある(これだと端にあるデータの登場回数が少ない)
        #バッチを作ったのは、GPUのメモリに全部は乗らないから

        x = Variable(torch.from_numpy(tmp_image).cuda())
        y = Variable(torch.from_numpy(tmp_mask).cuda())

        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        #10回に一回validateする
        #ピクセル単位でどれだけ正しく予測できているか
        if i % log_frequency == 0:

            tmp_num_mask = num_mask[r:r+20]

            out_val = net(val_x)

            training_accuracy = pro.calculate_accuracy(out, tmp_num_mask)
            validation_accuracy = pro.calculate_accuracy(out_val, validation_num_mask)

            log.loss.append(loss.data[0])
            log.training_accuracy.append(training_accuracy)
            log.validation_accuracy.append(validation_accuracy)

            tmp_end_time = time.time()

            try:
                tmp_time = tmp_end_time - tmp_start_time
                est_time = ((learning_times - i) / log_frequency) * tmp_time
                est_time = est_time // 60
            except:
                est_time = 0

            tmp_start_time = time.time()

            print('=========================================================')
            print('training times:      %s/%s' % (str(i), str(learning_times)))
            print('training accuracy:   %s' % str(training_accuracy))
            print('validation accuracy: %s' % str(validation_accuracy))
            print('loss:                %s' % str(loss.data[0]))
            print('estimated time:      %d' % est_time)
            print('=========================================================')

    torch.save(net, 'model/unet_c/%d' % seed)

    pro.make_dir('log')
    pro.make_dir('log/unet_c')

    pro.save(log, 'log/unet_c', str(seed))

    end_time = time.time()
    time_taken = (end_time - start_time) // 60

    print('saved model as model/unet_c/%d' % seed)
    print('took %d minutes' % time_taken)


def view(seed):
    image, mask = pro.load_unet_c_data(seed,mode=2)
    n = int(len(image) // 4)
    for i in range(n):
        start = i * 4
        img = image[start:start+4]
        msk = mask[start:start+4]
        net = torch.load('model/unet_c/%d' % seed)
        net.cuda()
        x = Variable(torch.from_numpy(img).cuda())
        out = net(x)
        _, pred = torch.max(out,1) #(n,360,360)で要素は0,1,2の配列
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
    pro.make_dir('model/unet_c')
    files = os.listdir('model/unet_c')
    seed = len(files)
    train(seed)
    view(seed)