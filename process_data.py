import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import cv2 as cv
from collections import defaultdict
import torch

class Log():
    def __init__(self, seed, learning_times, log_frequency):
        self.seed = seed
        self.learning_times = learning_times #何回学習したか
        self.log_frequency = log_frequency #何回毎にログを取ったか
        self.loss = []
        self.training_accuracy = []
        self.validation_accuracy = []

#画像をグレースケールに変換
def rgb2gray(rgb):
    gray = np.dot(rgb, [0.299, 0.587, 0.114])
    gray = gray / 255.0
    return gray

#折りたたんで拡張
def replicate(input_array, h_or_v, m):
    n = input_array.shape[0]
    a = np.identity(n)

    if h_or_v == 'h':
        a = np.hstack((a[:,:m][:,::-1],a,a[:,n-m:][:,::-1]))
        return np.dot(input_array, a)

    elif h_or_v == 'v':
        a = np.vstack((a[:m,:][::-1,:],a,a[n-m:,:][::-1,:]))
        return np.dot(a, input_array)

    else:
        print('put "h" or "v" as 2nd parameter')

#ミラーリング
def mirror(input_array, output_size):
    n = input_array.shape[0]
    m = (output_size - n) // 2

    input_array = replicate(input_array, 'h', m)
    input_array = replicate(input_array, 'v', m)

    return input_array

def save(obj,directory,filename):
    if os.path.exists(directory):
        pass
    else:
        os.mkdir(directory)

    path = '%s/%s' % (directory, filename)

    with open(path, 'wb') as f:
        pickle.dump(obj,f)

#マスクデータを作る
def create_mask_label(cpath, npath, size):
    cytoplasm,nucleus = [cv.imread(x)[3:,:,2]/255 for x in [cpath,npath]]

    cytoplasm,nucleus = [cv.resize(x,(size,size)) for x in [cytoplasm,nucleus]]

    cytoplasm,nucleus = [np.round(x) for x in [cytoplasm,nucleus]]

    for i in range(size):
        for j in range(size):
            if cytoplasm[i][j] == 0 and nucleus[i][j] == 1:
                cytoplasm[i][j] += 1

    background = np.ones((size,size)) - cytoplasm

    cytoplasm = cytoplasm - nucleus

    mask = []

    _ = [mask.append(x) for x in [background,cytoplasm,nucleus]]

    mask = np.array(mask)

    mask_num = mask[0]*0 + mask[1]*1 + mask[2]*2

    return mask, mask_num

def create_ncratio(mask):
    #maskは(3,n,n)
    cytoplasm = mask[1]
    nucleus = mask[2]
    c = np.sum(cytoplasm + nucleus)
    n = np.sum(nucleus)
    percentage = n / c
    #ncr = [0]*100
    #ncr[p] += 1

    return percentage

def create_c_and_n(cpath, npath):
    cyt, nuc = [cv.imread(x)[3:,:,2]/255 for x in [cpath, npath]]
    ones = np.ones((360,360))
    cyt_, nuc_ = ones - cyt, ones - nuc
    cytoplasm, nucleus = [], []
    _ = [cytoplasm.append(x) for x in [cyt_, cyt]]
    _ = [nucleus.append(x) for x in [nuc_, nuc]]
    cytoplasm, nucleus = [np.array(x) for x in [cytoplasm, nucleus]]
    return cytoplasm, nucleus, cyt, nuc


#answerの値を10段階に分け、それぞれの正解率をだす
def validate_ncr(answer_list, prediction_list):
    a = [0] * 10
    b = [0] * 10
    d = defaultdict(int)
    for x,y in zip(answer_list, prediction_list):
        i = int(x // 0.1)
        a[i] += 1
        diff = np.absolute(x-y)
        j = int(diff // 0.01)
        d[j] += 1
        if diff <= 0.05:
            b[i] += 1
        else:
            pass
    c = []
    for x,y in zip(a,b):
        if x == 0:
            c.append(0)
        else:
            p = np.round(y/x,decimals=3)
            c.append(p)

    return a,b,c,d

def tasu(tz,z,num):
    if tz == num:
        if z == num:
            tp[num] += 1
        else:
            fn[num] += 1
    else:
        if z == num:
            fp[num] += 1
        else:
            #tn[num] += 1
            pass

def validate_mask(mask, pred_mask):
    #mask,pred_maskは(n,360,360)
    tp = [0] * 3
    fp = [0] * 3
    #tn = [0] * 3
    fn = [0] * 3
    for tx, x in zip(mask,pred_mask):
        for ty, y in zip(tx, x):
            for tz, z in zip(ty, y):
                #0
                tasu(0)
                #1
                tasu(1)
                #2
                tasu(2)
    tp,fp,fn = [np.array(x) for x in [tp,fp,fn]]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * recall * precision) / (recall + precision)
    return f_measure


def make_dir(directory):
    if os.path.exists(directory):
        pass
    else:
        os.mkdir(directory)


def calculate_accuracy(out, mask):
    _, pred = torch.max(out, 1)
    pred = pred.cpu()
    pred = pred.data.numpy()
    pred = pred.reshape(mask.shape)
    correct = len(np.where(pred == mask)[0])
    pixels = mask.size
    accuracy = correct / pixels
    return accuracy
    

#=========================================================
#=========================================================
#LOADING FUNCTIONS
#=========================================================
#=========================================================


def load(path):
    with open(path,'rb') as f:
        a = pickle.load(f)
    return a

def load_image():
    print('loading image')
    x = np.array([]).reshape(0,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/image' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_mask():
    print('loading mask')
    x = np.array([]).reshape(0,3,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/mask' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_num_mask():
    print('loading num_mask')
    x = np.array([]).reshape(0,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/num_mask' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_ncratio():
    print('loading ncratio')
    x = []
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/ncratio' % cell)
        x = np.hstack((x,y))
    print('loading done')
    return x

def load_cytoplasm():
    print('loading cytoplasm')
    x = np.array([]).reshape(0,2,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/cytoplasm' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_nucleus():
    print('loading nucleus')
    x = np.array([]).reshape(0,2,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/nucleus' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_num_cytoplasm():
    print('loading num_cytoplasm')
    x = np.array([]).reshape(0,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/num_cytoplasm' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_num_nucleus():
    print('loading num_nucleus')
    x = np.array([]).reshape(0,360,360)
    cells = os.listdir('data')
    if '.DS_Store' in cells:
        cells.remove('.DS_Store')
    for cell in cells:
        y = load('data/%s/num_nucleus' % cell)
        x = np.vstack((x,y))
    print('loading done')
    return x

def load_unet2_data(seed,mode=0):

    if mode == 0:
        print('loading training data for U-Net2')
        image, mask, num_mask = load_image(), load_mask(), load_num_mask()
        for x in [image,mask,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[:850].reshape(850,1,360,360).astype(np.float32)
        mask = mask[:850].reshape(850,3,360,360).astype(np.float32)
        num_mask = num_mask[:850].astype(np.int32)
        print('loading done')
        return image, mask, num_mask

    elif mode == 1:
        print('loading test data for U-Net2')
        image = load_image()
        ncratio = load_ncratio()
        num_mask = load_num_mask()
        for x in [image,ncratio,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        ncratio = ncratio[850:]
        num_mask = num_mask[850:].astype(np.int32)
        print('loading done')
        return image, ncratio, num_mask

    else:
        print('loading view data for U-Net2')
        image = load_image()
        num_mask = load_num_mask()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        num_mask = num_mask[850:].reshape(200,3,360,360).astype(np.int32)
        print('loading done')
        return image, mask


def load_unet3_data(seed,mode=0):

    if mode == 0:
        print('loading training data for U-Net3')
        image = load_image()
        mask = load_mask()
        num_mask = load_num_mask()
        ncratio = load_ncratio()
        for x in [image,mask,num_mask,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[:850].reshape(850,1,360,360).astype(np.float32)
        mask = mask[:850].reshape(850,3,360,360).astype(np.float32)
        num_mask = num_mask[:850].astype(np.int32)
        ncratio = np.array(ncratio[:850]).reshape(850,1).astype(np.float32)
        print('loading done')
        return image, mask, num_mask, ncratio

    elif mode == 1:
        print('loading test data for U-Net3')
        image = load_image()
        ncratio = load_ncratio()
        num_mask = load_num_mask()
        for x in [image,ncratio,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        ncratio = ncratio[850:]
        num_mask = num_mask[850:].astype(np.int32)
        print('loading done')
        return image, ncratio, num_mask

    else:
        print('loading view data for U-Net3')
        image = load_image()
        num_mask = load_num_mask()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        num_mask = num_mask[850:].astype(np.int32)
        print('loading done')
        return image, mask


def load_unet_c_data(seed, mode=0):
    if mode == 0:
        print('loading training data for U-Net-C')
        image = load_image()
        cytoplasm = load_cytoplasm()
        num_cytoplasm = load_num_cytoplasm()
        for x in [image, cytoplasm, num_cytoplasm]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[:850].reshape(850,1,360,360).astype(np.float32)
        cytoplasm = cytoplasm[:850].reshape(850,2,360,360).astype(np.float32)
        num_cytoplasm = num_cytoplasm[:850].astype(np.int32)
        print('loading done')
        return image, cytoplasm, num_cytoplasm

    if mode == 2:
        print('loading data for U-Net-C')
        image = load_image()
        mask = load_num_cytoplasm()
        for x in [image, mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        mask = mask[850:].reshape(200,360,360).astype(np.int32)
        print('loading done')
        return image, mask

def load_unet_n_data(seed, mode=0):
    if mode == 0:
        print('loading training data for U-Net-N')
        image = load_image()
        nucleus = load_nucleus()
        num_nucleus = load_num_nucleus()
        for x in [image, nucleus, num_nucleus]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[:850].reshape(850,1,360,360).astype(np.float32)
        nucleus = nucleus[:850].reshape(850,2,360,360).astype(np.float32)
        num_nucleus = num_nucleus[:850].astype(np.int32)
        print('loading done')
        return image, nucleus, num_nucleus

    if mode == 2:
        print('loading data for U-Net-N')
        image = load_image()
        mask = load_num_nucleus()
        for x in [image, mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        image = image[850:].reshape(200,1,360,360).astype(np.float32)
        mask = mask[850:].reshape(200,360,360).astype(np.int32)
        print('loading done')
        return image, mask

def load_test_data():
    print('loading')
    image = load('mini_data/image')
    mask = load('mini_data/mask')
    nmask = load('mini_data/num_mask')
    ncratio = load('mini_data/ncratio')
    return image, mask, nmask, ncratio

def load_cnn_data(seed,is_train=True):
    print('loading data for CNN')
    image = load_image()
    ncratio = load_ncratio()
    for x in [image,ncratio]:
        np.random.seed(seed)
        np.random.shuffle(x)
    if is_train:
        print('loading done')
        return image[:850], ncratio[:850]
    else:
        print('loading done')
        return image[850:], ncratio[850:]


