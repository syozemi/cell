import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import cv2 as cv
from collections import defaultdict


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
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        pass

    path = '%s/%s' % (directory, filename)

    with open(path, 'wb') as f:
        pickle.dump(obj,f)

#マスクデータを作る
def create_mask_label(cpath, npath, size):
    cytoplasm,nucleus = [cv.imread(x)[3:,:,2]/255 for x in [cpath,npath]]

    cytoplasm,nucleus = [cv.resize(x,(size,size)) for x in [cytoplasm,nucleus]]

    #cytoplasm, nucleus = [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 20, 2) for x in [cytoplasm,nucleus]]
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

    d = int((n / c) // 0.1)
    p = int((n / c) // 0.01)
    ncr = [0]*100
    ncr[p] += 1

    return np.array(ncr), p, d


#answerの値を10段階に分け、それぞれの正解率をだす
def validate(answer_list, prediction_list):
    a = [0] * 10
    b = [0] * 10
    d = defaultdict(int)
    for x,y in zip(answer_list, prediction_list):
        i = int(x // 10)
        a[i] += 1
        diff = np.absolute(x-y)
        d[diff] += 1
        if diff <= 5:
            b[i] += 1
        else:
            pass
    c = [x/y for x,y in zip(b,a)]
    return a,b,c,d


    

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
    print('loading image data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/image' % folder
        if i == 0:
            image = load(path)
        else:
            image = np.vstack((image,load(path)))

    print('loading done')

    return image

def load_mask():
    print('loading mask data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/mask' % folder
        if i == 0:
            mask = load(path)
        else:
            mask = np.vstack((mask,load(path)))

    print('loading done')

    return mask

def load_num_mask():
    print('loading num_mask data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/num_mask' % folder
        if i == 0:
            num_mask = load(path)
        else:
            num_mask = np.vstack((num_mask,load(path)))

    print('loading done')

    return num_mask

def load_num_ncratio():
    print('loading ncratio data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/num_ncratio' % folder
        if i == 0:
            num_ncratio = load(path)
        else:
            num_ncratio = np.hstack((num_ncratio, load(path)))

    print('loading done')

    return num_ncratio

def load_num_ncratio10():
    print('loading num_ncratio10 data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/num_ncratio10' % folder
        if i == 0:
            num_ncratio10 = load(path)
        else:
            num_ncratio10 = np.hstack((num_ncratio10, load(path)))
    print('loading done')

    return num_ncratio10

def load_raw_image():
    print('loading image data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/raw_image' % folder
        if i == 0:
            image = load(path)
        else:
            image = np.vstack((image,load(path)))

    print('loading done')

    return image

def load_raw_mask():
    print('loading mask data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/raw_mask' % folder
        if i == 0:
            mask = load(path)
        else:
            mask = np.vstack((mask,load(path)))

    print('loading done')

    return mask

def load_raw_num_mask():
    print('loading num_mask data')

    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        path = 'data/%s/raw_num_mask' % folder
        if i == 0:
            num_mask = load(path)
        else:
            num_mask = np.vstack((num_mask,load(path)))

    print('loading done')

    return num_mask


def load_unet_data(seed,mode=0):
    print('loading data for U-Net')

    if mode == 0:
        image = load_image()
        mask = load_mask()
        num_mask = load_num_mask()
        for x in [image,mask,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[:250], mask[:250], num_mask[:250]

    elif mode == 1:
        image = load_image()
        ncratio = load_num_ncratio10()
        for x in [image,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], ncratio[250:]

    else:
        image = load_image()
        mask = load_mask()
        ncratio = load_num_ncratio()
        for x in [image,mask,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], mask[250:], ncratio[250:]

def load_unet2_data(seed,mode=0):
    if mode == 0:
        print('loading training data for U-Net2')
        image = load_raw_image()
        mask = load_raw_mask()
        num_mask = load_raw_num_mask()
        for x in [image,mask,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[:250], mask[:250], num_mask[:250]

    elif mode == 1:
        print('loading test data for U-Net2')
        image = load_raw_image()
        ncratio = load_num_ncratio()
        for x in [image,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], ncratio[250:]

    else:
        print('loading view data for U-Net2')
        image = load_raw_image()
        mask = load_raw_mask()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], mask[250:]

def load_unet3_data(seed,mode=0):
    if mode == 0:
        print('loading training data for U-Net3')
        image = load_raw_image()
        mask = load_raw_mask()
        num_mask = load_raw_num_mask()
        ncratio = load_num_ncratio()
        for x in [image,mask,num_mask,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[:250], mask[:250], num_mask[:250], ncratio[:250]

    elif mode == 1:
        print('loading test data for U-Net3')
        image = load_raw_image()
        ncratio = load_num_ncratio()
        for x in [image,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], ncratio[250:]

    else:
        print('loading view data for U-Net3')
        image = load_raw_image()
        mask = load_raw_mask()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[250:], mask[250:]

def load_cnn_data(seed,is_train=True):
    print('loading data for CNN')
    image = load_image()
    ncratio = load_num_ncratio10()
    for x in [image,ncratio]:
        np.random.seed(seed)
        np.random.shuffle(x)
    if is_train:
        print('loading done')
        return image[:250], ncratio[:250]
    else:
        print('loading done')
        return image[250:], ncratio[250:]


