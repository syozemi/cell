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


#answerの値を10段階に分け、それぞれの正解率をだす
def validate(answer_list, prediction_list):
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
            c.append(y/x)

    return a,b,c,d

def make_dir(directory):
    if os.path.exists(directory):
        pass
    else:
        os.mkdir(directory)

    

#=========================================================
#=========================================================
#LOADING FUNCTIONS
#=========================================================
#=========================================================


def load(path):
    with open(path,'rb') as f:
        a = pickle.load(f)
    return a

def load_image360():
    print('loading image360 data')

    image = np.array([]).reshape(0,1,360,360)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/image360' % (name,folder)
            img = load(path).reshape(-1,1,360,360)
            image = np.vstack((image,img))

    print('loading done')

    return image

def load_image572():
    print('loading image572 data')
    
    image = np.array([]).reshape(0,1,572,572)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/image572' % (name,folder)
            img = load(path).reshape(-1,1,572,572)
            image = np.vstack((image,img))

    print('loading done')

    return image

def load_mask360():
    print('loading mask360 data')
    
    x = np.array([]).reshape(0,3,360,360)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/mask360' % (name,folder)
            x = np.vstack((x,load(path)))

    print('loading done')

    return x

def load_mask388():
    print('loading mask388 data')
    
    x = np.array([]).reshape(0,3,388,388)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/mask388' % (name,folder)
            x = np.vstack((x,load(path)))

    print('loading done')

    return x

def load_num_mask360():
    print('loading num_mask360 data')
    
    x = np.array([]).reshape(0,1,360,360)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/num_mask360' % (name,folder)
            msk = load(path).reshape(-1,1,360,360)
            x = np.vstack((x,msk))

    print('loading done')

    return x

def load_num_mask388():
    print('loading num_mask572 data')
    
    x = np.array([]).reshape(0,1,388,388)
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/num_mask388' % (name,folder)
            msk = load(path).reshape(-1,1,388,388)
            x = np.vstack((x,msk))

    print('loading done')

    return x

def load_num_ncratio360():
    print('loading num_ncratio360 data')
    
    x = []
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/num_ncratio360' % (name,folder)
            x = np.hstack((x,load(path)))

    print('loading done')

    return x

def load_num_ncratio388():
    print('loading num_ncratio388 data')
    
    x = []
    for name in ['image', 'image02']:
        folders = os.listdir('data/%s' % name)
        for i,folder in enumerate(folders):
            path = 'data/%s/%s/num_ncratio388' % (name,folder)
            x = np.hstack((x,load(path)))

    print('loading done')

    return x


def load_unet_data(seed,mode=0):
    print('loading data for U-Net')

    if mode == 0:
        image = load_image572()
        mask = load_mask388()
        num_mask = load_num_mask388()
        for x in [image,mask,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[:850], mask[:850], num_mask[:850]

    elif mode == 1:
        image = load_image572()
        ncratio = load_num_ncratio360()
        for x in [image,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[850:], ncratio[850:]

    else:
        image = load_image572()
        mask = load_num_mask388()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[850:], mask[850:]

def load_unet2_data(seed,mode=0):
    if mode == 0:
        print('loading training data for U-Net2')
        image = load_image360()
        mask = load_mask360()
        num_mask = load_num_mask360()
        for x in [image,mask,num_mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[:850], mask[:850], num_mask[:850]

    elif mode == 1:
        print('loading test data for U-Net2')
        image = load_image360()
        ncratio = load_num_ncratio360()
        for x in [image,ncratio]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[850:], ncratio[850:]

    else:
        print('loading view data for U-Net2')
        image = load_image360()
        mask = load_num_mask360()
        for x in [image,mask]:
            np.random.seed(seed)
            np.random.shuffle(x)
        print('loading done')
        return image[850:], mask[850:]

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
    image = load_image360()
    ncratio = load_num_ncratio()
    for x in [image,ncratio]:
        np.random.seed(seed)
        np.random.shuffle(x)
    if is_train:
        print('loading done')
        return image[:850], ncratio[:850]
    else:
        print('loading done')
        return image[850:], ncratio[850:]


