import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import cv2 as cv

#画像をグレースケールに変換。
def rgb2gray(rgb):
    gray = np.dot(rgb, [0.299, 0.587, 0.114])
    gray = gray / 255.0
    return gray

#折りたたんで拡張。
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
    path = directory + '/' + filename
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

def create_mask_label(cpath, npath, size):
    cytoplasm,nucleus = [cv.imread(x)[3:,:,2]/255 for x in [cpath,npath]]
    #cytoplasm = cv.imread(cpath)[3:,:,2]
    #nucleus = cv.imread(npath)[3:,:,2]

    cytoplasm,nucleus = [cv.resize(x,(size,size)) for x in [cytoplasm,nucleus]]
    #cytoplasm = cv.resize(cytoplasm,(size,size))
    #nucleus = cv.resize(nucleus,(size,size))

    cytoplasm,nucleus = [np.round(x) for x in [cytoplasm,nucleus]]
    #cytoplasm = np.round(cytoplasm)
    #nucleus = np.round(nucleus)

    for i in range(size):
        for j in range(size):
            if cytoplasm[i][j] == 0 and nucleus[i][j] == 1:
                cytoplasm[i][j] += 1

    background = np.ones((size,size)) - cytoplasm

    cytoplasm = cytoplasm - nucleus

    mask = []
    _,_,_ = [mask.append(x) for x in [background,cytoplasm,nucleus]]
    #mask.append(background)
    #mask.append(cytoplasm)
    #mask.append(nucleus)
    mask = np.array(mask)

    return mask


#==================================================
#LOADING FUNCTIONS
#==================================================

def load(path):
    with open(path,'rb') as f:
        a = pickle.load(f)
    return a


def load_data_cnn():
    print('loading data for cnn...')
    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/image360' % folder
        ncpath = 'data/%s/ncratio10' % folder        
        if i == 0:
            image, ncratio = [load(x) for x in [ipath,ncpath]]
        else:
            img,ncr = [load(x) for x in [ipath,ncpath]]
            image,ncratio = [np.vstack(x) for x in [(image,img),(ncratio,ncr)]]
    print('loading done')
    return image, ncratio

def load_data_unet():
    print('loading data for unet...')
    folders = os.listdir('data')
    image,mask = [],[]
    for i,folder in enumerate(folders):
        try:
            ipath = 'data/%s/image572' % folder
            mpath = 'data/%s/mask' % folder
            img, msk = [load(x) for x in [ipath, mpath]]
            image.append(img)
            mask.append(msk)
        except:
            pass
    image,mask = [np.array(x) for x in [image,mask]]
    image = image.reshape(7,50,572,572,1)
    print('loading done')
    return image, mask  

def load_data_cnn_torch():
    print('loading data for cnn...')
    folders = os.listdir('data')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/image360' % folder
        n10path = 'data/%s/ncratio_num10' % folder
        n100path = 'data/%s/ncratio_num100' % folder
        nc10path = 'data/%s/ncratio10' % folder
        nc100path = 'data/%s/ncratio100' % folder
        if i == 0:
            image, n10, n100, nc10, nc100 = [load(x) for x in [ipath, n10path, n100path, nc10path, nc100path]]
        else:
            img, nn10, nn100, ncn10, ncn100 = [load(x) for x in [ipath, n10path, n100path, nc10path, nc100path]]
            image, nc10, nc100 = [np.vstack(x) for x in [(image,img),(nc10,ncn10),(nc100,ncn100)]]
            n10, n100 = [np.hstack(x) for x in [(n10,nn10),(n100,nn100)]]
    print('loading done')
    return image, n10, n100, nc10, nc100  

def load_data_unet_torch():
    print('loading')
    folders = os.listdir('data')
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/image' % folder
        mpath = 'data/%s/mask' % folder
        if i == 0:
            image,mask = [load(x) for x in [ipath,mpath]]
        else:
            img,msk = [load(x) for x in [ipath,mpath]]
            image, mask = [np.vstack(x) for x in [(image,img),(mask,msk)]]
    print('loading done')
    return image, mask

def load_data_unet_torch2():
    print('loading')
    folders = os.listdir('data')
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/image' % folder
        mpath = 'data/%s/mask' % folder
        if i == 0:
            image,mask = [load(x) for x in [ipath,mpath]]
        else:
            img,msk = [load(x) for x in [ipath,mpath]]
            image, mask = [np.vstack(x) for x in [(image,img),(mask,msk)]]
    print('loading done')
    return image, mask

def load_data_wnet():
    print('loading')
    folders = os.listdir('data')
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/wnet' % folder
        mpath = 'data/%s/mask' % folder
        mmpath = 'data/%s/tmask' % folder
        if i == 0:
            image,mask,tmask = [load(x) for x in [ipath,mpath,mmpath]]
        else:
            img,msk,tmsk = [load(x) for x in [ipath,mpath,mmpath]]
            image, mask, tmask = [np.vstack(x) for x in [(image,img),(mask,msk),(tmask,tmsk)]]
    print('loading done')
    return image.astype(np.float32), mask.astype(np.float32), tmask

def load_data_wnet_for_test():
    print('loading')
    folders = os.listdir('data')
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/wnet' % folder
        mpath = 'data/%s/mask' % folder
        mmpath = 'data/%s/image360' % folder
        if i == 0:
            image,mask,tmask = [load(x) for x in [ipath,mpath,mmpath]]
        else:
            img,msk,tmsk = [load(x) for x in [ipath,mpath,mmpath]]
            image, mask, tmask = [np.vstack(x) for x in [(image,img),(mask,msk),(tmask,tmsk)]]
    print('loading done')
    return image.astype(np.float32), mask.astype(np.float32), tmask

