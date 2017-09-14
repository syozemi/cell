import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

#shapeで指定したサイズでinput_arrayを真ん中で切り抜く
def crop(input_array, shape):
    shape_ = input_array.shape
    a,b,c,d = shape_[0],shape_[1],shape[0],shape[1]
    nx = (a - c) // 2
    ny = (b - d) // 2
    res = np.zeros(c * d * 3).reshape(c, d, 3)
    for i in range(c):
        for j in range(d):
            res[i][j] = input_array[nx + i][ny + j]

    return res

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

def flip(x):
    if x == 0.:
        return 1.
    else:
        return 0.

def create_mask_label(cell, nucleus):
    l = []
    a = [flip(x) for x in cell.flatten()]
    a = np.array(a).reshape(cell.shape)
    cell = cell - nucleus
    a,cell,nucleus = [x.T for x in [a,cell,nucleus]]
    _ = [l.append(x) for x in [a,cell,nucleus]]
    return np.array(l).T

def create_torch_mask_label(cell,nucleus):
    l = cell + nucleus
    return l.astype(np.int)

def load(path):
    with open(path,'rb') as f:
        return pickle.load(f)

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
        ipath = 'data/%s/image572' % folder
        mpath = 'data/%s/mask' % folder
        mmpath = 'data/%s/tmask' % folder
        if i == 0:
            image,mask,tmask = [load(x) for x in [ipath,mpath,mmpath]]
        else:
            img,msk,tmsk = [load(x) for x in [ipath,mpath,mmpath]]
            image, mask, tmask = [np.vstack(x) for x in [(image,img),(mask,msk),(tmask,tmsk)]]
    print('loading done')
    return image, mask, tmask

def load_data_unet_torch2():
    print('loading')
    folders = os.listdir('data')
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for i,folder in enumerate(folders):
        ipath = 'data/%s/image284' % folder
        mpath = 'data/%s/mask' % folder
        mmpath = 'data/%s/tmask' % folder
        if i == 0:
            image,mask,tmask = [load(x) for x in [ipath,mpath,mmpath]]
        else:
            img,msk,tmsk = [load(x) for x in [ipath,mpath,mmpath]]
            image, mask, tmask = [np.vstack(x) for x in [(image,img),(mask,msk),(tmask,tmsk)]]
    image = image.reshape(-1,1,284,284).astype(np.float32)
    mask = mask.reshape(350,196,196,3).astype(np.float32)
    mask = np.swapaxes(mask,1,3)
    mask = np.swapaxes(mask,2,3)
    print('loading done')
    return image, mask, tmask

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



def check():
    img, ncr = load_data_cnn()
    image, mask = load_data_unet()

    n = 50

    l = random.sample(range(len(img)),1)

    i = img[l[0]]
    r = ncr[l[0]]
    im = image[l[0]]
    c = mask[l[0],:,:,1]
    nu = mask[l[0],:,:,2]

    fig = plt.figure(figsize=(6,6))

    subplot = fig.add_subplot(2,2,1)
    subplot.imshow(i,cmap='gray')
    subplot = fig.add_subplot(2,2,2)
    subplot.imshow(im,cmap='gray')
    subplot = fig.add_subplot(2,2,3)
    subplot.imshow(c,cmap='gray')
    subplot = fig.add_subplot(2,2,4)
    subplot.imshow(nu,cmap='gray')

    plt.show()

    cel = np.sum(c)
    nuc = np.sum(nu)


    for x in c[180]:
        print(x)
    print(int((nuc/cel)//0.1))
    print(r)