import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

#画像をnumpyのarrayとして取得して返す。
def img_to_np(img_path):
	img = plt.imread(img_path)
	return img

#指定したサイズになるようにinput_arrayの真ん中を切り取って返す。output_shapeはtupleかlistで。
#U-Netのup-convolutionで使う。
def rescale(input_array, output_shape):
	i = len(input_array.shape)
	y, x = input_array.shape[i-3], input_array.shape[i-2]
	y_, x_ = output_shape[0], output_shape[1]
	y_start = (y - y_) // 2
	x_start = (x - x_) // 2
	if len(input_array.shape) == 3:
		return input_array[y_start: y_start + y_, x_start: x_start + x_, :]
	elif len(input_array.shape) == 2:
		return input_array[y_start: y_start + y_, x_start: x_start + x_]
	else:
		return input_array[:, y_start: y_start + y_, x_start: x_start + x_, :]

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
    
#画像のarrayをロードする。
def load_array(name):
	with open(name, 'rb') as f:
		x = pickle.load(f)
		return x

#画像のarrayを保存する。
def save_array(z, name):
	with open(name, 'wb') as f:
		pickle.dump(z, f)

#画像をグレースケールに変換。
def rgb2gray(rgb):
    gray = np.dot(rgb, [0.299, 0.587, 0.114])
    gray = gray / 255.0
    return gray

#まとめてグレースケールに変換。
def rgb2gray_array(rgb_array):
	l = []
	for x in rgb_array:
		x_ = rgb2gray(x)
		l.append(x_)
	return np.array(l)

def flatten(matrix):
    return matrix.reshape(matrix.shape[0], -1)

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

#nc比を計算する。
def n_c_ratio(n,c):
	n_size = len(np.where(n!=0)[0])
	c_size = len(np.where(c!=0)[0])
	return n_size / c_size

#画像をもらって、その回転と反転の回転を返す。
def rotate_and_inverte(image):
    x1 = image
    x2 = np.rot90(x1)
    x3 = np.rot90(x2)
    x4 = np.rot90(x3)
    x5 = image[:,::-1]
    x6 = np.rot90(x5)
    x7 = np.rot90(x6)
    x8 = np.rot90(x7)
    return x1, x2, x3, x4, x5, x6, x7, x8

def mirror_all(matrix,size):
    for x in matrix:
        x = mirror(x,size)
    return matrix

def create_label(matrix, size):
    label = []
    for i,x in enumerate(matrix):
        print(i,'\r',end='')
        x_ = mirror(x, size)
        l = []
        for y in x_:
            for z in y:
                if z == 0.:
                    w = [1.,0.]
                else:
                    w = [0.,1.]
                l.append(w)
        l = np.array(l).reshape(size,size,2)
        label.append(l)
    return np.array(label)


def save(objecto, path):
    with open(path, 'wb') as f:
        pickle.dump(objecto, f)
    return None


def random_crop(img,cell,nucleus,size=(200,200)):
    x,y = img.shape
    start_x = random.sample(range(x-size[1]),1)[0]
    start_y = random.sample(range(y-size[0]),1)[0]
    img = img[start_y:start_y+size[0], start_x:start_x+size[1]]
    img = mirror(img,572)
    cell = cell[start_y:start_y+size[0], start_x:start_x+size[1]]
    cell = mirror(cell,572)
    nucleus = nucleus[start_y:start_y+size[0], start_x:start_x+size[1]]
    nucleus = mirror(nucleus,572)
    return img,cell,nucleus

def flip(x):
    if x == 0.:
        return 1.
    else:
        return 0.

def create_mask_label(cell, nucleus):
    l = []
    a = [flip(x) for x in cell.flatten()]
    a = np.array(a).reshape(cell.shape)
    a,cell,nucleus = [x.T for x in [a,cell,nucleus]]
    _ = [l.append(x) for x in [a,cell,nucleus]]
    #l.append(a)
    #l.append(cell)
    #l.append(nucleus)
    return np.array(l).T

def load(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def load_data():
    i0,i1,m0,m1,r0,r1 = [load(x) for x in ['data/image0','data/image1','data/mask0','data/mask1','data/ncratio0','data/ncratio1']]
    image,mask,ncratio = [np.vstack(x) for x in [(i0,i1),(m0,m1),(r0,r1)]]
    return image, mask, ncratio

def create_random_batch(x,y,n):
    l = random.sample(range(len(x)),n)
    batch_x = [x[i] for i in l]
    batch_y = [y[i] for i in l]
    batch_x, batch_y = map(np.array(), [batch_x,batch_y])
    return batch_x, batch_y

