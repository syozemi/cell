import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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
    a = shape_[0]
    b = shape_[1]
    c = shape[0]
    d = shape[1]
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
    gray = gray / 256.0
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





























