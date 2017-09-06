import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random


if not os.path.exists('data'):
    os.mkdir('data')

image0,image1,mask0,mask1,ncratio0,ncratio1 = [],[],[],[],[],[]

folders = os.listdir('cell_data/image')
del folders[0]

for folder in folders:
    files = os.listdir('cell_data/image/%s' % folder)
    for i,file in enumerate(files):
        #画像のパス
        image_path = 'cell_data/image/%s/%s' % (folder,file)
        cell_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.0.png'))
        nucleus_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.1.png'))

        #画像を360*360の行列として取得する
        image_array,cell_array,nucleus_array = [plt.imread(x)[3:,:,:] for x in [image_path,cell_path,nucleus_path]]

        #画像をグレースケールに変換する
        image_array = pro.rgb2gray(image_array)

        #細胞と核のマスクを0,1の行列にする
        cell_array, nucleus_array = [x[:,:,0] for x in [cell_array, nucleus_array]]

        #ミラーリングして572*572の行列にする
        image_array,cell_array,nucleus_array = [pro.mirror(x,572) for x in [image_array,cell_array,nucleus_array]]

        #細胞と核のマスクから、ラベルを作る
        mask_array = pro.create_mask_label(cell_array, nucleus_array)

        #nc比を作る
        c,n = [np.sum(x) for x in [mask_array[:,:,1], mask_array[:,:,2]]]
        nc = int((n/c)//0.1)
        ncl = [0.]*10
        ncl[nc] = 1.

        if i % 2 == 0:
            image0.append(image_array)
            mask0.append(mask_array)
            ncratio0.append(ncl)
        else:
            image1.append(image_array)
            mask1.append(mask_array)
            ncratio1.append(ncl)

        print(i,'\r',end='')
    print(folder)

image0,image1,mask0,mask1,ncratio0,ncratio1 = [np.array(x) for x in [image0,image1,mask0,mask1,ncratio0,ncratio1]]

#保存する
pro.save(image0,'data/image0')
pro.save(mask0,'data/mask0')
pro.save(ncratio0,'data/ncratio0')
pro.save(image1,'data/image1')
pro.save(mask1,'data/mask1')
pro.save(ncratio1,'data/ncratio1')








