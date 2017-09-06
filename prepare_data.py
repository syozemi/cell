import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random


if os.path.exists('data'):
    pass
else:
    os.mkdir('data')

if not os.path.exists('cell_data'):
    print('no cell_data')
else:
    image0 = []
    mask0 = []
    ncratio0 = []
    image1 = []
    mask1 = []
    ncratio1 = []

    folders = os.listdir('cell_data/image')
    for folder in folders:
        if folder == '.DS_Store':
            pass
        else:
            files = os.listdir('cell_data/image/' + folder)
            print(folder)
            for i,file in enumerate(files):
                #画像のパス
                image_path = 'cell_data/image/' + folder + '/' + file
                cell_path = 'cell_data/mask/' + folder + '/' + file.replace('.jpg', '.mask.0.png')
                nucleus_path = 'cell_data/mask/' + folder + '/' + file.replace('.jpg', '.mask.1.png')

                #画像を360*360の行列として取得する
                image_array = plt.imread(image_path)
                image_array = image_array[3:,:,:]
                cell_array = plt.imread(cell_path)
                cell_array = cell_array[3:,:,:]
                nucleus_array = plt.imread(nucleus_path)
                nucleus_array = nucleus_array[3:,:,:]

                #画像をグレースケールに変換する
                image_array = pro.rgb2gray(image_array)

                #細胞と核のマスクを0,1の行列にする
                cell_array = cell_array[:,:,0]
                nucleus_array = nucleus_array[:,:,0]

                #ミラーリングして572*572の行列にする
                image_array = pro.mirror(image_array,572)
                cell_array = pro.mirror(cell_array,572)
                nucleus_array = pro.mirror(nucleus_array,572)

                #細胞と核のマスクから、ラベルを作る
                mask_array = pro.create_mask_label(cell_array, nucleus_array)

                #nc比を作る
                c = np.sum(mask_array[:,:,1])
                n = np.sum(mask_array[:,:,2])
                nc = n / c
                nc = int(nc // 0.1)
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

    image0 = np.array(image0)
    image1 = np.array(image1)
    mask0 = np.array(mask0)
    mask1 = np.array(mask1)
    ncratio0 = np.array(ncratio0)
    ncratio1 = np.array(ncratio1)

    #保存する
    pro.save(image0,'data/image0')
    pro.save(mask0,'data/mask0')
    pro.save(ncratio0,'data/ncratio0')
    pro.save(image1,'data/image1')
    pro.save(mask1,'data/mask1')
    pro.save(ncratio1,'data/ncratio1')








