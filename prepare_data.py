import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random
import cv2 as cv


if not os.path.exists('data'):
    os.mkdir('data')
else:
    pass

folders = os.listdir('cell_data/image')

for folder in folders:
    try:
        image,image572,image572m,mask,ncratio,ncratio_ = [],[],[],[],[],[]
        files = os.listdir('cell_data/image/%s' % folder)
        for i,file in enumerate(files):

            #画像のパス
            image_path = 'cell_data/image/%s/%s' % (folder,file)
            cell_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.0.png'))
            nucleus_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.1.png'))

            #画像を360*360の行列として取得する
            image_array = cv.imread(image_path,0)[3:,:] / 255
            cell_array,nucleus_array = [cv.imread(x)[3:,:,2]/255 for x in [cell_path,nucleus_path]]

            #拡大
            image_array_572 = cv.resize(image_array, (572,572))
            image_array_572m = pro.mirror(image_array, 572)

            #細胞と核のマスクから、ラベルを作る
            mask_array = pro.create_mask_label(cell_array, nucleus_array)

            #nc比を作る
            c,n = [np.sum(x) for x in [cell_array,nucleus_array]]
            nc = int((n/c)//0.1)
            nc_ = int((n/c)//0.01)
            ncl = [0.]*10
            ncl_ = [0.]*100
            ncl[nc] = 1.
            ncl_[nc_] = 1.

            image.append(image_array)
            mask.append(mask_array)
            ncratio.append(ncl)
            ncratio_.append(ncl_)
            image572.append(image_array_572)
            image572m.append(image_array_572m)

            print(i,'\r',end='')
        image,image572,image572m,mask,ncratio,ncratio_ = [np.array(x) for x in [image,image572,image572m,mask,ncratio,ncratio_]]
        pro.save(image,'data/%s' % folder, 'image360')
        pro.save(image572,'data/%s' % folder, 'image572')
        pro.save(image572m, 'data/%s' % folder, 'image572m')
        pro.save(mask,'data/%s' % folder, 'mask')
        pro.save(ncratio,'data/%s' % folder, 'ncratio10')
        pro.save(ncratio_, 'data/%s' % folder, 'ncratio100')   
        print(folder + ' done')
    except:
        print('unable to process ' + folder)








