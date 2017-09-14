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
        image,image284,image572m,mask,tmask,maskm,ncratio,ncratio_num,ncratio_,ncratio_num_ = [],[],[],[],[],[],[],[],[],[]
        files = os.listdir('cell_data/image/%s' % folder)
        for i,file in enumerate(files):

            #画像のパス
            image_path = 'cell_data/image/%s/%s' % (folder,file)
            cell_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.0.png'))
            nucleus_path = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.1.png'))

            #画像を360*360の行列として取得する
            image_array = cv.imread(image_path,0)[3:,:] / 255
            cell_array,nucleus_array = [cv.imread(x)[3:,:,2] for x in [cell_path,nucleus_path]]
            cell_array,nucleus_array = [x/255 for x in [cell_array,nucleus_array]]

            #画像を284に縮小する
            image284_array = cv.resize(image_array,(284,284))

            #細胞と核のマスクから、ラベルを作る
            cell_array,nucleus_array = [cv.resize(x,(196,196)) for x in [cell_array,nucleus_array]]
            mask_array = pro.create_mask_label(cell_array, nucleus_array)
            tmask_array = pro.create_torch_mask_label(cell_array,nucleus_array)
            #cell_array_m,nucleus_array_m = [pro.mirror(x,388) for x in [cell_array,nucleus_array]]
            #mask_array_m = pro.create_mask_label(cell_array_m, nucleus_array_m)

            #nc比を作る
            c,n = [np.sum(x) for x in [cell_array,nucleus_array]]
            nc = int((n/c)//0.1)
            nc_ = int((n/c)//0.01)
            ncl = [0.]*10
            ncl_ = [0.]*100
            ncl[nc] = 1.
            ncl_[nc_] = 1.

            image.append(image_array)
            image284.append(image284_array)
            mask.append(mask_array)
            tmask.append(tmask_array)
            #maskm.append(mask_array_m)
            ncratio.append(ncl)
            ncratio_.append(ncl_)
            ncratio_num.append(nc)
            ncratio_num_.append(nc_)

            print(i,'\r',end='')
        image,image284,image572m,mask,tmask,maskm,ncratio,ncratio_,ncratio_num,ncratio_num_ = [np.array(x) for x in [image,image284,image572m,mask,tmask,maskm,ncratio,ncratio_,ncratio_num,ncratio_num_]]
        pro.save(image,'data/%s' % folder, 'image360')
        pro.save(image284,'data/%s' % folder, 'image284')
        #pro.save(image572m, 'data/%s' % folder, 'image572m')
        pro.save(mask,'data/%s' % folder, 'mask')
        pro.save(tmask, 'data/%s' % folder, 'tmask')
        #pro.save(maskm, 'data/%s' % folder, 'maskm')
        pro.save(ncratio,'data/%s' % folder, 'ncratio10')
        pro.save(ncratio_, 'data/%s' % folder, 'ncratio100')
        pro.save(ncratio_num, 'data/%s' % folder, 'ncratio_num10')
        pro.save(ncratio_num_, 'data/%s' % folder, 'ncratio_num100')   
        print(folder + ' done')
    except Exception as e:
        print(str(e))
        print('unable to process ' + folder)








