import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random
import cv2 as cv
from tqdm import tqdm


if os.path.exists('data'):
    pass
else:
    os.mkdir('data')

for name in ['image','image02']:

    print(name)

    if os.path.exists('data/%s' % name):
        pass
    else:
        os.mkdir('data/%s' % name)

    folders = os.listdir('cell_data/%s' % name)
    for folder in folders:
        #for aug in aug_type:
        files = os.listdir('cell_data/%s/%s' % (name,folder))
        try:
            image360,image572,mask360,mask388,num_mask360,num_mask388,num_ncratio360,num_ncratio388 = [],[],[],[],[],[],[],[]
            #files = os.listdir('picture/image/%s/%s' % (folder,aug))
            for file in tqdm(files):
                try:
                    #画像のパス
                    #ipath = 'picture/image/%s/%s/%s' % (folder,aug,file)
                    #cpath = 'picture/mask/%s/%s/%s' % (folder,aug,file.replace('.jpg', '.mask.0.png'))
                    #npath = 'picture/mask/%s/%s/%s' % (folder,aug,file.replace('.jpg', '.mask.1.png'))
                    ipath = 'cell_data/%s/%s/%s' % (name, folder, file)
                    cpath = 'cell_data/%s/%s/%s' % (name.replace('image', 'mask'), folder, file.replace('.jpg', '.mask.0.png'))
                    npath = 'cell_data/%s/%s/%s' % (name.replace('image', 'mask'), folder, file.replace('.jpg', '.mask.1.png'))

                    #画像をグレースケールで取得して正規化し拡大する
                    #img = cv.resize(cv.imread(ipath,0)/255,(572,572))
                    img360 = cv.imread(ipath,0)[3:,:]/255
                    img572 = cv.resize(img360,(572,572))

                    #３クラスのマスクを作る
                    msk360, num_msk360 = pro.create_mask_label(cpath,npath,360)
                    msk388, num_msk388 = pro.create_mask_label(cpath,npath,388)

                    #マスクからnc比を計算する
                    num_ncr360 = pro.create_ncratio(msk360)
                    num_ncr388 = pro.create_ncratio(msk388)

                    _ = [x.append(y) for x,y in [(image360,img360),(image572,img572),(mask360,msk360),
                        (mask388,msk388),(num_mask360,num_msk360),(num_mask388,num_msk388),
                        (num_ncratio360,num_ncr360),(num_ncratio388,num_ncr388)]]

                except Exception as e:
                    print(str(e))
                    print(files+' error')

            image360,image572,mask360,mask388,num_mask360,num_mask388 = [np.array(x) for x in [image360,image572,mask360,mask388,num_mask360,num_mask388]]

            _ = [pro.save(x, 'data/%s/%s' % (name,folder), y) for x,y in [(image360,'image360'), (image572, 'image572'), (mask360,'mask360'), (mask388,'mask388'), (num_mask360, 'num_mask360'), (num_mask388, 'num_mask388'), (num_ncratio360, 'num_ncratio360'), (num_ncratio388, 'num_ncratio388')]]

            print('%s done' % folder)
                    
        except Exception as e:
            print(str(e))
            print('unable to process ' + folder)


