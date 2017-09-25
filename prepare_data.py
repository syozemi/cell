import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random
import cv2 as cv
from tqdm import tqdm


pro.make_dir('data')

folders = os.listdir('cell_data/image')

if '.DS_Store' in folders:
    folders.remove('.DS_Store')

for folder in folders:

    image,mask,num_mask,ncratio = [],[],[],[]

    for name in ['image', 'image02']:

        try:
            files = os.listdir('cell_data/%s/%s' % (name,folder))


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
                    img = cv.imread(ipath,0)[3:,:]/255

                    #３クラスのマスクを作る
                    msk, num_msk = pro.create_mask_label(cpath,npath,360)
                    
                    #マスクからnc比を計算する
                    ncr = pro.create_ncratio(msk)

                    _ = [x.append(y) for x,y in [(image, img), (mask, msk), (num_mask, num_msk), (ncratio, ncr)]]

                except Exception as e:
                    print(str(e))
                    print(files+' error')

        except:
            pass

    image, mask, num_mask = [np.array(x) for x in [image, mask, num_mask]]

    _ = [pro.save(x, 'data/%s/' % folder, y) for x,y in [(image, 'image'), (mask, 'mask'), (num_mask, 'num_mask'), (ncratio, 'ncratio')]]

    print('%s done' % folder)
                    



