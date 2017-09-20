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
#aug_type = os.listdir('picture/image/Band')
for folder in folders:
    #for aug in aug_type:
        files = os.listdir('cell_data/image/%s' % folder)
        try:
            image,mask,num_mask,ncratio,num_ncratio,num_ncratio10 = [],[],[],[],[],[]
            #files = os.listdir('picture/image/%s/%s' % (folder,aug))
            for i,file in enumerate(files):
                try:
                    #画像のパス
                    #ipath = 'picture/image/%s/%s/%s' % (folder,aug,file)
                    #cpath = 'picture/mask/%s/%s/%s' % (folder,aug,file.replace('.jpg', '.mask.0.png'))
                    #npath = 'picture/mask/%s/%s/%s' % (folder,aug,file.replace('.jpg', '.mask.1.png'))
                    ipath = 'cell_data/image/%s/%s' % (folder,file)
                    cpath = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.0.png'))
                    npath = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.1.png'))

                    #画像をグレースケールで取得して正規化し拡大する
                    #img = cv.resize(cv.imread(ipath,0)/255,(572,572))
                    img = cv.imread(ipath,0)[3:,:]/255

                    #３クラスのマスクを作る
                    msk, num_msk = pro.create_mask_label(cpath,npath,360)

                    #マスクからnc比を計算する
                    ncr, num_ncr, num_ncr10 = pro.create_ncratio(msk)

                    image.append(img)
                    mask.append(msk)
                    num_mask.append(num_msk)
                    ncratio.append(ncr)
                    num_ncratio.append(num_ncr)
                    num_ncratio10.append(num_ncr10)

                except Exception as e:
                    print(str(e))
                    print(files+' error')

                print(str(i), '\r', end='')


            image = np.array(image)
            mask = np.array(mask)
            num_mask = np.array(num_mask)
            ncratio = np.array(ncratio)
            num_ncratio = np.array(num_ncratio)
            num_ncratio10 = np.array(num_ncratio10)

            pro.save(image, 'data/%s' % folder, 'raw_image')
            pro.save(mask, 'data/%s' % folder, 'raw_mask')
            pro.save(num_mask, 'data/%s' % folder, 'raw_num_mask')
            pro.save(ncratio, 'data/%s' % folder, 'raw_ncratio')
            pro.save(num_ncratio, 'data/%s' % folder, 'raw_num_ncratio')
            pro.save(num_ncratio10, 'data/%s' % folder, 'raw_num_ncratio10')

            print('%s done' % folder)
                    
        except Exception as e:
            print(str(e))
            print('unable to process ' + folder)


