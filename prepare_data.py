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
        images,masks,ncratio = [],[],[]
        files = os.listdir('cell_data/image/%s' % folder)
        for i,file in enumerate(files):

            try:
                #画像のパス
                ipath = 'cell_data/image/%s/%s' % (folder,file)
                cpath = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.0.png'))
                npath = 'cell_data/mask/%s/%s' % (folder,file.replace('.jpg', '.mask.1.png'))

                #画像をグレースケールで取得して拡大し正規化する
                image = cv.resize(cv.imread(ipath,0)/255,(572,572))

                #３クラスのマスクを作る
                mask = pro.create_mask_label(cpath,npath,388)

                #マスクからnc比を計算する
                ncr = pro.create_ncratio(mask)

                images.append(image)
                masks.append(mask)
                ncratio.append(ncr)

            except Exception as e:
                print(str(e))
                print(files+' error')

            print(str(i), '\r', end='')

        images = np.array(images)
        masks = np.array(masks)
        ncratio = np.array(ncratio)
        pro.save(images, 'data/%s' % folder, 'image')
        pro.save(masks, 'data/%s' % folder, 'mask')
        pro.save(ncratio, 'data/%s' % folder, 'ncratio')
        print('%s done' % folder)
                
    except Exception as e:
        print(str(e))
        print('unable to process ' + folder)


