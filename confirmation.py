#
#データの画像とマスクが一致しているかどうか確認するだけのプログラム。
#ランダムに選んだ画像とマスクを並べて表示する。
#

import os
import matplotlib.pyplot as plt
import process_data as pro
import random

folders = os.listdir('cell_data/image/')

for folder in folders:
    if folder == '.DS_Store':
        pass
    else:
        files = os.listdir('cell_data/image/'+folder)
        r = random.sample(range(len(files)), 3)
        fig = plt.figure(figsize=(7,7))
        for i,j in enumerate(r):
            name = files[i]
            image_path = 'cell_data/image/' + folder + '/' + name
            mask_cell_path = 'cell_data/mask/' + folder + '/' + name.replace('.jpg', '.mask.0.png')
            mask_nucleus_path = 'cell_data/mask/' + folder + '/' + name.replace('.jpg', '.mask.1.png')
            image = plt.imread(image_path)
            cell = plt.imread(mask_cell_path)
            nucleus = plt.imread(mask_nucleus_path)
            subplot = fig.add_subplot(3,3,i*3+1)
            subplot.set_title(folder+'/'+name)
            subplot.imshow(image)
            subplot = fig.add_subplot(3,3,i*3+2)
            subplot.imshow(cell)
            subplot = fig.add_subplot(3,3,i*3+3)
            subplot.imshow(nucleus)
        fig.tight_layout()
        plt.show()


























