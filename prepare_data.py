import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random

def makedir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

if not os.path.exists('cell_data'):
    print('no cell_data')
else:
    image = []
    cell = []
    nucleus = []
    image_ = []
    cell_ = []
    nucleus_ = []

    folders = os.listdir('cell_data/image')
    for folder in folders:
        if folder == '.DS_Store':
            pass
        else:
            files = os.listdir('cell_data/image/' + folder)
            for i,file in enumerate(files):
                image_path = 'cell_data/image/' + folder + '/' + file
                cell_path = 'cell_data/mask/' + folder + '/' + file.replace('.jpg', '.mask.0.png')
                nucleus_path = 'cell_data/mask/' + folder + '/' + file.replace('.jpg', '.mask.1.png')
                image_array = plt.imread(image_path)
                image_array = image_array[3:,:,:]
                cell_array = plt.imread(cell_path)
                cell_array = cell_array[3:,:,:]
                nucleus_array = plt.imread(nucleus_path)
                nucleus_array = nucleus_array[3:,:,:]
                image.append(image_array)
                cell.append(cell_array)
                nucleus.append(nucleus_array)
                image_array = pro.rgb2gray(image_array)
                cell_array = cell_array[:,:,0]
                nucleus_array = nucleus_array[:,:,0]
                image_array,cell_array,nucleus_array = pro.random_crop(image_array,cell_array,nucleus_array)
                image_.append(image_array)
                cell_.append(cell_array)
                nucleus_.append(nucleus_array)
                print(i,'\r',end='')

    image = np.array(image)
    image = pro.rgb2gray_array(image)
    image = image / 255
    image = pro.mirror_all(image,572)
    pro.save(image,'data/image')

    print('image saved')

    image_ = np.array(image_)
    image_ = image_ / 255
    pro.save(image_,'data/image_')

    print('image_ saved')

    cell = np.array(cell)
    nucleus = np.array(nucleus)
    cell_ = np.array(cell_)
    nucleus_ = np.array(nucleus_)
    cell = cell[:,:,:,0]
    nucleus = nucleus[:,:,:,0]

    cell_label = []
    nucleus_label = []
    cell_label_ = []
    nucleus_label_ = []

    cell = pro.create_label(cell,572)
    print('cell')
    nucleus = pro.create_label(nucleus,572)
    print('nucleus')
    cell_ = pro.create_label(cell_,572)
    print('cell_')
    nucleus_ = pro.create_label(nucleus_,572)
    print('nucleus_')

    pro.save(cell,'data/cell')
    pro.save(nucleus,'data/nucleus')
    pro.save(cell_, 'data/cell_')
    pro.save(nucleus_, 'data/nucleus_')

    print('done')





































