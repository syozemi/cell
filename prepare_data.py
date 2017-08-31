import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import process_data as processer

if not os.path.exists('band'):
    print('no band')
else:
    if os.path.exists('data'):
        pass
    else:
        os.mkdir('data')

    image = []
    cell = []
    nucleus = []

    files = os.listdir('band/')

    for file in files:
        if '.bmp' in file:
            image_path = 'band/' + file
            cell_path = 'band/' + file.replace('.bmp', '.mask.0.png')
            nucleus_path = 'band/' + file.replace('.bmp', '.mask.1.png')

            image_array = processer.img_to_np(image_path)
            cell_array = processer.img_to_np(cell_path)
            nucleus_array = processer.img_to_np(nucleus_path)
            image_array = image_array[3:,:,:]
            cell_array = cell_array[3:,:,:]
            nucleus_array = nucleus_array[3:,:,:]

            image.append(image_array)
            cell.append(cell_array)
            nucleus.append(nucleus_array)

        else:
            pass

    image = np.array(image)
    cell = np.array(cell)
    nucleus = np.array(nucleus)

    image = processer.rgb2gray_array(image)

    image_ = []
    for x in image:
        for y in processer.rotate_and_inverte(x):
            image_.append(y)
    image_ = np.array(image_)
    image_ = processer.flatten(image_)

    image = processer.flatten(image)

    with open('data/image', 'wb') as f:
        pickle.dump(image, f)

    with open('data/image_', 'wb') as f:
        pickle.dump(image_, f)

    cell = cell[:, :, :, 0]
    nucleus = nucleus[:, :, :, 0]
    ncratio = []

    for i in range(len(cell)):
        cell_sum = np.sum(cell[i])
        nucleus_sum = np.sum(nucleus[i])
        ncratio.append(nucleus_sum / cell_sum)

    ncratio = np.array(ncratio)

    with open('data/cell', 'wb') as f:
        pickle.dump(cell, f)
    with open('data/nucleus', 'wb') as f:
        pickle.dump(nucleus, f)
    with open('data/ncratio', 'wb') as f:
        pickle.dump(ncratio, f)


    ncratio10 = []
    ncratio100 = []

    for x in ncratio:
        i10 = int(x // 0.1)
        i100 = int(x // 0.01)
        nc10l = [0]*10
        nc100l = [0]*100
        nc10l[i10] += 1
        nc100l[i100] += 1
        ncratio10.append(nc10l)
        ncratio100.append(nc100l)

    ncratio10 = np.array(ncratio10)
    ncratio100 = np.array(ncratio100)

    with open('data/ncratio10', 'wb') as f:
        pickle.dump(ncratio10, f)
    with open('data/ncratio100', 'wb') as f:
        pickle.dump(ncratio100, f)


    ncratio_ = []
    ncratio10_ = []
    ncratio100_ = []

    for x in ncratio:
        for _ in range(8):
            ncratio_.append(x)

    for x in ncratio10:
        for _ in range(8):
            ncratio10_.append(x)

    for x in ncratio100:
        for _ in range(8):
            ncratio100_.append(x)

    ncratio_ = np.array(ncratio_)
    ncratio10_ = np.array(ncratio10_)
    ncratio100_ = np.array(ncratio100_)

    with open('data/ncratio_', 'wb') as f:
        pickle.dump(ncratio_, f)
    with open('data/ncratio10_', 'wb') as f:
        pickle.dump(ncratio10_, f)
    with open('data/ncratio100_', 'wb') as f:
        pickle.dump(ncratio100_, f)

