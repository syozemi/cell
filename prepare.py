import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import process_data as processor

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

            image_array = processor.img_to_np(image_path)
            cell_array = processor.img_to_np(cell_path)
            nucleus_array = processor.img_to_np(nucleus_path)
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

    image = processor.rgb2gray_array(image)

    processor.save(image, 'data/image')

    image572 = []

    for x in image:
        y = processor.mirror(x, 572)
        image572.append(y)

    image572 = np.array(image572)
    processor.save(image572, 'data/image572')

    cell = cell[:, :, :, 0]
    nucleus = nucleus[:, :, :, 0]
    ncratio = []

    for i in range(len(cell)):
        cell_sum = np.sum(cell[i])
        nucleus_sum = np.sum(nucleus[i])
        ncratio.append(nucleus_sum / cell_sum)

    ncratio = np.array(ncratio)

    processor.save(cell, 'data/cell')
    processor.save(nucleus, 'data/nucleus')
    processor.save(ncratio, 'data/ncratio')

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

    processor.save(ncratio10, 'data/ncratio10')
    processor.save(ncratio100, 'data/ncratio100')

    cell_label = processor.create_label(cell, 572)
    nucleus_label = processor.create_label(nucleus, 572)

    processor.save(cell_label, 'data/cell_label')
    processor.save(nucleus_label, 'data/nucleus_label')














