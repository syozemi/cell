#Opencvをインストールしてから使う
#cell_dataを同じディレクトリに置く

import cv2
import os
import shutil
import numpy as np
import process_data as processer

def folder_copy(copyfrom, copyto):
    for filename in os.listdir(copyfrom):
        filepath = os.path.join(copyfrom, filename)
        if os.path.isdir(filepath):
            folder_copy(filepath, copyto)
        elif os.path.isfile(filepath):
            copypath = os.path.join(copyto, filename)
            shutil.copy(filepath, copypath)
            print('{0}から{1}にファイルをコピー'.format(filepath, copypath))

if os.path.exists('picture'):
    pass
else:
    os.mkdir('picture')

files = os.listdir('cell_data/')
files_folder = [f for f in files if os.path.isdir(os.path.join('cell_data', f))]

for f1 in files_folder:
    cells = os.listdir('cell_data/' + str(f1))
    for f2 in cells:
        copyfrom = 'cell_data/' + str(f1) + '/' + str(f2)
        copyto = 'picture/' + str(f2)
        if not os.path.isdir(copyto):
            os.mkdir(copyto)
            print('{0}にフォルダを作成'.format(copyto))
        folder_copy(copyfrom, copyto)

'''
picture/
　├ folder/
　│　 　└arg/
　│  　 ├ 360/
　│   　│　├ gray/
　│   　│　├ shift_and_deformation/
　│   　│　└ rotate_and_transpose/
　│   　├ 388/
　│   　│　├ gray/
　│   　│　├ shift_and_deformation/
　│   　│　└ rotate_and_transpose/
　│   　└ 572/
　│   　  ├ gray/
　│   　  ├ shift_and_deformation/
　│    　 └ rotate_and_transpose/
　│
　├ folder/
　│　  └arg/
 ...  　├...　
       ... 
 
360:image
388:mask
572:image
'''

folders = os.listdir('picture/')
size = ['360', '388', '572']
process = ['gray', 'shift_and_deformation', 'rotate_and_transpose']

for folder in folders:
    
    if os.path.exists('picture/' + str(folder) + '/aug'):
        pass
    else:
        os.mkdir('picture/' + str(folder) + '/aug')
        
    for s in size:
        if os.path.exists('picture/' + str(folder) + '/aug/' + s):
            pass
        else:
            os.mkdir('picture/' + str(folder) + '/aug/' + s)
        
        for p in process:
            if os.path.exists('picture/' + str(folder) + '/aug/' + s + '/' + p):
                pass
            else:
                os.mkdir('picture/' + str(folder) + '/aug/' + s + '/' + p)

    files = os.listdir('picture/' + str(folder) + '/')

    for file in files:
        if '.jpg' in file:
            image_path = 'picture/' + str(folder) + '/' + file
            cell_path = 'picture/' + str(folder) + '/' + file.replace('.jpg', '.mask.0.png')
            nucleus_path = 'picture/' + str(folder) + '/' + file.replace('.jpg', '.mask.1.png')

            image = cv2.imread(image_path)
            cell = cv2.imread(cell_path)
            nucleus = cv2.imread(nucleus_path)
            
            #grayscale
            image = processer.rgb2gray(image)
            
            #resize
            image360 = cv2.resize(image, (360,360))
            image572 = cv2.resize(image, (572,572))
            cell388 = cv2.resize(cell, (388,388))
            nucleus388 = cv2.resize(nucleus, (388,388))
            
            cv2.imwrite('picture/' + str(folder) + '/aug/360/gray/' + file, image360)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/gray/' + file, image572)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/gray/' + file.replace('.jpg', '.mask.0.png'), cell388)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/gray/' + file.replace('.jpg', '.mask.1.png'), nucleus388)

            #shift and deformation
            x1 = np.random.randint(30,150)
            x2 = np.random.randint(210,330)
            y1 = np.random.randint(30,150)
            y2 = np.random.randint(210,330)
            
            image_trim = image[x1:x2, y1:y2]
            cell_trim = cell[x1:x2, y1:y2]
            nucleus_trim = nucleus[x1:x2, y1:y2]
            
            image_deform360 = cv2.resize(image_trim, (360,360))
            image_deform572 = cv2.resize(image_trim, (572,572))
            cell_deform = cv2.resize(cell_trim, (388,388))
            nucleus_deform = cv2.resize(nucleus_trim, (388,388))
       
            cv2.imwrite('picture/' + str(folder) + '/aug/360/shift_and_deformation/' + file, image_deform360)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/shift_and_deformation/' + file, image_deform572)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/shift_and_deformation/' + file.replace('.jpg', '.mask.0.png'), cell_deform)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/shift_and_deformation/' + file.replace('.jpg', '.mask.1.png'), nucleus_deform)
            
            #rotate and transpose 
            center_image360 = tuple(np.array([image360.shape[1] / 2, image360.shape[0] / 2]))
            center_image572 = tuple(np.array([image572.shape[1] / 2, image572.shape[0] / 2]))
            center_cell388 = tuple(np.array([cell388.shape[1] / 2, cell388.shape[0] / 2]))
            center_nucleus388 = tuple(np.array([cell388.shape[1] / 2, cell388.shape[0] / 2]))
            
            size_image360 = tuple(np.array([image360.shape[1], image360.shape[0]]))
            size_image572 = tuple(np.array([image572.shape[1], image572.shape[0]]))
            size_cell388 = tuple(np.array([cell388.shape[1], cell388.shape[0]]))
            size_nucleus388 = tuple(np.array([cell388.shape[1], cell388.shape[0]]))
            
            affine_matrix_image360 = cv2.getRotationMatrix2D(center_image360, 90.0, 1.0)
            affine_matrix_image572 = cv2.getRotationMatrix2D(center_image572, 90.0, 1.0)
            affine_matrix_cell388 = cv2.getRotationMatrix2D(center_cell388, 90.0, 1.0)
            affine_matrix_nucleus388 = cv2.getRotationMatrix2D(center_nucleus388, 90.0, 1.0)
            
            image360_10 = cv2.warpAffine(image360, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_10 = cv2.warpAffine(image572, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_10 = cv2.warpAffine(cell388, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_10 = cv2.warpAffine(nucleus388, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            image360_20 = cv2.warpAffine(image360_10, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_20 = cv2.warpAffine(image572_10, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_20 = cv2.warpAffine(cell388_10, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_20 = cv2.warpAffine(nucleus388_10, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            image360_30 = cv2.warpAffine(image360_20, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_30 = cv2.warpAffine(image572_20, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_30 = cv2.warpAffine(cell388_20, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_30 = cv2.warpAffine(nucleus388_20, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            image360_01 = image360[:,::-1]
            image572_01 = image572[:,::-1]
            cell388_01 = cell388[:,::-1]
            nucleus388_01 = nucleus388[:,::-1]
            
            image360_11 = cv2.warpAffine(image360_01, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_11 = cv2.warpAffine(image572_01, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_11 = cv2.warpAffine(cell388_01, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_11 = cv2.warpAffine(nucleus388_01, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            image360_21 = cv2.warpAffine(image360_11, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_21 = cv2.warpAffine(image572_11, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_21 = cv2.warpAffine(cell388_11, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_21 = cv2.warpAffine(nucleus388_11, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            image360_31 = cv2.warpAffine(image360_21, affine_matrix_image360, size_image360, flags=cv2.INTER_CUBIC)
            image572_31 = cv2.warpAffine(image572_21, affine_matrix_image572, size_image572, flags=cv2.INTER_CUBIC)
            cell388_31 = cv2.warpAffine(cell388_21, affine_matrix_cell388, size_cell388, flags=cv2.INTER_CUBIC)
            nucleus388_31 = cv2.warpAffine(nucleus388_21, affine_matrix_nucleus388, size_nucleus388, flags=cv2.INTER_CUBIC)
            
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.10.jpg'), image360_10)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.20.jpg'), image360_20)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.30.jpg'), image360_30)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.01.jpg'), image360_01)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.11.jpg'), image360_11)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.21.jpg'), image360_21)
            cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.31.jpg'), image360_31)
            
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.10.jpg'), image572_10)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.20.jpg'), image572_20)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.30.jpg'), image572_30)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.01.jpg'), image572_01)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.11.jpg'), image572_11)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.21.jpg'), image572_21)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.31.jpg'), image572_31)
            
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.10.mask.0.png'), cell388_10)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.20.mask.0.png'), cell388_20)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.30.mask.0.png'), cell388_30)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.01.mask.0.png'), cell388_01)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.11.mask.0.png'), cell388_11)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.21.mask.0.png'), cell388_21)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.31.mask.0.png'), cell388_31)
            
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.10.mask.1.png'), nucleus388_10)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.20.mask.1.png'), nucleus388_20)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.30.mask.1.png'), nucleus388_30)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.01.mask.1.png'), nucleus388_01)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.11.mask.1.png'), nucleus388_11)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.21.mask.1.png'), nucleus388_21)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.31.mask.1.png'), nucleus388_31)

        else:
            pass