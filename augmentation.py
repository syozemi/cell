# -*- coding: utf-8 -*-
#Opencvをインストールしてから使う
#cell_data, data_augmentation.pyを同じディレクトリに置く

import cv2
import os
import data_augmentation as aug
import shutil

#shift and deformationを1枚の画像に対して行う回数
shift_and_deformation_time = 10

#フォルダ1つあたりの画像の最大枚数
folder_size = 50

if not os.path.exists('picture'):
    os.mkdir('picture')
    
if not os.path.exists('cell_copy'):
    os.mkdir('cell_copy')

#cell_data内の画像をpictureにコピー
files = os.listdir('cell_data/')
files_folder = [f for f in files if os.path.isdir(os.path.join('cell_data', f))]

for f1 in files_folder:
    cells = os.listdir('cell_data/' + str(f1))
    
    if not os.path.isdir('picture/' + str(f1)):
        os.mkdir('picture/' + str(f1))
        
        
    for f2 in cells:
        copyfrom = 'cell_data/' + str(f1) + '/' + str(f2)
        copyto = 'cell_copy/' + str(f2)
        
        if not os.path.isdir('picture/' + str(f1) + '/' + str(f2)):
            os.mkdir('picture/' + str(f1) + '/' + str(f2))

        if not os.path.isdir(copyto):
            os.mkdir(copyto)
            print('{0}にフォルダを作成'.format(copyto))
        aug.folder_copy(copyfrom, copyto)

'''
picture/
　├ image/
　│   ├folder/
　│   │ ├ raw/
　│   │ ├ shift_and_deform/
　│   │ ├ rot_and_trans01/
　│   │...
　│   │ └ rot_and_trans31/
　│   │
　│   ├folder/
　│   │ ├raw/
　│   │ ...
　│   ...
　│   
　└ mask/
　    ├folder/
　    │ ├ raw/
　    │...
     ...
'''




sorts = os.listdir('cell_copy/')

for sort in sorts:
    files = os.listdir('cell_copy/' + str(sort) + '/')
    
    save_image = 'picture/image/' + str(sort) + '/'
    save_mask = 'picture/mask/' + str(sort) + '/'
    
    i = 0
    j = 1
    sd_i = 0
    sd_j = 1
    
    #画像、アノテーションのセットを1つずつ読み込み、処理後の画像を保存
    for file in files:
        if '.jpg' in file:
            image_path = 'cell_copy/' + str(sort) + '/' + file
            cell_path = 'cell_copy/'  + str(sort) + '/' + file.replace('.jpg', '.mask.0.png')
            nucleus_path = 'cell_copy/' + str(sort) + '/' + file.replace('.jpg', '.mask.1.png')
    
            image = cv2.imread(image_path)
            cell = cv2.imread(cell_path)
            nucleus = cv2.imread(nucleus_path)
            
            #crop
            image_crop, cell_crop, nucleus_crop = aug.crop(image, cell, nucleus, 360, 360)
            
            if i < folder_size:
                i += 1
            else:
                i = 1
                j += 1
                
            if not os.path.exists(save_image + 'raw_' + str(j) + '/'):
                os.mkdir(save_image + 'raw_' + str(j))
            
            if not os.path.exists(save_mask + 'raw_' + str(j) + '/'):
                os.mkdir(save_mask + 'raw_' + str(j))
            
            cv2.imwrite(save_image + 'raw_' + str(j) + '/' + file, image_crop)
            cv2.imwrite(save_mask + 'raw_' + str(j) + '/' + file.replace('.jpg', '.mask.0.png'), cell_crop)
            cv2.imwrite(save_mask + 'raw_' + str(j) + '/' + file.replace('.jpg', '.mask.1.png'), nucleus_crop)

 
            #shift and deformation
            t = 0
            for t in range(shift_and_deformation_time):
                t += 1
                if sd_i < folder_size:
                    sd_i += 1
                else:
                    sd_i = 1
                    sd_j += 1
                
                if not os.path.exists(save_image + 'shift_and_deform_' + str(sd_j) + '/'):
                    os.mkdir(save_image + 'shift_and_deform_' + str(sd_j))
                    
                if not os.path.exists(save_mask + 'shift_and_deform_' + str(sd_j) + '/'):
                    os.mkdir(save_mask + 'shift_and_deform_' + str(sd_j))
                    
                image_deform, cell_deform, nucleus_deform = aug.shift_and_deformation(image, cell, nucleus, 360, 360)
           
                cv2.imwrite(save_image + 'shift_and_deform_' + str(sd_j) + '/' + file.replace('.jpg', '.' + str(t) + '.jpg'), image_deform)
                cv2.imwrite(save_mask + 'shift_and_deform_' + str(sd_j) + '/' + file.replace('.jpg', '.' + str(t) + '.mask.0.png'), cell_deform)
                cv2.imwrite(save_mask + 'shift_and_deform_' + str(sd_j) + '/' + file.replace('.jpg', '.' + str(t) + '.mask.1.png'), nucleus_deform)
          
            #rotate and transpose 
            #1桁目は回転の回数、2桁目は転置の回数
            r = ['00', '10', '20', '30']
            t = ['01', '11', '21', '31'] 
            r_t = r + t
            
            image_rt_00 = image_crop
            cell_rt_00 = cell_crop
            nucleus_rt_00 = nucleus_crop
            
            #transpose
            image_rt_01 = image_crop[:,::-1]
            cell_rt_01 = cell_crop[:,::-1]
            nucleus_rt_01 = nucleus_crop[:,::-1]
            
            for n in range(1, 4):
                exec("image_rt_%s, cell_rt_%s, nucleus_rt_%s = aug.rotate(image_rt_%s, cell_rt_%s, nucleus_rt_%s)" 
                     % (r[n], r[n], r[n], r[n-1], r[n-1], r[n-1]))
            
                exec("image_rt_%s, cell_rt_%s, nucleus_rt_%s = aug.rotate(image_rt_%s, cell_rt_%s,  nucleus_rt_%s)" 
                     % (t[n], t[n], t[n], t[n-1], t[n-1], t[n-1]))
    
            for n in range(1, 8):
                if not os.path.exists(save_image + 'rot_and_trans' + r_t[n] + '_' + str(j) + '/'):
                    os.mkdir(save_image + 'rot_and_trans' + r_t[n] + '_' + str(j))
                    
                if not os.path.exists(save_mask + 'rot_and_trans' + r_t[n] + '_' + str(j) + '/'):
                    os.mkdir(save_mask + 'rot_and_trans' + r_t[n] + '_' + str(j))
                    
                exec("cv2.imwrite(save_image + 'rot_and_trans%s' + '_' + str(j) + '/' + file, image_rt_%s)" % (r_t[n], r_t[n]))
                exec("cv2.imwrite(save_mask + 'rot_and_trans%s' + '_' + str(j) + '/' + file.replace('.jpg', '.mask.0.png'), cell_rt_%s)" % (r_t[n], r_t[n]))
                exec("cv2.imwrite(save_mask + 'rot_and_trans%s' + '_' + str(j) + '/' + file.replace('.jpg', '.mask.1.png'), nucleus_rt_%s)" % (r_t[n], r_t[n]))
       
        else:
            pass

shutil.rmtree('cell_copy')
Raw
 aug_processor.py(old)
# -*- coding: utf-8 -*-
#Opencvをインストールしてから使う
#cell_data, augmentation.pyを同じディレクトリに置く

import cv2
import os
import augmentation as aug

#shift and deformationを1枚の画像に対して行う回数
shift_and_deformation_time = 10

if os.path.exists('picture'):
    pass
else:
    os.mkdir('picture')

#cell_data内の画像をpictureにコピー
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
        aug.folder_copy(copyfrom, copyto)

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

#上記のディレクトリを構成
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

    #画像、アノテーションのセットを1つずつ読み込み、処理後の画像を保存
    for file in files:
        if '.jpg' in file:
            image_path = 'picture/' + str(folder) + '/' + file
            cell_path = 'picture/' + str(folder) + '/' + file.replace('.jpg', '.mask.0.png')
            nucleus_path = 'picture/' + str(folder) + '/' + file.replace('.jpg', '.mask.1.png')

            image_rgb = cv2.imread(image_path)
            cell = cv2.imread(cell_path)
            nucleus = cv2.imread(nucleus_path)
            
            #grayscale
            image = aug.rgb2gray(image_rgb)
            
            #resize
            image360, image572, cell388, nucleus388 = aug.resize(image, cell, nucleus)
            
            cv2.imwrite('picture/' + str(folder) + '/aug/360/gray/' + file, image360)
            cv2.imwrite('picture/' + str(folder) + '/aug/572/gray/' + file, image572)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/gray/' + file.replace('.jpg', '.mask.0.png'), cell388)
            cv2.imwrite('picture/' + str(folder) + '/aug/388/gray/' + file.replace('.jpg', '.mask.1.png'), nucleus388)

            #shift and deformation
            i = 0
            for i in range(shift_and_deformation_time):
                i += 1
                
                image_deform360, image_deform572, cell_deform, nucleus_deform = aug.shift_and_deformation(image, cell, nucleus)
           
                cv2.imwrite('picture/' + str(folder) + '/aug/360/shift_and_deformation/' + file.replace('.jpg', '.' + str(i) + '.jpg'), image_deform360)
                cv2.imwrite('picture/' + str(folder) + '/aug/572/shift_and_deformation/' + file.replace('.jpg', '.' + str(i) + '.jpg'), image_deform572)
                cv2.imwrite('picture/' + str(folder) + '/aug/388/shift_and_deformation/' + file.replace('.jpg', '.' + str(i) + '.mask.0.png'), cell_deform)
                cv2.imwrite('picture/' + str(folder) + '/aug/388/shift_and_deformation/' + file.replace('.jpg', '.' + str(i) + '.mask.1.png'), nucleus_deform)
            
            #rotate and transpose 
            #1桁目は回転の回数、2桁目は転置の回数
            r = ['00', '10', '20', '30']
            t = ['01', '11', '21', '31'] 
            r_t = r + t
            
            image360_00 = image360
            image572_00 = image572
            cell388_00 = cell388
            nucleus388_00 = nucleus388
            
            #transpose
            image360_01 = image360[:,::-1]
            image572_01 = image572[:,::-1]
            cell388_01 = cell388[:,::-1]
            nucleus388_01 = nucleus388[:,::-1]
            
            for n in range(1, 4):
                exec("image360_%s, image572_%s, cell388_%s,  nucleus388_%s = aug.rotate(image360_%s, image572_%s, cell388_%s,  nucleus388_%s)" 
                     % (r[n], r[n], r[n], r[n], r[n-1], r[n-1], r[n-1], r[n-1]))
            
                exec("image360_%s, image572_%s, cell388_%s,  nucleus388_%s = aug.rotate(image360_%s, image572_%s, cell388_%s,  nucleus388_%s)" 
                     % (t[n], t[n], t[n], t[n], t[n-1], t[n-1], t[n-1], t[n-1]))
    
            for n in range(1, 8):
                exec("cv2.imwrite('picture/' + str(folder) + '/aug/360/rotate_and_transpose/' + file.replace('.jpg', '.%s.jpg'), image360_%s)" % (r_t[n], r_t[n]))
                exec("cv2.imwrite('picture/' + str(folder) + '/aug/572/rotate_and_transpose/' + file.replace('.jpg', '.%s.jpg'), image572_%s)" % (r_t[n], r_t[n]))
                exec("cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.%s.mask.0.png'), cell388_%s)" % (r_t[n], r_t[n]))
                exec("cv2.imwrite('picture/' + str(folder) + '/aug/388/rotate_and_transpose/' + file.replace('.jpg', '.%s.mask.1.png'), nucleus388_%s)" % (r_t[n], r_t[n]))
       
        else:
            pass