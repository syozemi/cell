# -*- coding: utf-8 -*-
#Opencvをインストールしてから使う

import cv2
import os
import shutil
import numpy as np

def folder_copy(copyfrom, copyto):
    for filename in os.listdir(copyfrom):
        filepath = os.path.join(copyfrom, filename)
        
        if os.path.isdir(filepath):
            folder_copy(filepath, copyto)
            
        elif os.path.isfile(filepath):
            copypath = os.path.join(copyto, filename)
            shutil.copy(filepath, copypath)
            print('{0}から{1}にファイルをコピー'.format(filepath, copypath))

'''
#グレースケールにする   
def rgb2gray(image_rgb):
    image = np.dot(image_rgb, [0.299, 0.587, 0.114])
    image = image / 255.0
    
    return image
'''    

'''
#imageをimage_rの、cell,nucleusをmask_rの正方形に拡大/縮小
def resize(image, cell, nucleus, image_r, mask_r):  
    image_resize = cv2.resize(image, (image_r,image_r), interpolation=cv2.INTER_NEAREST)
    cell_resize = cv2.resize(cell, (mask_r,mask_r), interpolation=cv2.INTER_NEAREST)
    nucleus_resize = cv2.resize(nucleus, (mask_r,mask_r), interpolation=cv2.INTER_NEAREST)
    return image_resize, cell_resize, nucleus_resize
'''

#imageをimage_cの、cell,nucleusをmask_cの正方形で真ん中を切り抜く
def crop(image, cell, nucleus, image_c, mask_c):
    image_sizex = image.shape[0]
    image_sizey = image.shape[1]
    crop_x = int((image_sizex - image_c) / 2)
    crop_y = int((image_sizey - image_c) / 2)

    image_crop = image[crop_x:(crop_x + image_c), crop_y:(crop_y + image_c)]
    cell_crop = cell[crop_x:(crop_x + mask_c), crop_y:(crop_y + mask_c)]
    nucleus_crop = nucleus[crop_x:(crop_x + mask_c), crop_y:(crop_y + mask_c)]
    
    return image_crop, cell_crop, nucleus_crop
    

#x1~x2,y1~y2の範囲で切り抜き、imageをimage_sizeの、cell,nucleusをmask_sizeの正方形に引き延ばす
#np.random.randintの範囲は調整の余地あり
def shift_and_deformation(image, cell, nucleus, image_size, mask_size):
    image_sizex = image.shape[0]
    image_sizey = image.shape[1]
    
    x1 = np.random.randint(image_sizex / 9, image_sizex * 4 / 9)
    x2 = np.random.randint(image_sizex * 5 / 9, image_sizex * 8 / 9)
    y1 = np.random.randint(image_sizey / 9, image_sizey * 4 / 9)
    y2 = np.random.randint(image_sizey * 5 / 9, image_sizey * 8 / 9)
            
    image_trim = image[x1:x2, y1:y2]
    cell_trim = cell[x1:x2, y1:y2]
    nucleus_trim = nucleus[x1:x2, y1:y2]
        
    image_deform = cv2.resize(image_trim, (image_size,image_size), interpolation=cv2.INTER_NEAREST)
    cell_deform = cv2.resize(cell_trim, (mask_size,mask_size), interpolation=cv2.INTER_NEAREST)
    nucleus_deform = cv2.resize(nucleus_trim, (mask_size,mask_size), interpolation=cv2.INTER_NEAREST)

    return image_deform, cell_deform, nucleus_deform
   
#回転（転置は　image[:,::-1]　とすればよい）
def rotate(image_resize, cell_resize, nucleus_resize): 
    center_image = tuple(np.array([image_resize.shape[1] / 2, image_resize.shape[0] / 2]))
    center_cell = tuple(np.array([cell_resize.shape[1] / 2, cell_resize.shape[0] / 2]))
    center_nucleus = tuple(np.array([nucleus_resize.shape[1] / 2, nucleus_resize.shape[0] / 2]))
    
    size_image = tuple(np.array([image_resize.shape[1], image_resize.shape[0]]))
    size_cell = tuple(np.array([cell_resize.shape[1], cell_resize.shape[0]]))
    size_nucleus = tuple(np.array([nucleus_resize.shape[1], nucleus_resize.shape[0]]))
    
    affine_matrix_image = cv2.getRotationMatrix2D(center_image, 90.0, 1.0)
    affine_matrix_cell = cv2.getRotationMatrix2D(center_cell, 90.0, 1.0)
    affine_matrix_nucleus = cv2.getRotationMatrix2D(center_nucleus, 90.0, 1.0)
    
    image_r = cv2.warpAffine(image_resize, affine_matrix_image, size_image, flags=cv2.INTER_NEAREST)
    cell_r = cv2.warpAffine(cell_resize, affine_matrix_cell, size_cell, flags=cv2.INTER_NEAREST)
    nucleus_r = cv2.warpAffine(nucleus_resize, affine_matrix_nucleus, size_nucleus, flags=cv2.INTER_NEAREST)

    return image_r, cell_r, nucleus_r