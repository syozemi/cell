import os
import matplotlib.pyplot as plt
import process_data as pro
import random
import pickle
import numpy as np
import time
import batch
import cv2 as cv
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

image,mask,tmask = pro.load_data_wnet()

print(image.shape)

print(mask.shape)

n = random.randint(0,350)

image = image[n,...]
mask = mask[n,...]

i = image[0]
c = image[1]
n = image[2]

cc = mask[:,:,1]
nn = mask[:,:,2]

fig = plt.figure(figsize=(10,10))
sub = fig.add_subplot(3,2,1)
sub.imshow(i,cmap='gray')
sub = fig.add_subplot(3,2,3)
sub.imshow(cc,cmap='gray')
sub = fig.add_subplot(3,2,4)
sub.imshow(nn,cmap='gray')
sub = fig.add_subplot(3,2,5)
sub.imshow(c,cmap='gray')
sub = fig.add_subplot(3,2,6)
sub.imshow(n,cmap='gray')
plt.show()





















