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
from collections import defaultdict

image,mask,tmask = pro.load_data_unet_torch2()

print(image.shape)

print(mask.shape)

n = random.randint(0,350)

image = image[n,...].reshape(284,284)
mask = mask[n,...]

cc = mask[1,:,:]
nn = mask[2,:,:]

d = defaultdict(int)

for x in cc:
    for y in x:
        d[y] += 1

for x in nn:
    for y in x:
        d2[y] += 1

print(d)

print(d2)

fig = plt.figure(figsize=(10,10))
sub = fig.add_subplot(1,3,1)
sub.imshow(image,cmap='gray')
sub = fig.add_subplot(1,3,2)
sub.imshow(cc,cmap='gray')
sub = fig.add_subplot(1,3,3)
sub.imshow(nn,cmap='gray')
plt.show()

print(n)



















