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

n = random.randint(0,350)

image = image[n,...]

c = image[1]
n = image[2]

fig = plt.figure(figsize=(8,8))
sub = fig.add_subplot(1,2,1)
sub.imshow(c,cmap='gray')
sub = fig.add_subplot(1,2,2)
sub.imshow(n,cmap='gray')





















