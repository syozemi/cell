import os
import matplotlib.pyplot as plt
#import process_data as pro
import random
import pickle
import numpy as np
import time
import cv2 as cv
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import process_data as pro

image, mask, num_mask = pro.load_unet2_data(0,mode=0)

b = mask[:,0,:,:]
c = mask[:,1,:,:]
n = mask[:,2,:,:]

bb = np.sum(b)
cc = np.sum(c)
nn = np.sum(n)

print(bb)
print(cc)
print(nn)

