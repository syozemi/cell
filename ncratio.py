import os
import numpy as np
import process_data as pro
import pickle
import matplotlib.pyplot as plt
import random
import cv2 as cv
from tqdm import tqdm
from collections import defaultdict

result = {}

cell_list1 = os.listdir('data/image')
cell_list2 = os.listdir('data/image02')

for x in tqdm(cell_list1):
    if x in cell_list2:
        d = defaultdict(int)
        path1 = 'data/image/%s/%s' % (x,num_ncratio360)
        path2 = 'data/image02/%s' % (x,num_ncratio360)
        ncr1 = load(path1)
        ncr2 = load(path2)
        ncr = np.hstack((ncr1,ncr2))
    else:
        path = 'data/image/%s/%s' % (x,num_ncratio360)
        ncr = load(path)
    print(ncr.shape)
    n = len(ncr)
    mean = sum(ncr) / n
    for y in ncr:
        d[int(y//0.01)] += 1
    res = (n,mean,d)
    result[x] = res

print(result)






