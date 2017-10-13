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
from tqdm import tqdm


ones = torch.ones(3,1,5,5)

ones = torch.sum(torch.sum(ones,2),3)

print(ones.size())

print(ones)


















