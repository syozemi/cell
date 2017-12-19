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
import sys

from c import Net,Conv,Up,MulWeight

model = 'model/unet2/1'

net = torch.load(model, location:'cpu')

