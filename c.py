import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import process_data as pro
import pickle
import random
from collections import defaultdict

import torch_unet


image,mask,ncratio = pro.load_data_unet_ncr()

image = image.reshape(350,1,572,572).astype(np.float32)
mask = mask.reshape(350,3,388,388).astype(np.float32)

torch_unet.eval('model/torchmodel_unet0',image,ncratio)

































