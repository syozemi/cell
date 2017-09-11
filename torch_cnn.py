import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import batch
import process_data as pro


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
            nn.Linear(64*9*9,100),
            nn.ReLU())
        self.fc = nn.Linear(100,10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.layer6(out)
        out = self.fc(out)
        return out

image,ncn10,ncn100,ncr10,ncr100 = pro.load_data_cnn_torch()

image = image.reshape(350,1,360,360).astype(np.float32)

train_x = batch.Batch(image)
train_y = batch.Batch(ncn10)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
learningtime = 50
for i in range(learningtime):
    batch_x = train_x.next_batch(50)
    batch_y = train_y.next_batch(50)
    x = Variable(torch.from_numpy(batch_x))
    y = Variable(torch.from_numpy(batch_y))
    optimizer.zero_grad()
    out = net(x)
    loss = criterion(out,y)
    loss.backward()
    optimizer.step()
    print(loss)
    print(str(i)+'/'+str(learningtime))
torch.save(net, 'model/torchmodel')

