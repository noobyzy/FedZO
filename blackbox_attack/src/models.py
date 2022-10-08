import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


'''
CIFAR-10 : 

test acc  = 8,230/10,000

train acc = 49,849/50,000
-------------------------------
label | correct predict / total
-------------------------------
0     | 4,987 / 5,000
-------------------------------
1     | 4.995 / 5,000
-------------------------------
2     | 4,980 / 5,000
-------------------------------
3     | 4,949 / 5,000
-------------------------------
4     | 4,992 / 5,000
-------------------------------
5     | 4,979 / 5,000
-------------------------------
6     | 4,988 / 5,000
-------------------------------
7     | 4,996 / 5,000
-------------------------------
8     | 4,993 / 5,000
-------------------------------
9     | 4,990 / 5,000
-------------------------------
'''
class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict


'''
MNIST : 

test acc  = 9,948/10,000

train acc = 59,997/60,000
-------------------------------
label | correct predict / total
-------------------------------
0     | 5,923 / 5,923
-------------------------------
1     | 6,742 / 6,742
-------------------------------
2     | 5,958 / 5,958
-------------------------------
3     | 6,131 / 6,131
-------------------------------
4     | 5,841 / 5,842
-------------------------------
5     | 5,421 / 5,421
-------------------------------
6     | 5,918 / 5,918
-------------------------------
7     | 6,265 / 6,265
-------------------------------
8     | 5,851 / 5,851
-------------------------------
9     | 5,947 / 5,949
-------------------------------

===============================

FMNIST : 

test acc = 9,205/10,000

train acc = 59,650/60,000
-------------------------------
label | correct predict / total
-------------------------------
0     | 5,970 / 6,000
-------------------------------
1     | 5,999 / 6,000
-------------------------------
2     | 5,948 / 6,000
-------------------------------
3     | 5,988 / 6,000
-------------------------------
4     | 5,963 / 6,000
-------------------------------
5     | 5,993 / 6,000
-------------------------------
6     | 5,873 / 6,000
-------------------------------
7     | 5,918 / 6,000
-------------------------------
8     | 5,999 / 6,000
-------------------------------
9     | 5,999 / 6,000
-------------------------------
'''

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200,200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,1,28,28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict



