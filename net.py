"""
File: net.py
Author: Suibin Sun
Created Date: 2023-12-25, 6:27:31 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 6:27:31 pm
-----
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, nr_channel, nr_output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(nr_channel, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1_width = 500 if nr_channel == 3 else 320
        self.fc1 = nn.Linear(self.fc1_width, 50)
        self.fc2 = nn.Linear(50, nr_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Mnist_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(in_features=4 * 4 * 50, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 4 * 4 * 50)  # flattening
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class Cifar10_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(16 * 5 * 5)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.tanh(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)  # flattening
        # x = self.batch_norm(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class AttackNet(nn.Module):
    def __init__(self, in_features, out_features=2):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=out_features)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        print(x.shape)
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
