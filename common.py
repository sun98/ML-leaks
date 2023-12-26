"""
File: common.py
Author: Suibin Sun
Created Date: 2023-12-25, 8:27:28 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 8:27:28 pm
-----
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

batch_size = 256
learning_rate = 0.001
nr_epochs_mnist = 2
nr_epochs_cifar10 = 20

poison_rate_train = 0.01
poison_size = 5

transform_mnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ConvertImageDtype(torch.float),
    ]
)

transform_cifar = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class MNISTNet(nn.Module):
    def __init__(self, nr_channel, nr_output):
        super(MNISTNet, self).__init__()
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


class CIFAR10Net(nn.Module):
    def __init__(self, nr_channel, nr_output):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=nr_channel, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=nr_output)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.tanh(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


criterion = nn.CrossEntropyLoss()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, trainloader, optimizer, nr_epochs, testloader):
    for epoch in range(nr_epochs):
        model.train()
        print(f"training in epoch {epoch}/{nr_epochs}")
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            test_model(model, testloader)
    return model


def test_model(model, testloader):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print("Accuracy of %1s : %2d %%" % (i, 100 * class_correct[i] / class_total[i]))
