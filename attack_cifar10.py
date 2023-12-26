"""
File: attack_cifar10.py
Author: Suibin Sun
Created Date: 2023-12-25, 8:46:48 pm
-----
Last Modified by: Suibin Sun
Last Modified: 2023-12-25, 8:46:48 pm
-----
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from PIL import Image
import numpy as np


num_epochs_target = 10
num_epochs_shadow = 10
num_epochs_attack = 10

# NOTE: this code is only tested on CPU, there may be some issues on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# define transform and load dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = dsets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_set = TensorDataset(torch.tensor(train_set.data), torch.tensor(train_set.targets))
test_set = dsets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# %% split the dataset in 4 equal parts
split_size = len(train_set) // 4

data_train_shadow = Subset(train_set, range(0, split_size))
data_out_shadow = Subset(train_set, range(split_size, split_size * 2))
data_train_attack = Subset(train_set, range(0, split_size * 2))
data_train_target = Subset(train_set, range(split_size * 2, split_size * 3))
data_nonmember_target = Subset(train_set, range(split_size * 3, len(train_set)))
data_eval_attack = Subset(train_set, range(split_size * 2, len(train_set)))


# make sure the splitted dataset are transformed
class TransformedTensorDataset(TensorDataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.tensors = (data_tensor, target_tensor)

    def __getitem__(self, index):
        data = self.data_tensor[index]
        target = self.target_tensor[index]
        data = Image.fromarray(data.numpy())
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return self.data_tensor.size(0)


def subset_to_tensor(subset):
    return TransformedTensorDataset(
        torch.stack([subset[i][0] for i in range(len(subset))]),
        torch.tensor([subset[i][1] for i in range(len(subset))]),
        transform=transform,
    )


data_train_shadow = subset_to_tensor(data_train_shadow)
data_out_shadow = subset_to_tensor(data_out_shadow)
data_train_attack = subset_to_tensor(data_train_attack)
data_train_target = subset_to_tensor(data_train_target)
data_nonmember_target = subset_to_tensor(data_nonmember_target)
data_eval_attack = subset_to_tensor(data_eval_attack)

# Create labels for training of attack model
label_train_attack = torch.cat(
    (torch.ones(len(data_train_shadow)), torch.zeros(len(data_out_shadow))), dim=0
)
data_train_attack = TransformedTensorDataset(
    data_train_attack.tensors[0], label_train_attack, transform=transform
)

# create labels for evaluating the attack model
label_eval_attack = torch.cat(
    (torch.ones(len(data_train_target)), torch.zeros(len(data_nonmember_target))), dim=0
)
data_eval_attack = TransformedTensorDataset(
    data_eval_attack.tensors[0], label_eval_attack, transform=transform
)


loader_train_shadow = DataLoader(data_train_shadow, batch_size=64, shuffle=True)
loader_out_shadow = DataLoader(data_out_shadow, batch_size=64, shuffle=True)
loader_train_attack = DataLoader(data_train_attack, batch_size=64, shuffle=True)
loader_train_target = DataLoader(data_train_target, batch_size=64, shuffle=True)
loader_nonmember_target = DataLoader(data_nonmember_target, batch_size=64, shuffle=True)
loader_eval_attack = DataLoader(data_eval_attack, batch_size=64, shuffle=True)


# %% define the CNN for the target model and the shadow model
class CNN(nn.Module):
    def __init__(self, input_size=3):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(
            in_channels=input_size, out_channels=48, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if input_size == 3:
            self.fc_features = 6 * 6 * 96
        else:
            self.fc_features = 5 * 5 * 96
        self.fc1 = nn.Linear(in_features=self.fc_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_features)  # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


target_model = CNN().to(device)
shadow_model = CNN().to(device)
target_loss = nn.CrossEntropyLoss()
shadow_loss = nn.CrossEntropyLoss()


# %% train the target model
optimizer_target = torch.optim.Adam(target_model.parameters(), lr=0.001)
loss = 999
for epoch in range(num_epochs_target):
    print(f"epoch {epoch}/{num_epochs_target} loss={loss}")
    loss = 0
    for i, (images, labels) in tqdm(enumerate(loader_train_target)):
        images = images.to(device)
        labels = labels.to(device)
        outputs_target = target_model(images)
        loss_target = target_loss(outputs_target, labels)
        optimizer_target.zero_grad()
        loss_target.backward()
        optimizer_target.step()
        loss += loss_target
target_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader_train_target:
        images = images.to(device)
        labels = labels.to(device)
        target = target_model(images)
        _, prediction = torch.max(target, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
    print(f"target model accu (train set): {correct}/{total}={correct/total*100:.2f}%")
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader_nonmember_target:
        images = images.to(device)
        labels = labels.to(device)
        target = target_model(images)
        _, prediction = torch.max(target, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
    print(f"target model accu (test set): {correct}/{total}={correct/total*100:.2f}%")

# %% train the shadow model
optimizer_shadow = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
loss = 999
for epoch in range(num_epochs_shadow):
    print(f"epoch {epoch}/{num_epochs_target} loss={loss}")
    loss = 0
    for i, (images, labels) in tqdm(enumerate(loader_train_shadow)):
        images = images.to(device)
        labels = labels.to(device)
        outputs_shadow = shadow_model(images)
        loss_shadow = shadow_loss(outputs_shadow, labels)
        optimizer_shadow.zero_grad()
        loss_shadow.backward()
        optimizer_shadow.step()
        loss += loss_shadow
shadow_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader_nonmember_target:
        images = images.to(device)
        labels = labels.to(device)
        target = shadow_model(images)
        _, prediction = torch.max(target, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
    print(f"shadow model accu: {correct}/{total}={correct/total*100:.2f}%")


# %% definition of attack model
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


attack_model = AttackModel().to(device)

# %% train the attack model

optimizer_attack = torch.optim.Adam(attack_model.parameters(), lr=0.001)
loss = 999
for epoch in range(num_epochs_attack):
    print(f"epoch {epoch}/{num_epochs_attack}, loss={loss}")
    loss = 0
    for idx, (images, labels) in tqdm(enumerate(loader_train_attack)):
        images = images.to(device)
        labels = labels.to(device)
        posteriors_shadow = F.softmax(shadow_model(images), dim=1)
        top3_posteriors = torch.topk(posteriors_shadow, 3, dim=1)[0]
        labels_attack = labels.float().unsqueeze(1)

        # attack_outputs = attack_model(posteriors_shadow)
        attack_outputs = attack_model(top3_posteriors)
        loss_attack = F.binary_cross_entropy(attack_outputs, labels_attack)

        optimizer_attack.zero_grad()
        loss_attack.backward()
        optimizer_attack.step()
        loss += loss_attack

# %% eval the performance of attack model
attack_model.eval()
total = 0

all_pred = torch.empty((1))
all_labels = torch.empty((1))

with torch.no_grad():
    for images, labels in loader_eval_attack:
        images = images.to(device)
        labels = labels.to(device)
        posteriors_target = F.softmax(target_model(images), dim=1)
        top3_posteriors = torch.topk(posteriors_target, 3, dim=1)[0]

        # attack_outputs = attack_model(posteriors_target)
        attack_outputs = attack_model(top3_posteriors)

        all_pred = torch.cat((all_pred, attack_outputs.squeeze()), dim=0)
        all_labels = torch.cat((all_labels, labels.squeeze()), dim=0)

# %%
for pred_thre in np.arange(0.20, 0.65, 0.05):
    all_pred_bin = all_pred > pred_thre
    all_labels_bin = all_labels > 0.5  # only 0 or 1

    TP = (all_pred_bin & all_labels_bin).sum().item()
    TN = ((~all_pred_bin) & (~all_labels_bin)).sum().item()
    FP = (all_pred_bin & (~all_labels_bin)).sum().item()
    FN = ((~all_pred_bin) & all_labels_bin).sum().item()

    print(
        f"Attack Model Accuracy for thre={pred_thre:.2f} TP={TP} TN={TN} FP={FP} FN={FN} ",
        end="",
    )
    if (TP + FP) == 0 or (TP + FN) == 0:
        print("metrics invalid")
        continue
    accu = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"accu={accu:.2f} " f"prec={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
