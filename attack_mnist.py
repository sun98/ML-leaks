"""
File: attack_mnist.py
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


num_epochs = 50
num_epochs_attack = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define transform and load dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_set = dsets.MNIST(root="./data", train=True, download=True, transform=transform)
train_set = TensorDataset(train_set.data, train_set.targets)
test_set = dsets.MNIST(root="./data", train=False, download=True, transform=transform)
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
        assert data_tensor.size(0) == target_tensor.size(
            0
        )  # assuming first dimension is data size
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.tensors = (data_tensor, target_tensor)

    def __getitem__(self, index):
        data = self.data_tensor[index]
        target = self.target_tensor[index]
        data = Image.fromarray(data.numpy(), mode="L")
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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


target_model = CNN().to(device)
shadow_model = CNN().to(device)

# %% train the target model and the shadow model
optimizer_target = torch.optim.SGD(target_model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    for i, (images, labels) in tqdm(enumerate(loader_train_target)):
        outputs_target = target_model(images)
        loss_target = F.nll_loss(outputs_target, labels)
        optimizer_target.zero_grad()
        loss_target.backward()
        optimizer_target.step()
target_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader_nonmember_target:
        target = target_model(images)
        _, prediction = torch.max(target, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
    print(f"target model accu: {correct}/{total}={correct/total*100:.2f}%")

optimizer_shadow = torch.optim.SGD(shadow_model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    for i, (images, labels) in tqdm(enumerate(loader_train_shadow)):
        outputs_shadow = shadow_model(images)
        loss_shadow = F.nll_loss(outputs_shadow, labels)
        optimizer_shadow.zero_grad()
        loss_shadow.backward()
        optimizer_shadow.step()
shadow_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader_nonmember_target:
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

optimizer_attack = torch.optim.SGD(attack_model.parameters(), lr=0.001)
loss_attack = 1
for epoch in range(num_epochs_attack):
    print(f"epoch {epoch}/{num_epochs_attack}, loss={loss_attack}")
    for idx, (images, labels) in tqdm(enumerate(loader_train_attack)):
        posteriors_shadow = shadow_model(images)
        top3_posteriors = torch.topk(posteriors_shadow, 3, dim=1)[0]
        labels_attack = labels.float().unsqueeze(1)

        # attack_outputs = attack_model(posteriors_shadow)
        attack_outputs = attack_model(top3_posteriors)
        loss_attack = F.binary_cross_entropy(attack_outputs, labels_attack)

        optimizer_attack.zero_grad()
        loss_attack.backward()
        optimizer_attack.step()

# %% eval the performance of attack model
attack_model.eval()
total = 0
TP, TN, FP, FN = 0, 0, 0, 0

with torch.no_grad():
    for idx, (images, labels) in enumerate(loader_eval_attack):
        posteriors_target = target_model(images)
        top3_posteriors = torch.topk(posteriors_target, 3, dim=1)[0]

        # attack_outputs = attack_model(posteriors_target)
        attack_outputs = attack_model(top3_posteriors)
        # print(attack_outputs)

        pred = attack_outputs > 0.5
        real = labels.float() > 0.5
        if idx % 100 == 0:
            print(
                f"pred={pred.sum().item() / len(pred)} real={real.sum().item() / len(real)}"
            )
        TP += (pred & real).sum().item()
        TN += ((~pred) & (~real)).sum().item()
        FP += (pred & (~real)).sum().item()
        FN += ((~pred) & real).sum().item()


accu = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
print(
    f"Attack Model Accuracy on Test Set: accu={accu:.2f}, prec={precision:.2f} "
    f"recall={recall:.2f}, f1={f1:.2f}"
)
