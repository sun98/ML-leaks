# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from net import Net, AttackNet


# %% 加载MNIST数据集并进行预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ConvertImageDtype(torch.float),
    ]
)

dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# %% 切分数据集为训练集和影子模型的训练集
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, shadow_dataset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
shadow_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)

# %% 假设你已经有了一个预训练的目标模型和一个影子模型
target_model = Net(1, 10)
shadow_model = Net(1, 10)

# %% 训练影子模型
# optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# num_epochs = 10

# for epoch in range(num_epochs):
#     print(f"epoch {epoch}/{num_epochs}")
#     for images, labels in tqdm(shadow_loader):
#         optimizer.zero_grad()
#         outputs = shadow_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# %% 将目标模型和影子模型设置为评估模式
target_model.eval()
shadow_model.eval()

# %% 计算目标模型和影子模型在训练集上的后验概率
train_targets = []
train_shadow_posteriors = []
with torch.no_grad():
    for images, labels in trainloader:
        targets = labels.numpy()
        train_targets.extend(targets.tolist())

        # 获取目标模型的后验概率
        target_posteriors = target_model(images)
        # _, predicted = torch.max(target_posteriors.data, 1)
        posteriors = F.softmax(target_posteriors, dim=1)
        # print(f"posteriors: {posteriors.shape} {posteriors[:, predicted].shape}")
        train_shadow_posteriors.extend(posteriors.tolist())

# train_shadow_posteriors_top3 = []
# with torch.no_grad():
#     for images, labels in trainloader:
#         targets = labels.numpy()
#         train_targets.extend(targets.tolist())

#         # 获取目标模型的后验概率
#         target_posteriors = target_model(images)
#         posteriors = F.softmax(target_posteriors, dim=1)

#         # 获取每个样本后验概率最高的三个类别及其概率
#         _, top3_indices = torch.topk(posteriors, k=3, dim=1)
#         top3_probabilities = torch.gather(posteriors, dim=1, index=top3_indices)

#         # 扁平化并存储最高三个后验概率
#         top3_probabilities_flattened = top3_probabilities.reshape(-1)
#         train_shadow_posteriors_top3.extend(top3_probabilities_flattened.tolist())

# %% 计算目标模型和影子模型在测试集上的后验概率
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

test_targets = []
test_shadow_posteriors = []
with torch.no_grad():
    for images, labels in testloader:
        targets = labels.numpy()
        test_targets.extend(targets.tolist())

        # 获取影子模型的后验概率
        shadow_posteriors = shadow_model(images)
        # _, predicted = torch.max(shadow_posteriors.data, 1)
        posteriors = F.softmax(shadow_posteriors, dim=1)
        test_shadow_posteriors.extend(posteriors.tolist())

# %% 使用训练集和影子模型的后验概率训练攻击模型
attack_dataset = [
    (torch.tensor(posterior), torch.tensor(int(target)))
    for posterior, target in zip(train_shadow_posteriors, train_targets)
]
attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

attack_model = AttackNet(in_features=10, out_features=2)
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 10

for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    for inputs, targets in tqdm(attack_loader):
        optimizer.zero_grad()
        outputs = attack_model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

# %% 使用测试集和目标模型的后验概率评估攻击模型
attack_predictions = []
with torch.no_grad():
    for posterior in test_shadow_posteriors:
        input_tensor = torch.tensor([posterior])
        prediction = torch.sigmoid(attack_model(input_tensor)).item()
        attack_predictions.append(prediction >= 0.5)

# %% 计算精度和召回率
precision = precision_score(test_targets, attack_predictions)
recall = recall_score(test_targets, attack_predictions)

print("Precision: ", precision)
print("Recall: ", recall)
