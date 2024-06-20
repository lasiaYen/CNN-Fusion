import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json


# 定义Siamese网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=3, mode='bilinear', align_corners=True)
        self.fc1 = nn.Linear(256 * 15 * 30, 128)
        self.fc2 = nn.Linear(128, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        combined = torch.cat([output1, output2], dim=1)
        combined = combined.view(combined.size(0), -1)
        combined = F.relu(self.fc1(combined))
        output = self.fc2(combined)
        return output


# 自定义数据集
class PatchDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 读取标签文件
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        self.folder_list = list(self.labels.keys())

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        imgA_path = f'{self.root_dir}/{folder_name}/A.jpeg'
        imgB_path = f'{self.root_dir}/{folder_name}/B.jpeg'
        imageA = Image.open(imgA_path)
        imageB = Image.open(imgB_path)
        label = self.labels[folder_name]

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB, label


# 数据加载和预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

root_dir = './dataset/data'
labels_file = './dataset/label/labels.json'
epoch = 15
batch_size = 128
lr = 0.0001
momentum = 0.9
weight_decay = 0.0005


if __name__ == "__main__":
    dataset = PatchDataset(
        root_dir=root_dir, labels_file=labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # 训练Siamese网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

    min_loss = float('inf')
    weight = None

    for e in range(epoch):
        epoch_loss = 0
        batch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            patch1, patch2, labels = data
            patch1, patch2, labels = patch1.to(
                device), patch2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patch1, patch2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 9:
                print(
                    f'[Epoch {e + 1}, Batch {i + 1}] Loss: {batch_loss / 10}')
                batch_loss = 0.0
        print(epoch_loss)
        if (epoch_loss < min_loss):
            weight = model.state_dict()
            min_loss = epoch_loss

            # 保存模型权重
    torch.save(weight, './weight/siamese_model_weights.pth')
    print('Finished Training')
