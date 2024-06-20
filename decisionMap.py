import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from train import SiameseNetwork
from smallRegionFilter import smallRegionFilter
# 定义转换后的Siamese网络


class SiameseNetworkConverted(nn.Module):
    def __init__(self):
        super(SiameseNetworkConverted, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_fc1 = nn.Conv2d(
            512, 128, stride=1, padding=7, kernel_size=15)  # 转换后的全连接层
        self.conv_fc2 = nn.Conv2d(
            128, 2,   kernel_size=1)  # 转换后的全连接层

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
        combined = torch.cat([output1, output2], dim=1)  # 在channel维度上拼接
        combined = self.conv_fc1(combined)
        output = self.conv_fc2(combined)
        return output


def load_weights_and_convert(model, original_model_weights):
    original_model = SiameseNetwork()
    original_model.load_state_dict(torch.load(original_model_weights))

    # 复制卷积层的权重
    model.conv1.load_state_dict(original_model.conv1.state_dict())
    model.conv2.load_state_dict(original_model.conv2.state_dict())
    model.conv3.load_state_dict(original_model.conv3.state_dict())

    # 转换并复制全连接层的权重
    fc1_weight = original_model.fc1.weight.view(128, 512, 15, 15)
    fc1_bias = original_model.fc1.bias
    model.conv_fc1.weight.data.copy_(fc1_weight)
    model.conv_fc1.bias.data.copy_(fc1_bias)

    fc2_weight = original_model.fc2.weight.view(2, 128, 1, 1)
    fc2_bias = original_model.fc2.bias
    model.conv_fc2.weight.data.copy_(fc2_weight)
    model.conv_fc2.bias.data.copy_(fc2_bias)

    return model


# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)


# 加载并转换模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
converted_model = SiameseNetworkConverted().to(device)
converted_model = load_weights_and_convert(
    converted_model, './weight/pre-train.pth')
converted_model.eval()


# 加载两张源图像
imageA_path = './fusionData/far.jpg'
imageB_path = './fusionData/near.jpg'
imageA = load_and_preprocess_image(imageA_path)
imageB = load_and_preprocess_image(imageB_path)

# 推理生成得分图
with torch.no_grad():
    output = converted_model(imageA, imageB)
    # 只保留 softMax 后第一个标签的值，即取值为 0 的概率
    score_map = F.softmax(output, dim=1)[:, 1, :, :].cpu().numpy()

score_map_image = (score_map * 255).squeeze().astype(np.uint8)

# 当取值为 0 的概率大于 0.5 时，即认为在当前像素中，ImageA 比 ImageB 模糊，所以当 decision_map 为 0 时，选中 ImageB 的像素
decision_map_image = np.where(
    score_map > 0.5, 0, 255).squeeze().astype(np.uint8)

decision_map_filtered_image = smallRegionFilter(decision_map_image)
decision_map_filtered_image = smallRegionFilter(decision_map_filtered_image)

cv2.imwrite('./decisionMap/score_map.jpg', score_map_image)
cv2.imwrite('./decisionMap/decision_map.jpg', decision_map_image)
cv2.imwrite('./decisionMap/decision_map_filtered.jpg',
            decision_map_filtered_image)
