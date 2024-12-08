import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, droprate=0.8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout(p=droprate)  # 添加Dropout层

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # 应用Dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_classes, droprate=0.5):
        super(ResNet3D, self).__init__()
        self.in_channels = 16  # 初始通道数
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2, bias=False)  # 输入1个通道
        self.bn1 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, 2, stride=1, droprate=droprate)  # 第一层
        self.layer2 = self._make_layer(block, 32, 2, stride=2, droprate=droprate)  # 第二层
        self.layer3 = self._make_layer(block, 64, 2, stride=2, droprate=droprate)  # 第三层
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)  # 输出层

    def _make_layer(self, block, out_channels, blocks, stride, droprate):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, droprate))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, droprate=droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet3D_5Layers(num_classes):
    return ResNet3D(BasicBlock, num_classes, droprate=0.8)  # 设置Dropout率