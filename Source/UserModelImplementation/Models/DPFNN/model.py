# In[1] 导入所需工具包
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import time
from torch.nn import functional as F
from math import floor, ceil


# In[1] 定义卷积核
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


# In[1] 定义残差块
class DPFNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(DPFNNBlock, self).__init__()
        self.conv1_1 = conv3x3(in_channels, out_channels, stride)
        self.conv1_2 = conv3x3(in_channels, out_channels, stride)
        self.conv1_3 = conv3x3(in_channels, out_channels, stride)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        self.bn1_3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv2_1 = conv3x3(out_channels, out_channels)
        self.conv2_2 = conv3x3(out_channels, out_channels)
        self.conv2_3 = conv3x3(out_channels, out_channels)
        self.bn2_1 = nn.BatchNorm2d(out_channels)
        self.bn2_2 = nn.BatchNorm2d(out_channels)
        self.bn2_3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1_1(x)
        out1 = self.bn1_1(out1)

        out2 = self.conv1_2(x)
        out2 = self.bn1_2(out2)

        out3 = self.conv1_3(x)
        out3 = self.bn1_3(out3)
        # 三条支路均通过3X3卷积
        out1 = out1 + out2 + out3
        out1 = self.relu1_1(out1)

        out2 = out1 + out2 + out3
        out2 = self.relu1_2(out2)

        out3 = out1 + out2 + out3
        out3 = self.relu1_3(out3)
        # 经过卷积之后进行跨层连接
        out1 = self.conv2_1(out1)
        out1 = self.bn2_1(out1)

        out2 = self.conv2_2(out2)
        out2 = self.bn2_2(out2)

        out3 = self.conv2_1(out3)
        out3 = self.bn2_3(out3)

        # 下采样
        if self.downsample:
            residual = self.downsample(x)
        out1 += 2 * residual
        out2 += 2 * residual
        out3 += 2 * residual
        out = out1 + out2 + out3
        out = self.relu(out)
        return out

# In[1] 搭建残差神经网络


class DPFNNModel(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(DPFNNModel, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 构建残差块,恒等映射
        # in_channels == out_channels and stride = 1 所以这里我们构建残差块,没有下采样
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        # 不构建残差块,进行了下采样
        # layers中记录的是数字,表示对应位置的残差块数目
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        # 不构建残差块,进行了下采样
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(3136, 128)
        self.normfc12 = nn.LayerNorm((128), eps=1e-5)
        self.fc2 = nn.Linear(128, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 当out_channels = 32时,in_channels也变成32了
        self.in_channels = out_channels
        # blocks是残差块的数目
        # 残差块之后的网络结构,是out_channels->out_channels的
        # 可以说,make_layer做的是输出尺寸相同的所有网络结构
        # 由于输出尺寸会改变,我们用make_layers去生成一大块对应尺寸完整网络结构
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # layer1是三块in_channels等于16的网络结构，包括三个恒等映射
        out = self.layer1(out)
        # layer2包括了16->32下采样,然后是32的三个恒等映射
        out = self.layer2(out)
        # layer3包括了32->64的下采样,然后是64的三个恒等映射
        out = self.layer3(out)
        # out = self.avg_pool(out)
        # 全连接压缩
        # out.size(0)可以看作是batch_size
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.normfc12(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
