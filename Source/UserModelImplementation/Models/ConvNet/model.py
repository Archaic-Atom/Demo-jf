# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import time
from torch.nn import functional as F
from math import floor, ceil


def conv(in_channels: int, out_channels: int, kernel_size: int=3,
         stride: int=1, padding: int=1, bias: bool=False,
         bn: bool=True, act: bool=True):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    ]

    if bn:
        layers.append(nn.BatchNorm2d(out_channels))

    if act:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10) -> object:
        super().__init__()
        self._in_channels = in_channels

        self.conv_0 = conv(1, 16)
        self.downsample_0 = self.make_layer(16, 32, 2)
        self.downsample_1 = self.make_layer(32, 64, 2)
        self.downsample_2 = self.make_layer(64, 128, 2)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, in_channels: int, out_channels: int, layers_num: int) -> object:
        layers = [conv(in_channels, out_channels, stride=2)]
        for _ in range(layers_num):
            layers.append(conv(out_channels, out_channels, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.downsample_0(x)
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
