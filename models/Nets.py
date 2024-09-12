#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)#输入层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()#dropout随机归零 Dropout 是一种正则化技术，用于在训练过程中随机丢弃部分神经元，以防止过拟合。
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)#隐含层

    def forward(self, x): #forward前向传播
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1]) #x.view()就是对tensor进行reshape 其中第一维的大小为 `-1`，表示 PyTorch 自动推断这维的大小；第二维的大小为 `x.shape[1]*x.shape[-2]*x.shape[-1]`，这是所有特征维度的乘积。 - 这种操作通常用于将多维的输入张量展平为一维，以便通过全连接层进行处理。
        x = self.layer_input(x)#Linear
        x = self.dropout(x)#
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)#第一个卷积层，将输入的通道数（args.num_channels）转换为 10 个输出通道，卷积核大小为 5x5。
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)#第二个卷积层，将 10 个通道的输入转换为 20 个输出通道，卷积核大小为 5x5。
        self.conv2_drop = nn.Dropout2d()#在第二个卷积层之后应用的 dropout 层。nn.Dropout2d() 用于在卷积层的输出中随机丢弃一些通道，以帮助防止过拟合。
        self.fc1 = nn.Linear(320, 50)#第一个全连接层，将输入特征大小 320 转换为 50 个输出特征。
        self.fc2 = nn.Linear(50, args.num_classes)#第二个全连接层，将 50 个特征转换为 args.num_classes 个输出特征，用于分类。

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])#将多维的卷积层输出展平为一维，以便输入到全连接层。x.shape[1]*x.shape[2]*x.shape[3] 计算出展平后的特征数量
        x = F.relu(self.fc1(x))#将展平后的张量通过第一个全连接层，并应用 ReLU 激活函数
        x = F.dropout(x, training=self.training)#对全连接层的输出应用 dropout，以帮助防止过拟合。training=self.training 确保只有在训练模式下才会应用 dropout。
        x = self.fc2(x) #将 dropout 之后的输出通过第二个全连接层，得到最终的分类结果
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
