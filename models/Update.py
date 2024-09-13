#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):  # 定义了一个名为 LocalUpdate 的类，主要用于处理本地训练的逻辑
    def __init__(self, args, dataset=None, idxs=None): # 构造函数，用于初始化类的实例
                                # 包含训练参数的命名空间，例如学习率、批量大小等
                                # 数据集对象，通常是一个 PyTorch 数据集
                                # 指定的数据集索引，用于选择特定的训练样本
        self.args = args 
        self.loss_func = nn.CrossEntropyLoss() # self.loss_func = nn.CrossEntropyLoss(): 定义损失函数为交叉熵损失，用于分类问题
        self.selected_clients = [] # 初始化一个空列表，可能用于存储选中的客户端（在联邦学习中）
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)# self.ldr_train = DataLoader(...): 创建一个数据加载器，使用 DatasetSplit 将数据集分割为指定的索引，并设置批量大小和是否打乱数据

    def train(self, net):
        net.train() # net.train(): 将模型设置为训练模式，这会启用 Dropout 和 BatchNorm 等层的训练行为
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)# 使用随机梯度下降（SGD）优化器，设置学习率和动量。net.parameters() 获取模型的可训练参数

        epoch_loss = [] # 初始化一个列表，用于记录每个 epoch 的平均损失
        for iter in range(self.args.local_ep): # 遍历本地训练的 epoch 数量
            batch_loss = []# 初始化一个列表，用于记录每个批次的损失
            for batch_idx, (images, labels) in enumerate(self.ldr_train):# 遍历数据加载器中的每个批次，images 是输入数据，labels 是目标标签
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad() # 清零模型的梯度，以便进行新的反向传播
                log_probs = net(images) # 通过模型进行前向传播，得到每个类别的对数概率
                loss = self.loss_func(log_probs, labels) # 计算当前批次的损失
                loss.backward() # loss.backward(): 进行反向传播，计算梯度
                optimizer.step() # optimizer.step(): 更新模型参数
                if self.args.verbose and batch_idx % 10 == 0:   # 如果 verbose 为真，并且当前批次索引是 10 的倍数，打印当前 epoch、已处理样本数、总样本数和当前批次的损失
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item()) # 将当前 epoch 的平均损失添加到 epoch_loss 列表中 
                                               # item() 方法是 PyTorch 中一个非常实用的工具，特别是在处理损失值、模型输出或其他需要从张量中提取单个数值的场景中。使用时需要确保张量是标量，以避免引发错误
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) # 将当前 epoch 的平均损失添加到 epoch_loss 列表中
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
                # 通过torch.save(model.state_dict(),"model_name.pth")将model的参数保存为python的字典类型的本地文件中
                # https://blog.csdn.net/niuxuerui11/article/details/115674422?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1-115674422-blog-90722261.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.2&utm_relevant_index=4

