#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib #这行代码导入了 matplotlib 库。matplotlib 是一个用于绘制图形和可视化数据的库，广泛应用于科学计算和数据分析
matplotlib.use('Agg') # 'Agg' 是一个非交互式后端，适合于生成图像文件（如 PNG、PDF 等），而不需要显示在屏幕上
import matplotlib.pyplot as plt # 这行代码从 matplotlib 库中导入 pyplot 模块，并将其命名为 plt。pyplot 是 matplotlib 的一个子模块，提供了一系列用于绘图的函数，使用起来非常方便。通过 plt，你可以使用常见的绘图函数，如 plt.plot()、plt.show()、plt.savefig() 等
import copy# copy 模块提供了用于对象复制的功能。你可以使用 copy.copy() 进行浅拷贝，或者使用 copy.deepcopy() 进行深拷贝。在深度学习中，尤其是在处理模型参数或数据时，拷贝对象可能是必要的，以避免意外修改原始对象
import numpy as np # numpy 是 Python 中用于科学计算的核心库，提供了高性能的多维数组对象和用于数组操作的工具。通过将其命名为 np，你可以方便地调用其函数，例如 np.array()、np.mean()、np.zeros() 等
from torchvision import datasets, transforms # transforms：torchvision.transforms 提供了一系列用于数据预处理和增强的工具，例如图像缩放、裁剪、归一化等。通过这些工具，可以在加载数据时对图像进行必要的转换
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate   #实现本地模型更新的逻辑
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img    # 用于测试模型的性能，特别是在图像分类任务中


if __name__ == '__main__':
    # parse args
    args = args_parser() # 调用 args_parser() 函数来解析命令行参数。这个函数通常定义了可用的参数（如学习率、批量大小、数据集名称等），并返回一个包含这些参数的对象 args
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')#这行代码根据是否可用 GPU 来设置计算设备
              # torch.cuda.is_available() 检查是否有可用的 GPU  
              # args.gpu 是用户指定的 GPU 设备。如果没有可用的 GPU，或者用户指定的 GPU 无效，则使用 CPU。最终，args.device 将包含要使用的设备（如 cuda:0 或 cpu）

    # 数据集加载和用户分割
    if args.dataset == 'mnist': # 如果是mnist数据集
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('FL/FedAvg/federated-learning/data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('FL/FedAvg/federated-learning/data/mnist/', train=False, download=True, transform=trans_mnist)
        # 用户采样
        if args.iid:# 根据 args.iid 的值决定是进行 IID 还是 Non-IID 采样，dict_users 是一个字典，保存了每个用户的数据索引
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':# 如果是cifar数据集
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('FL/FedAvg/federated-learning/data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('FL/FedAvg/federated-learning/data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape # 获取每张图片的大小

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train() # 将模型设置为训练模式。这是 PyTorch 中的一个重要步骤，因为某些层（如 Dropout 和 BatchNorm）在训练和评估模式下的行为不同

    # copy weights
    w_glob = net_glob.state_dict()#获取当前模型的权重，并将其存储在 w_glob 中。state_dict() 方法返回一个包含所有模型参数的字典，可以用于后续的模型更新或保存

    # training
    loss_train = []
    cv_loss, cv_acc = [], [] # 用于存储交叉验证的损失和准确率
    val_loss_pre, counter = 0, 0 #用于跟踪验证损失和计数器
    net_best = None #用于存储最佳模型和最佳损失
    best_loss = None
    val_acc_list, net_list = [], [] # 用于存储验证准确率和网络列表

       # 处理所有客户端的情况
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]# 创建一个列表 w_locals，其中每个元素都是当前全局权重 w_glob 的副本，长度等于用户数量
    for iter in range(args.epochs):# 使用一个循环遍历指定的训练轮数（args.epochs）
        loss_locals = [] # 在每次迭代开始时，初始化 loss_locals 列表，用于存储每个用户的本地损失
        if not args.all_clients: # 如果不使用所有客户端，则重置 w_locals 列表
            w_locals = []
            # 随机选择用户进行训练
        m = max(int(args.frac * args.num_users), 1)# 计算参与训练的用户数量 m，为用户总数的一个分数(抽取客户端的比例)（args.frac），确保至少有一个用户参与
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)# 随机选择 m 个用户的索引，确保不重复选择
            # 本地训练
        for idx in idxs_users:# 对每个选择的用户进行本地训练
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])# 创建 LocalUpdate 实例，传入参数、训练数据集和该用户的数据索引
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))# 使用 local.train() 方法训练本地模型，返回更新后的权重 w 和本地损失 loss
                # 更新权重w
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)# 如果使用所有客户端，更新 w_locals 列表中的对应用户的权重
            else:
                w_locals.append(copy.deepcopy(w))# 否则，将新权重添加到 w_locals 列表中
            loss_locals.append(copy.deepcopy(loss))# 同时，将本地损失添加到 loss_locals 列表中
        # 更新全局权重
        w_glob = FedAvg(w_locals) # 使用 FedAvg 函数对所有用户的权重进行聚合，计算新的全局权重 w_glob

        # copy weight to net_glob 更新全局模型
        net_glob.load_state_dict(w_glob)# 将新的全局权重加载到全局模型 net_glob 中

        # print loss 打印损失
        loss_avg = sum(loss_locals) / len(loss_locals) # 计算平均损失 loss_avg，并打印当前轮次的平均损失
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg)) # 将平均损失添加到 loss_train 列表中，以便后续分析
        loss_train.append(loss_avg)
    """
    总结
    这段代码实现了一个联邦学习的训练过程，主要包括以下步骤：

    初始化所需变量。
    根据是否使用所有客户端来决定权重的处理方式。
    在每个训练轮次中，随机选择用户并进行本地训练。
    聚合各个用户的权重，更新全局模型。
    记录和打印每轮的平均损失。
    这种方法在联邦学习中非常有效，因为它允许多个用户在本地训练模型，同时只共享模型参数而不共享原始数据，从而保护用户隐私。
    """


    # plot loss curve  绘制损失曲线
    plt.figure()# plt.figure(): 创建一个新的图形窗口，用于绘制图表。每次调用此函数都会生成一个新的空白图形。
    plt.plot(range(len(loss_train)), loss_train) # range(len(loss_train)): 生成一个从 0 到 loss_train 列表长度的整数序列，表示训练的轮次
                                                 # plt.plot(...): 绘制损失曲线，横轴为训练轮次，纵轴为训练损失
    plt.ylabel('train_loss')# plt.ylabel('train_loss'): 设置 y 轴的标签为 "train_loss"，以便在图表上标识该轴表示训练损失
    plt.savefig('FL/FedAvg/federated-learning/save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
                            # plt.savefig(...): 保存绘制的图形到指定的路径。路径中使用了 format 方法填充参数：
    # testing
    net_glob.eval() #net_glob.eval(): 将全局模型设置为评估模式。在评估模式下，某些层（如 Dropout 和 BatchNorm）会以不同的方式工作，以确保模型在测试时的表现更加稳定
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))#调用 test_img 函数对训练集进行测试，返回训练集的准确率 acc_train 和损失 loss_train
    print("Testing accuracy: {:.2f}".format(acc_test))#同样地，对测试集进行测试，返回测试集的准确率 acc_test 和损失 loss_test
    """
    总结
    这段代码的功能可以分为两个主要部分：

    绘制和保存损失曲线：通过 matplotlib 绘制训练损失曲线，并将其保存为图像文件，以便后续分析和可视化。

    模型评估：将全局模型设置为评估模式，并使用测试函数计算并打印训练集和测试集的准确率和损失。这样可以评估模型在训练和测试数据上的表现。
    """
