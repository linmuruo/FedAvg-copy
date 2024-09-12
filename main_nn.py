#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#这段代码是一个用于训练和测试神经网络的 Python 脚本，主要使用 PyTorch 和 torchvision 库

import matplotlib #用于绘图，设置为 'Agg' 后端以支持无界面环境下的绘图。
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F # 包含各种神经网络功能（如损失函数）
from torch.utils.data import DataLoader #用于加载数据集
import torch.optim as optim #优化器库
from torchvision import datasets, transforms

from utils.options import args_parser #参数解析器，返回值是一个arg对象
from models.Nets import MLP, CNNMnist, CNNCifar #不同类型的神经网络模型

#做2次修改

def test(net_g, data_loader):#PyTorch 的 DataLoader 对象，用于提供测试数据集的批次
    # testing
    net_g.eval()   #将模型设置为评估模式。这会禁用 dropout 和 batch normalization 等训练时特有的操作，以确保模型在测试时的稳定性和一致性
    test_loss = 0  #用于累积每个批次的损失，以便在最后计算平均损失
    correct = 0  #用于累计正确预测的数量
    l = len(data_loader) #记录测试数据集的批次数量，虽然在后续代码中没有实际使用
        # 遍历数据加载器
    for idx, (data, target) in enumerate(data_loader):#使用 enumerate(data_loader) 遍历数据加载器，data 是输入数据，target 是对应的标签
            # 将数据和目标转移到设备
        data, target = data.to(args.device), target.to(args.device)#使用 .to(args.device) 将数据和标签转移到指定的计算设备（CPU 或 GPU），以便进行计算
            # 模型前向传播
        log_probs = net_g(data)# 将输入数据传入模型，得到预测的对数概率（log_probs）。这是模型对每个类别的预测值
            # 计算损失
        test_loss += F.cross_entropy(log_probs, target).item()#使用 F.cross_entropy 计算当前批次的损失，并将其累加到 test_loss 中。这是多类分类问题中常用的损失函数
            # 获取预测结果
        y_pred = log_probs.data.max(1, keepdim=True)[1]#通过 log_probs.data.max(1, keepdim=True)[1] 获取每个样本的预测类别。max(1) 返回每行的最大值和索引，这里我们只需要索引（即预测的类别）
                                    # https://blog.csdn.net/weixin_43978703/article/details/122308319
            # 累计正确预测数量                                                                                                                                                                                                                                                                                                                                             
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()#使用 y_pred.eq(target.data.view_as(y_pred)) 比较预测结果和真实标签，返回一个布尔张量
                        #将布尔张量转换为长整型并在 CPU 上求和，得到当前批次的正确预测数量，并累加到 correct 中
            #计算平均损失
    test_loss /= len(data_loader.dataset)#将总损失 test_loss 除以测试数据集的样本数量，以得到平均损失 
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(#打印测试集的平均损失和准确率。准确率的计算方式为正确预测数量除以总样本数，乘以 100 得到百分比
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss# 返回正确预测的数量和平均损失，以便在主程序中进行进一步处理或记录

#总结
#该 test 函数实现了模型在测试数据集上的评估过程，包括前向传播、损失计算和准确率统计。通过将模型设置为评估模式，确保了测试过程的稳定性。整体结构清晰且易于理解，适合用于多类分类问题的模型评估。


if __name__ == '__main__':
    #这段代码是一个完整的深度学习训练和测试流程的实现，主要使用 PyTorch 框架。
    # parse args
    args = args_parser()  # 调用参数解析器，获取用户输入的参数（如学习率、批量大小、数据集名称等）
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')#根据用户指定的 GPU 设备和可用性设置计算设备（CPU 或 GPU）
                #这里是默认先GPU训练，如果无英伟达显卡则调用CPU
            
    # 设置随机种子
    torch.manual_seed(args.seed) #设置随机种子以确保实验的可重复性 
                                 #设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同
        # https://blog.csdn.net/weixin_44211968/article/details/123769010?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-123769010-blog-112174334.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.2&utm_relevant_index=4

    # 数据集加载和分割用户
    if args.dataset == 'mnist': #根据用户指定的数据集名称（MNIST 或 CIFAR-10）加载相应的训练数据集
        dataset_train = datasets.MNIST('./data/mnist/',  train=True, download=True,#训练集，数据集直接从网络下载
                transform=transforms.Compose([
                    transforms.ToTensor(),   #totensor转换格式
                    transforms.Normalize((0.1307,), (0.3081,))   #归一化 https://blog.csdn.net/qq_38765642/article/details/109779370?ops_request_misc=%257B%2522request%255Fid%2522%253A%25224F1AA035-47D6-4C32-AC27-49576252B20C%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=4F1AA035-47D6-4C32-AC27-49576252B20C&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-109779370-null-null.142^v100^pc_search_result_base2&utm_term=transforms.Normalize&spm=1018.2226.3001.4187
                ]))#transform 定义数据预处理步骤，包括将图像转换为张量和归一化
        img_size = dataset_train[0][0].shape #img_size: 获取输入图像的大小（第一张）
    elif args.dataset == 'cifar': #根据用户指定的数据集名称（MNIST 或 CIFAR-10）加载相应的训练数据集
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#mean,std https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch.nn.functional.normalize
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    else:
        exit('Error: unrecognized dataset')

    # 模型构建
            # 根据用户输入的模型类型（CNN 或 MLP）构建相应的模型实例，并将其转移到指定的计算设备
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        # 这段代码的目的是计算输入特征的总维度，并使用该维度创建一个 MLP 模型实例
        # 通过这种方式，MLP 模型能够接受不同大小的输入图像，确保模型的输入层与数据的实际维度相匹配。
            # 对于多层感知机（MLP），计算输入维度并构建模型
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)# 这边循环是因为 通过遍历 img_size 中的每个维度 x，将 len_in 乘以该维度的大小。这实际上计算了输入图像的总特征数量
                                    #dim_out=args.num_classes: 设置输出层的维度为类别数量，通常是数据集中的类的数量（例如，MNIST 有 10 个类，CIFAR-10 有 10 个类）。
    else:
        exit('Error: unrecognized model111')
    print(net_glob)

    # 训练过程
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)#使用随机梯度下降（SGD）优化器
                        #获取模型 net_glob 中所有可训练的参数，以便优化器能够更新这些参数
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)#数据加载器
        #DataLoader: 用于将数据集 dataset_train 加载为批次
        #shuffle=True: 在每个 epoch 开始时随机打乱数据，以提高模型的泛化能力，防止模型学习到数据的顺序

    list_loss = [] # 创建一个空列表 list_loss，用于记录每个 epoch 的平均损失，以便后续分析和可视化
        #开始训练
    net_glob.train()# net_glob.train(): 将模型设置为训练模式。这会启用 dropout 和 batch normalization 等训练特有的层行为
    for epoch in range(args.epochs):# for epoch in range(args.epochs): 开始训练循环，遍历指定的 epoch 数量
            # 处理每个批次
        batch_loss = [] # 在每个 epoch 开始时初始化一个空列表 batch_loss，用于记录当前 epoch 中每个批次的损失
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)# 将当前批次的数据和目标标签转移到指定的计算设备（CPU 或 GPU），以便进行计算
            optimizer.zero_grad()# 清零优化器中存储的梯度。PyTorch 在每次反向传播时会累积梯度，因此在每次迭代开始时需要清零，以避免梯度累积
            output = net_glob(data)# 将当前批次的数据传入模型，进行前向传播，得到模型的输出 output
            loss = F.cross_entropy(output, target)# 使用交叉熵损失函数计算模型输出与目标标签之间的损失。交叉熵常用于分类问题
            loss.backward()# 进行反向传播，计算梯度。此时，模型参数的梯度会根据损失的反向传播算法自动计算
            optimizer.step()# 使用优化器更新模型参数，根据计算得到的梯度调整参数
            # 打印训练进度
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(#每 50 个批次打印一次训练进度，包括当前 epoch、已处理样本数、总样本数、当前进度百分比和当前批次的损失值
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))#loss.item(): 获取当前损失的标量值
            batch_loss.append(loss.item()) # 将当前批次的损失值添加到 batch_loss 列表中，以便后续计算平均损失
            # 计算并打印平均损失
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)# 计算当前 epoch 的平均损失 loss_avg，并打印出来
        list_loss.append(loss_avg) # 将平均损失添加到 list_loss 列表中，以便后续分析和可视化

    # 绘制Loss 
    #这段代码用于绘制训练过程中损失的变化，并将图像保存为 PNG 文件。以下是对每一行代码的详细解析
    plt.figure() # plt.figure(): 创建一个新的图形窗口。每次调用此函数都会生成一个新的图形，可以在其中绘制数据。此时，所有后续的绘图命令都会在这个新创建的图形上进行
    plt.plot(range(len(list_loss)), list_loss)#plt.plot(...): 这是 Matplotlib 中用于绘制二维线图的函数
        # range(len(list_loss)): 生成一个从 0 到 len(list_loss)-1 的整数序列，表示每个 epoch 的索引
        # list_loss: 这是一个包含每个 epoch 平均损失的列表。list_loss 的长度与 epoch 的数量相同
        # 这行代码将 x 轴设置为 epoch 的索引，y 轴设置为对应的训练损失，从而绘制出损失随训练 epoch 变化的曲线
    plt.xlabel('epochs')# plt.xlabel('epochs'): 设置 x 轴的标签为 "epochs"，表示横坐标所代表的含义是训练的 epoch 数
    plt.ylabel('train loss')# plt.ylabel('train loss'): 设置 y 轴的标签为 "train loss"，表示纵坐标所代表的含义是训练过程中的损失值
     #保存图像
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))#plt.savefig(...): 将当前图形保存为文件

    # 开始测试
    if args.dataset == 'mnist': # 检查 args.dataset 的值，以确定使用哪个数据集进行测试。如果数据集是 "mnist"，则执行对应的代码块
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,# 从 PyTorch 的 torchvision 库中加载 MNIST 数据集
                   transform=transforms.Compose([# 对数据进行预处理
                       transforms.ToTensor(),# 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量
                       transforms.Normalize((0.1307,), (0.3081,))#对图像进行标准化。这里的均值和标准差是根据 MNIST 数据集计算的，确保每张图像的像素值在训练时的分布保持一致。
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)# 测试时不打乱数据顺序，因为需要保证每次测试的一致性
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),# 将图像转换为张量
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])# 对 CIFAR-10 数据集进行标准化，均值和标准差均为 0.5，适用于 RGB 图像
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
                        #target_transform=None: 不对目标标签进行变换
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')# 打印测试集中的样本数量，len(dataset_test) 返回数据集的大小
    test_acc, test_loss = test(net_glob, test_loader)#test(net_glob, test_loader): 调用 test 函数进行模型评估
            #net_glob: 需要测试的模型 test_loader: 测试数据加载器
