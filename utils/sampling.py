#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users) # 计算每个用户应该分配的样本数量。这里使用了整除运算，确保每个用户获得相同数量的数据
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] #dict_users：一个空字典，用于存储每个用户对应的数据索引 all_idxs：一个列表，包含所有数据样本的索引，从 0 到 len(dataset) - 1
        # 为每个用户分配数据
    for i in range(num_users):# 使用 for 循环遍历每个用户的索引（从 0 到 num_users - 1）
            # 在每次迭代中，执行以下操作
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))# np.random.choice(all_idxs, num_items, replace=False)
                    # 从 all_idxs 中随机选择 num_items 个索引，replace=False 确保不重复选择
                    # 这些索引被转换为集合（set），以便后续操作中可以方便地进行集合运算
                    # 将选中的索引存储在 dict_users 字典中，以用户的索引为键
        all_idxs = list(set(all_idxs) - dict_users[i])# 更新 all_idxs，从中移除已经分配给当前用户的索引，确保下一个用户不会获得相同的数据
    return dict_users# 返回 dict_users 字典，包含每个用户的索引集合

#这个函数的主要功能是从 MNIST 数据集中随机且均匀地分配样本给指定数量的用户，确保每个用户获得的样本是独立同分布的。这种方法在联邦学习中非常常见，因为它模拟了多个用户在不同设备上进行学习的场景。通过随机选择，确保了数据的多样性和代表性


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
