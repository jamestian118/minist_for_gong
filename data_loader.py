# 读取minist数据集
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


# batch_size (int): 每个批次的大小，即每次迭代中返回的数据样本数。
# shuffle (bool): 是否在每个 epoch 之前打乱数据。如果设置为 True，则在每个 epoch 之前重新排列数据集以获得更好的训练效果。
#
# torchvision.datasets.MNIST
# root：下载数据的目录；
# train决定是否下载的是训练集；
# download为true时会主动下载数据集到指定目录中，如果已存在则不会下载
# transform是接收PIL图片并返回转换后版本图片的转换函数，
#
# transform 函数是一个 PyTorch 转换操作，它将图像转换为张量并对其进行标准化，其中均值为 0.1307，标准差为 0.3081。
# torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
# transforms.ToTensor()函数的作用是将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
# transforms.Normalize()用均值和标准差归一化张量图像
# 归一化：把数据变成[0,1]或者[-1,1]之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速。
# 即将每一个图像像素的值减去均值，除以标准差，如此使它们有相似的尺度，从而更容易地训练模型。

def load_mnist_data(batch_size, train_size, val_split):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    indices = np.arange(len(full_train_dataset))
    np.random.shuffle(indices)

    class_indices = {label: [] for label in range(10)}
    for idx in indices:
        label = full_train_dataset[idx][1]
        class_indices[label].append(idx)

    train_indices = []
    for label in range(10):  # 0到9总共有10种标签
        n_samples = int(train_size * len(class_indices[label]))
        train_indices.extend(class_indices[label][:n_samples])

    if val_split:
        val_size = int(val_split * len(train_indices))
        np.random.shuffle(train_indices)
        val_indices = train_indices[-val_size:]
        train_indices = train_indices[:-val_size]

        val_dataset = Subset(full_train_dataset, val_indices)

    # cnn:为剩余的训练样本创建无标签数据加载器
    unlabeled_indices = []
    for label in range(10):
        n_samples = int(train_size * len(class_indices[label]))
        unlabeled_indices.extend(class_indices[label][n_samples:])
    unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)
    unlabeled_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=unlabeled_sampler)

    # 创建训练集和验证集的采样器
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # fcnn
    # return train_loader, val_loader, test_loader

    # cnn
    return train_loader, val_loader, test_loader, unlabeled_loader
