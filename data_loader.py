# 读取minist数据集
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

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
    for label in range(10):
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
    # fcn
    # return train_loader, val_loader, test_loader
    # c
    # nn
    return train_loader, val_loader, test_loader, unlabeled_loader
