import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets


def mnist_loader(batch_size=256, train=True):
    preprocess = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True),
    ])
    dataset = datasets.MNIST(root='dataset', train=train, transform=preprocess, download=False)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=False)
    return dataloader


def svhn_dataloader(batch_size=4, train=True):
    preprocess = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True),
    ])
    dataset = datasets.SVHN(root='dataset/SVHN', split='train' if train else 'test', transform=preprocess, download=False)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=3, pin_memory=True, drop_last=False)
    return dataloader


def sample_mnist():
    preprocess = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True),
    ])
    mnist = datasets.MNIST(root='dataset', train=True, download=False, transform=preprocess)
    indexs = torch.randperm(len(mnist))
    X, Y = [], []
    for i in indexs:
        img = torch.from_numpy(np.array(mnist[i][0]))
        X.append(img.unsqueeze(0))
        Y.append(torch.from_numpy(np.array(mnist[i][1])))
    X, Y = torch.stack(X, dim=0), torch.stack(Y, dim=0)
    return X, Y


def sample_target(n: int):
    preprocess = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True),
    ])
    svhn = datasets.SVHN(root='dataset/SVHN', split='train', transform=preprocess, download=True)
    X, Y = [], []
    class_num = 10 * [n]
    for img, label in svhn:
        if len(X) == n * 10:
            break
        if class_num[label] > 0:
            X.append(torch.from_numpy(np.array(img)))
            Y.append(torch.from_numpy(np.array(label)))
            class_num[label] -= 1
    return torch.stack(X, dim=0), torch.stack(Y, dim=0)
