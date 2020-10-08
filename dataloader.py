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


def sample_source():
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
    return torch.stack(X, dim=0), torch.stack(Y, dim=0)


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


def sample_groups(Xs, Ys, Xt, Yt):
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    shot_num = Xt.shape[0] // 10

    def s_idx(c: int):
        idx = torch.nonzero(Ys.eq(c))
        return idx[torch.randperm(len(idx))][:shot_num * 2].squeeze()

    t_idx = lambda c: torch.nonzero(Yt.eq(c))[:shot_num].squeeze()
    shuffled_class = torch.randperm(10)
    s_matrix = torch.stack(list(map(s_idx, shuffled_class)), dim=0)
    t_matrix = torch.stack(list(map(t_idx, shuffled_class)), dim=0)
    G1, G2, G3, G4, Y1, Y2, Y3, Y4 = [], [], [], [], [], [], [], []
    for i in range(10):
        for j in range(shot_num):
            G1.append((Xs[s_matrix[i][j * 2]], Xs[s_matrix[i][j * 2 + 1]]))
            Y1.append((Ys[s_matrix[i][j * 2]], Ys[s_matrix[i][j * 2 + 1]]))
            G2.append((Xs[s_matrix[i][j]], Xt[t_matrix[i][j]]))
            Y2.append((Ys[s_matrix[i][j]], Yt[t_matrix[i][j]]))
            G3.append((Xs[s_matrix[i][j]], Xs[s_matrix[(i + 1) % 10][j]]))
            Y3.append((Ys[s_matrix[i][j]], Ys[s_matrix[(i + 1) % 10][j]]))
            G4.append((Xs[s_matrix[i][j]], Xt[t_matrix[(i + 1) % 10][j]]))
            Y4.append((Ys[s_matrix[i][j]], Yt[t_matrix[(i + 1) % 10][j]]))
    group, groupy = [G1, G2, G3, G4], [Y1, Y2, Y3, Y4]
    for g in group:
        assert len(g) == Xt.shape[0]
    return group, groupy
