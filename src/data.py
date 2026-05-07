import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np


DATA_PATH = Path(__file__).parents[1] / 'data'


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_quadratic(num_data, num_label, d, im_d, noise_scale, regression=False):
    C = torch.randn((im_d, d))
    Z = torch.randn((num_data, im_d))
    noise = torch.randn((num_data, d))

    X = Z @ C + noise_scale * noise
    X /= torch.linalg.norm(X, dim=1).view(-1, 1)
    # X *= (d ** 0.5)

    A = torch.randn((d, d)) 
    A = (A + A.T) / 2
    A -= torch.trace(A) / d

    y = (X @ A @ X.T).diag()
    if not regression:
        y_ = torch.zeros_like(y)
        for i in range(num_label):
            q = torch.quantile(y, i / num_label)
            y_[y >= q] = i
        y = y_.to(torch.int64)
    else:
        y = torch.relu(y).view(-1, 1)

    train_dataset = CustomDataset(X[:int(num_data / 2)], y[:int(num_data / 2)])
    valid_dataset = CustomDataset(X[int(num_data / 2):int(num_data / 4 * 3)], y[int(num_data / 2):int(num_data / 4 * 3)])
    test_dataset = CustomDataset(X[int(num_data / 4 * 3):], y[int(num_data / 4 * 3):])
    return train_dataset, valid_dataset, test_dataset


def get_mnist():
    """
    Returns the MNIST dataset.
    """
    trainset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    return trainset, testset


def get_celeba():
    """
    Returns the CelebA dataset.
    """
    trainset = torchvision.datasets.CelebA(
        root=DATA_PATH,
        split='train',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    testset = torchvision.datasets.CelebA(
        root=DATA_PATH,
        split='test',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    return trainset, testset


def get_svhn():
    """
    Returns the SVHN dataset.
    """
    trainset = torchvision.datasets.SVHN(
        root=DATA_PATH,
        split='train',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    testset = torchvision.datasets.SVHN(
        root=DATA_PATH,
        split='test',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    return trainset, testset


def get_cifar10():
    """
    Returns the CIFAR-10 dataset.
    """
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    return trainset, testset


def get_cifar100():
    """
    Returns the CIFAR-100 dataset.
    """
    preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=preprocess
    )
    testset = torchvision.datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=preprocess
    )
    return trainset, testset


def randomize_label(trainset, testset, p, c=10):
    train_index = np.random.binomial(n=1, p=p, size=len(trainset.targets)).astype(bool)
    test_index = np.random.binomial(n=1, p=p, size=len(testset.targets)).astype(bool)
    train_random_label = np.random.randint(low=0, high=c, size=sum(train_index))
    test_random_label = np.random.randint(low=0, high=c, size=sum(test_index))

    trainset.targets = np.array(trainset.targets)
    trainset.targets[train_index] = train_random_label
    testset.targets = np.array(testset.targets)
    testset.targets[test_index] = test_random_label
    return trainset, testset
