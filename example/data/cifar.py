import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar100:
    def __init__(self, batch_size, threads, size=(32, 32)):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar10:
    def __init__(self, batch_size, threads, size=(32, 32)):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar10Sub:
    def __init__(self, batch_size, threads, size=(32, 32)):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        mask = torch.utils.data.Subset(train_set, list(range(0, batch_size)))

        self.train = torch.utils.data.DataLoader(mask, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class FashionMNIST:
    def __init__(self, batch_size, threads, size=(32, 32)):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        mask = torch.utils.data.Subset(train_set, list(range(0, 800)))
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(mask, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class MNIST:
    def __init__(self, batch_size, threads, size=(32, 32)):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        mask = torch.utils.data.Subset(train_set, list(range(0, 1000)))
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(mask, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
