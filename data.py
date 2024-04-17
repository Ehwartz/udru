import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
import random
import math
from itertools import combinations
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class TempDataset(Dataset):
    def __init__(self, data, targets):
        super(TempDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)


class MNIST(Dataset):
    def __init__(self, root, flatten, train=True):
        super(MNIST, self).__init__()

        mds = torchvision.datasets.MNIST(root=root, download=True, train=train)

        if flatten:
            self.data = mds.data.view(-1, 784) / 255
        else:
            self.data = mds.data / 255
        self.n = self.data.size(0)
        self.targets = torch.zeros(size=[self.data.size(0), 10])
        self.targets[torch.arange(self.n), mds.targets] = 1
        self.labels = mds.targets.clone().detach()
        del mds

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


class FashionMNIST(Dataset):
    def __init__(self, root, flatten, train=True):
        super(FashionMNIST, self).__init__()

        mds = torchvision.datasets.FashionMNIST(root=root, download=True, train=train)

        if flatten:
            self.data = mds.data.view(-1, 784) / 255
        else:
            self.data = mds.data.view(-1, 1, 28, 28) / 255
        self.n = self.data.size(0)
        self.targets = torch.zeros(size=[self.data.size(0), 10])
        self.targets[torch.arange(self.n), mds.targets] = 1
        self.labels = mds.targets.clone().detach()
        del mds

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


class CIFAR(Dataset):
    def __init__(self, root, train=True):
        super(CIFAR, self).__init__()

        cds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.data = (torch.from_numpy(cds.data) / 255).permute(0, 3, 1, 2)
        self.n = self.data.size(0)
        self.targets = torch.zeros(size=[self.data.size(0), 10])
        self.targets[torch.arange(self.n), cds.targets] = 1
        self.labels = torch.tensor(cds.targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


class MixedDataset(Dataset):
    def __init__(self, dataset, mixed_labels):
        super(MixedDataset, self).__init__()
        self.data = dataset.data
        self.n = self.data.size(0)
        dim = len(mixed_labels)
        self.true_targets = dataset.targets
        self.targets = torch.zeros(size=[self.data.size(0), dim])
        self.labels = torch.argmax(dataset.targets, dim=-1)

        for d in range(dim):
            indices = []
            for label in mixed_labels[d]:
                indices.append(torch.where(self.labels == label)[0])
            indices = torch.concat(indices, dim=0)
            self.targets[indices, d] = 1

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])


class PartialLabelSet(Dataset):
    def __init__(self, dataset, partial_rate, targets_root=None):
        super(PartialLabelSet, self).__init__()
        self.data = dataset.data
        self.labels = dataset
        self.n = self.data.size(0)
        self.labels = torch.argmax(dataset.targets, dim=-1)
        if targets_root:
            self.targets = torch.load(targets_root)
        else:
            self.targets = dataset.targets
            # partial_indices = torch.where(torch.rand(size=self.targets.size()) < partial_rate)
            # self.targets[partial_indices] = 1

            K = self.targets.size()[-1]
            candidates = list(range(K))
            pls = torch.tensor([list(c) for c in combinations(candidates, int(partial_rate * K))])
            # print(pls)
            n_pl = pls.size(0)

            random_indices = torch.randint(low=0, high=n_pl, size=[self.targets.size()[0]])
            # print(pls[random_indices])
            for i in range(int(partial_rate * K)):
                self.targets[torch.arange(self.targets.size()[0]), pls[random_indices][:, i]] = 1

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


class NoisyLabelSet(Dataset):
    def __init__(self, dataset, noisy_rate, targets_root=None):
        super(NoisyLabelSet, self).__init__()
        self.data = dataset.data
        self.labels = dataset
        self.n = self.data.size(0)
        self.labels = torch.argmax(dataset.targets, dim=-1)
        if targets_root:
            self.targets = torch.load(targets_root)
        else:
            self.targets = dataset.targets
            indices = torch.arange(len(dataset))
            nos = torch.rand(size=[len(dataset)]) < noisy_rate
            n = torch.sum(nos * 1)
            self.targets[nos] *= 0
            self.targets[indices[nos], torch.randint(self.targets.size(-1), size=[int(n)])] = 1

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


def generate_unlearn_set(dataset, n, percentage, ul_labels, rm_labels, remain=False):
    uld = dataset.filter(ul_labels)
    rmd = dataset.filter(rm_labels)
    uln = len(uld)
    indices = list(range(uln))
    random.shuffle(indices)
    if n:
        ulset = TempDataset(data=uld.data[indices[:n]],
                            targets=uld.targets[indices[:n]])
    elif percentage:
        ulset = TempDataset(data=uld.data[indices[:int(uln * percentage)]],
                            targets=uld.targets[indices[:int(uln * percentage)]])
    else:
        ulset = None
    if remain:
        rmn = len(rmd)
        indices = list(range(rmn))
        random.shuffle(indices)
        rmset = TempDataset(data=rmd.data, targets=rmd.targets)
        return ulset, rmset

    else:
        return ulset


def generate_unlearn_set_indices(dataset, n, percentage, ul_labels, rm_labels, remain=False):
    print('len dataset: ', len(dataset))
    uld_indices = dataset.filter_indices(ul_labels)
    rmd_indices = dataset.filter_indices(rm_labels)
    uln = len(uld_indices)
    indices = list(range(uln))
    random.shuffle(indices)
    if n:
        ulset_indices = uld_indices[indices[:n]]
    elif percentage:
        ulset_indices = uld_indices[indices[:int(uln * percentage)]]

    else:
        ulset_indices = None
    if remain:
        rmn = len(rmd_indices)
        indices = list(range(rmn))
        random.shuffle(indices)
        rmset_indices = rmd_indices[indices]

        return ulset_indices, rmset_indices

    else:
        return ulset_indices


def generate_ul_and_rm_labels(K_ul, K):
    candidates = list(range(K))
    ull = [list(c) for c in combinations(candidates, K_ul)]
    rml = []
    for uli in ull:
        cand = candidates.copy()
        for label in uli:
            cand.remove(label)
        rml.append(cand)
    return ull, rml




