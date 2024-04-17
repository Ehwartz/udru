import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
import time
from tqdm import tqdm
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def ce_loss(outputs, targets):
    softmax = torch.softmax(outputs, dim=1)
    n_log = -torch.log(softmax)
    l = torch.sum(n_log * targets)
    return l


def loss_pl(outputs, targets):
    sfm = torch.softmax(outputs, dim=1)
    l = torch.sum((1 - 10 * targets) * sfm)
    return l


def partial_loss(output1, targets):
    output = torch.softmax(output1, dim=1)
    l = targets * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    return loss


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()

    return average_loss


class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(torch.argmax(targets, dim=-1), 1))

        loss = (((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] -
                ((1 - (self.k ** self.q)) / self.q) * self.weight[indexes])
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes, device):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(torch.argmax(targets, dim=-1), 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk)  # .type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1).to(device)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.FloatTensor).to(device)  # .type(torch.cuda.FloatTensor)


def valid(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)
    total = 0.0
    correct = 0.0
    for i, (x, labels, indices) in enumerate(tqdm(dataloader, leave=False)):
        x = Variable(x).to(device)
        labels = Variable(labels).to(device)
        outputs = model(x)
        correct += torch.sum(torch.argmax(labels, dim=1) == torch.argmax(outputs, dim=1))
        total += labels.size(0)

    if total == 0:
        return 0, 0, 0
    else:
        return correct / total, correct, total


def train(model, loss, train_set, valid_set, batch_size, lr, weight_decay, n_epoch, epoch, device, save_path):
    model.train()
    model.to(device)
    if not batch_size:
        batch_size = 64
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    if valid_set:
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epoch):
        for iep in range(epoch):
            loss_sum = 0
            for _, (x, labels, indices) in enumerate(tqdm(train_loader, leave=False)):
                x = Variable(x).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                outputs = model(x)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()
                loss_sum += l.data
            torch.save(model.state_dict(), save_path)
            acc, cor, tot = valid(model, train_loader, device)
            info = f'i:{i}\t,iep:{iep}\t ,accuracy on train set: {acc:.4f}\t, correct/total: {cor}/{tot}'
            print(info)
            if valid_set:
                acc, cor, tot = valid(model, valid_loader, device)
                info = f'i:{i}\t,iep:{iep}\t ,accuracy on valid set: {acc:.4f}\t, correct/total: {cor}/{tot}'
                print(info)
    return [], []


def train_nl(model, train_set, valid_set, batch_size, lr, weight_decay, n_epoch, epoch, device, save_path):
    if not batch_size:
        batch_size = 64
    model.to(device)
    criterion = TruncatedLoss(trainset_size=len(train_set))
    criterion = criterion.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    for i in range(n_epoch):
        for iep in range(epoch):
            # update weight
            model.eval()
            if (i * iep + iep) >= 40 and (i * iep + iep) % 40 == 0 :
                for x, y, indices in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    criterion.update_weight(output, y, indices, device)
            model.train()
            loss_sum = 0

            for _, (x, y, indices) in enumerate(tqdm(train_loader, leave=False)):
                x = Variable(x).to(device)
                y = Variable(y).to(device)
                optimizer.zero_grad()
                outputs = model(x)
                ls = criterion(outputs, y, indices)
                ls.backward()
                optimizer.step()
                loss_sum += ls.data
            torch.save(model.state_dict(), save_path)

            acc, cor, tot = valid(model, valid_loader, device)
            info = f'i:{i}\t,iep:{iep} ,accuracy on valid set: {acc}\t, correct/total: {cor}/{tot}'
            print(info)

