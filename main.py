import argparse
import torch.nn as nn
import torch
import data
from models.mlp import MLP
from models.resnet import ResNet, BottleNeck
from models.cnn import CNN
import utils
import os
from valid_method import valid_method_sl, valid_method_pll, valid_method_nll

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'], type=str,
                    help='dataset name')
parser.add_argument('--epoch', default=64, type=int)
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--data_type', default='SL', choices=['SL', 'PLL', 'NLL'], type=str, help='data type')
parser.add_argument('--partial_rate', default=0.2, type=float, help='partial rate of PLL labels')
parser.add_argument('--noisy_rate', default=0.2, type=float, help='noise rate of NLL labels')
parser.add_argument('--method', default='udru', type=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset_name
    lr = args.lr
    epoch = args.epoch
    data_type = args.data_type
    weight_decay = args.weight_decay
    method = args.method
    batch_size = args.batch_size

    os.makedirs('./dataset', exist_ok=True)
    if dataset_name == 'mnist':
        train_set = data.MNIST(root='./dataset', train=True, flatten=True)
        valid_set = data.MNIST(root='./dataset', train=False, flatten=True)
        batch_size = None
        model = MLP([784, 512, 256, 128, 10], bias=False, activ=nn.ReLU)
        model_name = 'mlp'

    if dataset_name == 'fashionmnist':
        train_set = data.FashionMNIST(root='./dataset', flatten=False, train=True)
        valid_set = data.FashionMNIST(root='./dataset', flatten=False, train=False)
        batch_size = None
        model = CNN(num_classes=10)
        model_name = 'cnn'

    if dataset_name == 'cifar10':
        train_set = data.CIFAR(root='./dataset', train=True)
        valid_set = data.CIFAR(root='./dataset', train=False)
        batch_size = 256
        model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=10)  # resnet50
        model_name = 'resnet'

    n_epoch = 1

    if data_type == 'SL':
        pass
    elif data_type == 'PLL':
        targets_root = f'./weights/pll' + f'/{dataset_name}_pll_targets.pth'
        if epoch == 0:
            train_set_pll = data.PartialLabelSet(train_set, partial_rate=args.partial_rate, targets_root=targets_root)
        else:
            train_set_pll = data.PartialLabelSet(train_set, partial_rate=args.partial_rate, targets_root=None)
            os.makedirs(f'./weights/pll', exist_ok=True)
            torch.save(train_set_pll.targets, targets_root)

    elif data_type == 'NLL':
        targets_root = f'./weights/nll' + f'/{dataset_name}_nll_targets.pth'
        if epoch == 0:
            train_set_nll = data.NoisyLabelSet(train_set, noisy_rate=args.noisy_rate, targets_root=targets_root)
        else:
            train_set_nll = data.NoisyLabelSet(train_set, noisy_rate=args.noisy_rate, targets_root=None)
            os.makedirs(f'./weights/nll', exist_ok=True)
            torch.save(train_set_nll.targets, targets_root)

    model_root = './weights/' + dataset_name + '/' + model_name + '_' + args.data_type + '.pth'
    os.makedirs('./weights/' + dataset_name, exist_ok=True)
    if data_type == 'SL':
        utils.train(model=model, loss=utils.ce_loss,
                    train_set=train_set, valid_set=valid_set,
                    batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                    n_epoch=n_epoch, epoch=epoch,
                    device=device, save_path=model_root)
        valid_method_sl.main(model, model_name, model_root, method, dataset_name, train_set, valid_set)

    elif data_type == 'PLL':
        utils.train(model=model, loss=utils.cc_loss,
                    train_set=train_set_pll, valid_set=valid_set,
                    batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                    n_epoch=n_epoch, epoch=epoch,
                    device=device, save_path=model_root)
        valid_method_pll.main(model, model_name, model_root, method, dataset_name, train_set, valid_set, targets_root)

    elif data_type == 'NLL':
        utils.train_nl(model=model,
                       train_set=train_set_nll, valid_set=valid_set,
                       batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                       n_epoch=n_epoch, epoch=epoch,
                       device=device, save_path=model_root)
        valid_method_nll.main(model, model_name, model_root, method, dataset_name, train_set, valid_set, targets_root)


if __name__ == '__main__':
    main()
