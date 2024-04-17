

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import data
import utils
from utils import valid
from models.mlp import MLP
from models.resnet import ResNet, BottleNeck
from models.cnn import CNN
from algorithms.udru import udrur
from algorithms.kpriors import kpr


def valid_method(model, model_root, ull, rml, train_set, valid_set, percentage,
                 method, beta, lr, dp, dm, batch_size, epoch, numlbs, n_sample, max_iter, device,
                 save_result_path):
    flsls_train = []
    ncpls_train = []
    dts_train = []

    flsls_valid = []
    ncpls_valid = []
    dts_valid = []

    acc_uls_train_original = []
    acc_rms_train_original = []
    acc_uls_train = []
    acc_rms_train = []

    acc_uls_valid_original = []
    acc_rms_valid_original = []
    acc_uls_valid = []
    acc_rms_valid = []

    for i, (ul_labels, rm_labels) in enumerate(zip(ull, rml)):
        # unlearn train
        print('unlearn labels: ', ul_labels)
        model.load_state_dict(torch.load(model_root))
        model.to(torch.device('cpu'))
        ulset_train, rmset_train = data.generate_unlearn_set(dataset=train_set,
                                                             n=None, percentage=percentage,
                                                             ul_labels=ul_labels, rm_labels=rm_labels,
                                                             remain=True)
        model.to(device)
        # original acc on train set
        acc_ul_train_original, _, _ = valid(model=model,
                                            dataloader=DataLoader(dataset=ulset_train, batch_size=64),
                                            device=device)
        acc_rm_train_original, _, _ = valid(model=model,
                                            dataloader=DataLoader(dataset=rmset_train, batch_size=64),
                                            device=device)

        acc_uls_train_original.append(acc_ul_train_original)
        acc_rms_train_original.append(acc_rm_train_original)

        info = f'original acc: ' \
               f'acc_ul_train: {float(acc_ul_train_original):.4f},   ' \
               f'acc_rm_train: {float(acc_rm_train_original):.4f}'
        print(info)

        t0 = time.time()
        if ulset_train.data.size(0) == 0:
            print('size == 0')
            continue
        if method == 'udru':
            flsl, ncpl = udrur(model=model, ulset=ulset_train, rm_labels=rm_labels, beta=beta, mapping=False,
                               numlbs=numlbs, bsz_mp=batch_size, init_range=1e-2, epoch_mp=128,
                               lr=lr, dp=dp, dm=dm, batch_size=batch_size, epoch=epoch, record_acc=True,
                               device=device)
        elif method == 'kpriors':
            flsl, ncpl = kpr(model, ulset_train, lr, batch_size, epoch, utils.ce_loss, device)
        elif method == 'retrain':
            # model.__init__([784, 1024, 512, 256, 10], bias=False, activ=nn.ReLU)
            # [784, 512, 256, 10], bias=False, activ=nn.ReLU
            model.__init__(num_classes=10)
            flsl, ncpl = utils.train(model=model, loss=utils.ce_loss,
                                     train_set=rmset_train, valid_set=None,
                                     batch_size=2048, lr=1e-4, weight_decay=1e-4,
                                     n_epoch=1, epoch=1024,
                                     device=device, save_path=f'./weights/tmp.pth')
        else:
            flsl, ncpl = [], []

        t1 = time.time()
        dt = t1 - t0
        # acc on train set
        acc_ul_train, _, _ = valid(model=model,
                                   dataloader=DataLoader(dataset=ulset_train, batch_size=64),
                                   device=device)
        acc_rm_train, _, _ = valid(model=model,
                                   dataloader=DataLoader(dataset=rmset_train, batch_size=64),
                                   device=device)

        flsls_train.append(flsl)
        ncpls_train.append(ncpl)
        dts_train.append(dt)

        acc_uls_train.append(acc_ul_train)
        acc_rms_train.append(acc_rm_train)

        info = f'unlearn  acc: ' \
               f'acc_ul_train: {float(acc_ul_train):.4f},   ' \
               f'acc_rm_train: {float(acc_rm_train):.4f},   ' \
               f'dt: {dt:.4f}'
        print(info)

    save_dict = {'flsls_train': flsls_train,
                 'ncpls_train': ncpls_train,
                 'dts_train': dts_train,
                 'flsls_valid': flsls_valid,
                 'ncpls_valid': ncpls_valid,
                 'dts_valid': dts_valid,
                 'acc_uls_train': acc_uls_train,
                 'acc_rms_train': acc_rms_train,
                 'acc_uls_valid': acc_uls_valid,
                 'acc_rms_valid': acc_rms_valid,
                 'acc_uls_train_original': acc_uls_train_original,
                 'acc_rms_train_original': acc_rms_train_original,
                 'acc_uls_valid_original': acc_uls_valid_original,
                 'acc_rms_valid_original': acc_rms_valid_original,
                 }
    torch.save(save_dict, save_result_path)


def main(model,
         model_name,
         model_root,
         method, dataset_name, train_set, valid_set):
    if dataset_name == 'mnist':
        batch_size = None
        lr = 1e-3
        beta = 1.0
        epoch = 1024
        dp = 0.0001

    if dataset_name == 'fashionmnist':
        batch_size = None
        lr = 1e-3
        beta = 1.0
        epoch = 1024
        dp = 0.001

    if dataset_name == 'cifar10':
        batch_size = 128
        lr = 2e-5
        beta = 1.0
        epoch = 256
        dp = 0.5


    dm = None
    K_ul = 1
    K = train_set.targets.size(1)
    K_rm = K - K_ul
    numlbs = [4] * K_rm

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    n_sample = None
    max_iter = None

    save_root = './results_sl'
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root + f'/{method}/', exist_ok=True)

    percentages = [0.2, 0.4, 0.6, 0.8, 1]
    for i, percentage in enumerate(percentages):
        ull, rml = data.generate_ul_and_rm_labels(K_ul=K_ul, K=K)
        os.makedirs(save_root + f'/{method}/' + dataset_name, exist_ok=True)
        save_result_path = save_root + f'/{method}/' + dataset_name + f'/{model_name}_sl_percentage{i}.pth'
        print(ull, rml)
        print(save_result_path)
        valid_method(model, model_root, ull, rml, train_set, valid_set, percentage,
                     method, beta, lr, dp, dm, batch_size, epoch, numlbs, n_sample, max_iter, device,
                     save_result_path)


