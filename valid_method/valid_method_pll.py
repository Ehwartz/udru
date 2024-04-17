
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


def valid_method(model, model_root, ull, rml, train_set, valid_set, targets_root, percentage,
                 method, beta, mapping, lr, dp, dm, batch_size, epoch, numlbs, n_sample, max_iter, device,
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

    partial_train_set = data.PartialLabelSet(dataset=train_set, partial_rate=0.3,
                                             targets_root=targets_root)
    # partial_valid_set = data.PartialDataset(dataset=valid_set, partial_rate=0.3,
    #                                         targets_root='./weights/partial_targets_valid.pth')

    for i, (ul_labels, rm_labels) in enumerate(zip(ull, rml)):
        # unlearn train partial
        model.load_state_dict(torch.load(model_root))
        model.to(torch.device('cpu'))
        ulset_train_indices, rmset_train_indices = data.generate_unlearn_set_indices(dataset=partial_train_set,
                                                                                     n=None, percentage=percentage,
                                                                                     ul_labels=ul_labels,
                                                                                     rm_labels=rm_labels,
                                                                                     remain=True)
        ulx, uly, _ = partial_train_set[ulset_train_indices]
        ulset_train = data.TempDataset(data=ulx, targets=uly)

        acx, acy, _ = train_set[ulset_train_indices]
        ulset_train_acc = data.TempDataset(data=acx, targets=acy)
        acx, acy, _ = train_set[rmset_train_indices]
        rmset_train_acc = data.TempDataset(data=acx, targets=acy)

        model.to(device)
        # original acc on train set
        acc_ul_train_original, _, _ = valid(model=model,
                                            dataloader=DataLoader(dataset=ulset_train_acc, batch_size=64),
                                            device=device)
        acc_rm_train_original, _, _ = valid(model=model,
                                            dataloader=DataLoader(dataset=rmset_train_acc, batch_size=64),
                                            device=device)

        acc_uls_train_original.append(acc_ul_train_original)
        acc_rms_train_original.append(acc_rm_train_original)

        info = f'original acc: ' \
               f'acc_ul_train: {float(acc_ul_train_original):.4f},   ' \
               f'acc_rm_train: {float(acc_rm_train_original):.4f}'
        print(info)

        t0 = time.time()
        if method == 'udru':
            flsl, ncpl = udrur(model=model, ulset=ulset_train, rm_labels=rm_labels, beta=beta, mapping=mapping,
                               numlbs=numlbs, bsz_mp=batch_size, init_range=1e-2, epoch_mp=128,
                               lr=lr, dp=dp, dm=dm, batch_size=batch_size, epoch=epoch, record_acc=True,
                               device=device)

        else:
            flsl, ncpl = [], []

        t1 = time.time()
        dt = t1 - t0
        # acc on train set
        acc_ul_train, _, _ = valid(model=model,
                                   dataloader=DataLoader(dataset=ulset_train_acc, batch_size=64),
                                   device=device)
        acc_rm_train, _, _ = valid(model=model,
                                   dataloader=DataLoader(dataset=rmset_train_acc, batch_size=64),
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
         method, dataset_name, train_set, valid_set, targets_root):
    if dataset_name == 'mnist':
        batch_size = None
        lr = 1e-5
        beta = 1.0
        epoch = 2048
        dp = 0.0001

    if dataset_name == 'fashionmnist':
        batch_size = None
        lr = 1e-5
        beta = 0.1  # balance ln(beta)
        epoch = 2048
        dp = 0.00001

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

    save_root = './results_pll/'
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(save_root + f'/{method}/', exist_ok=True)

    percentages = [0.2, 0.4, 0.6, 0.8, 1]
    for i, percentage in enumerate(percentages):
        ull, rml = data.generate_ul_and_rm_labels(K_ul=K_ul, K=K)
        os.makedirs(save_root + f'/{method}/' + dataset_name, exist_ok=True)
        save_result_path = save_root + f'/{method}/' + dataset_name + f'/{model_name}_pll_percentage{i}.pth'
        print(ull, rml)
        print(save_result_path)
        valid_method(model, model_root, ull, rml, train_set, valid_set, targets_root, percentage,
                     method, beta, False, lr, dp, dm, batch_size, epoch, numlbs, n_sample, max_iter, device,
                     save_result_path)

