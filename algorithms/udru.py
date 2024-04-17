import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from data import TempDataset
import utils
from tqdm import tqdm


def min_div_target(model, x, y, beta=1.0):
    n = x.size(0)
    output0 = model(x).detach()
    pos = torch.where(y == 1)
    cpl = 1 - y
    cpls = torch.sum(cpl, dim=-1)
    zp = torch.exp(output0)
    zp[pos] = 0
    zps = torch.sum(zp, dim=-1)
    zpm = (zps / cpls).view([n, 1])
    zt = zp + y * zpm * beta
    ft = torch.log(zt).detach()
    return ft


def min_div_noisy_target(model, x, y, beta=1.0):
    n = x.size(0)
    output0 = model(x).detach()
    # print(output0)

    p = torch.softmax(output0, dim=-1)
    yc = y.detach().clone()
    # yc = torch.zeros_like(y)
    # print(torch.argmax(output0, dim=-1))
    yc[torch.arange(n), torch.argmax(output0, dim=-1)] = 1
    # print(yc)
    # print(torch.sum(yc, dim=-1))
    pos = torch.where(yc == 1)
    cpl = 1 - yc
    cpls = torch.sum(cpl, dim=-1)
    zp = torch.exp(output0)
    zp[pos] = 0
    zps = torch.sum(zp, dim=-1)
    zpm = (zps / cpls).view([n, 1])
    zt = zp + yc * zpm * beta
    ft = torch.log(zt).detach()
    return ft


def fobsc_loss(model, pVec0, output, target, delta):
    pVec = torch.nn.utils.parameters_to_vector(model.parameters())
    return delta * F.mse_loss(pVec, pVec0) + F.mse_loss(output, target)


def Jf(model, ulx, ult, mpx, mpt, dp, dm, pvc0, mapping=True):
    pvc = torch.nn.utils.parameters_to_vector(model.parameters())
    ulf = model(ulx)
    if mapping:
        mpf = model(mpx)
        return (F.mse_loss(ulf, ult, reduction='mean') +
                F.mse_loss(pvc, pvc0, reduction='sum') * dp +
                F.mse_loss(mpf, mpt, reduction='mean') * dm)
    else:
        return (F.mse_loss(ulf, ult, reduction='mean') +
                F.mse_loss(pvc, pvc0, reduction='sum') * dp)


def generate_target_set(model, ulset, batch_size, beta, device):
    targets = []
    if batch_size:
        ulloader = DataLoader(dataset=ulset, batch_size=batch_size)
    else:
        ulloader = DataLoader(dataset=ulset, batch_size=64)

    n_batch = len(ulloader)
    for i, data in enumerate(ulloader):
        # print(f'generate target set idx: {i} / {n_batch} ')
        ulx, uly, _ = data
        ulx = ulx.to(device)
        uly = uly.to(device)
        ult = min_div_target(model, ulx, uly, beta)
        targets.append(ult)
    targets = torch.concat(targets, dim=0)
    return TempDataset(data=ulset.data, targets=targets)


def generate_noisy_target_set(model, ulset, batch_size, beta, device):
    targets = []
    if batch_size:
        ulloader = DataLoader(dataset=ulset, batch_size=batch_size)
    else:
        ulloader = DataLoader(dataset=ulset, batch_size=64)

    n_batch = len(ulloader)
    for i, data in enumerate(ulloader):
        # print(f'generate target set idx: {i} / {n_batch} ')
        ulx, uly, _ = data
        ulx = ulx.to(device)
        uly = uly.to(device)
        ult = min_div_noisy_target(model, ulx, uly, beta)
        targets.append(ult)
    targets = torch.concat(targets, dim=0)
    return TempDataset(data=ulset.data, targets=targets)


def udruu(model, tgset, mpset, lr, dp, dm, batch_size, epoch, device):
    # ulset.data.to(device)
    # ulset.targets.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    pvc0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
    tgloader = DataLoader(dataset=tgset, batch_size=batch_size)
    uln = len(tgset)
    mpn = len(mpset)
    flsl = []
    for iep in range(epoch):
        for tgx, tgt, tgi in tgloader:
            tgx = tgx.to(device)
            tgt = tgt.to(device)
            # ult = min_div_target(model, ulx, uly)
            mpi = torch.randint(high=mpn, size=[batch_size]).to(device)
            mpx, mpt, _ = mpset[mpi]
            fls = Jf(model, tgx, tgt, mpx, mpt, dp, dm, pvc0)
            fls.backward()
            flsl.append(fls)
            optimizer.step()

    return torch.tensor(flsl)


def get_correct_pred_idx(model, dataset, device, batch_size=64):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    idxl = []
    for data in dataloader:
        xs, ys, indices = data
        xs = xs.to(device)
        ys = ys.to(device)
        correct_pred_idx = torch.where(torch.argmax(model(xs), dim=-1) == torch.argmax(ys, dim=-1))[0]
        idxl.append(indices[correct_pred_idx])
    idxl = torch.concat(idxl)
    return idxl


def get_correct_pred_set(model, dataset, device, batch_size=64):
    cpidx = get_correct_pred_idx(model=model, dataset=dataset, device=device, batch_size=batch_size)
    return TempDataset(data=dataset.data[cpidx], targets=dataset.targets[cpidx]), cpidx


def random_sample_set(dataset, n_sample):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_sample]
    ds, ts, _ = dataset[indices]
    return TempDataset(data=ds, targets=ts)


def random_sample_data(dataset, n_sample):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_sample]
    xs, ts, _ = dataset[indices]
    return xs, ts


def reverse(model, x, y, lr, epoch):
    optimizer = torch.optim.Adam(params=[x], lr=lr)
    for iep in range(epoch):
        optimizer.zero_grad()
        output = model(x)
        ls = utils.ce_loss(output, y) + torch.mean(x)
        ls.backward()
        optimizer.step()
    return x.detach()


def generate_mapping_set(model, init_range, xsz, ysz, labels: list, numlbs: list, bsz_mp, lr, epoch, device):
    xs = []
    ys = []
    # model.to(device)
    for label, num in zip(labels, numlbs):
        # print(f'generate mapping set idx: {label} , {num}')
        xr = torch.rand(size=[bsz_mp] + list(xsz)) * init_range
        xr.requires_grad_(True)
        yr = torch.zeros(size=[bsz_mp] + list(ysz))
        yr[:, label] = 1
        xr = xr.to(device).detach()
        yr = yr.to(device)
        xr = reverse(model, xr, yr, lr, epoch)
        xs.append(xr)
        ys.append(yr)
    ts = []
    for x in xs:
        t = model(x).detach()
        t.to(device)
        ts.append(t)
    xs = torch.concat(xs, dim=0)
    ts = torch.concat(ts, dim=0)
    return TempDataset(data=xs, targets=ts)


def udrus(model, ulset, tgset, mpset, lr, dp, dm, batch_size, epoch, n_sample, max_iter, device):
    flsl = []
    ncpl = []
    model.train()
    for i in range(max_iter):
        correct_pred_ulset, cpidx = get_correct_pred_set(model=model, dataset=ulset, device=device)
        n_correct_pred = len(correct_pred_ulset)
        ncpl.append(n_correct_pred)
        if n_correct_pred == 0:
            break
        rs_tgset = random_sample_set(TempDataset(data=tgset.data[cpidx],
                                                 targets=tgset.targets[cpidx]),
                                     n_sample)

        fls = udruu(model, rs_tgset, mpset, lr, dp, dm, batch_size, epoch, device)
        flsl.append(fls)
    return torch.concat(flsl), ncpl


def udrur(model, ulset, rm_labels, beta, mapping, numlbs, bsz_mp, init_range, epoch_mp,
          lr, dp, dm, batch_size, epoch, record_acc, device):
    model.to(device)

    # ulset = get_correct_pred_set(model=model, dataset=ulset, device=device, batch_size=batch_size)[0]
    if batch_size:
        filter_bsz = batch_size
    else:
        filter_bsz = 64
    ulset = dataset_filter(model=model, dataset=ulset, device=device, batch_size=filter_bsz)[0]

    x0, y0, _ = ulset[0]
    xsz = x0.size()
    ysz = y0.size()
    if not bsz_mp:
        bsz_mp = 64
    if mapping:
        mpset = generate_mapping_set(model, init_range, xsz, ysz, rm_labels, numlbs, bsz_mp, lr, epoch_mp, device)
    else:
        mpset = None

    if batch_size:

        tgset = generate_target_set(model=model, ulset=ulset, batch_size=batch_size, beta=beta, device=device)
        tgloader = DataLoader(dataset=tgset, batch_size=batch_size)

        dp = dp / len(tgloader)
        pvc0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        jfls = []
        for iep in tqdm(range(epoch), leave=False):
            optimizer.zero_grad()
            jfl_sum = torch.tensor(data=[0.0]).to(device)
            for i, data in enumerate(tgloader):
                if mapping:
                    mpx, mpt = random_sample_data(mpset, batch_size)
                    mpx = mpx.to(device)
                    mpt = mpt.to(device)
                else:
                    mpx = None
                    mpt = None
                ulx, ult, _ = data
                ulx = ulx.to(device)
                ult = ult.to(device)
                jfl = Jf(model, ulx, ult, mpx, mpt, dp, dm, pvc0, mapping=mapping)
                jfl.backward()
                jfl_sum += jfl.detach()
            jfls.append(jfl_sum)
            optimizer.step()
            # print(f'fobsc iep: {iep}/{epoch}\t, jfl_sum: {float(jfl_sum)}')
    else:
        ulx, uly, _ = ulset[:]
        ulx = ulx.to(device)
        uly = uly.to(device)
        ult = min_div_target(model, ulx, uly, beta).to(device)
        pvc0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
        n_sample = len(ulset)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        jfls = []
        for iep in tqdm(range(epoch), leave=False):
            optimizer.zero_grad()
            if mapping:
                mpx, mpt = random_sample_data(mpset, n_sample)
                mpx = mpx.to(device)
                mpt = mpt.to(device)
            else:
                mpx = None
                mpt = None
            jfl = Jf(model, ulx, ult, mpx, mpt, dp, dm, pvc0, mapping=mapping)
            jfl.backward()
            optimizer.step()
            jfls.append(jfl)
    return jfls, []


def udrur_noisy(model, ulset, rm_labels, beta, mapping, numlbs, bsz_mp, init_range, epoch_mp,
                lr, dp, dm, batch_size, epoch, record_acc, device):
    model.to(device)

    # ulset = get_correct_pred_set(model=model, dataset=ulset, device=device, batch_size=batch_size)[0]

    x0, y0, _ = ulset[0]
    xsz = x0.size()
    ysz = y0.size()
    if not bsz_mp:
        bsz_mp = 64
    if mapping:
        mpset = generate_mapping_set(model, init_range, xsz, ysz, rm_labels, numlbs, bsz_mp, lr, epoch_mp, device)
    else:
        mpset = None

    if batch_size:

        tgset = generate_noisy_target_set(model=model, ulset=ulset, batch_size=batch_size, beta=beta, device=device)
        tgloader = DataLoader(dataset=tgset, batch_size=batch_size)

        dp = dp / len(tgloader)
        pvc0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        jfls = []
        for iep in range(epoch):
            optimizer.zero_grad()
            jfl_sum = torch.tensor(data=[0.0]).to(device)
            for i, data in enumerate(tgloader):
                if mapping:
                    mpx, mpt = random_sample_data(mpset, batch_size)
                    mpx = mpx.to(device)
                    mpt = mpt.to(device)
                else:
                    mpx = None
                    mpt = None
                ulx, ult, _ = data
                ulx = ulx.to(device)
                ult = ult.to(device)
                jfl = Jf(model, ulx, ult, mpx, mpt, dp, dm, pvc0, mapping=mapping)
                jfl.backward()
                jfl_sum += jfl.detach()
            jfls.append(jfl_sum)
            optimizer.step()
            # print(f'fobsc iep: {iep}/{epoch}\t, jfl_sum: {float(jfl_sum)}')
    else:
        ulx, uly, _ = ulset[:]
        ulx = ulx.to(device)
        uly = uly.to(device)
        ult = min_div_noisy_target(model, ulx, uly, beta).to(device)
        pvc0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
        n_sample = len(ulset)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        jfls = []
        for iep in range(epoch):
            optimizer.zero_grad()
            if mapping:
                mpx, mpt = random_sample_data(mpset, n_sample)
                mpx = mpx.to(device)
                mpt = mpt.to(device)
            else:
                mpx = None
                mpt = None
            jfl = Jf(model, ulx, ult, mpx, mpt, dp, dm, pvc0, mapping=mapping)
            jfl.backward()
            optimizer.step()
            jfls.append(jfl)
    return jfls, []


def index_filter(model, dataset, device, batch_size=64):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    idxl = []
    for data in dataloader:
        xs, ys, indices = data
        xs = xs.to(device)
        ys = ys.to(device)
        output = model(xs)
        agmx = torch.argmax(output, dim=-1).unsqueeze(dim=-1)
        idx = indices[ys.gather(dim=1, index=agmx).squeeze() == 1]
        if idx.ndim != 1:
            idx = idx.view(-1)
        idxl.append(idx)
    idxl = torch.concat(idxl)
    return idxl


def dataset_filter(model, dataset, device, batch_size=64):
    idxl = index_filter(model, dataset, device, batch_size=batch_size)
    return TempDataset(data=dataset.data[idxl], targets=dataset.targets[idxl]), idxl


