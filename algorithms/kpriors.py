import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ce_loss
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from tqdm import tqdm

def kap(df, dw, output, output0, pVec, pVec0):
    return df(output, output0) + dw(pVec, pVec0)


def kpr(model, ulset, lr, batch_size, epoch,loss, device):
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    pVec0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    output0s = []
    if batch_size:
        ulloader = DataLoader(dataset=ulset, batch_size=batch_size)
        for i, data in enumerate(ulloader):
            ulx, uly, idx = data
            ulx = ulx.to(device)
            uly = uly.to(device)
            output0 = torch.softmax(model(ulx), dim=-1).detach().clone()
            output0s.append(output0)
        output0s = torch.concat(output0s, dim=0)
    else:
        ulx, uly, _ = ulset[:]
        ulx = ulx.to(device)
        uly = uly.to(device)
        output0 = torch.softmax(model(ulx), dim=-1).detach().clone()
    df = loss
    dw = F.mse_loss
    flsl = []
    ncpl = []
    alpha = 1.
    if batch_size:
        ulloader = DataLoader(dataset=ulset, batch_size=batch_size)
        for iep in tqdm(range(epoch), leave=False):
            ls_sum = 0
            for i, data in enumerate(ulloader):
                ulx, uly, idx = data
                ulx = ulx.to(device)
                uly = uly.to(device)
                output = torch.softmax(model(ulx), dim=-1)
                pVec = parameters_to_vector(model.parameters())
                output0 = output0s[idx]
                ls = - loss(output, uly) + alpha*kap(df, dw, output, output0, pVec, pVec0)
                ls.backward()
                optimizer.step()
                ls_sum += ls.data
            flsl.append(ls_sum)
    else:
        ulx, uly, _ = ulset[:]
        ulx = ulx.to(device)
        uly = uly.to(device)
        for iep in tqdm(range(epoch), leave=False):
            output = torch.softmax(model(ulx), dim=-1)
            pVec = parameters_to_vector(model.parameters())
            ls = - loss(output, uly) + alpha*kap(df, dw, output, output0, pVec, pVec0)
            ls.backward()
            optimizer.step()
            flsl.append(ls.data)
    return flsl, ncpl
    #
    #
    # for i in range(epoch):
    #     optimizer.zero_grad()
    #     output = torch.softmax(model(x), dim=-1)
    #     pVec = parameters_to_vector(model.parameters())
    #     ls = - loss(output, y) + kap(df, dw, output, output0, pVec, pVec0)
    #     ls.backward()
    #     optimizer.step()


if __name__ == '__main__':
    import torch.nn as nn
    from models.mlp import MLP
    from data import MNIST

    model = MLP(features=[784, 1024, 512, 256, 10], bias=False, activ=nn.ReLU)
    model.load_state_dict(torch.load('../weights/MNIST/m0.pth', map_location='cpu'))

    dataset = MNIST(root='../datasets', flatten=True)
    x, y, _ = dataset[:1]
    print(x, y)
    kpr(model=model, data_to_unlearn=(x, y), lr=1e-4, epoch=100)
