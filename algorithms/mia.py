import torch
import torch.nn as nn
from models.mlp import MLP
import utils
import data
from torch.utils.data import DataLoader
from tqdm import tqdm


class Attacker(MLP):
    def __init__(self, features: list, bias, activ):
        super(Attacker, self).__init__(features, bias, activ)


def generate_attack_set(model, dataset, path=None,loss=utils.ce_loss,
                        attacker_name='attacker_tmp'):
    model.to(torch.device('cpu'))
    if path:
        return data.TempDataset(torch.load(path + attacker_name + '_data.pth'),
                                torch.load(path + attacker_name + '_targets.pth'))
    n_data = len(dataset)
    datal = []
    for i in tqdm(range(n_data), leave=False):
        paramsl = []
        model.zero_grad()
        xf = dataset[i:i + 1][0]
        for layer in model.layers:
            xf = layer(xf)
            paramsl.append(xf)
        ls = loss(xf, dataset[i:i + 1][1])
        ls.backward()
        for p in model.parameters():
            paramsl.append(p.grad)
        param2v = torch.nn.utils.parameters_to_vector(paramsl)
        datal.append(param2v)

    mia_targets = torch.zeros(size=[n_data, 2])
    mia_targets[:n_data, 0] = 1

    mia_data = torch.concat(datal, dim=0).view(n_data, -1)

    # torch.save(mia_data, './attacker/' + attacker_name + '_data.pth')
    # torch.save(mia_targets, './attacker/' + attacker_name + '_targets.pth')
    return data.TempDataset(mia_data, mia_targets)


def generate_attack_train_set(model, member_set, non_member_set, loss=utils.ce_loss, path=None, train=True,
                              attacker_name='attacker'):
    model.to(torch.device('cpu'))
    if path:
        return data.TempDataset(torch.load(path + attacker_name + '_data.pth'),
                                torch.load(path + attacker_name + '_targets.pth'))
    n_member = len(member_set)
    n_non_member = len(non_member_set)
    datal = []
    for i in tqdm(range(n_non_member), leave=False):
        paramsl = []
        model.zero_grad()
        xf = member_set[i:i + 1][0]
        for layer in model.layers:
            xf = layer(xf)
            paramsl.append(xf)
        ls = loss(xf, member_set[i:i + 1][1])
        ls.backward()
        for p in model.parameters():
            paramsl.append(p.grad)
        param2v = torch.nn.utils.parameters_to_vector(paramsl)
        datal.append(param2v)
    if train:
        for i in tqdm(range(n_non_member), leave=False):
            paramsl = []
            model.zero_grad()
            xf = non_member_set[i:i + 1][0]
            for layer in model.layers:
                xf = layer(xf)
                paramsl.append(xf)
            ls = loss(xf, non_member_set[i:i + 1][1])
            ls.backward()
            for p in model.parameters():
                paramsl.append(p.grad)
            param2v = torch.nn.utils.parameters_to_vector(paramsl)
            datal.append(param2v)
        mia_targets = torch.zeros(size=[n_non_member + n_non_member, 2])
        mia_targets[:n_non_member, 0] = 1
        mia_targets[n_non_member:, 1] = 1
        mia_data = torch.concat(datal, dim=0).view(n_non_member + n_non_member, -1)
    else:
        mia_targets = torch.zeros(size=[n_non_member, 2])
        mia_targets[:n_non_member, 0] = 1
        mia_data = torch.concat(datal, dim=0).view(n_non_member, -1)

    # torch.save(mia_data, './attacker/' + attacker_name + '_data.pth')
    # torch.save(mia_targets, './attacker/' + attacker_name + '_targets.pth')
    return data.TempDataset(mia_data, mia_targets)


def train_attacker(attacker, dataloader, optimizer, epoch, device, save_path, loss=utils.ce_loss):
    utils.train(attacker, dataloader, optimizer, epoch, device, save_path, loss)


def vec2attacker_input(model, xyi, loss=utils.ce_loss):
    x, y, _ = xyi
    n = x.size(0)
    retl = []
    for i in range(n):
        paramsl = []
        model.zero_grad()
        xf = x[i:i + 1]
        for layer in model.layers:
            xf = layer(xf)
            paramsl.append(xf)
        ls = loss(xf, y[i:i + 1])
        ls.backward()
        for p in model.parameters():
            paramsl.append(p.grad)
        param2v = torch.nn.utils.parameters_to_vector(paramsl)
        retl.append(param2v)
    return torch.concat(retl, dim=0).view(n, -1)


def main(partial=False):
    batch_size = 64
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = MLP(features=[784, 1024, 512, 256, 10], bias=False, activ=nn.ReLU)
    if partial:
        model.load_state_dict(torch.load('../weights/MNIST/mnist_partial.pth'))
        member_set = data.PartialLabelSet(dataset=data.MNIST(root='../datasets', train=True, flatten=True),
                                          partial_rate=0.2,
                                          targets_root='../weights/partial_targets_train.pth')
        non_member_set = data.PartialLabelSet(dataset=data.MNIST(root='../datasets', train=False, flatten=True),
                                              partial_rate=0.2,
                                              targets_root='../weights/partial_targets_valid.pth')
    else:
        model.load_state_dict(torch.load('../weights/MNIST/m0.pth'))
        member_set = data.MNIST(root='../dataset', flatten=True, train=True)
        non_member_set = data.MNIST(root='../dataset', flatten=True, train=False)

    attack_train_set = generate_attack_set(model, member_set, non_member_set)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    epoch = 100
    attacker = Attacker(features=[10, 128, 64, 2], bias=False, activ=nn.ReLU)
    dataloader = DataLoader(dataset=attack_train_set, batch_size=batch_size)
    train_attacker(attacker=attacker,
                   dataloader=dataloader,
                   optimizer=optimizer,
                   epoch=epoch,
                   device=device,
                   save_path='../weights/attacker.pth',
                   loss=utils.ce_loss)


if __name__ == '__main__':
    model = MLP(features=[784, 1024, 512, 256, 10], bias=False, activ=nn.ReLU)
    model.load_state_dict(torch.load('../weights/MNIST/m0.pth', map_location=torch.device('cpu')))
    member_set = data.MNIST(root='../dataset', flatten=True, train=True)
    non_member_set = data.MNIST(root='../dataset', flatten=True, train=False)
    dataset = generate_attack_set(model, member_set, non_member_set)
    print(dataset.data)
    print(dataset.data.size())
