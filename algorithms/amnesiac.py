import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm


def amnesiac_train(model: nn.Module, dataloader: DataLoader, optimizer, epoch, device, save_path, loss, reload=False):
    model.to(device)
    pVec = torch.nn.utils.parameters_to_vector(model.parameters())
    param_numel = pVec.numel()
    n_data = len(dataloader.dataset)
    # record = torch.zeros(size=[n_data, param_numel]).to(device)
    model.train()
    if reload:
        for index in tqdm(range(n_data)):
            torch.save(torch.zeros(size=[param_numel]), f'./amn_record/rec{index}.pth')
    for ep in range(epoch):
        loss_sum = 0
        for i, data in enumerate(tqdm(dataloader)):
            x, y, index = data
            record = torch.load(f'./amn_record/rec{int(index)}.pth').to(device)
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            optimizer.zero_grad()
            output = model(x)
            l = loss(output, y)
            l.backward()
            grads = []
            for p in model.parameters():
                grads.append(p.grad.data)
            gVec = torch.nn.utils.parameters_to_vector(grads)
            record += gVec * optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()
            loss_sum += l.data
            torch.save(record, f'./amn_record/rec{int(index)}.pth')
        print(loss_sum)
    torch.save(model.state_dict(), save_path)
    # torch.save(record, record_path)


def amnesiac_unlearn(model, indices_to_unlearn:list, device):
    pVec = torch.nn.utils.parameters_to_vector(model.parameters())
    for index in indices_to_unlearn:
        record = torch.load(f'./amn_record/rec{int(index)}.pth').to(device)
        pVec += record
    torch.nn.utils.vector_to_parameters(pVec, model.parameters())
    return model


if __name__ == '__main__':
    import data
    import utils
    from models.mlp import MLP
    train_set = data.MNIST(root='./dataset', flatten=True, train=False)
    model = MLP(features=[784, 512, 10], bias=False, activ=nn.ReLU)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    amnesiac_train(model=model,
                   dataloader=train_loader,
                   optimizer=optimizer,
                   epoch=50,
                   device=device,
                   save_path='./weights/MNIST/amnesiac_w.pth',
                   loss=utils.ce_loss,
                   record_path='./weights/MNIST/amnesiac_r.pth')
