import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, features: list, bias, activ):
        super(MLP, self).__init__()
        self.layers = self._make_layer(features, bias, activ)
        self.features = features
        self.bias = bias
        self.activ = activ

    @staticmethod
    def _make_layer(features, bias, activ):
        n_layer = len(features) - 1
        layers = []
        for i in range(n_layer-1):
            layers.append(nn.Linear(in_features=features[i], out_features=features[i + 1], bias=bias))
            layers.append(activ())
        layers.append(nn.Linear(in_features=features[-2], out_features=features[-1], bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    model = MLP(features=[4, 4, 4, 6, 1], bias=True, activ=nn.ReLU)
    print(model)
    x = torch.rand(size=[1, 4])
    print(model(x))