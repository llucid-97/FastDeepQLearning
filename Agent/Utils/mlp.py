import torch
from torch import nn, Tensor


def xavier_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class VanillaMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes: tuple, activation_class=nn.ReLU):
        super().__init__()

        layers = []
        if len(hidden_sizes) > 0:
            layers.append(nn.Linear(in_features, hidden_sizes[0]))
            layers.append(activation_class())
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(activation_class())
            layers.append(nn.Linear(hidden_sizes[-1], out_features))
        else:
            layers.append(nn.Linear(in_features, out_features))

        self.net = nn.Sequential(*layers)
        self.apply(xavier_init_)

    def forward(self, x):
        return self.net(x)


class SkipMLP(nn.Module):
    """feed forward network that uses skip connections for better gradient flow (but more memory-expensive)
    TODO: implement better memory efficiency using https://github.com/gpleiss/efficient_densenet_pytorch
    """

    def __init__(self, in_features, out_features, hidden_sizes: tuple, activation_class=nn.ReLU):
        super().__init__()

        self._depth = len(hidden_sizes)
        Linear = nn.Linear

        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                Linear(in_features + sum(hidden_sizes[:i]), hidden_sizes[i]),
                nn.LeakyReLU(inplace=True),
            )
            for i in range(len(hidden_sizes))
        ])

        penult_dim = in_features + sum(hidden_sizes)
        self.head = Linear(penult_dim, out_features)
        self.apply(xavier_init_)

    def forward(self, x):
        for i, hidden_layer in enumerate(self.feature_extractor):
            y = hidden_layer(x)
            x = torch.cat((x, y), dim=-1)

        return self.head(x)


MLP = VanillaMLP
