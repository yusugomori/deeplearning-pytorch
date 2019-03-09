import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
