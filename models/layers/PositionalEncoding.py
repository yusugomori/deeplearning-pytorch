import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    '''
    Positional encoding layer with sinusoid
    '''
    def __init__(self, output_dim,
                 max_len=6000,
                 device='cpu'):
        super().__init__()
        self.output_dim = output_dim
        self.max_len = max_len
        pe = self.initializer()
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        '''
        # Argument
            x: (batch, sequence)
        '''
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        return x + Variable(pe, requires_grad=False)

    def initializer(self):
        pe = \
            np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                      for pos in range(self.max_len)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return torch.from_numpy(pe).float()
