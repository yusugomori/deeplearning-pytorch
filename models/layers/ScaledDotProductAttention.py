import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_model,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.scaler = np.sqrt(d_model)

    def forward(self, q, k, v, mask=None):
        score = torch.einsum('jik,lik->jil', (q, k)) / self.scaler
        score = score - torch.max(score,
                                  dim=-1,
                                  keepdim=True)[0]  # softmax max trick

        score = torch.exp(score)
        if mask is not None:
            # suppose `mask` is a mask of source
            # in source-target-attention, source is `k` and `v`
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(0)
            # score = score * mask.float().to(self.device)
            score.data.masked_fill_(mask, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum('jik,kil->jil', (a, v))

        return c
