import torch
import torch.nn as nn


class DotProductAttention(nn.Module):
    def __init__(self,
                 # d_model,
                 device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        score = torch.einsum('ijk,ilk->ijl', (q, k))
        score = score - torch.max(score,
                                  dim=-1,
                                  keepdim=True)[0]  # softmax max trick

        score = torch.exp(score)
        if mask is not None:
            # suppose `mask` is a mask of source
            # in source-target-attention, source is `k` and `v`
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).repeat(1, score.size(1), 1)
            # score = score * mask.float().to(self.device)
            score.data.masked_fill_(mask, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum('ijk,ikl->ijl', (a, v))

        return c
