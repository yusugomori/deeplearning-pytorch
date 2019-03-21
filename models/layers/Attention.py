import torch
import torch.nn as nn


class Attention(nn.Module):
    '''
    Reference:
        "Effective Approaches to Attention-based Neural Machine Translation"
        https://arxiv.org/abs/1508.04025
    '''
    def __init__(self,
                 output_dim,
                 hidden_dim,  # suppose dim(hs) = dim(ht)
                 device='cpu'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.W_a = nn.Parameter(torch.Tensor(hidden_dim,
                                             hidden_dim))

        self.W_c = nn.Parameter(torch.Tensor(hidden_dim + hidden_dim,
                                             output_dim))

        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_normal_(self.W_a)
        nn.init.xavier_normal_(self.W_c)

    def forward(self, ht, hs, source=None, pad_value=0):
        '''
        # Argument
            ht, hs: (sequence, batch, out_features)
            source: (sequence, batch)
        '''
        score = torch.einsum('jik,kl->jil', (hs, self.W_a))
        score = torch.einsum('jik,lik->jil', (ht, score))

        score = score - torch.max(score,
                                  dim=-1,
                                  keepdim=True)[0]  # softmax max trick

        score = torch.exp(score)
        if source is not None:
            # mask_source = (source.t() != pad_value).unsqueeze(0)
            # score = score * mask_source.float().to(self.device)
            mask_source = source.t().eq(pad_value).unsqueeze(0)
            score.data.masked_fill_(mask_source, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum('jik,kil->jil', (a, hs))

        h = torch.cat((c, ht), -1)
        return torch.tanh(torch.einsum('jik,kl->jil', (h, self.W_c)) + self.b)
