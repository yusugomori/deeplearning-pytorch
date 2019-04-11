import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
# from torch.utils.data import Dataset, DataLoader
from utils.datasets.small_parallel_enja import load_small_parallel_enja
from utils.preprocessing.sequence import pad_sequences, sort
from sklearn.utils import shuffle
from layers import PositionalEncoding
from layers import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=20,
                 bos_value=1,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               max_len=max_len,
                               device=device)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               max_len=max_len,
                               device=device)
        self.out = nn.Linear(d_model, depth_target)
        nn.init.xavier_normal_(self.out.weight)

        self._BOS = bos_value
        self._max_len = max_len

    def forward(self, source, target=None):
        source_mask = self.sequence_mask(source)

        hs = self.encoder(source, mask=source_mask)

        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.size(1)
            target_mask = self.sequence_mask(target).unsqueeze(1)
            subsequent_mask = self.subsequence_mask(target)
            target_mask = torch.gt(target_mask + subsequent_mask, 0)

            y = self.decoder(target, hs,
                             mask=target_mask,
                             source_mask=source_mask)
            output = self.out(y)
        else:
            batch_size = source.size(0)
            len_target_sequences = self._max_len

            output = torch.ones((batch_size, 1),
                                dtype=torch.long,
                                device=self.device) * self._BOS

            for t in range(len_target_sequences - 1):
                target_mask = self.subsequence_mask(output)
                out = self.decoder(output, hs,
                                   mask=target_mask,
                                   source_mask=source_mask)
                out = self.out(out)[:, -1:, :]
                out = out.max(-1)[1]
                output = torch.cat((output, out), dim=1)

        return output

    def sequence_mask(self, x):
        return x.eq(0)

    def subsequence_mask(self, x):
        shape = (x.size(1), x.size(1))
        mask = torch.triu(torch.ones(shape, dtype=torch.uint8),
                          diagonal=1)
        return mask.unsqueeze(0).repeat(x.size(0), 1, 1).to(self.device)


class Encoder(nn.Module):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_source,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.encs = nn.ModuleList([
            EncoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         max_len=max_len,
                         device=device) for _ in range(N)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for enc in self.encs:
            y = enc(y, mask=mask)

        return y


class EncoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128,
                 device='cpu'):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y


class Decoder(nn.Module):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_target,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.decs = nn.ModuleList([
            DecoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         max_len=max_len,
                         device=device) for _ in range(N)])

    def forward(self, x, hs,
                mask=None,
                source_mask=None):
        x = self.embedding(x)
        y = self.pe(x)

        for dec in self.decs:
            y = dec(y, hs,
                    mask=mask,
                    source_mask=source_mask)

        return y


class DecoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128,
                 device='cpu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, hs,
                mask=None,
                source_mask=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=source_mask)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y


class FFN(nn.Module):
    '''
    Position-wise Feed-Forward Networks
    '''
    def __init__(self, d_model, d_ff,
                 device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        # self.l1 = nn.Conv1d(d_model, d_ff, 1)
        # self.l2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        y = self.l2(x)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def valid_step(x, t):
        model.eval()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        return loss, preds

    def test_step(x):
        model.eval()
        preds = model(x)
        return preds

    def ids_to_sentence(ids, i2w):
        return [i2w[id] for id in ids]

    '''
    Load data
    '''
    class ParallelDataLoader(object):
        def __init__(self, dataset,
                     batch_size=128,
                     shuffle=False,
                     random_state=None):
            if type(dataset) is not tuple:
                raise ValueError('argument `dataset` must be tuple,'
                                 ' not {}.'.format(type(dataset)))
            self.dataset = list(zip(dataset[0], dataset[1]))
            self.batch_size = batch_size
            self.shuffle = shuffle
            if random_state is None:
                random_state = np.random.RandomState(1234)
            self.random_state = random_state
            self._idx = 0

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(self.dataset):
                self._reorder()
                raise StopIteration()

            x, y = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
            x, y = sort(x, y, order='descend')
            x = pad_sequences(x, padding='post')
            y = pad_sequences(y, padding='post')

            x = torch.LongTensor(x)  # not use .t()
            y = torch.LongTensor(y)  # not use .t()

            self._idx += self.batch_size

            return x, y

        def _reorder(self):
            if self.shuffle:
                self.dataset = shuffle(self.dataset,
                                       random_state=self.random_state)
            self._idx = 0

    (x_train, y_train), \
        (x_test, y_test), \
        (num_x, num_y), \
        (w2i_x, w2i_y), (i2w_x, i2w_y) = \
        load_small_parallel_enja(to_ja=True)

    train_dataloader = ParallelDataLoader((x_train, y_train),
                                          shuffle=True)
    valid_dataloader = ParallelDataLoader((x_test, y_test))
    test_dataloader = ParallelDataLoader((x_test, y_test),
                                         batch_size=1,
                                         shuffle=True)

    '''
    Build model
    '''
    model = Transformer(num_x,
                        num_y,
                        N=3,
                        h=4,
                        d_model=128,
                        d_ff=256,
                        max_len=20,
                        device=device).to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optimizer = optimizers.Adam(model.parameters())

    '''
    Train model
    '''
    epochs = 20

    for epoch in range(epochs):
        print('-' * 20)
        print('Epoch: {}'.format(epoch+1))

        train_loss = 0.
        valid_loss = 0.

        for idx, (source, target) in enumerate(train_dataloader):
            source, target = source.to(device), target.to(device)
            loss, _ = train_step(source, target)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        for (source, target) in valid_dataloader:
            source, target = source.to(device), target.to(device)
            loss, _ = valid_step(source, target)
            valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)
        print('Valid loss: {:.3}'.format(valid_loss))

        for idx, (source, target) in enumerate(test_dataloader):
            source, target = source.to(device), target.to(device)
            out = test_step(source)
            out = out.view(-1).tolist()
            out = ' '.join(ids_to_sentence(out, i2w_y))
            source = ' '.join(ids_to_sentence(source.view(-1).tolist(), i2w_x))
            target = ' '.join(ids_to_sentence(target.view(-1).tolist(), i2w_y))
            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 10:
                break
