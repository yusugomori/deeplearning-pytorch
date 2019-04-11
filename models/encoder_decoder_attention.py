import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
# from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.datasets.small_parallel_enja import load_small_parallel_enja
from utils.preprocessing.sequence import pad_sequences, sort
from sklearn.utils import shuffle
from layers import Attention


class EncoderDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 bos_value=1,
                 max_len=20,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, device=device)
        self.decoder = Decoder(hidden_dim, output_dim, device=device)

        self._BOS = bos_value
        self._max_len = max_len
        self.output_dim = output_dim

    def forward(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.size()[1]
        if target is not None:
            len_target_sequences = target.size()[0]
        else:
            len_target_sequences = self._max_len

        hs, states = self.encoder(source)

        y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=device) * self._BOS
        output = torch.zeros((len_target_sequences,
                              batch_size,
                              self.output_dim),
                             device=device)

        for t in range(len_target_sequences):
            out, states = self.decoder(y, hs, states, source=source)
            output[t] = out

            if use_teacher_forcing and target is not None:
                y = target[t].unsqueeze(0)
            else:
                y = out.max(-1)[1]

        return output


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, x):
        len_source_sequences = (x.t() > 0).sum(dim=-1)
        x = self.embedding(x)
        pack = pack_padded_sequence(x, len_source_sequences)
        y, states = self.lstm(pack)
        y, _ = pad_packed_sequence(y)

        return y, states


class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.attn = Attention(hidden_dim, hidden_dim, device=device)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hs, states, source=None):
        x = self.embedding(x)
        x, states = self.lstm(x, states)
        x = self.attn(x, hs, source=source)
        y = self.out(x)
        # y = torch.log_softmax(x, dim=-1)

        return y, states


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t,
                   teacher_forcing_rate=0.5,
                   pad_value=0):
        use_teacher_forcing = (random.random() < teacher_forcing_rate)
        model.train()
        preds = model(x, t, use_teacher_forcing=use_teacher_forcing)
        loss = compute_loss(t.contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def valid_step(x, t):
        model.eval()
        preds = model(x, t, use_teacher_forcing=False)
        loss = compute_loss(t.contiguous().view(-1),
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

            x = torch.LongTensor(x).t()
            y = torch.LongTensor(y).t()

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
        load_small_parallel_enja(to_ja=True, add_bos=False)

    train_dataloader = ParallelDataLoader((x_train, y_train),
                                          shuffle=True)
    valid_dataloader = ParallelDataLoader((x_test, y_test))
    test_dataloader = ParallelDataLoader((x_test, y_test),
                                         batch_size=1,
                                         shuffle=True)

    '''
    Build model
    '''
    input_dim = num_x
    hidden_dim = 128
    output_dim = num_y

    model = EncoderDecoder(input_dim,
                           hidden_dim,
                           output_dim,
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

        for (source, target) in train_dataloader:
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
            out = out.max(dim=-1)[1].view(-1).tolist()
            out = ' '.join(ids_to_sentence(out, i2w_y))
            source = ' '.join(ids_to_sentence(source.view(-1).tolist(), i2w_x))
            target = ' '.join(ids_to_sentence(target.view(-1).tolist(), i2w_y))
            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 10:
                break
