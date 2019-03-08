import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
# from torch.utils.data import Dataset, DataLoader
from utils.datasets.small_parallel_enja import load_small_parallel_enja
from utils.preprocessing.sequence import pad_sequences, sort


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(self.dataset):
                self._reorder()
                raise StopIteration()

            x, y = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
            x, y = sort(x, y)
            x = pad_sequences(x, padding='post')
            y = pad_sequences(y, padding='post')

            self._idx += self.batch_size

            return x, y

        def _reorder(self):
            if self.shuffle:
                self.data = shuffle(self.dataset,
                                    random_state=self.random_state)
            self._idx = 0

    (x_train, y_train), \
        (x_test, y_test), \
        (num_x, num_y), \
        (w2i_x, w2i_y), (i2w_x, i2w_y) = \
        load_small_parallel_enja(to_ja=True, add_bos=False)

    train_dataloader = ParallelDataLoader((x_train, y_train), shuffle=True)
    valid_dataloader = ParallelDataLoader((x_test, y_test))
    test_dataloader = ParallelDataLoader((x_test, y_test), batch_size=1)
