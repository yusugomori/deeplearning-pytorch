import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from layers import GlobalAvgPool2d


class ResNet34(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=1)
        self.block1 = nn.ModuleList([
            self._building_block(64) for _ in range(3)
        ])
        self.conv2 = nn.Conv2d(64, 128,
                               kernel_size=(1, 1),
                               stride=(2, 2))
        self.block2 = nn.ModuleList([
            self._building_block(128) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(128, 256,
                               kernel_size=(1, 1),
                               stride=(2, 2))
        self.block3 = nn.ModuleList([
            self._building_block(256) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(256, 512,
                               kernel_size=(1, 1),
                               stride=(2, 2))
        self.block4 = nn.ModuleList([
            self._building_block(512) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(512, 1000)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)

        return y

    def _building_block(self, channel_out=64):
        channel_in = channel_out
        return Block(channel_in, channel_out)


class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_out, channel_out,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        y = self.relu2(x + h)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def test_step(x, t):
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)

        return loss, preds

    '''
    Load data
    '''
    root = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'fashion_mnist')
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=True,
                                          transform=transform)
    mnist_test = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=False,
                                          transform=transform)

    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)
    test_dataloader = DataLoader(mnist_test,
                                 batch_size=100,
                                 shuffle=False)

    '''
    Build model
    '''
    model = ResNet34(10).to(device)
    criterion = nn.NLLLoss()
    optimizer = optimizers.Adam(model.parameters(), weight_decay=0.01)

    '''
    Train model
    '''
    epochs = 1

    for epoch in range(epochs):
        train_loss = 0.
        test_loss = 0.
        test_acc = 0.

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, _ = train_step(x, t)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        for (x, t) in test_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t)
            test_loss += loss.item()
            test_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
            epoch+1,
            test_loss,
            test_acc
        ))
