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


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.out(x)
        y = torch.log_softmax(x, dim=-1)

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
    root = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = \
        torchvision.datasets.MNIST(root=root,
                                   download=True,
                                   train=True,
                                   transform=transform)
    mnist_test = \
        torchvision.datasets.MNIST(root=root,
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
    model = LeNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = optimizers.Adam(model.parameters())

    '''
    Train model
    '''
    epochs = 10

    for epoch in range(epochs):
        train_loss = 0.
        test_loss = 0.
        test_acc = 0.

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, _ = train_step(x, t)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        if epoch % 5 == 4 or epoch == epochs - 1:
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
