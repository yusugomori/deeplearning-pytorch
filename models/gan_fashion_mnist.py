import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers import GlobalAvgPool2d


class GAN(nn.Module):
    '''
    Simple Generative Adversarial Network
    '''
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

    def forward(self, x):
        x = self.G(x)
        y = self.D(x)

        return y

    def set_trainable(self, net, trainable=True):
        params = net.parameters()
        for p in params:
            p.requires_grad = trainable


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(256, 512,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(512*7*7, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.LeakyReLU(0.2)
        self.out = nn.Linear(1024, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.dropout1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.dropout2(h)
        h = h.view(-1, 512*7*7)
        h = self.fc(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.out(h)
        y = torch.sigmoid(h)

        return y


class Generator(nn.Module):
    def __init__(self,
                 input_dim=100):
        super().__init__()
        self.linear = nn.Linear(input_dim, 256*14*14)
        self.bn1 = nn.BatchNorm1d(256*14*14)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 128,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 1,
                               kernel_size=(1, 1))

    def forward(self, x):
        h = self.linear(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = h.view(-1, 256, 14, 14)
        h = nn.functional.interpolate(h, size=(28, 28))
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.conv3(h)
        y = torch.sigmoid(h)

        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x):
        batch_size = x.size(0)
        model.D.train()
        model.G.train()

        # train D
        # real images
        preds = model.D(x).squeeze()  # preds with true images
        t = torch.ones(batch_size).float().to(device)
        loss_D_real = compute_loss(t, preds)
        # fake images
        noise = gen_noise(batch_size)
        z = model.G(noise)
        preds = model.D(z.detach()).squeeze()  # preds with fake images
        t = torch.zeros(batch_size).float().to(device)
        loss_D_fake = compute_loss(t, preds)

        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # train G
        noise = gen_noise(batch_size)
        z = model.G(noise)
        preds = model.D(z).squeeze()  # preds with fake images
        t = torch.ones(batch_size).float().to(device)  # label as true
        loss_G = compute_loss(t, preds)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        return loss_D, loss_G

    def generate(batch_size=10):
        model.eval()
        noise = gen_noise(batch_size)
        gen = model.G(noise)

        return gen

    def gen_noise(batch_size):
        return torch.empty(batch_size, 100).uniform_(0, 1).to(device)

    '''
    Load data
    '''
    root = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'fashion_mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x / 255.])
    mnist_train = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=True,
                                          transform=transform)
    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)

    '''
    Build model
    '''
    model = GAN().to(device)
    criterion = nn.BCELoss()
    optimizer_D = optimizers.Adam(model.D.parameters())
    optimizer_G = optimizers.Adam(model.G.parameters())

    '''
    Train model
    '''
    epochs = 100
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'output')

    for epoch in range(epochs):
        train_loss_D = 0.
        train_loss_G = 0.
        test_loss = 0.

        for (x, _) in train_dataloader:
            x = x.to(device)
            loss_D, loss_G = train_step(x)

            train_loss_D += loss_D.item()
            train_loss_G += loss_G.item()

        train_loss_D /= len(train_dataloader)
        train_loss_G /= len(train_dataloader)

        print('Epoch: {}, D Cost: {:.3f}, G Cost: {:.3f}'.format(
            epoch+1,
            train_loss_D,
            train_loss_G
        ))

        if epoch % 5 == 4 or epoch == epochs - 1:
            images = generate(batch_size=16)
            images = images.squeeze().detach().cpu().numpy()
            plt.figure(figsize=(6, 6))
            for i, image in enumerate(images):
                plt.subplot(4, 4, i+1)
                plt.imshow(image, cmap='binary')
                plt.axis('off')
            plt.tight_layout()
            # plt.show()
            template = '{}/gan_fashion_mnist_epoch_{:0>4}.png'
            plt.savefig(template.format(out_path, epoch), dpi=300)
