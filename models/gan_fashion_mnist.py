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


class Generator(nn.Module):
    def __init__(self,
                 input_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, 256*14*14)
        self.bn1 = nn.BatchNorm1d(256*14*14)
        self.conv1 = nn.Conv2d(256, 128,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1,
                               kernel_size=(1, 1))

    def forward(self, x):
        h = self.linear(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = h.view(-1, 256, 14, 14)
        h = nn.functional.interpolate(h, size=(28, 28))
        h = self.conv1(h)
        h = self.bn2(h)
        h = torch.relu(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = torch.relu(h)
        h = self.conv3(h)
        y = torch.sigmoid(h)

        return y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256,
                               kernel_size=(3, 3),
                               padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(256, 512,
                               kernel_size=(3, 3),
                               padding=1)
        self.dropout2 = nn.Dropout(0.3)
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(512, 1000)
        self.out = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.dropout1(h)
        h = self.conv2(h)
        h = torch.relu(h)
        h = self.dropout2(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        y = self.out(h)
        y = torch.log_softmax(h, dim=-1)

        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step_D(x):
        model.set_trainable(model.D, True)
        model.train()
        noise = gen_noise(x.size(0))
        gen = model.G(noise)
        x = torch.cat((x, gen))
        t = torch.zeros(x.size(0)).long().to(device)
        t[:t.size(0)//2] = 1

        preds = model.D(x)
        loss = compute_loss(t, preds)
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

        return loss, preds

    def train_step_G():
        model.set_trainable(model.D, False)
        model.train()
        noise = gen_noise(x.size(0))
        t = torch.ones(x.size(0)).long().to(device)
        preds = model(noise)
        loss = compute_loss(t, preds)
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        return loss, preds

    def generate(batch_size=10):
        model.eval()
        noise = gen_noise(batch_size)
        gen = model.G(noise)

        return gen

    def gen_noise(batch_size):
        return torch.empty(batch_size, 128).uniform_(0, 1).to(device)

    '''
    Load data
    '''
    root = os.path.join(os.path.dirname(__file__), '..', 'data')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x / 255.])
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
    model = GAN().to(device)
    criterion = nn.NLLLoss()
    optimizer_G = optimizers.Adam(model.G.parameters())
    optimizer_D = optimizers.Adam(model.D.parameters())

    '''
    Train model
    '''
    epochs = 10000
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'output')

    for epoch in range(epochs):
        train_loss_D = 0.
        train_loss_G = 0.
        test_loss = 0.

        for (x, _) in train_dataloader:
            x = x.to(device)

            loss, _ = train_step_D(x)  # train D
            train_loss_D += loss.item()

            loss, _ = train_step_G()  # train G
            train_loss_G += loss.item()

        train_loss_D /= len(train_dataloader)
        train_loss_G /= len(train_dataloader)

        print('Epoch: {}, D Cost: {:.3f}, G Cost: {:.3f}'.format(
            epoch+1,
            train_loss_D,
            train_loss_G
        ))

        # Generate images
        if epoch % 10 == 9 or epoch == epochs - 1:
            images = generate(batch_size=10)
            images = images.squeeze().detach().cpu().numpy()
            plt.figure(figsize=(8, 4))
            for i, image in enumerate(images):
                plt.subplot(2, 5, i+1)
                plt.imshow(image, cmap='binary')
                plt.axis('off')
            plt.tight_layout()
            template = '{}/gan_fashion_mnist_epoch_{:0>4}.png'
            plt.savefig(template.format(out_path, epoch), dpi=300)
