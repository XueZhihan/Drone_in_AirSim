import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input, size=3072):
        out = input.view(input.size(0), 1024, 1,  3)
        return out

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=3072, z_dim=32, device=None):
        super(VAE, self).__init__()
        #self.device = device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(1024, 128, kernel_size=(5, 7), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(6, 8), stride=2), # 25
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 8), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(8, 6), stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        #esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD