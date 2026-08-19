"""manigalati/usad의 지정 커밋에서 가져온 USAD 모델 코어다.
출처: e25af45c8e1c32783aed94fc7e5ab85effef2b1a의 usad.py.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(window_size, int(window_size / 2))
        self.linear2 = nn.Linear(int(window_size / 2), int(window_size / 4))
        self.linear3 = nn.Linear(int(window_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, window):
        output = self.relu(self.linear1(window))
        output = self.relu(self.linear2(output))
        return self.relu(self.linear3(output))


class Decoder(nn.Module):
    def __init__(self, latent_size, window_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(window_size / 4))
        self.linear2 = nn.Linear(int(window_size / 4), int(window_size / 2))
        self.linear3 = nn.Linear(int(window_size / 2), window_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent):
        output = self.relu(self.linear1(latent))
        output = self.relu(self.linear2(output))
        return self.sigmoid(self.linear3(output))


class UsadModel(nn.Module):
    def __init__(self, window_size, latent_size):
        super().__init__()
        self.encoder = Encoder(window_size, latent_size)
        self.decoder1 = Decoder(latent_size, window_size)
        self.decoder2 = Decoder(latent_size, window_size)

    def forward(self, batch):
        encoded = self.encoder(batch)
        reconstruction1 = self.decoder1(encoded)
        reconstruction2 = self.decoder2(encoded)
        reconstruction3 = self.decoder2(self.encoder(reconstruction1))
        return reconstruction1, reconstruction2, reconstruction3

    def training_step(self, batch, epoch):
        reconstruction1, reconstruction2, reconstruction3 = self(batch)
        loss1 = (
            torch.mean((batch - reconstruction1) ** 2) / epoch
            + (1 - 1 / epoch) * torch.mean((batch - reconstruction3) ** 2)
        )
        loss2 = (
            torch.mean((batch - reconstruction2) ** 2) / epoch
            - (1 - 1 / epoch) * torch.mean((batch - reconstruction3) ** 2)
        )
        return loss1, loss2
