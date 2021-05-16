"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""

import os
import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(Encoder(), Bottleneck(), Decoder())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forwards pass of the model.

        :param x: input tensor of the model
        :return: output tensor of the model
        """
        return self.model(x)

    def save(self, path: str) -> None:
        """
        Save the current model to provided path.

        :param path: path to the model directory
        """
        model_path = os.path.join(path, "model")
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, "best_model.pt"))

    def load(self, path: str) -> None:
        """
        Load a saved model from provided directory.

        :param path: path to the model
        """
        print("\033[33m#### Loading Model ####\033[0m")

        try:
            self.model.load_state_dict(torch.load(os.path.join(path, "model", "best_model.pt")))
            print("\033[33mModel successfully loaded\033[0m")

        except FileNotFoundError:
            print("\033[91mNo file 'best_model.pt' found\033[0m")
            print("\033[33mContinue with initial model parameters\033[0m")


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 2. conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 3. conv block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 4. conv block
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the encoder.

        :param x: input of the encoder
        :return: output of the encoder
        """
        return self.encoder(x)


class Bottleneck(nn.Module):

    def __init__(self):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Bottleneck

        :param x: input of the bottleneck
        :return: output of the bottleneck
        """
        return self.bottleneck(x)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            # 1. conv block
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 2. conv block
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 3. conv block
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 4. conv block
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the decoder.

        :param x: input of the decoder
        :return: output of the decoder
        """
        return self.decoder(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = torch.nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 2. conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 3. conv block
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 4. conv block
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0),
            # 5. conv block
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the discriminator.

        :param x: input of the discriminator
        :return: output of the discriminator
        """
        return self.discriminator(x)

    def save(self, path: str) -> None:
        """
        Save the current model to provided path.

        :param path: path to the model directory
        """
        model_path = os.path.join(path, "model")
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.discriminator.state_dict(), os.path.join(model_path, "discriminator.pt"))

    def load(self, path: str) -> None:
        """
        Load a saved model from provided directory.

        :param path: path to the model
        """
        print("\033[33m#### Loading Discriminator ####\033[0m")

        try:
            self.discriminator.load_state_dict(torch.load(os.path.join(path, "model", "discriminator.pt")))
            print("\033[33mDiscriminator successfully loaded\033[0m")

        except FileNotFoundError:
            print("\033[91mNo file 'discriminator.pt' found\033[0m")
            print("\033[33mContinue with initial discriminator parameters\033[0m")
