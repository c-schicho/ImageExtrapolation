"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""

import torch
import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *


class Trainer:

    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module, l_rate: float, weight_decay: float,
                 epochs: int, log_path: str, device: torch.device = torch.device("cuda:0")):
        """
        :param generator: generator network to train
        :param discriminator: discriminator network
        :param l_rate: learning rate
        :param weight_decay: weight decay during learning
        :param epochs: number of epochs
        :param log_path: path to output directory
        :param device: torch.device
        """
        # generator
        self.generator = generator.to(device)
        self.g_optimizer = optim.Adam(params=self.generator.parameters(), lr=l_rate, weight_decay=weight_decay)
        self.g_loss_fun = nn.L1Loss()
        # discriminator
        self.discriminator = discriminator.to(device)
        self.d_optimizer = optim.Adam(params=self.discriminator.parameters(), lr=l_rate, weight_decay=weight_decay)
        self.d_loss_fun = nn.MSELoss()

        self.epochs = epochs
        self.path = log_path
        self.device = device

    def train_model(self, train_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                    plot_at: int, eval_at: int) -> None:
        """
        Performs training of the model. Saves the best model during training, plots of the input and predictions
        (plot_at), saves states during training to a log file and plots the training stats in the end.

        :param train_loader: data loader containing train data
        :param validation_loader: data loader containing validation data
        :param plot_at: plot interval (batch % plot_at == 0)
        :param eval_at: validation interval (epoch % val_at == 0)
        """
        summary_path = os.path.join(self.path, "tensorboard")
        os.makedirs(summary_path, exist_ok=True)
        writer = SummaryWriter(log_dir=summary_path)

        top_loss = torch.tensor(float('+inf'), device=self.device)
        eval_loss = torch.tensor(float('+inf'), device=self.device)

        print(f"\n\033[33m#### Starting Training against Discriminator ####\033[0m\n")

        for epoch in range(1, self.epochs+1):
            for batch, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                         ncols=90, desc=f"Epoch {epoch}/{self.epochs}"):
                # prepare data
                inputs, targets, ids = data
                inputs = inputs.to(self.device)
                targets = targets.type(torch.FloatTensor).to(self.device)

                # predict targets (generator)
                g_outputs = self.generator(inputs)

                # real inputs (discriminator)
                d_outputs_real = self.discriminator(targets)
                d_outputs_real = torch.squeeze(d_outputs_real)
                gt_real = torch.ones_like(d_outputs_real) * 0.9  # ground truth for real inputs
                valid_loss = self.d_loss_fun(d_outputs_real, gt_real)

                # generator inputs (fake) (discriminator)
                d_outputs_fake = self.discriminator(g_outputs.detach())
                gt_fake = torch.zeros_like(d_outputs_fake) * 0.1  # ground truth for fake inputs
                fake_loss = self.d_loss_fun(d_outputs_fake, gt_fake)

                # train discriminator
                self.discriminator.zero_grad()
                d_loss = (valid_loss + fake_loss) * 0.5
                d_loss.backward()
                self.d_optimizer.step()

                # train generator
                self.g_optimizer.zero_grad()
                d_outputs = self.discriminator(g_outputs)
                d_outputs = torch.squeeze(d_outputs)
                g_loss = self.g_loss_fun(d_outputs, gt_real)
                g_loss.backward()
                self.g_optimizer.step()

                # print train loss after each batch
                print(f"; train loss: {torch.mean(g_loss).item()}")
                print(f"; discriminator loss: {torch.mean(d_loss).item()}")

            if epoch % eval_at == 0:
                eval_loss = self.evaluate_model(data_loader=validation_loader)

                # write losses to tensorboard
                writer.add_scalar(tag="train loss", scalar_value=g_loss.cpu().item(), global_step=epoch)
                writer.add_scalar(tag="validation loss", scalar_value=eval_loss.cpu().item(), global_step=epoch)

                print(f"\033[33mEvaluation loss after {epoch}. epoch: {eval_loss.item()}\033[0m\n")

            if epoch % plot_at == 0:
                print(f"\033[33m\n#### Plotting Predictions ####\033[0m")
                plot(inputs=inputs, predictions=g_outputs, targets=targets, log_path=self.path, update=epoch)
                print(f"\033[33m#### Finished Plotting Predictions ####\033[0m\n")

            if eval_loss.item() < top_loss.item():
                self.generator.save(path=self.path)
                self.discriminator.save(path=self.path)
                top_loss = eval_loss

        writer.close()
        self.discriminator.save(path=self.path)
        print(f"\033[33m#### Finished Training ####\033[0m\n")

    def evaluate_model(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Evaluate provided model on provided data loader.

        :param data_loader: data loader to evaluate the model on
        :return: averaged loss on the data loader
        """
        print(f"\033[33m\n#### Evaluating Model ####\033[0m")

        loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data in tqdm.tqdm(data_loader, desc="Processing dataset", ncols=90):
                inputs, targets, ids = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.generator(inputs)

                loss += (torch.stack([self.g_loss_fun(output, target) for
                                      output, target in zip(outputs, targets)]).sum() / len(data_loader))

        print(f"\033[33m#### Finished Evaluating Model ####\033[0m\n")

        return loss
