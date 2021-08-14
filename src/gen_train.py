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


class GenTrainer:

    def __init__(self, generator: torch.nn.Module, l_rate: float, l_rate_decay: bool, weight_decay: float, epochs: int,
                 log_path: str, device: torch.device = torch.device("cuda:0")):
        """
        :param generator: generator network to train
        :param l_rate: learning rate
        :param l_rate_decay: whether to reduce learning rate during training or not
        :param weight_decay: weight decay during learning
        :param epochs: number of epochs
        :param log_path: path to output directory
        :param device: torch.device
        """
        # generator
        self.generator = generator.to(device)
        self.g_optimizer = optim.Adam(params=self.generator.parameters(), lr=l_rate, weight_decay=weight_decay)
        self.g_loss_fun = nn.L1Loss()

        self.epochs = epochs
        self.path = log_path
        self.device = device

        self.l_rate = l_rate
        self.l_rate_decay = l_rate_decay
        self.weight_decay = weight_decay

    def train_model(self, train_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                    l_rate_decay_at: int, plot_at: int, eval_at: int) -> None:
        """
        Performs training of the model. Saves the best model during training, plots of the input and predictions
        (plot_at), saves states during training to a log file and plots the training stats in the end.

        :param train_loader: data loader containing train data
        :param validation_loader: data loader containing validation data
        :param l_rate_decay_at: epoch where the learning rate should get reduced (epoch % l_rate_decay_at == 0)
        :param plot_at: plot interval (batch % plot_at == 0)
        :param eval_at: validation interval (epoch % val_at == 0)
        """
        summary_path = os.path.join(self.path, "results")
        os.makedirs(summary_path, exist_ok=True)
        writer = SummaryWriter(log_dir=summary_path)

        eval_loss = top_loss = self.evaluate_model(data_loader=validation_loader)

        print(f"\n\033[33m#### Starting Training ####\033[0m\n")

        for epoch in range(1, self.epochs+1):
            # reduce learning rate after every l_rate_decay_at epoch
            if epoch % l_rate_decay_at == 0 and self.l_rate_decay:
                self.l_rate *= 0.5
                self.g_optimizer = optim.Adam(params=self.generator.parameters(), lr=self.l_rate,
                                              weight_decay=self.weight_decay)

            for batch, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                         ncols=90, desc=f"Epoch {epoch}/{self.epochs}"):
                # prepare data
                inputs, targets, ids = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # train generator
                self.g_optimizer.zero_grad()
                outputs = self.generator(inputs)
                predictions = outputs[:, 0][inputs[:, 1] == 0]
                target_predictions = targets[:, 0][inputs[:, 1] == 0]
                loss = self.g_loss_fun(predictions, target_predictions)
                # loss = self.g_loss_fun(outputs, targets)
                loss.backward()
                self.g_optimizer.step()

            if epoch % eval_at == 0:
                eval_loss = self.evaluate_model(data_loader=validation_loader)

                # write losses to tensorboard
                writer.add_scalar(tag="train loss", scalar_value=loss.cpu().item(), global_step=epoch)
                writer.add_scalar(tag="validation loss", scalar_value=eval_loss.cpu().item(), global_step=epoch)

                # write stats of the model to tensorboard
                for i, param in enumerate(self.generator.parameters()):
                    writer.add_histogram(tag=f"trainable parameter {i}", values=param.cpu(), global_step=epoch)
                    writer.add_histogram(tag=f"gradient of trainable parameter {i}", values=param.grad.cpu(),
                                         global_step=epoch)

                print(f"\033[33mEvaluation loss after {epoch}. epoch: {eval_loss.item()}\033[0m\n")

            if epoch % plot_at == 0:
                print(f"\033[33m\n#### Plotting Predictions ####\033[0m")
                plot(inputs=inputs, predictions=outputs, targets=targets, log_path=self.path, update=epoch)
                print(f"\033[33m#### Finished Plotting Predictions ####\033[0m\n")

            if eval_loss.item() < top_loss.item():
                self.generator.save(path=self.path)
                top_loss = eval_loss

        writer.close()
        print(f"\033[33m#### Finished Training ####\033[0m\n")
        print(f"\033[33mLoss of best_model.pt: {top_loss.item()}\033[0m\n")

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
