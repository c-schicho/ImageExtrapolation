"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""


import os
import tqdm
import torch
import dill as pkl
import numpy as np
from torchvision import transforms
from dataloader import DataPipeline, SubmissionDataLoader
from model import Generator, Discriminator
from gen_train import GenTrainer
from gan_train import GanTrainer
from utils import *


def main(challenge: bool, training: bool, gan_train: bool, testing: bool, load_model: bool, img_path: str,
         log_path: str, sub_path: str, img_shape: int, batch_size: int, epochs: int, l_rate: float, l_rate_decay: bool,
         l_rate_decay_at: int, weight_decay: float, plot_at: int, eval_at: int) -> None:
    """
    Function to execute the image extrapolation.

    :param challenge: whether to load and predict values for submission or not
    :param training: whether the model should be trained or not
    :param gan_train: whether to train the model against a discriminator or not
    :param testing: whether the model should be tested or not
    :param load_model: whether a pretrained model should be loaded or not
    :param img_path: path to the image folder
    :param log_path: path to the output folder
    :param sub_path: path to the submission image data
    :param img_shape: shape of the resized train images
    :param batch_size: mini-batch size
    :param epochs: number of train epochs
    :param l_rate: learning rate
    :param l_rate_decay: whether to reduce learning rate during training or not
    :param l_rate_decay_at: epoch when the learning rate should get reduced
    :param weight_decay: learning rate decay
    :param plot_at: epoch when the model should save plots to the log_path (epoch % plot_at == 0)
    :param eval_at: epoch when the model should be evaluated (epoch % eval_at == 0)
    """
    generator = Generator()
    discriminator = Discriminator()

    if not challenge:  # creates no submission file
        # random image augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(img_shape, img_shape)),
            transforms.RandomHorizontalFlip(p=0.33),
        ])

        data_pipeline = DataPipeline(img_path=img_path, batch_size=batch_size, log_path=log_path, transform=transform)

        if load_model:
            generator.load(path=log_path)
            #discriminator.load(path=log_path)

        if training:
            train_loader = data_pipeline.get_train_loader()
            valid_loader = data_pipeline.get_validation_loader()

            if gan_train:  # train with discriminator (dcgan)
                trainer = GanTrainer(generator=generator, discriminator=discriminator, l_rate=l_rate,
                                     weight_decay=weight_decay, epochs=epochs, log_path=log_path)
            else:  # train with usual loss function
                trainer = GenTrainer(generator=generator, l_rate=l_rate, l_rate_decay=l_rate_decay,
                                     weight_decay=weight_decay, epochs=epochs, log_path=log_path)

            trainer.train_model(train_loader=train_loader, validation_loader=valid_loader,
                                l_rate_decay_at=l_rate_decay_at, plot_at=plot_at, eval_at=eval_at)

        if testing:  # test current model on test set
            test_loader = data_pipeline.get_test_loader()
            try:
                test_loss = trainer.evaluate_model(data_loader=test_loader)
            except NameError:  # training did not happen before (during the runtime) / no trainer initialized
                trainer = GenTrainer(generator=generator, l_rate=l_rate, l_rate_decay=l_rate_decay,
                                     weight_decay=weight_decay, epochs=epochs, log_path=log_path)
                test_loss = trainer.evaluate_model(data_loader=test_loader)

            print(f"Loss based on the test data: {test_loss.item()}")

    else:  # challenge submission
        print(f"\033[33m#### Creating submission file ####\033[0m\n")

        output_path = os.path.join(log_path, "submission")
        os.makedirs(output_path, exist_ok=True)

        device = "cuda:0"
        generator.load(path=log_path)
        generator = generator.to(device)

        sub_data = SubmissionDataLoader(sub_path=sub_path, img_path=img_path)
        sub_loader = sub_data.get_submission_loader()

        # mean and std for de-normalization of the network output
        mean = sub_data.get_mean()
        std = sub_data.get_std()

        # predict target values
        predictions = []
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(sub_loader), total=len(sub_loader), ncols=90,
                                     desc="Processing Submission data"):
                inputs, ids = data
                inputs = inputs.to(device)

                outputs = generator(inputs)
                outputs_s = torch.squeeze(outputs)
                prediction = outputs_s[inputs.cpu().detach().numpy()[0][1] == 0]  # get only border pixels
                predictions.append(np.asarray(((prediction.cpu() * std) + mean), dtype=np.uint8))  # de-normalize

                # plot predictions
                plot_submission(inputs=inputs, predictions=outputs, log_path=output_path, update=i)

        # save submission file containing predicted target values
        with open(os.path.join(output_path, "submission.pkl"), 'wb') as f:
            pkl.dump(predictions, f)

        print(f"\033[33m#### Finished creating submission file ####\033[0m")


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file", type=str)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    main(**config)
