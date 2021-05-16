"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""


import os.path
import numpy as np
import tqdm
import torch
import dill as pkl
from torchvision import transforms
from dataloader import DataPipeline, SubmissionDataLoader
from model import Generator, Discriminator
from train import Trainer as Trainer
from gan_train import Trainer as GanTrainer


def main(challenge: bool, training: bool, gan_train: bool, testing: bool, load_model: bool, img_path: str,
         log_path: str, sub_path: str, img_shape: int, batch_size: int, epochs: int, l_rate: float, weight_decay: float,
         plot_at: int, eval_at: int) -> None:
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
    :param weight_decay: learning rate decay
    :param plot_at: when the model should save plots to the log_path (epoch % plot_at == 0)
    :param eval_at: when the model should be evaluated (epoch % eval_at == 0)
    """
    generator = Generator()
    discriminator = Discriminator()

    if not challenge:
        # random image augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(img_shape, img_shape)),
            transforms.RandomHorizontalFlip(p=0.33),
        ])

        data_pipeline = DataPipeline(img_path=img_path, batch_size=batch_size, log_path=log_path, transform=transform)

        if load_model:
            generator.load(path=log_path)
            discriminator.load(path=log_path)

        if training:
            train_loader = data_pipeline.get_train_loader()
            valid_loader = data_pipeline.get_validation_loader()

            if gan_train:
                trainer = GanTrainer(generator=generator, discriminator=discriminator, l_rate=l_rate,
                                     weight_decay=weight_decay, epochs=epochs, log_path=log_path)
            else:
                trainer = Trainer(generator=generator, l_rate=l_rate, weight_decay=weight_decay, epochs=epochs,
                                  log_path=log_path)

            trainer.train_model(train_loader=train_loader, validation_loader=valid_loader, plot_at=plot_at,
                                eval_at=eval_at)

        if testing:
            test_loader = data_pipeline.get_test_loader()
            try:
                test_loss = trainer.evaluate_model(data_loader=test_loader)
            except NameError:  # training did not happen before (during the runtime)
                trainer = Trainer(generator=generator, l_rate=l_rate, weight_decay=weight_decay, epochs=epochs,
                                  log_path=log_path)
                test_loss = trainer.evaluate_model(data_loader=test_loader)

            print(f"Loss based on the test data: {test_loss.item()}")

    # challenge submission
    else:
        print(f"\033[33m#### Creating submission file ####\033[0m\n")

        device = "cuda:0"
        generator.load(path=log_path)
        generator = generator.to(device)

        sub_data = SubmissionDataLoader(sub_path=sub_path, img_path=img_path)
        sub_loader = sub_data.get_submission_loader()

        mean = sub_data.get_mean()
        std = sub_data.get_std()

        # predict target values
        predictions = []
        with torch.no_grad():
            for data in tqdm.tqdm(sub_loader, desc="Processing Submission data", ncols=90):
                inputs, ids = data
                inputs = inputs.to(device)

                outputs = generator(inputs)
                outputs = torch.squeeze(outputs)
                prediction = outputs[inputs.cpu().detach().numpy()[0][1] == 0]
                predictions.append(np.asarray(((prediction.cpu() * std) + mean), dtype=np.uint8))

        # save submission file containing predicted target values
        output_path = os.path.join(log_path, "submission")
        os.makedirs(output_path, exist_ok=True)
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
