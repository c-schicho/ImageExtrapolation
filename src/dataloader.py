"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""

import os
import glob
import random
import torch
import gzip
import dill as pkl
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime
from typing import Tuple


class DataPipeline:

    def __init__(self, img_path: str, batch_size: int, log_path: str, shuffle: bool = False,
                 transform: torchvision.transforms = None, seed: bool = True):
        """
        :param img_path: path to the image files
        :param batch_size: size ob batches
        :param shuffle: whether to shuffle the image data or not
        :param transform: torchvision.transform object
        :param seed: whether to set a random seed (reproducibility)
        """
        if seed:
            torch.manual_seed(42)
            np.random.seed(42)

        self.batch_size = batch_size
        self.shuffle = shuffle

        __data_set = ImageDataset(img_path=img_path, transform=transform, seed=seed)

        # create indices for dataset splitting
        __n_samples = len(__data_set)
        __shuffled_idx = np.random.permutation(__n_samples)
        __train_idx = __shuffled_idx[int(__n_samples/5) * 2 : ]
        __test_idx = __shuffled_idx[:int(__n_samples/5)]
        __val_idx = __shuffled_idx[int(__n_samples/5) : int(__n_samples/5)*2]

        # save chosen indices to a file (for reproduction)
        path = os.path.join(log_path, "dataloader")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "dataset_indices.txt"), "a") as f:
            f.write(f"""{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            train indices: {__test_idx}
            test indices: {__test_idx}
            validation indices: {__val_idx}\n""")

        # create subsets of the whole dataset
        self.train_data = Subset(__data_set, indices=__train_idx)
        self.test_data = Subset(__data_set, indices=__test_idx)
        self.val_data = Subset(__data_set, indices=__val_idx)

    def get_train_loader(self) -> torch.utils.data.DataLoader:
        """
        :return: pytorch data loader containing the preprocessed train images
        """
        return DataLoader(self.train_data, shuffle=self.shuffle, batch_size=self.batch_size,
                          num_workers=4, collate_fn=self._collate_fn)

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        """
        :return: pytorch data loader containing the preprocessed test images
        """
        return DataLoader(self.test_data, shuffle=self.shuffle, batch_size=self.batch_size,
                          num_workers=4, collate_fn=self._collate_fn)

    def get_validation_loader(self) -> torch.utils.data.DataLoader:
        """
        :return: pytorch data loader containing the preprocessed validation images
        """
        return DataLoader(self.val_data, shuffle=self.shuffle, batch_size=self.batch_size,
                          num_workers=4, collate_fn=self._collate_fn)

    def _collate_fn(self, batch_as_list: list) -> Tuple[torch.Tensor, ...]:
        """
        Function for custom stacking to be passed to the torch.utils.data.DataLoader

        :param batch_as_list: mini batch as list
        :return: processed mini batch as list
        """
        inputs = [sample[0] for sample in batch_as_list]
        stacked_inputs = torch.stack([input_img for input_img in inputs], dim=0)

        targets = [sample[1] for sample in batch_as_list]
        stacked_targets = torch.stack([torch.tensor([target], dtype=torch.float32) for target in targets], dim=0)

        ids = [sample[2] for sample in batch_as_list]
        stacked_ids = torch.stack([torch.tensor(idx, dtype=torch.float32) for idx in ids], dim=0)

        return stacked_inputs, stacked_targets, stacked_ids


class ImageDataset(Dataset):

    def __init__(self, img_path: str, transform: torchvision.transforms = None, seed: bool = True):
        """
        :param img_path: path to the image files
        :param transform: torchvision.transform object
        :param seed: whether to set a random seed (reproducibility)
        """
        super(ImageDataset, self).__init__()

        if seed:
            torch.manual_seed(42)
            np.random.seed(42)

        # load computed means and standard deviations for image normalizing
        with gzip.open(os.path.join(img_path, "analysis", "04_means_stds.pklz"), 'rb') as f:
            data = pkl.load(f)

        self.norm_mean = np.mean(data["means"])
        self.norm_std = np.mean(data["stds"])

        self.img_paths = sorted(glob.glob(os.path.join(img_path, "**", "*.jpg"), recursive=True))
        self.transform = transform

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """
        :param idx: index of the desired image data
        :return: image data (augmented if augmentation is set to true)
        """
        # load image and normalize it
        img = Image.open(self.img_paths[idx])

        if self.transform:
            img = self.transform(img)

        img = (np.asarray(img) - self.norm_mean) / self.norm_std
        x = img.shape[0]
        y = img.shape[1]
        # set random borders
        border_x = tuple((random.randint(5, 10), random.randint(5, 10)))
        border_y = tuple((random.randint(5, 10), random.randint(5, 10)))
        # creating known mask (unknown = 0, known = 1)
        known_array = np.ones_like(img)
        height_idx = np.array([[i] for i in range(x) if i < border_x[0] or i >= x - border_x[1]])
        width_idx = np.array([i for i in range(y) if i < border_y[0] or i >= y - border_y[1]])
        np.put_along_axis(known_array, height_idx, 0, axis=0)  # set border values to 0 (rows / x-values)
        for i in range(x):
            np.put(known_array[i], width_idx, 0)  # set border values to 0 (columns / y-values)

        target = img[known_array == 0]  # store values of defined borders

        input_array = img.copy()
        np.putmask(input_array, known_array == 0, 0)  # set values of defined borders to 0

        inputs = torch.Tensor([input_array, known_array])
        targets = img

        return tuple([inputs, targets, idx])


class SubmissionDataLoader:

    def __init__(self, sub_path: str, img_path: str):
        """
        :param sub_path: path to the submission image data
        :param img_path: path to the image data folder
        """
        self.data_set = SubmissionDataset(sub_path=sub_path, img_path=img_path)

    def get_submission_loader(self) -> DataLoader:
        """
        :return: pytorch data loader containing image data for submission
        """
        return DataLoader(self.data_set, shuffle=False, batch_size=1, num_workers=0)

    def get_mean(self) -> np.ndarray:
        """
        :return: mean used for normalizing image data
        """
        return self.data_set.norm_mean

    def get_std(self) -> np.ndarray:
        """
        :return: standard deviation used for normalizing image data
        """
        return self.data_set.norm_std


class SubmissionDataset(Dataset):

    def __init__(self, img_path: str, sub_path: str):
        """
        :param img_path: path to the image data folder
        :param sub_path: path to the submission image data
        """
        super(SubmissionDataset, self).__init__()

        # load computed means and standard deviations for image normalizing
        with gzip.open(os.path.join(img_path, "analysis", "04_means_stds.pklz"), 'rb') as f:
            data = pkl.load(f)

        # load submission test data set
        with open(os.path.join(sub_path), 'rb') as f:
            img_data = pkl.load(f)

        self.norm_mean = np.mean(data["means"])
        self.norm_std = np.mean(data["stds"])
        self.input_arrays = img_data["input_arrays"]
        self.known_arrays = img_data["known_arrays"]
        self.sample_ids = img_data["sample_ids"]

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self.input_arrays)

    def __getitem__(self, idx: int) -> Tuple:
        """
        :param idx: index of the desired image data
        :return: normalized submission image data
        """
        known_array = self.known_arrays[idx]

        # load image and normalize it
        input_array = (np.asarray(self.input_arrays[idx]) - self.norm_mean) / self.norm_std
        np.putmask(input_array, known_array == 0, 0)

        inputs = torch.Tensor([input_array, known_array])

        idx = self.sample_ids[idx]

        return tuple([inputs, idx])
