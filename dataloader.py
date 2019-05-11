import torch
import pandas as pd
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms


"""
Dataset for Digit Data
and Transformation functions
"""


class DigitDataset(Dataset):
    """ Digit Dataset """

    def __init__(self, csv_file, root_dir, train=False, transform=None):
        """

        :param csv_file: file name for date
        :param root_dir: directory to the csv file
        :param train: train dataset or test dataset
        :param transform: Optional transform functions
        """
        self.digit_data = pd.read_csv(root_dir + csv_file)
        self.train = train
        self.transform = transform

    def __len__(self):
        """
        Return length of the dataset

        :return: length of dataset
        """
        return len(self.digit_data)

    def __getitem__(self, item):
        """
        Iterator of the dataset

        :param item: index of data
        :return: one item
        """
        if self.train:
            digit = self.digit_data.iloc[item, 1:].values
            digit = digit.astype('float').reshape((28, 28))
            label = self.digit_data.iloc[item, 0]
        else:
            digit = self.digit_data.iloc[item, :].values
            digit = digit.astype('float').reshape((28, 28))
            label = 0
        sample = [digit, label]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


class Regularize(object):
    """ Regularize digit pixel value """

    def __init__(self, max_pixel=255):
        """

        :param max_pixel: max pixel value
        """
        self.max_pixel = max_pixel

    def __call__(self, digit):
        """

        :param digit: digit image
        :return:
        """
        assert isinstance(digit, np.ndarray)
        digit = digit / self.max_pixel
        return digit


class ToTensor(object):
    """ Covert ndarrays to Tensors """

    def __call__(self, digit):
        """

        :param digit: digit image
        :return:
        """
        assert isinstance(digit, np.ndarray)
        digit = digit.reshape((1, 28, 28))
        digit = torch.from_numpy(digit)
        digit = digit.float()
        return digit


def digits_per_class(dataset, indices):
    """
    Compute number of digits per class base on indices

    :param dataset: dataset of all digits
    :param indices: list of indices
    :return: number of digits for all classes
    """
    assert isinstance(dataset, DigitDataset)
    assert isinstance(indices, list)
    digit_num = [0 for num in range(10)]
    for idx in indices:
        digit_num[dataset[idx][1]] += 1
    return digit_num


def train_validate_split(dataset, test_ratio=0.2):
    """
    Divide dataset into training and validation randomly
    and with balanced number of digits in each class

    :param dataset: dataset of all digits
    :param test_ratio: ratio of validation data
    :return: lists of indices of train and validate dataset
    """
    assert isinstance(dataset, DigitDataset)
    overall_indices = [num for num in range(len(dataset))]
    overall_class_num = digits_per_class(dataset, overall_indices)
    test_class_num = [int(num*test_ratio) for num in overall_class_num]
    tmp_test_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
    train_indices = []
    val_indices = []
    for idx in overall_indices:
        tmp_label = dataset[idx][1]
        if tmp_test_class_num[tmp_label] < test_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_test_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
    return train_indices, val_indices


if __name__ == '__main__':
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    data = DigitDataset('train.csv', './data/', train=True, transform=composed_transform)
    print(len(data))
    dataloader = DataLoader(data, batch_size=5, shuffle=False, num_workers=4)
    for i, d in enumerate(dataloader, 0):
        if i == 0:
            digit, label = d
            print(label)
            break
    class_num = digits_per_class(data, [num for num in range(len(data))])
    print(class_num)
    train_idx, val_idx = train_validate_split(data)
    print(len(train_idx), len(val_idx), len(train_idx) + len(val_idx))
    train_class_num = digits_per_class(data, train_idx)
    print(train_class_num)
    val_class_num = digits_per_class(data, val_idx)
    print(val_class_num)
    print(val_idx[:100])
