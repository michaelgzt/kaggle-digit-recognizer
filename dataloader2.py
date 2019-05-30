import torch
import pandas as pd
import numpy as np
import math
import time
from random import shuffle, randint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms
from PIL import Image


"""
Dataset for Digit Data
and Transformation functions
"""


class DigitDataset(Dataset):
    """ Digit Dataset """

    def __init__(self, csv_file, root_dir, train=False, argument=False, transform=None):
        """

        :param csv_file: file name for data
        :param root_dir: directory to the csv file
        :param train: train dataset or test dataset
        :param transform: Optional transform functions
        """
        self.digit_df = pd.read_csv(root_dir + csv_file)
        self.transform = transform
        self.train = train
        self.argument = argument

    def __len__(self):
        """
        Return length of the dataset

        :return: length of dataset
        """
        if self.argument:
            return 2 * len(self.digit_df)
        else:
            return len(self.digit_df)

    def __getitem__(self, item):
        """
        Iterator of the dataset

        :param item: index of data
        :return: one item
        """
        if item < len(self.digit_df):
            if self.train:
                digit = self.digit_df.iloc[item, 1:].values
                digit = digit.astype('float').reshape((28, 28))
                label = self.digit_df.iloc[item, 0]
            else:
                digit = self.digit_df.iloc[item, :].values
                digit = digit.astype('float').reshape((28, 28))
                label = 0
        else:
            assert self.argument and self.train
            digit = self.digit_df.iloc[item % len(self.digit_df), 1:].values
            digit = digit.astype('float').reshape((28, 28))
            rand_theta = (randint(-20, 20) / 180) * math.pi
            rand_x = randint(-2, 2)
            rand_y = randint(-2, 2)
            rand_scale = randint(9, 11) * 0.1
            digit = digit_argument(digit, rand_theta, [rand_x, rand_y], rand_scale)
            label = self.digit_df.iloc[item % len(self.digit_df), 0]
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


def digit_argument(digit, angle, translate, scale):
    """
    Data Argumentation on Digit Image using Pillow

    :param digit: numpy digit image
    :param angle: rotation
    :param translate: translation
    :param scale: scaling
    :return: argument numpy digit image
    """
    digit_img = Image.fromarray(digit)
    t1 = np.array([[1, 0, 14],
                   [0, 1, 14],
                   [0, 0, 1]])
    t2 = np.array([[math.cos(angle), math.sin(angle), 0],
                   [-math.sin(angle), math.cos(angle), 0],
                   [0, 0, 1]])
    t3 = np.array([[scale, 0, 0],
                   [0, scale, 0],
                   [0, 0, 1]])
    t4 = np.array([[1, 0, -14],
                   [0, 1, -14],
                   [0, 0, 1]])
    t5 = np.array([[1, 0, translate[0]],
                   [0, 1, translate[1]],
                   [0, 0, 1]])
    t_inv = np.linalg.inv(t1 @ t2 @ t3 @ t4 @ t5)
    digit_img = digit_img.transform((28, 28),
                                    Image.AFFINE,
                                    data=t_inv.flatten()[:6],
                                    resample=Image.BILINEAR)
    digit_arg = np.asarray(digit_img)
    return digit_arg


def digits_per_class(digit_df, indices):
    """
    Compute number of digits per class base on indices

    :param digit_df: dataframe of all digits
    :param indices: list of indices
    :return: number of digits for all classes
    """
    assert isinstance(digit_df, pd.DataFrame)
    assert isinstance(indices, list)
    digit_num = [0 for num in range(10)]
    for idx in indices:
        label = digit_df.iloc[idx, 0]
        digit_num[label] += 1
    return digit_num


def train_validate_split(digit_df, test_ratio=0.2, argument=False):
    """
    Divide dataset into training and validation randomly
    and with balanced number of digits in each class

    :param dataset: dataset of all digits
    :param test_ratio: ratio of validation data
    :param argument: if or not use argument data
    :return: lists of indices of train and validate dataset
    """
    assert isinstance(digit_df, pd.DataFrame)
    digit_num = len(digit_df)
    overall_indices = [num for num in range(digit_num)]
    overall_class_num = digits_per_class(digit_df, overall_indices)
    test_class_num = [int(num*test_ratio) for num in overall_class_num]
    tmp_test_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
    train_indices = []
    val_indices = []
    for idx in overall_indices:
        tmp_label = digit_df.iloc[idx, 0]
        if tmp_test_class_num[tmp_label] < test_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_test_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
            if argument:
                train_indices.append(idx + digit_num)
    return train_indices, val_indices


if __name__ == '__main__':
    data = DigitDataset('train.csv', './data/', train=True, argument=True)
    print(len(data))
    digit = data[42000][0]
    plt.imshow(digit, cmap='gray')
    plt.show()
