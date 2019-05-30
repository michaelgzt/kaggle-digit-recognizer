import torch
import pandas as pd
import numpy as np
import math
import time
from random import shuffle, randint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms


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
        self.digit_data = DigitDataset.gen_dataset(csv_file, root_dir, argument, train)
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
        sample = self.digit_data[item].copy()
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample

    @staticmethod
    def gen_dataset(csv_file, root_dir, argument, train):
        """


        :param csv_file: file name for data
        :param root_dir: directory to the csv file
        :param argument: if true use data argument
        :param train: train dataset or test dataset
        :return: all data
        """
        data_df = pd.read_csv(root_dir + csv_file)
        all_data = []
        for num in range(len(data_df)):
            if train:
                digit = data_df.iloc[num, 1:].values
                digit = digit.astype('float').reshape((28, 28))
                label = data_df.iloc[num, 0]
            else:
                digit = data_df.iloc[num, :].values
                digit = digit.astype('float').reshape((28, 28))
                label = 0
            sample = [digit, label]
            all_data.append(sample)
            if argument:
                rand_theta = randint(-10, 10)
                rand_x = randint(-3, 3)
                rand_y = randint(-3, 3)
                new_digit = affine_transform_rotation(digit, rand_theta)
                new_digit = affine_transform_translation(new_digit, rand_x, rand_y)
                new_sample = [new_digit, label]
                all_data.append(new_sample)
        return all_data


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


def train_validate_split(dataset, test_ratio=0.1):
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


def affine_transform_translation(image, x, y):
    """
    Perform translation on digit image

    :param image: numpy image
    :param x: translation on x
    :param y: translation on y
    :return: new image
    """
    digit_location = np.argwhere(image > 0)
    transform_location = np.int_(np.ones((digit_location.shape[0], 3)))
    transform_location[:, :2] = digit_location
    translation_matrix = np.array([[1, 0, x],
                                   [0, 1, y],
                                   [0, 0, 1]])
    transform_location = np.int_(np.dot(translation_matrix, transform_location.T)).T
    """
    Check if the digit is out of image
    """
    max_x = np.amax(transform_location[:, 0])
    min_x = np.amin(transform_location[:, 0])
    max_y = np.amax(transform_location[:, 1])
    min_y = np.amin(transform_location[:, 1])
    if max_x > 27 or min_x < 0 or max_y > 27 or min_y < 0:
        back_x = 0
        back_y = 0
        if max_x > 27:
            back_x = 27 - max_x
        elif min_x < 0:
            back_x = -min_x
        if max_y > 27:
            back_y = 27 - max_y
        elif min_y < 0:
            back_y = -min_y
        back_translation_matrix = np.array([[1, 0, back_x],
                                            [0, 1, back_y],
                                            [0, 0, 1]])
        transform_location = np.int_(np.dot(back_translation_matrix, transform_location.T)).T
    """
    Generate new image
    """
    new_image = np.zeros((28, 28))
    new_image[transform_location[:, 0], transform_location[:, 1]] = image[digit_location[:, 0], digit_location[:, 1]]
    return new_image


def affine_transform_rotation(image, theta):
    """
    Preform rotation on digit image

    :param image: numpy image
    :param theta: rotation
    :return: new image
    """
    digit_location = np.argwhere(image > 0)
    """
    move digit to [0, 0]
    """
    digit_mass_center = digit_location.mean(axis=0)
    transform_location = np.int_(np.ones((digit_location.shape[0], 3)))
    transform_location[:, :2] = digit_location
    translation_matrix = np.array([[1, 0, -digit_mass_center[0]],
                                   [0, 1, -digit_mass_center[1]],
                                   [0, 0, 1]])
    transform_location = np.int_(np.round(np.dot(translation_matrix, transform_location.T))).T
    """
    rotate image
    """
    theta = (theta / 360) * (2 * math.pi)
    rotation_matrix = np.array([[math.cos(theta), math.sin(theta), 0],
                                [-math.sin(theta), math.cos(theta), 0],
                                [0, 0, 1]])
    transform_location = np.int_(np.round(np.dot(rotation_matrix, transform_location.T))).T
    """
    move back digit
    """
    back_translation_matrix = np.array([[1, 0, digit_mass_center[0]],
                                        [0, 1, digit_mass_center[1]],
                                        [0, 0, 1]])
    transform_location = np.int_(np.round(np.dot(back_translation_matrix, transform_location.T))).T
    """
    Check if the digit is out of image
    """
    max_x = np.amax(transform_location[:, 0])
    min_x = np.amin(transform_location[:, 0])
    max_y = np.amax(transform_location[:, 1])
    min_y = np.amin(transform_location[:, 1])
    if max_x > 27 or min_x < 0 or max_y > 27 or min_y < 0:
        back_x = 0
        back_y = 0
        if max_x > 27:
            back_x = 27 - max_x
        elif min_x < 0:
            back_x = -min_x
        if max_y > 27:
            back_y = 27 - max_y
        elif min_y < 0:
            back_y = -min_y
        back_translation_matrix = np.array([[1, 0, back_x],
                                            [0, 1, back_y],
                                            [0, 0, 1]])
        transform_location = np.int_(np.dot(back_translation_matrix, transform_location.T)).T
    """
    Generate new image
    """
    new_image = np.zeros((28, 28))
    new_image[transform_location[:, 0], transform_location[:, 1]] = image[digit_location[:, 0], digit_location[:, 1]]
    """
    Fill the empty pixels
    """
    max_x = np.amax(transform_location[:, 0])
    min_x = np.amin(transform_location[:, 0])
    max_y = np.amax(transform_location[:, 1])
    min_y = np.amin(transform_location[:, 1])
    for tmp_x in range(min_x, max_x):
        for tmp_y in range(min_y, max_y):
            pixel = new_image[tmp_x, tmp_y]
            if pixel == 0:
                up_pixel = new_image[tmp_x-1, tmp_y]
                down_pixel = new_image[tmp_x+1, tmp_y]
                left_pixel = new_image[tmp_x, tmp_y-1]
                right_pixel = new_image[tmp_x, tmp_y+1]
                if up_pixel > 0 and down_pixel > 0 and right_pixel > 0 and left_pixel > 0:
                    new_image[tmp_x, tmp_y] = (up_pixel + down_pixel + right_pixel + left_pixel) / 4
                elif up_pixel > 0 and down_pixel > 0:
                    new_image[tmp_x, tmp_y] = (up_pixel + down_pixel) / 2
                elif left_pixel > 0 and right_pixel > 0:
                    new_image[tmp_x, tmp_y] = (left_pixel + right_pixel) / 2
    return new_image


if __name__ == '__main__':
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    time1 = time.time()
    data = DigitDataset('train.csv', './data/', train=True, argument=False)
    print(len(data))
    class_num = digits_per_class(data, [num for num in range(len(data))])
    print(class_num)
    train_idx, val_idx = train_validate_split(data)
    print(len(train_idx), len(val_idx), len(train_idx) + len(val_idx))
    train_class_num = digits_per_class(data, train_idx)
    print(train_class_num)
    val_class_num = digits_per_class(data, val_idx)
    print(val_class_num)
