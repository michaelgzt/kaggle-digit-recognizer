import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader2 import *
from network import *


"""
Training Digit Recognition network
"""


def training(network, criterion, optimizer, epoch_num, test=True):
    """
    Training network

    :param network: Network using
    :param criterion: loss function
    :param optimizer: optimizer function
    :param epoch_num: training epoch
    :param test: if true include validation
    :return: trained network
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Start Training with", device, epoch_num, "overall epoch")
    network.to(device)
    """
    Create Dataloader for training and validating
    """
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    digit_dataset = DigitDataset('train.csv', './data/', train=True, transform=composed_transform, argument=True)
    if test:
        train_indices, val_indices = train_validate_split(digit_dataset.digit_df, argument=True)
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        val_sampler = sampler.SubsetRandomSampler(val_indices)
        train_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        print("Training with validation, ", "Overall Data:", len(train_indices)+len(val_indices))
        print("Training Data:", len(train_indices), "Validate Data:", len(val_indices))
    else:
        train_dataloader = DataLoader(
            digit_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_dataloader = None
        print("Training all data, ", "Overall Data:", len(digit_dataset))
    """
    Start Training
    """
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            digits, labels = data
            digits, labels = digits.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(digits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500.))
                running_loss = 0.0
        if test:
            train_accuracy = validating(network, train_dataloader)
            val_accuracy = validating(network, val_dataloader)
            print('Training accuracy epoch %d: %.3f' % (epoch, train_accuracy))
            print('Validation accuracy epoch %d: %.3f' % (epoch, val_accuracy))
    return network


def validating(network, loader):
    """
    Validating the network during and after training

    :param network: trained network
    :param loader: dataloader
    :return: accuracy
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct_num = 0
    total_num = 0
    for i, data in enumerate(loader, 0):
        digits, labels = data
        total_num += labels.size(0)
        digits, labels = digits.to(device), labels.to(device)
        outputs = network(digits)
        _, predicted = torch.max(outputs, 1)
        correct_num += ((predicted == labels).sum().to("cpu")).item()
    accuracy = correct_num / total_num
    return accuracy


def testing(network):
    """
    Generate test result for Kaggle

    :param network: trained network
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    composed_transform = transforms.Compose([Regularize(), ToTensor()])
    digit_dataset = DigitDataset('test.csv', './data/', train=False, transform=composed_transform)
    test_dataloader = DataLoader(
        digit_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_results = []
    for i, data in enumerate(test_dataloader, 0):
        digits, label = data
        digits = digits.to(device)
        outputs = network(digits)
        _, predicted = torch.max(outputs, 1)
        test_results += np.int_(predicted.to("cpu").numpy().squeeze()).tolist()
    """
    Write test results to a csv file
    """
    test_df = pd.read_csv("./data/sample_submission.csv")
    assert (len(test_df) == len(test_results))
    test_df.loc[:, 'Label'] = test_results
    test_df.to_csv('./data/test_results.csv', index=False)
    print("Test Results for Kaggle Generated ...")


if __name__ == '__main__':
    # lenet = BasicLeNet()
    lenet = EnhancedLeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet.parameters())
    lenet = training(lenet, criterion, optimizer, 50, test=False)
    testing(lenet)

