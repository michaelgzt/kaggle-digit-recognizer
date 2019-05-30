import torch
import torch.nn as nn


"""
Networks for digit recognition
"""


class BasicLeNet(nn.Module):
    """ Basic LeNet-5 as defined in LeCun's paper"""

    def __init__(self):
        super(BasicLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        """
        Feedforward function

        :param x: input batch
        :return: x
        """
        x = self.features(x)
        x = x.view(x.size(0), 128*4*4)
        x = self.classifier(x)
        return x


class EnhancedLeNet(nn.Module):

    def __init__(self):
        super(EnhancedLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Feedforward function

        :param x: input batch
        :return: x
        """
        x = self.features(x)
        x = x.view(x.size(0), 128*7*7)
        x = self.classifier(x)
        return x
