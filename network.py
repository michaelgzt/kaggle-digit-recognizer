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
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        """
        Feedforward function

        :param x: input batch
        :return: x
        """
        x = self.features(x)
        x = x.view(x.size(0), 16*4*4)
        x = self.classifier(x)
        return x
