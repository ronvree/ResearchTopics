


"""


    "The observation model is Gaussian with mean parameterized by a deconvolutional neural network and identity
     covariance"

"""


import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F


class ObservationModel(nn.Module):

    def __init__(self, input_size: int, args: argparse.Namespace):
        super().__init__()
        self.embedding_size = args.observation_model_embedding_size

        self.dense = nn.Linear(input_size, self.embedding_size)
        self.conv1 = nn.ConvTranspose2d(self.embedding_size, 128, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        xs = self.dense(xs)
        xs = xs.view(-1, self.embedding_size, 1, 1)
        xs = self.conv1(xs)
        xs = F.relu(xs)
        xs = self.conv2(xs)
        xs = F.relu(xs)
        xs = self.conv3(xs)
        xs = F.relu(xs)
        xs = self.conv4(xs)

        return xs

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser('Observation model')

        parser.add_argument('--observation_model_embedding_size',
                            type=int,
                            default=200,  # TODO -- did they also use 200?
                            help='')  # TODO

        return parser





