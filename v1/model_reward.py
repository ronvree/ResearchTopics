


"""


    " The reward model is a scalar Gaussian with mean parameterized by a feed-forward neural network and unit variance"

"""


import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F


class RewardModel(nn.Module):

    def __init__(self, input_size: int, args: argparse.Namespace):
        """

        :param args:
        """
        super().__init__()

        self.dense1 = nn.Linear(input_size, args.reward_model_num_hidden)
        self.dense2 = nn.Linear(args.reward_model_num_hidden, args.reward_model_num_hidden)
        self.dense3 = nn.Linear(args.reward_model_num_hidden, 1)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        xs = self.dense1(xs)
        xs = F.relu(xs)
        xs = self.dense2(xs)
        xs = F.relu(xs)
        xs = self.dense3(xs)

        return xs.squeeze(1)  # Remove last dimension because it is always of size 1

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser('Reward model')

        parser.add_argument('--reward_model_num_hidden',
                            type=int,
                            default=200,  # 200 is the value from the original paper
                            help='')  # TODO

        return parser


