import argparse

import numpy as np

from collections import deque

import torch


class ExperienceReplay:

    def __init__(self, args: argparse.Namespace):

        # Items in the buffer are 4-tuples (O, A, R, D) where
        #  - O is a tensor of observations of shape (bs, T, channels, observation_width, observation_height)
        #  - A is a tensor of actions of shape (bs, T, action_size)
        #  - R is a tensor of rewards of shape (bs, T)
        #  - D is a tensor of done flags of shape (bs, T)
        # where bs is the batch size and T is the episode length
        if args.exp_replay_buffer_size == np.inf:
            self.buffer = list()
        else:
            self.buffer = deque(maxlen=args.exp_replay_buffer_size)

    def append_batch(self,
                     observations: torch.Tensor,
                     actions: torch.Tensor,
                     rewards: torch.Tensor,
                     dones: torch.Tensor
                     ):
        # Check if batch sizes are equal
        batch_size = observations.size(0)
        assert batch_size == actions.size(0)
        assert batch_size == rewards.size(0)
        assert batch_size == dones.size(0)

        self.buffer.append((observations, actions, rewards, dones))

    def get_dataloader(self):
        pass  # TODO


