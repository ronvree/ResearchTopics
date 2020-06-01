import argparse
import tqdm

import torch


from environment import *
from planet import *


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Train PlaNet')


def get_args():
    return build_argument_parser().parse_args()


def train_planet(args: argparse.Namespace = None):
    args = args or get_args()

    # Determine if GPU or CPU should be used
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Build the environment
    env = build_env_from_args(args)

    # Initialize dataset D with S random seed episodes

    # Intialize model parameters theta randomly

    converged = False
    while not converged:

        # Model Fitting
        pass  # TODO

        # Data Collection
        pass  # TODO


if __name__ == '__main__':
    train_planet()
