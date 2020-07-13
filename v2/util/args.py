
import os
import argparse
import pickle

import torch
import torch.optim
from torch.optim import Optimizer


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args


def optimizer_from_args(model_parameters, args: argparse.Namespace) -> Optimizer:

    # Get optimizer name
    name = args.optimizer
    # Sanitize input argument
    name = name.lower()

    if name == 'sgd':
        raise NotImplementedError  # TODO
    if name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=1e-3  # TODO -- specify in args
                                     )
        return optimizer

    raise Exception('Optimizer argument not recognized!')


