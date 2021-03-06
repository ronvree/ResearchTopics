

import argparse

from collections import deque

import numpy as np
import numpy.random as rnd

import torch
import torch.utils.data

from torch.utils.data import Dataset, TensorDataset

from v2.util.episode import Episode
from v2.util.func import batch_tensors


class ExperienceReplay:

    """

        Implementation of a (possibly infinite) buffer of data collected from experience

        Can be converted to a PyTorch DataSet

        The dataset contains data sample tuples (o, a, r, o', a') consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: (1,) )
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: action_shape)

        The samples are stored sequentially in a deque object

        Data samples can be sampled from the dataset in batches

    """

    def __init__(self, args: argparse.Namespace):
        assert args.experience_buffer_size > 0

        self._max_size = args.experience_buffer_size
        self._data = deque(maxlen=self._max_size)

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser = None):
        """
        Add the arguments required for the ExperienceReplay to the argument parser
        Creates a new parser if none is given
        :param parser: (optional) parser to which the arguments should be added
        :return: an argument parser for parsing arguments required for the ExperienceReplay
        """
        parser = parser or argparse.ArgumentParser('Experience Replay Hyperparameters')

        parser.add_argument('--experience_size',
                            type=int,
                            default=np.inf,
                            help='Maximum size of the experience replay dataset')

    @staticmethod
    def episode_to_samples(episode: Episode) -> tuple:
        """
        Transform an episode to a tuple of (o, a, r, o', a') data samples
        """
        # Store samples in a list
        samples = list()
        # An episode consists of:
        # o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        for i in range(0, len(episode), 3):  # Jump from observation to observation in the episodes
            # Get the next sequence of (o, a, r, o', a')
            sample = episode[i:i+5]
            # Check if the sample is complete (this does not happen at the end of the episode)
            if len(sample) == 5:
                samples.append(tuple(sample))
        # Return all samples
        return tuple(samples)

    @property
    def size(self) -> int:
        """
        :return: the number of samples that are currently in the dataset
        """
        return len(self)

    @property
    def max_size(self):
        """
        :return: the max number of samples that can be stored in the dataset. Samples are removed in a FIFO manner
        """
        return self._max_size

    def __len__(self) -> int:
        """
        :return: the number of samples that are currently in the dataset
        """
        return len(self._data)

    def draw_sample(self, batch_size: int = 1) -> tuple:
        """
        Draw a (uniformly distributed) random sample (batch) from the dataset (with replacement)

        One sample (batch) is a (o, a, r, o', a')-tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)

        :param batch_size: Optional batch size parameter. Default = 1
        :return: one data sample (batch)
        """
        assert batch_size > 0
        # Collect individual samples
        samples = []
        for i in range(batch_size):
            index = rnd.randint(self.size)
            sample = self._data[index]
            samples += [sample]
        # Return the samples including a batch dimension
        return ExperienceReplay._batch_samples(samples)

    def append(self, sample: tuple) -> None:
        """
        Append a single (o, a, r, o', a')-tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)

        The batch dimension is removed from the tensors and the individual samples are added to the dataset

        :param sample: the data tuple that should be added to the dataset
        """
        self.append_all((sample,))

    def append_all(self, samples: tuple) -> None:
        """
        Append multiple data sample( batche)s

        All samples in the `samples` tuple should be (o, a, r, o', a')-tuples consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)

        Samples from the same sequence (or batch index) are appended sequentially

        :param samples: A tuple of individual batched data samples
        """
        batch_size = samples[0][0].size(0)
        for i in range(batch_size):
            for o, a, r, o_, a_ in samples:
                self._data.append((o[i], a[i], r[i], o_[i], a_[i]))

    def append_episode(self, episode: Episode) -> None:
        """
        Append the data from an episode to this experience buffer
        :param episode: the episode from which data should be appended
        """
        self._data.extend(ExperienceReplay.episode_to_samples(episode))

    def append_episodes(self, episodes: list) -> None:
        """
        Append the data of multiple episodes to this experience buffer
        :param episodes: a list of episodes from which data should be appended
        """
        for episode in episodes:
            self.append_episode(episode)

    def as_dataset(self) -> TensorDataset:
        """
        Convert the dataset to a torch.utils.data.TensorDataset
        Iterating through this dataset will give the individual (o, a, r, o', a') samples
        :return: the entire dataset as a torch.utils.data.TensorDataset
        """
        # Separate the dataset into five tuples of all (o, a, r, o', a')
        o, a, r, o_, a_ = (*zip(*self._data),)
        # For all five tuples, concatenate the entries over one batch dimension
        # All tuples are now single tensors containing all data (with the data set size as batch size)
        o, a, r, o_, a_ = tuple(batch_tensors(*ts) for ts in (o, a, r, o_, a_))
        # Use these tensors to create a TensorDataset
        return TensorDataset(o, a, r, o_, a_)

    @staticmethod
    def _batch_samples(samples: tuple) -> tuple:
        """
        Transform a tuple of sample tuples to a single batched sample tuple

        The batch size is the length of the tuple of samples

        :param samples: a tuple of individual samples
            Each individual sample is a tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: (1,) )
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: action_shape)

        :return: a single sample batch tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)
        """
        # Separate the samples into five tuples of all (o, a, r, o', a')
        o, a, r, o_, a_ = tuple(*zip(*samples))
        # For all five tuples, concatenate the entries over one batch dimension
        o, a, r, o_, a_ = tuple(batch_tensors(*ts) for ts in (o, a, r, o_, a_))
        # Return as a single batched sample tuple
        return o, a, r, o_, a_

    @staticmethod
    def _unbatch_samples(sample: tuple) -> tuple:
        """
        Transform a single batched sample tuple to a tuple of sample tuples

        The batch size is the length of the tuple of samples

        :param sample: a single sample batch tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: (batch_size,) + observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: (batch_size,) + action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: batch_size)
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: (batch_size,) + observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: (batch_size,) + action_shape)
        :return: a tuple of individual samples
            Each individual sample is a tuple consisting of:
            - o: an observation tensor (at time t)          (dtype: float, shape: observation_shape)
            - a: an action tensor (at time t)               (dtype: float, shape: action_shape)
            - r: a reward tensor (at time t + 1)            (dtype: float, shape: (1,) )
            - o': an observation tensor (at time t + 1)     (dtype: float, shape: observation_shape)
            - a': an action tensor (at time t + 1)          (dtype: float, shape: action_shape)
        """
        return tuple(s for s in zip(*sample))


#
#
#
# class ExperienceReplay:
#
#     """
#         Implementation of a (possibly infinite) buffer of data collected from experience
#
#         Can be converted to a PyTorch DataSet
#
#         The dataset contains data sample tuples (o, a, r, f) consisting of:
#             - o: an observation tensor      (dtype: float, shape: observation_shape)
#             - a: an action tensor           (dtype: float, shape: action_shape)
#             - r: a reward tensor            (dtype: float, shape: (1,) )
#             - f: a termination flag tensor  (dtype: bool, shape: (1,) )
#
#         The samples are stored sequentially in a deque object
#
#         (Sequences of) data samples can be sampled from the dataset in batches
#
#     """
#
#     def __init__(self, args: argparse.Namespace):
#         """
#         Create a new experience replay buffer
#         :param args: an argparse.Namespace object containing parsed hyperparameters
#         """
#         assert args.experience_size > 0
#
#         self._max_size = args.experience_size
#         self._data = deque(maxlen=args.experience_size)
#
#     @staticmethod
#     def build_argument_parser(parser: argparse.ArgumentParser = None):
#         """
#         Add the arguments required for the ExperienceReplay to the argument parser
#         Creates a new parser if none is given
#         :param parser: (optional) parser to which the arguments should be added
#         :return: an argument parser for parsing arguments required for the ExperienceReplay
#         """
#         parser = parser or argparse.ArgumentParser('Experience Replay Hyperparameters')
#
#         parser.add_argument('--experience_size',
#                             type=int,
#                             default=np.inf,
#                             help='Maximum size of the experience replay dataset')
#
#         return parser
#
#     @property
#     def size(self) -> int:
#         """
#         :return: the number of samples that are currently in the dataset
#         """
#         return len(self)
#
#     @property
#     def max_size(self):
#         """
#         :return: the max number of samples that can be stored in the dataset. Samples are removed in a FIFO manner
#         """
#         return self._max_size
#
#     def __len__(self) -> int:
#         """
#         :return: the number of samples that are currently in the dataset
#         """
#         return len(self._data)
#
#     def draw_sample(self, batch_size: int = 1) -> tuple:
#         """
#         Draw a (uniformly distributed) random sample (batch) from the dataset
#
#         One sample (batch) is a tuple consisting of:
#             - an observation tensor     Shape: (batch_size,) + observation_shape
#             - an action tensor          Shape: (batch_size,) + action_shape
#             - a reward tensor           Shape: (batch_size,)
#             - a termination flag        Shape: (batch_size,)
#
#         :param batch_size: Optional batch size parameter. Default = 1
#         :return: one data sample (batch)
#         """
#         assert batch_size > 0
#         # Collect individual samples
#         samples = []
#         for i in range(batch_size):
#             index = rnd.randint(self.size)
#             sample = self._data[index]
#             samples += [sample]
#         # Return the samples including a batch dimension
#         return self._batch_samples(samples)
#
#     def draw_sequence(self, length: int, batch_size: int = 1) -> tuple:
#         """
#         Draw a (uniformly distributed) random sequence of subsequent samples from the dataset
#
#         One sample (batch) is a tuple consisting of:
#             - an observation tensor     Shape: (batch_size,) + observation_shape
#             - an action tensor          Shape: (batch_size,) + action_shape
#             - a reward tensor           Shape: (batch_size,)
#             - a termination flag        Shape: (batch_size,)
#
#         The sequence is stored as tuple (of individual sample tuples)
#
#         :param length: The length of the sequence
#         :param batch_size: Optional batch size parameter. Default = 1
#         :return: a sequence of (batched) data samples
#         """
#         assert 0 < length <= self.size
#         assert batch_size > 0
#
#         raise NotImplementedError
#
#
#         index = rnd.randint(self.size - length)
#         sequence = self._data[index:index + length]
#         return tuple(sequence)
#
#     def append(self, sample: tuple):
#         """
#         Append a single sample tuple consisting of:
#             - an observation tensor     Shape: (batch_size,) + observation_shape
#             - an action tensor          Shape: (batch_size,) + action_shape
#             - a reward tensor           Shape: (batch_size,)
#             - a termination flag        Shape: (batch_size,)
#
#         The batch dimension is removed from the tensors and the individual samples are added to the dataset
#
#         :param sample: the data tuple that should be added to the dataset
#         """
#         self.append_sequence((sample,))
#
#     def append_sequence(self, sequence: tuple):
#         """
#         Append a sequence of data sample batches
#         Samples from the same sequence (or batch index) are appended sequentially
#         :param sequence: A tuple of individual batched data samples
#         """
#         batch_size = sequence[0][0].size(0)
#         for i in range(batch_size):
#             for o, a, r, f in sequence:
#                 self._data.append((o[i], a[i], r[i], f[i]))
#
#     def as_dataset(self) -> TensorDataset:
#         """
#         Convert the dataset to a torch.utils.data.TensorDataset
#         Iterating through this dataset will give the individual (o, a, r, f) samples
#
#         :return: the entire dataset as a torch.utils.data.TensorDataset
#         """
#         # Separate the observations, actions, rewards, flags from the data set into four tuples
#         observations, actions, rewards, flags = tuple(*zip(*self._data))
#         # Concatenate all four tuples of tensors to one single batch tensor (over the entire dataset)
#         tensors = tuple(batch_tensors(*ts) for ts in (observations, actions, rewards, flags))
#         # Use these tensors to create a TensorDataset
#         return TensorDataset(*tensors)
#
#     def _batch_samples(self, samples: list) -> tuple:
#         """
#         Transform a list of sample tuples to a single batched sample tuple
#
#         The batch size is the length of the list of samples
#
#         Each individual sample is a tuple consisting of:
#             - an observation tensor     Shape: observation_shape
#             - an action tensor          Shape: action_shape
#             - a reward tensor           Shape: 1
#             - a termination flag        Shape: 1
#
#         :param samples: a list of individual samples
#         :return: a single sample batch tuple consisting of:
#             - an observation tensor     Shape: (batch_size,) + observation_shape
#             - an action tensor          Shape: (batch_size,) + action_shape
#             - a reward tensor           Shape: (batch_size,)
#             - a termination flag        Shape: (batch_size,)
#         """
#         batch_size = len(samples)
#
#         raise NotImplementedError
#
