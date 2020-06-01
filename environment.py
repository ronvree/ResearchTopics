import cv2
import numpy as np
import torch
import argparse


GYM_ENVS = [
    'Pendulum-v0',
    'MountainCarContinuous-v0',
    'Ant-v2',
    'HalfCheetah-v2',
    'Hopper-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2',
    'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2',
    'Reacher-v2',
    'Swimmer-v2',
    'Walker2d-v2'
]

CONTROL_SUITE_ENVS = [
    'cartpole-balance',
    'cartpole-swingup',
    'reacher-easy',
    'finger-spin',
    'cheetah-run',
    'ball_in_cup-catch',
    'walker-walk'
]

# CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}


def build_env_from_args(args: argparse.Namespace) -> "Environment":
    """
    Build an environment from parsed arguments
    :param args: a Namespace object containing parsed arguments
    :return: the built environment
    """
    name = args.env_name

    if name in GYM_ENVS:
        return GymEnv(args)
    if name in CONTROL_SUITE_ENVS:
        return ControlSuiteEnvironment(args)
    raise Exception('Environment name not recognized!')


class Environment:

    """
    Abstract class for defining an environment
    """

    def reset(self) -> tuple:
        """
        Set the environment to its initial state
        :return: a 3-tuple consisting of:
                    - the initial observation
                    - a flag indicating whether the environment has terminated
                    - a dict possibly containing additional information
        """
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> tuple:
        """
        Perform an action in the environment. Returns a reward and observation
        :param action: a Tensor representation of the action that should be performed in the environment
        :return: a 4-tuple consisting of:
                    - an (object) observation
                    - a (float) reward
                    - a (boolean) flag indicating whether the environment has terminated
                    - a dict possibly containing additional information
        """
        raise NotImplementedError

    def close(self) -> dict:
        """
        Close the environment
        :return: a dict possibly containing information
        """
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of observations
        """
        raise NotImplementedError

    @property
    def action_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of actions
        """
        raise NotImplementedError


class ControlSuiteEnvironment(Environment):

    pass  # TODO

    def __init__(self, args: argparse.Namespace):
        pass

    def reset(self) -> tuple:
        raise NotImplementedError

    def step(self, action) -> tuple:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def action_shape(self) -> tuple:
        raise NotImplementedError


class GymEnv(Environment):

    """
    Environment from the OpenAI gym
    """

    def __init__(self, args: argparse.Namespace):
        """
        :param args:
        """
        import gym

        self._env = gym.make(args.env_name)

    def reset(self) -> tuple:
        """
        Set the environment to its initial state
        :return: a 3-tuple consisting of:
                    - the initial observation (torch.Tensor)
                    - a flag indicating whether the environment has terminated
                    - a dict possibly containing additional information
        """
        observation = self._env.reset()
        observation = torch.from_numpy(observation)

        return observation, False, {}

    def step(self, action) -> tuple:  # TODO -- cast all values to tensors?
        """
        Perform an action in the environment. Returns a reward and observation
        :param action: a Tensor representation of the action that should be performed in the environment
        :return: a 4-tuple consisting of:
                    - a (torch.Tensor) observation
                    - a (torch.Tensor) reward consisting of 1 float
                    - a (boolean) flag indicating whether the environment has terminated
                    - a dict possibly containing additional information
        """
        action = action.detach().numpy()

        o, r, f, info = self._env.step(action)
        o = torch.from_numpy(o)  # TODO -- this is problematic...
        r = torch.FloatTensor([r])  # TODO -- to which device??
        return o, r, f, info

    def close(self):
        """
        Close the environment
        :return: a dict possibly containing information
        """
        self._env.close()
        return {}  # Closing env does not give information, so return empty dict

    def render(self) -> bool:
        """
        Shows the environment
        :return: the result of the OpenAI gym's render function
        """
        return self._env.render()  # TODO -- what does this return?

    @property
    def observation_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of observations
        """
        return self._env.observation_space.shape

    @property
    def action_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of actions
        """
        return self._env.action_space.shape


class EnvironmentBatcher:

    """
    Wrapper class for running multiple environments of the same type
    """

    def __init__(self, args: argparse.Namespace):
        assert args.num_env > 0

        self._num_env = args.num_env
        self._envs = [build_env_from_args(args) for _ in range(self._num_env)]

    @property
    def action_shape(self) -> tuple:
        return self._envs[0].action_shape

    @property
    def observation_shape(self) -> tuple:
        return self._envs[0].observation_shape

    @property
    def num_environments(self) -> int:
        return self._num_env

    def reset(self) -> tuple:
        """

        :return:
        """
        # Reset all environments, collect all results
        results = [env.reset() for env in self._envs]
        # Unzip all tuples into 3 tuples containing the observations, termination flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = torch.cat([o.view(1, *self.observation_shape) for o in results[0]], dim=0)
        # Return all results as a tuple
        return tuple(results)

    def step(self, actions) -> tuple:  # TODO -- what is the type of actions? pass as single tensor with batch dim or list of tensors?
        """

        :param actions:
        :return:
        """
        # TODO -- what is the return type? a tuple of tensors? a list of tuples with individual results?

        # Perform the actions on the environments
        results = [env.step(action) for env, action in zip(self._envs, actions)]
        # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = torch.cat([o.view(1, *self.observation_shape) for o in results[0]], dim=0)
        # Merge all rewards to one tensor
        results[1] = torch.cat([r.view(1, 1) for r in results[1]], dim=0)
        # Return all results as a tuple
        return tuple(results)

    def close(self) -> tuple:
        """

        :return:
        """
        return tuple(env.close() for env in self._envs)


if __name__ == '__main__':

    _args = argparse.Namespace()
    _args.env_name = GYM_ENVS[0]

    _args.num_env = 4

    # _env = GymEnv(_args)
    #
    # _env.reset()
    # for _ in range(10000):
    #     print(_env.render())
    #     _env.step(torch.randn(_env.action_shape))
    #
    # _env.close()

    _env_batch = EnvironmentBatcher(_args)

    _env_batch.reset()

    for _ in range(10000):

        _actions = torch.randn(_args.num_env, *_env_batch.action_shape)

        _env_batch.step(_actions)

        # for _env in _env_batch._envs:
        #     _env.render()

    _env_batch.close()



