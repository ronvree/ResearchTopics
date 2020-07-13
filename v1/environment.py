import cv2
import argparse
from copy import deepcopy

import numpy as np

import torch


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

    # TODO -- action repeat
    # TODO -- finite time steps
    # TODO -- select which device
    # TODO -- rendering
    # TODO -- set states

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

    def get_state(self):  # TODO -- maybe add warning about complete observability
        """
        Get the environment state
        Note: Using this function assumes complete observability!
        :return: the environment state
        """
        raise NotImplementedError

    def set_state(self, state):
        """
        Set the environment state
        :param state: the environment will be set to this state
        """
        raise NotImplementedError

    def clone(self, set_init_state: bool = False) -> "Environment":
        """
        Get a deep copy of the environment
        :param set_init_state: When set to True, the current environment state will be considered the initial state.
                               self.reset() will set the environment state to this initial state
        :return: a deep copy of this environment
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

    def __init__(self, args: argparse.Namespace):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        domain, task = args.env_name.split('-')

        self._env = suite.load(domain_name=domain, task_name=task)

        self._env = pixels.Wrapper(self._env)  # TODO

    def reset(self) -> tuple:
        """

        :return:
        """
        result = self._env.reset()

        obs, _, terminal, info = self._process_result(result)

        return obs, terminal, info

    def step(self, action) -> tuple:
        """

        :param action:
        :return:
        """
        action = action.detach().numpy()

        result = self._env.step(action)

        return self._process_result(result)

    def close(self):
        """

        :return:
        """
        return self._env.close()

    @property
    def observation_shape(self) -> tuple:
        # return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in
        #             self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

        pass  # TODO

        raise NotImplementedError

    @property
    def action_shape(self) -> tuple:
        raise self._env.action_spec().shape

    def _process_result(self, result) -> tuple:
        """

        :param result:
        :return:
        """
        observation = [np.asarray([obs]) if isinstance(obs, float) else obs for obs in result.observation.values()]
        observation = np.concatenate(observation)
        observation = torch.FloatTensor(observation)  # TODO -- device

        reward = result.reward
        reward = torch.FloatTensor([result.reward]) if reward is not None else None

        terminal = result.last()

        return observation, reward, terminal, {}


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

        self._envs = [build_env_from_args(args) for _ in range(args.num_env)]

    @property
    def action_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of actions (excluding batch dimension)
        """
        return self._envs[0].action_shape

    @property
    def observation_shape(self) -> tuple:
        """
        :return: a tuple describing the shape of observations (excluding batch dimension)
        """
        return self._envs[0].observation_shape

    @property
    def num_environments(self) -> int:
        """
        :return: the number of environments in this batch of environments
        """
        return len(self._envs)

    def reset(self) -> tuple:
        """
        Reset all environments
        :return: a 3-tuple consisting of:
                    - a single tensor containing all initial observations
                    - a tuple containing all termination flags
                    - a tuple containing all info dicts
        """
        # Reset all environments, collect all results
        results = [env.reset() for env in self._envs]
        # Unzip all tuples into 3 tuples containing the observations, termination flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = torch.cat([o.view(1, *self.observation_shape) for o in results[0]], dim=0)
        # Return all results as a tuple
        return tuple(results)

    def step(self, actions: torch.Tensor) -> tuple:
        """
        Perform an action in each of the environments
        :param actions: a tensor containing a batch of actions. Shape: (batch_size,) + action_shape
        :return: a 4-tuple consisting of:
                    - a single tensor containing all observations. Shape: (batch_size,) + observation_shape
                    - a single tensor containing all rewards. Shape: (batch_size, 1)
                    - a tuple containing all termination flags
                    - a tuple containing all info dicts
        """  # TODO -- should rewards be shaped (batch_size,) ?
        # Perform the actions on the environments
        results = [env.step(action) for env, action in zip(self._envs, actions)]
        # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = torch.cat([o.view(1, *self.observation_shape) for o in results[0]], dim=0)
        # Merge all rewards to one tensor
        results[1] = torch.cat([r for r in results[1]], dim=0)
        # Return all results as a tuple
        return tuple(results)

    def close(self) -> tuple:
        """
        Close all environments
        :return: a tuple containing the results of closing each environment
        """
        return tuple(env.close() for env in self._envs)


if __name__ == '__main__':

    _args = argparse.Namespace()
    # _args.env_name = GYM_ENVS[0]
    _args.env_name = CONTROL_SUITE_ENVS[0]

    _args.num_env = 1

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



