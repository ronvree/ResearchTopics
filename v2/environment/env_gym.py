
import argparse

from copy import deepcopy

import numpy as np

import torch

from v2.environment.env import Environment
from v2.util.func import batch_tensors


# A list of all supported Gym environment names

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


class GymEnv(Environment):

    """
    Wrapper around OpenAI Gym environments
    """

    # TODO -- docs

    def __init__(self,
                 args: argparse.Namespace
                 ):
        assert args.batch_size > 0
        assert args.env_name in GYM_ENVS
        assert args.max_episode_length > 0

        import gym

        # Wrap the Gym's Viewer constructor so the Viewer invisible
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(_self, *_args, **_kwargs):
            org_constructor(_self, *_args, **_kwargs)
            _self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor

        # Initialize the OpenAI Gym environments
        self._envs = [gym.make(args.env_name) for _ in range(args.batch_size)]

        # Time step counter
        self._t = 0
        # Set time step limit
        self._max_t = args.max_episode_length
        # Check whether images or states should be observed
        self._state_obs = args.state_observations

    def _process_image(self, image):
        return np.moveaxis(image, 2, 0) / 256  # TODO

    @property
    def batch_size(self):
        return len(self._envs)

    def reset(self, no_observation: bool = False) -> tuple:
        # Set internal time step counter to 0
        self._t = 0
        # Get all initial observations from resetting the environment batch
        observations = [env.reset() for env in self._envs]
        # Create a flag tensor
        flags = torch.zeros(self.batch_size, dtype=torch.bool)
        # Create an info dict for each environment
        infos = tuple([dict() for _ in range(len(self._envs))])
        # Don't return an observation if no_observation flag is set
        if no_observation:
            return None, flags, infos
        elif not self._state_obs:
            # Get raw pixel observations of the environments
            pixels_tuple = self._pixels()
            # Process the image observations
            observations = [self._process_image(o) for o in pixels_tuple]
            # Add raw pixels to the info dict
            for info, pixels in zip(infos, pixels_tuple):
                info['pixels'] = pixels

        # Cast all observations to tensors
        observations = [torch.from_numpy(o).to(dtype=torch.float) for o in observations]
        # Concatenate all tensors in a newly created batch dimension
        # Results in a single observation tensor of shape: (batch_size,) + observation_shape
        observations = batch_tensors(*observations)
        # Return the results
        return observations, flags, infos

    def get_state(self) -> tuple:
        return self._t, tuple(deepcopy(tuple(env.unwrapped.state for env in self._envs)))

    def set_state(self, state: tuple):
        self._t, state = state
        for s, env in zip(state, self._envs):
            env.unwrapped.state = deepcopy(s)

    def get_seed(self):  # TODO -- why are openai seeds stored in lists??
        return tuple([env.seed()[0] for env in self._envs])

    def set_seed(self, seeds: tuple):
        for env, seed in zip(self._envs, seeds):
            env.seed(seed)

    def clone(self) -> "Environment":
        copy = deepcopy(self)
        return copy

    def step(self, action: torch.Tensor, no_observation: bool = False) -> tuple:
        # Increment the internal time step counter
        self._t += 1
        # Convert the tensor to suitable input
        action = action.detach().numpy()
        # Execute the actions in the environments
        results = [env.step(a) for a, env in zip(action, self._envs)]

        if no_observation:
            # Convert the results to tensors
            results = [(None,
                        torch.tensor(r, dtype=torch.float),
                        torch.tensor(f),
                        info)
                       for o, r, f, info in results]
            # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
            results = [*zip(*results)]
            # Merge all rewards to one tensor
            results[1] = batch_tensors(*results[1])
            # Merge all flags to one tensor
            results[2] = batch_tensors(*results[2])
            return None, results[1], results[2], results[3]

        # Check if an image observation should be made
        if not self._state_obs:
            # Get raw pixel observations of the environments
            pixels_tuple = self._pixels()
            # Convert them to suitable observations
            observations = [self._process_image(o) for o in pixels_tuple]
            # Merge the observations in the results
            results = [(o,) + result[1:] for o, result in zip(observations, results)]

            # Add all raw pixel observations to the info dicts
            for result, pixels in zip(results, pixels_tuple):
                result[3]['pixels'] = pixels

        # Convert the results to tensors
        results = [(torch.from_numpy(o).to(dtype=torch.float),
                    torch.tensor(r, dtype=torch.float),
                    torch.tensor(f),
                    info)
                   for o, r, f, info in results]

        # Unzip all tuples into 4 tuples containing the observations, rewards, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = batch_tensors(*results[0])
        # Merge all rewards to one tensor
        results[1] = batch_tensors(*results[1])
        # Merge all flags to one tensor
        results[2] = batch_tensors(*results[2])

        # Check max episode length condition. Update flags if required
        if self._t >= self._max_t:
            results[2] |= True  # Set all flags to true if max episode length is reached

        # Return all results as a tuple
        return tuple(results)

    def close(self) -> tuple:
        return tuple(env.close() for env in self._envs)

    def render(self, **kwargs):
        for env in self._envs:
            env.render(**kwargs)

    def _pixels(self) -> tuple:
        return tuple([env.render(mode='rgb_array').copy() for env in self._envs])

    def sample_random_action(self) -> torch.Tensor:
        actions = [env.action_space.sample() for env in self._envs]
        actions = [torch.from_numpy(a) for a in actions]
        actions = batch_tensors(*actions)
        return actions

    @property
    def observation_shape(self) -> tuple:
        return self._envs[0].observation_space.shape

    @property
    def action_shape(self) -> tuple:
        return self._envs[0].action_space.shape


if __name__ == '__main__':

    _args = argparse.Namespace()
    _args.env_name = GYM_ENVS[0]
    _args.batch_size = 3
    _args.max_episode_length = 1000
    _args.state_observations = True

    _env = GymEnv(_args)

    for _ in range(3):
        _env.reset()
        for _ in range(250):
            _env.render(mode='rgb_array')
            _env.step(_env.sample_random_action())

    _env.close()





