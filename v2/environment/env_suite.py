import argparse

from copy import deepcopy

import numpy as np

import torch

from v2.environment.env import Environment
from v2.util.func import batch_tensors


# List of all supported control suite environments names
CONTROL_SUITE_ENVS = [
    'cartpole-balance',
    'cartpole-swingup',
    'reacher-easy',
    'finger-spin',
    'cheetah-run',
    'ball_in_cup-catch',
    'walker-walk'
]


class ControlSuiteEnvironment(Environment):

    """
    Wrapper around Control Suite environments

    https://github.com/deepmind/dm_control
    """

    # TODO -- docs

    # TODO -- image observations

    def __init__(self, args: argparse.Namespace):
        assert args.batch_size > 0
        assert args.env_name in CONTROL_SUITE_ENVS
        assert args.max_episode_length > 0

        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        domain, task = args.env_name.split('-')

        self._envs = [suite.load(domain_name=domain, task_name=task)
                      for _ in range(args.batch_size)]
        # self._envs = [pixels.Wrapper(env) for env in self._envs]  # TODO

        # Time step counter
        self._t = 0
        # Set time step limit
        self._max_t = args.max_episode_length
        # Check whether images or states should be observed
        self._state_obs = args.state_observations

    @property
    def batch_size(self):
        return len(self._envs)

    @property
    def observation_shape(self) -> tuple:
        pass  # TODO

    @property
    def action_shape(self) -> tuple:
        return self._envs[0].action_spec().shape

    def _process_image(self, image):
        return torch.from_numpy(image) / 256  # TODO -- now converting to tensor twice!

    def reset(self) -> tuple:
        # Set internal time step counter to 0
        self._t = 0
        # Reset all environments
        results = [env.reset() for env in self._envs]
        # Process the control suite results
        results = [self._process_result(result) for result in results]

        # Get raw pixel observations
        pixels_tuple = self._pixels()

        # Filter the non-existent rewards
        results = [(o, t, info) for o, r, t, info in results]

        # If required, set observation to image observations
        if not self._state_obs:
            results = [(self._process_image(image), t, info) for image, (_, t, info) in zip(pixels_tuple, results)]

        # Unzip all tuples into 3 tuples containing the observations, flags and info dicts, respectively
        results = [*zip(*results)]
        # Merge all observations to one tensor
        results[0] = batch_tensors(*results[0])
        # Merge all flags to one tensor
        results[1] = batch_tensors(*results[1])

        # Add raw pixels to all info dicts
        for pixels, info in zip(pixels_tuple, results[2]):
            info['pixels'] = pixels

        # Return all results as a tuple
        return tuple(results)

    def get_state(self) -> tuple:
        states = []
        for env in self._envs:
            states.append(
                (
                    np.array(env.physics.data.qpos),
                    np.array(env.physics.data.qvel),
                    np.array(env.physics.data.ctrl),
                )
            )
        return self._t, tuple(states)

    def set_state(self, state: tuple):
        self._t, state = state
        for env, (pos, vel, ctrl) in zip(self._envs, state):
            with env.physics.reset_context():
                env.physics.data.qpos[:] = pos
                env.physics.data.qvel[:] = vel
                env.physics.data.ctrl[:] = ctrl

    def get_seed(self) -> tuple:
        raise NotImplementedError  # TODO

    def set_seed(self, seed: tuple):
        raise NotImplementedError  # TODO

    def clone(self) -> "Environment":
        return deepcopy(self)

    def step(self, action: torch.Tensor) -> tuple:
        # Increment the internal time step counter
        self._t += 1
        # Convert the tensor to suitable input
        action = action.detach().numpy()
        # Execute the actions in the environments
        results = [env.step(a) for a, env in zip(action, self._envs)]

        # Get raw pixels from all environments
        pixels_tuple = self._pixels()  # TODO -- image observations

        # Process the control suite results
        results = [self._process_result(result) for result in results]
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

        # Add all raw pixel observations to the info dicts
        for info, pixels in zip(results[3], pixels_tuple):
            info['pixels'] = pixels

        # Return all results as a tuple
        return tuple(results)

    def close(self) -> tuple:
        return tuple(env.close() for env in self._envs)

    def sample_random_action(self):
        actions = []
        for env in self._envs:
            spec = env.action_spec()
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
            actions += [torch.from_numpy(action)]
        actions = batch_tensors(*actions)
        return actions

    def _pixels(self) -> tuple:
        return tuple([env.physics.render() for env in self._envs])

    def _process_result(self, result) -> tuple:
        """

        :param result:
        :return:
        """
        observation = [np.asarray([obs]) if isinstance(obs, float) else obs for obs in result.observation.values()]
        observation = np.concatenate(observation)
        observation = torch.FloatTensor(observation)

        reward = result.reward
        reward = torch.FloatTensor([result.reward]) if reward is not None else torch.zeros(1)

        terminal = result.last()
        terminal = torch.tensor(terminal)

        return observation, reward, terminal, {}


if __name__ == '__main__':

    from dm_control import viewer

    _args = argparse.Namespace()
    _args.env_name = CONTROL_SUITE_ENVS[0]
    _args.batch_size = 3
    _args.max_episode_length = 1000
    _args.state_observations = True

    _env = ControlSuiteEnvironment(_args)

    _env.reset()

    # for _i in range(_args.batch_size):
    #     viewer.launch(_env._envs[_i], policy=lambda t: _env.sample_random_action()[_i])

    for _ in range(10000):
        _env.step(_env.sample_random_action())

    _env.close()
