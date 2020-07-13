
import tqdm
import argparse

import torch

from v2.environment.env import Environment


class CEM:

    """

    """

    def __init__(self, args: argparse.Namespace):
        """

        :param args:
        """
        assert args.planning_horizon > 0
        assert args.num_plan_iter > 0
        assert args.num_plan_candidates > 1
        assert args.num_plan_top_candidates > 1
        assert args.num_plan_candidates >= args.num_plan_top_candidates
        assert args.batch_size > 0

        self.horizon_distance = args.planning_horizon
        self.num_iter = args.num_plan_iter
        self.num_candidates = args.num_plan_candidates
        self.num_top_candidates = args.num_plan_top_candidates

        self.batch_size = args.batch_size

    def plan(self,
             env: Environment,
             device=torch.device('cpu')
             ):
        # TODO -- return info dict as well?
        """
        Plan an action using the Cross-Entropy Method
        :param env: the environment in which trajectories can be simulated
        :param device: the device on which the planning should be computed (cpu/gpu)
        :return: The action selected by planning
        """
        # Define size variables
        batch_size = self.batch_size
        action_shape = env.action_shape

        # Initialize action belief parameters
        param_shape = (batch_size, self.horizon_distance) + action_shape
        mean = torch.zeros(param_shape, device=device)
        std = torch.ones(param_shape, device=device)

        # Build planning progress bar
        iters = tqdm.tqdm(range(self.num_iter),
                          total=self.num_iter,
                          desc='Planner')
        iters.set_postfix_str(f'iters done 0/{self.num_iter}')

        # Optimize the action belief parameters
        for i in iters:
            # Sample all action sequences of all candidates from the current action belief
            candidates = self._sample_action_candidates(mean, std)
            # Score all candidates based on their estimated return
            returns = self._evaluate_action_candidates(candidates, env)
            # Reparameterize the policy distribution based on the best candidates
            mean, std = self._update_params(candidates, returns)

            # Update progress bar
            iters.set_postfix_str(f'iters done {i + 1}/{self.num_iter}')

        # Return first action mean
        return mean[:, 0]  # Shape: (batch_size,) + action_shape

    def _evaluate_action_candidates(self, candidates: torch.Tensor, environment: Environment) -> torch.Tensor:
        """
        Score each of the action sequence candidates
        :param candidates: Tensor containing each of the action sequence candidates
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param environment: Environment in which the action sequences are evaluated
        :return: tensor containing scores for each of the candidates
                    shape: (batch_size, num_candidates)
        """
        # Check which device should be used (cpu/gpu)
        device = candidates.device
        # Get the initial state of the environment
        init_env_state = environment.get_state()
        # Get the batch size
        batch_size = candidates.size(0)
        # Store scores for all candidates
        candidate_returns = list()
        # Loop through all candidate tensors
        for j, candidate in enumerate(candidates.split(1, dim=1)):
            # Remove the redundant candidate dimension of the single candidate action sequence (batch)
            # Shape: (batch_size, horizon_distance,) + action_shape
            candidate = candidate.squeeze(1)
            # Keep track of the total return
            reward_total = torch.zeros(batch_size, 1, device=device)
            # Execute all actions in the environment
            for tau, action in enumerate(candidate.split(1, dim=1)):
                # Remove the redundant planning horizon dimension
                # Shape: (batch_size,) + action_shape
                action = action.squeeze(1)
                # Execute the action
                observation, reward, flag, info = environment.step(action, no_observation=True)
                # Add to the total reward
                reward_total += reward.view(-1, 1)
            # Store the return estimate of this candidate
            candidate_returns.append(reward_total)
            # Reset environment for the next candidate
            environment.set_state(init_env_state)
        # Concatenate the returns to one tensor
        returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)
        return returns

    def _update_params(self, candidates: torch.Tensor, scores: torch.Tensor) -> tuple:

        # Get the batch size
        batch_size = candidates.size(0)
        # Select the top K candidate sequences based on the rewards obtained
        # Shape: (batch_size, num_top_candidates)
        top_candidate_ixs = torch.argsort(scores, dim=1)[:, -self.num_top_candidates:]

        # Iterate through each item in the batch
        new_mean = [None] * batch_size
        new_std = [None] * batch_size

        for batch_i, (candidates, top_ixs) in enumerate(zip(candidates, top_candidate_ixs)):

            top_candidates = candidates.index_select(0, top_ixs)  # Shape: (num_top_candidates, horizon_distance) + action_shape

            sample_mean = torch.sum(top_candidates, dim=0) / self.num_top_candidates
            sample_std = torch.sum(torch.abs(top_candidates - sample_mean), dim=0) / (self.num_top_candidates - 1)

            new_mean[batch_i] = sample_mean.unsqueeze(0)
            new_std[batch_i] = sample_std.unsqueeze(0)

        mean = torch.cat(new_mean, dim=0)
        std = torch.cat(new_std, dim=0)

        return mean, std

    def _sample_action_candidates(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Use the given parameterization of the action belief to sample number of action sequence candidates
        :param mean: a tensor containing a batch of mean values for the action parameters
                     Shape: (batch_size, horizon_distance, ) + action_shape
        :param std: a tensor containing a batch of std values for the action parameters
                     Shape: (batch_size, horizon_distance, ) + action_shape
        :return: a tensor containing a batch of candidate action sequences, sampled from the given parameters
                     Shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        """
        # Get the batch size from the parameter tensor
        batch_size = mean.size(0)
        # Create an extra 'candidate' dimension (using unsqueeze)
        # Expand the parameter tensors over this dimension. The other dimensions remain the same
        expanded_mean = mean.unsqueeze(1).expand(batch_size, self.num_candidates, *mean.shape[1:])
        expanded_std = std.unsqueeze(1).expand(batch_size, self.num_candidates, *std.shape[1:])
        # Sample the candidate action sequences
        # Shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        action_candidate_seqs = torch.normal(expanded_mean, expanded_std)
        return action_candidate_seqs

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser('Cross-Entropy Method')

        parser.add_argument('--planning_horizon',
                            type=int,
                            default=1,
                            help='Planning horizon distance')
        parser.add_argument('--num_plan_iter',
                            type=int,
                            default=1,
                            help='Optimization iterations')
        parser.add_argument('--num_plan_candidates',
                            type=int,
                            default=1,
                            help='Candidates per iteration')
        parser.add_argument('--num_plan_top_candidates',
                            type=int,
                            default=1,
                            help='Number of top candidates to fit')

        return parser


class AdaptedCEM(CEM):

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(args)
        self._model = model

    def _evaluate_action_candidates(self, candidates: torch.Tensor, environment: Environment):
        """
        Score each of the action sequence candidates
        :param candidates: Tensor containing each of the action sequence candidates
                            shape: (batch_size, num_candidates, horizon_distance,) + action_shape
        :param environment: Environment in which the action sequences are evaluated
        :return: tensor containing scores for each of the candidates
                    shape: (batch_size, num_candidates)
        """
        # Check which device should be used (cpu/gpu)
        device = candidates.device
        # Get the initial state of the environment
        init_env_state = environment.get_state()
        # Get the batch size
        batch_size = candidates.size(0)
        # Store scores for all candidates
        candidate_returns = list()
        # Loop through all candidate tensors
        for j, candidate in enumerate(candidates.split(1, dim=1)):
            # Remove the redundant candidate dimension of the single candidate action sequence (batch)
            # Shape: (batch_size, horizon_distance,) + action_shape
            candidate = candidate.squeeze(1)
            # Keep track of the total return
            reward_total = torch.zeros(batch_size, 1, device=device)
            # Separate the tensor into multiple action tensors
            actions = [action.squeeze(1) for action in candidate.split(1, dim=1)]
            # Execute all but the last action in the environment
            for tau, action in enumerate(actions[:-1]):
                # Execute the action
                observation, reward, flag, info = environment.step(action, no_observation=bool(len(actions) - tau - 2))  # TODO -- only get observation at last iter -- neater
                # Add to the total reward
                reward_total += reward.view(-1, 1)
            # Use the q-model to estimate the return of the final action
            reward_total += self._model(observation, actions[-1])
            # Store the return estimate of this candidate
            candidate_returns.append(reward_total)
            # Reset environment for the next candidate
            environment.set_state(init_env_state)
        # Concatenate the returns to one tensor
        returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)
        return returns


if __name__ == '__main__':

    from dm_control import viewer

    from v2.environment.util import build_env_from_args
    from v2.environment.env_suite import CONTROL_SUITE_ENVS
    from v2.environment.env_gym import GYM_ENVS

    _args = argparse.Namespace()
    # _args.env_name = CONTROL_SUITE_ENVS[0]
    _args.env_name = GYM_ENVS[0]
    _args.batch_size = 1
    _args.max_episode_length = 1000
    _args.state_observations = True

    _args.planning_horizon = 20
    _args.num_plan_iter = 5
    _args.num_plan_candidates = 20
    _args.num_plan_top_candidates = int(_args.num_plan_candidates // 1.3)

    _cem = CEM(_args)

    _env = build_env_from_args(_args)

    _env_clone = _env.clone()

    _env.reset()
    _env_clone.reset()

    _env_clone.set_state(_env.get_state())
    _env_clone.set_seed(_env.get_seed())

    with torch.no_grad():

        for _ in range(10000):
            # _env_clone.render()

            _action = _cem.plan(_env_clone)

            _env_clone.step(_action)

            _result = _env.step(_action)

            # print(_result[1])

        # _env.reset()
        # viewer.launch(_env._envs[0], policy=lambda t: _cem.plan(deepcopy(_env))[0])


