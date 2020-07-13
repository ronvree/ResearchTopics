
import argparse
from copy import deepcopy

import torch

from v1.environment import EnvironmentBatcher


class CEM:

    def __init__(self, args: argparse.Namespace):
        assert args.H > 0
        assert args.I > 0
        assert args.J > 1
        assert args.K > 1
        assert args.J >= args.K

        self.horizon_distance = args.H
        self.num_iter = args.I
        self.num_candidates = args.J
        self.num_top_candidates = args.K

    def plan(self,
             env: EnvironmentBatcher,
             device=torch.device('cpu')
             ):

        # TODO -- this assumes total observability
        init_states = env.get_states()

        # Define size variables
        batch_size = env.num_environments
        action_shape = env.action_shape

        # Initialize action belief parameters
        param_shape = (batch_size, self.horizon_distance) + action_shape
        mean = torch.zeros(param_shape, device=device)
        std = torch.ones(param_shape, device=device)

        # Optimize the action belief parameters
        for i in range(self.num_iter):
            # Sample all action sequences of all candidates from the current action belief
            batch_J_H_actions = self._sample_action_candidates(mean, std)  # Shape: (bs, J, H) + action_shape

            # Evaluate all candidate action sequences. Store the return for each candidate
            candidate_returns = [None] * self.num_candidates
            for j, batch_H_actions in zip(range(self.num_candidates), batch_J_H_actions.split(1, dim=1)):  # TODO -- this can just be enumerate

                # Remove the redundant candidate dimension of the single candidate action sequence (batch)
                # Shape: (bs, H) + action_shape
                batch_H_actions = batch_H_actions.view(batch_size, self.horizon_distance, *action_shape)

                # Evaluate the sequence batch on the environment. Store the returns
                returns = torch.zeros(batch_size, 1)
                for tau, batch_actions in zip(range(self.horizon_distance), batch_H_actions.split(1, dim=1)):  # TODO -- this can just be enumerate
                    # Remove the redundant sequence dimension of the single action (batch)
                    # Shape: (bs,) + action_shape
                    batch_actions = batch_actions.view(batch_size, *action_shape)
                    # Execute the actions in the environment
                    _, rs, _, _ = env.step(batch_actions)
                    # Add the return
                    returns += rs.view(batch_size, 1)

                # Store the return of this candidate
                candidate_returns[j] = returns

                # Reset the environment for the next candidate batch
                env.set_states(init_states)

            # Transform the return sequence to a tensor
            returns = torch.cat(candidate_returns, dim=1)  # Shape: (batch_size, num_candidates)

            # Select the top K candidate sequences based on the rewards obtained
            top_candidate_ixs = torch.argsort(returns, dim=1)[:, -self.num_top_candidates:]  # Shape: (batch_size, num_top_candidates)

            # Iterate through each item in the batch
            new_mean = [None] * batch_size
            new_std = [None] * batch_size
            for batch_i, (candidates, top_ixs) in enumerate(zip(batch_J_H_actions, top_candidate_ixs)):
                top_candidates = candidates.index_select(0, top_ixs)  # Shape: (num_top_candidates, horizon_distance) + action_shape

                sample_mean = torch.sum(top_candidates, dim=0) / self.num_top_candidates
                sample_std = torch.sum(torch.abs(top_candidates - sample_mean), dim=0) / (self.num_top_candidates - 1)

                new_mean[batch_i] = sample_mean.unsqueeze(0)
                new_std[batch_i] = sample_std.unsqueeze(0)

            mean = torch.cat(new_mean, dim=0)
            std = torch.cat(new_std, dim=0)

        return mean[:, 0]

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

        parser.add_argument('--H',
                            type=int,
                            default=1,
                            help='Planning horizon distance')
        parser.add_argument('--I',
                            type=int,
                            default=1,
                            help='Optimization iterations')
        parser.add_argument('--J',
                            type=int,
                            default=1,
                            help='Candidates per iteration')
        parser.add_argument('--K',
                            type=int,
                            default=1,
                            help='Number of top candidates to fit')

        return parser


if __name__ == '__main__':
    from v1.environment import EnvironmentBatcher
    from v1.environment.env_gym import GYM_ENVS

    _args = argparse.Namespace()
    _args.env_name = GYM_ENVS[0]
    _args.num_env = 1
    _args.disable_cuda = True

    _args.H = 20
    _args.I = 5
    _args.J = 40
    _args.K = int(_args.J // 1.3)

    # _env = GymEnv(_args)

    _envs = EnvironmentBatcher(_args)

    _envs_copy = deepcopy(_envs)
    _envs_copy.reset()

    _cem = CEM(_args)

    _envs.reset()
    for _i in range(10000):
        _actions = _cem.plan(_envs.clone())

        _observations, _rewards, _terminals, _infos = _envs.step(_actions)

        print(_rewards)

        _envs_copy.step(_actions)
        for _env in _envs_copy._envs:
            _env.render()



    print(_cem.plan(_envs))



