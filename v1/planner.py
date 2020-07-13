
import argparse

import torch


class CEM:

    def __init__(self, dynamics_model: callable, action_size: int, args: argparse.Namespace):

        self.num_optimization_iters = args.cem_num_optimization_iters
        self.num_candidates = args.cem_num_candidates
        self.num_top_candidates = args.cem_num_top_candidates
        self.planning_horizon = args.cem_planning_horizon

        self.action_size = action_size


    def plan_action(self) -> torch.Tensor:

        action_mean = torch.zeros(self.planning_horizon)  # TODO -- dims



        pass  # TODO

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser('Cross-Entropy Method Planner')

        parser.add_argument('--cem_num_optimization_iters',
                            type=int,
                            default=-1,  # TODO
                            help='')  # TODO
        parser.add_argument('--cem_num_candidates',
                            type=int,
                            default=-1,  # TODO
                            help='')  # TODO
        parser.add_argument('--cem_num_top_candidates',
                            type=int,
                            default=-1,
                            help='')  # TODO
        parser.add_argument('--cem_planning_horizon',
                            type=int,
                            default=-1,
                            help='')  # TODO

        return parser

