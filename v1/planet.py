import argparse

import torch


from v1.planner import CEM


class PlaNet(torch.nn.Module):

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.transition_model = ...  # TODO
        self.observation_model = ...  # TODO
        self.reward_model = ...  # TODO
        self.encoder = ...  # TODO

        self.planner = CEM(args)  # TODO

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parser or argparse.ArgumentParser('PlaNet')

        pass  # TODO

        return parser



