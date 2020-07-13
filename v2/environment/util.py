import argparse

from v2.environment.env import Environment
from v2.environment.env_gym import GYM_ENVS, GymEnv
from v2.environment.env_suite import CONTROL_SUITE_ENVS, ControlSuiteEnvironment


def build_env_from_args(args: argparse.Namespace) -> Environment:
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
