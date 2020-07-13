
import os
import cv2
import argparse
import tqdm

import torch

from v2.model import QModel
from v2.planner.cem import CEM, AdaptedCEM
from v2.util.args import optimizer_from_args, save_args
from v2.util.episode import Episode
from v2.util.experience_replay import ExperienceReplay
from v2.environment.util import build_env_from_args
from v2.environment.env import Environment
from v2.trainer import Trainer


# TODO -- colab sync
# TODO -- cuda compatibility
# TODO -- more modular, specify planner in args

# TODO -- different batch size when training?
from v2.util.logging import Log


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Adapted PlaNet')

    parser.add_argument('--batch_size',  # TODO -- trainer arg?
                        type=int,
                        default=8,
                        help='Number of data points per batch')
    parser.add_argument('--disable_cuda',
                        action='set_true')  # TODO
    parser.add_argument('--planner',
                        type=str,
                        default='adapted_cem',
                        help='Select which planner to use')

    # Optimizer args
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='Optimizer type')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Optimizer learning rate')

    parser.add_argument('--num_train_iter',  # referred to as 'Collect Interval' or 'C' in the original paper
                        type=int,  # TODO -- change to number of passes over the dataset?
                        default=1000,  # TODO
                        help='Number of data sequences sampled to train the model in each iteration')
    parser.add_argument('--num_seed_episodes',
                        type=int,
                        default=50,  # TODO
                        help='Number of episodes generated with a random policy to create an initial experience dataset'
                        )
    parser.add_argument('--num_data_episodes',  # TODO -- batch size in data collection phase
                        type=int,
                        default=1,
                        help='The number of episodes of data that are collected during the data collection phase')

    # Add arguments for building an environment
    Environment.build_argument_parser(parser)
    # Add arguments for building the dataset
    ExperienceReplay.build_argument_parser(parser)
    # Add planner arguments
    CEM.build_argument_parser(parser)
    # Add log args
    Log.build_argument_parser(parser)

    return parser


def get_args():
    args = build_argument_parser().parse_args()
    return args


def generate_random_episode_batch(env: Environment) -> list:
    """
    Generate a random trajectory in the environment using a uniformly distributed policy

    TODO -- comment
    :param env: The environment in which the trajectory should be generated
    :return:
    """
    # Reset the environment
    observation, _, _ = env.reset()
    # Get the batch size
    batch_size = observation.size(0)
    # Store the episode data in a list
    data = [observation]
    # Generate the trajectory
    env_terminated = torch.zeros(batch_size).bool()
    while not all(env_terminated):
        # Sample a random action from the environment action space
        action = env.sample_random_action()
        # Obtain o_{t+1}, r_{t+1}, f_{t+1} by executing a_t in the environment
        observation, reward, flag, info = env.step(action)
        # Extend the episode using the obtained information
        data.extend([action, reward, flag, observation])
        # Set environment termination flags
        env_terminated = flag
    # Return the data as episodes
    return data_to_episodes(data)


def data_to_episodes(data: list) -> list:
    """
    TODO
    :param data:
    :return:
    """
    if len(data) == 0:
        return list()
    # Get the batch size
    batch_size = data[0].size(0)
    # Convert the data to Episode objects
    episodes = list()
    # Data is stored as:
    # o_0, a_0, r_1, f_1, o_1, a_1, r_2, ... ,a_{T-1}, r_T, f_T, o_T
    # Only o_t, a_t, r_{t+1} are relevant
    # For each episode in the batch the termination flags need to be checked
    for i in range(batch_size):
        episode = Episode()
        for j in range(0, len(data), 4):
            # Get the relevant subsequence of the data
            seq = data[j:j+4]
            # Check whether the end of the data has been reached
            if len(seq) == 4:
                o, a, r, f = seq
                # Get the data relevant to this batch index
                o, a, r, f = o[i], a[i], r[i], f[i]
                # Add the data to the episode
                episode.append_all(o, a, r)
                # Check if the episode terminated. If so, add final observation and go to next episode
                if f:
                    # Final observation is the first entry after this sequence
                    o = data[j+4][i]
                    episode.append(o)
                    break
            else:  # Procedure only enters this code block if the episode data is incomplete (termination flag not set)
                # Subsequence only contains the final observation
                o = seq[i]
                # Append the final observation
                episode.append(o)
        # Add the episodes to the collection
        episodes.append(episode)
    # Return all episode objects
    return episodes


def init_data(dataset: ExperienceReplay,
              env: Environment,
              args: argparse.Namespace) -> tuple:

    info = dict()

    num_iters = args.num_seed_episodes // args.batch_size
    iters = tqdm.tqdm(range(num_iters),
                      total=num_iters,
                      desc='Generating random episodes')
    iters.set_postfix_str(f'completed 0/{num_iters * args.batch_size}')

    with torch.no_grad():
        for i in iters:
            # Generate a random episode batch
            episodes = generate_random_episode_batch(env)
            # Add the episodes to the dataset
            dataset.append_episodes(episodes)

            # Update progress bar
            iters.set_postfix_str(f'completed {(i + 1) * args.batch_size}/{num_iters * args.batch_size}')

    return dataset, info


class RandomPlanner:  # For debugging purposes

    def plan(self, env):
        return env.sample_random_action()


def run(args: argparse.Namespace):
    args = args or get_args()

    # Create a log
    log = Log(args)

    # Log the args/hyperparameters
    log.log_args(args)

    # Build the environment
    env = build_env_from_args(args)
    # Build an additional environment for planning
    env_planning = env.clone()

    # Initialize the dataset
    dataset = ExperienceReplay(args)

    # Add random seed episodes to the dataset
    log.log_message('Generating random episodes')
    init_data(dataset, env, args)

    # Initialize the model
    model = QModel(env.observation_shape,
                   env.action_shape,
                   args)

    # Build the model optimizer
    optimizer = optimizer_from_args(model.parameters(), args)  # TODO -- move optimizer to trainer?

    # Create a Trainer object for training the model
    trainer = Trainer(dataset, args)

    # Initialize the planner
    if args.planner.lower() == 'cem':
        planner = CEM(args)
    if args.planner.lower() == 'adapted_cem':
        planner = AdaptedCEM(model, args)
    if args.planner.lower() == 'debug':
        planner = RandomPlanner()

    # Start main loop
    log.log_message('Starting main loop')
    converged = False
    while not converged:

        log.log_message('Training the model')
        # Fit the model to the collected experience
        _, train_info = trainer.train(model,
                                      optimizer,
                                      batch_size=args.batch_size,
                                      num_iters=args.num_train_iter)
        # TODO -- log train info, -- avg loss

        # Collect new data using MPC
        log.log_message('Collecting data')
        with torch.no_grad():
            # Reset the environment, obtain the initial observation
            observation, env_terminated, info = env.reset()
            # Ensure the environment and planning environment are equivalent
            env_planning.reset()
            env_planning.set_state(env.get_state())
            env_planning.set_seed(env.get_seed())

            # TODO -- log env seed

            # Execute an episode (batch) in the environment
            episode_data = [observation]
            while not all(env_terminated):
                # Plan action a_t in an environment model
                action = planner.plan(env_planning)
                # Obtain o_{t+1}, r_{t+1}, f_{t+1} by executing a_t in the environment
                observation, reward, env_terminated, info = env.step(action)
                # Perform the action in the planning environment as well
                env_planning.step(action)

                print(reward)

                # Extend the episode using the obtained information
                episode_data.extend([action, reward, env_terminated, observation])

                cv2.imshow('Test', info[0]['pixels'])
                # cv2.waitKey(0)

            # Append the collected data to the dataset
            dataset.append_episodes(data_to_episodes(episode_data))

            # Check whether the algorithm should terminate
            converged = False  # TODO -- stop criterion


if __name__ == '__main__':

    from v2.environment.util import build_env_from_args
    from v2.environment.env_suite import CONTROL_SUITE_ENVS
    from v2.environment.env_gym import GYM_ENVS

    _args = argparse.Namespace()

    _args.log_directory = './test_log'
    _args.print_log = True

    _args.disable_cuda = True
    _args.optimizer = 'adam'
    _args.learning_rate = 1e-3
    _args.planner = 'adapted_cem'
    # _args.planner = 'cem'

    _args.num_train_iter = 1
    _args.num_seed_episodes = 2  # 30

    _args.experience_buffer_size = 10000

    # _args.env_name = CONTROL_SUITE_ENVS[1]
    _args.env_name = GYM_ENVS[0]
    _args.batch_size = 1
    _args.max_episode_length = 100
    _args.state_observations = False

    _args.planning_horizon = 20
    _args.num_plan_iter = 5
    _args.num_plan_candidates = 40
    _args.num_plan_top_candidates = int(_args.num_plan_candidates // 1.3)

    print(_args.env_name)

    run(_args)


