#!/usr/bin/env python

import sys
import numpy as np
from game.utils import print_me
import os

game2D = False

def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--agent',
        type=str,
        default="snn",
        choices=['human', 'random', 'snn'],
        help='Player agent to use.',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--fast-train',
        action='store_true',
        help='Disables GUI for fast training.',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Disables GUI for fast training.',
    )
    parser.add_argument(
        '--two-d',
        action='store_true',
        help='Import 2D game.',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot agent training statistics.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for random events.',
    )

    return parser.parse_args(args)


def create_snake_environment():
    """ Create a new Snake environment. """

    from game.environment import Environment, Environment2D

    global game2D
    if game2D:
        return Environment2D(level_map=["#############", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#.....S.....#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#...........#", 
                                        "#############"])
    else:
        return Environment()

def create_agent(name, model=None, verbose=0):
    """
    Create a specific type of Snake AI agent.

    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model.

    Returns:
        An instance of Snake agent.
    """

    from game.agent import HumanAgent, RandomAgent, SNNAgent, SNNAgent2D

    if name == 'human':
        return HumanAgent()
    elif name == 'random':
        return RandomAgent()
    elif name == 'snn':
        global game2D
        return (SNNAgent2D if game2D else SNNAgent)(model,verbose)

    raise KeyError('Unknown agent type: %s' % name)


def play_cli(env, agent, num_episodes=10):
    stats = []

    print('Playing:')

    for episode in range(num_episodes):
        timestep = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            action = agent.act(timestep.observation, timestep.reward)
            env.choose_action(action)
            timestep = env.timestep()
            game_over = timestep.is_episode_end
        agent.end_episode(timestep.observation, timestep.reward)

        stats.append([env.stats.fruits, env.stats.timesteps])

        summary = 'Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:3d}'
        print(summary.format(episode + 1, num_episodes, stats[-1][1], stats[-1][0]))

    fruits = [stat[0] for stat in stats]
    print('Fruits eaten: {:.1f} +/- stddev {:.1f}'.format(np.mean(fruits), np.std(fruits)))
    print('Fruits per 100 timesteps: {:.1f} '.format(np.mean([100 * stat[0] / stat[1] for stat in stats])))


def test(env, agent, num_episodes=10, num_runs=10):
    means = []
    print('Testing:')

    for run in range(num_runs):
        if os.path.isfile("./game/snn/weights.h5"):
            os.remove("./game/snn/weights.h5")
        agent.reset_agent()
        stats = []
        for episode in range(num_episodes):
            timestep = env.new_episode()
            agent.begin_episode()
            game_over = False

            while not game_over:
                action = agent.act(timestep.observation, timestep.reward)
                env.choose_action(action)
                timestep = env.timestep()
                game_over = timestep.is_episode_end
            weights = agent.end_episode(timestep.observation, timestep.reward)

            stats.append([env.stats.fruits, env.stats.timesteps])

            summary = 'Run {:3d} / {:3d} | Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:3d}'
            print(summary.format(run + 1, num_runs, episode + 1, num_episodes, stats[-1][1], stats[-1][0]))

        fruits = [stat[0] for stat in stats]
        means.append(np.mean(fruits))
        print('Run {:3d} / {:3d}'.format(run + 1, num_runs))
        print('Fruits eaten: {:.1f} +/- stddev {:.1f}'.format(np.mean(fruits), np.std(fruits)))
        print('Fruits per 100 timesteps: {:.1f} '.format(np.mean([100 * stat[0] / stat[1] for stat in stats])))
        print('W_L-F-R: %s-%s-%s' % (print_me(weights[0], '4.0f'), print_me(weights[1], '4.0f'), print_me(weights[2], '4.0f')))

    print('============================\nFinished running. Results:')
    print('After {:3d} runs and {:3d} episodes: Fruits eaten: {:.1f} +/- stddev {:.1f}'.format(num_runs, num_episodes, np.mean(means), np.std(means)))


def play_gui(env, agent, num_episodes):
    """
    Play using the specified Snake agent and the interactive graphical interface.

    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """
    from game.gui import PyGameGUI

    gui = PyGameGUI()
    gui.load_environment(env)
    gui.load_agent(agent)
    gui.run(num_episodes=num_episodes)


def use_seed(self, value):
    """ Initialize the random state to make results reproducible. """
    random.seed(value)
    np.random.seed(value)


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    if parsed_args.seed is not None:
        use_seed(parsed_args.seed)
        print("App is using seed: %d" % parsed_args.seed)

    if parsed_args.two_d:
        global game2D
        game2D = True

    env = create_snake_environment()
    agent = create_agent(parsed_args.agent, parsed_args.model, not (parsed_args.fast_train or parsed_args.test))

    if parsed_args.test:
        test(env, agent, num_episodes=parsed_args.num_episodes, num_runs=parsed_args.num_runs)
    else:
        (play_cli if parsed_args.fast_train else play_gui)(env, agent, num_episodes=parsed_args.num_episodes)

    if parsed_args.plot:
        agent.stats.plot()

if __name__ == '__main__':
    main()