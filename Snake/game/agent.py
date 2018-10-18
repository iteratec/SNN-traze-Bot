import random
import numpy as np

from .entities import CellType, SnakeAction, ALL_SNAKE_ACTIONS
from .utils import print_me, AgentStatistics
from .snn import snn as ai

class AgentBase(object):
    """ Represents an intelligent agent for the Snake environment. """

    def __init__(self, *args):
        pass

    def begin_episode(self, *args):
        """ Reset the agent for a new episode. """
        pass

    def act(self, observation, reward, *args):
        """
        Choose the next action to take.

        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        return None

    def end_episode(self, *args):
        """ Notify the agent that the episode has ended. """
        pass


class RandomAgent(AgentBase):
    """ Represents a snake agent that takes a random action at every step. """

    def act(self, observation, reward):
        return random.choice(ALL_SNAKE_ACTIONS)


class HumanAgent(AgentBase):
    """ Represents an agent that is operated by a human. """

    def act(self, observation, reward):
        # Do nothing by default. If the human player presses a keystroke,
        # this will be overridden on the GUI.
        return SnakeAction.MAINTAIN_DIRECTION


class SNNAgent(AgentBase):
    """ Represents a snake agent which actions come from a SNN. """

    def __init__(self, model=None, verbose=1):
        self.model = model
        self.verbose = verbose
        self.reset_agent()

    def begin_episode(self):
        self.w = self.snn.try_restore_model(self.model)
        self.stats.append(self.w)
        #self.snn.set_weights(np.array([3000.,3000.,0,0,0,0]), np.array([0,0,3000.,3000.,0,0]), np.array([0,0,0,0,3000.,3000.]))

    def set_reward(self, reward):
        self.rewards.append(reward)

        N = 1 # Number of element used in the rolling average
        reward = [float(sum(y)) / N for y in zip(*self.rewards[-N:])]
        self.snn.set_reward(reward)

        self.snn.reset_neurons()
        ai.nest_simulate()
        self.w = self.snn.get_results()[1]

        return reward

    def reset_agent(self):
        self.w = []
        self.rewards = []
        self.snn = ai.SnakeSNN()
        self.stats = AgentStatistics()

    def prepare_input(self, inputs):
        # Reflect negative values into positive ones onto the opposite sensor side and normalize them
        # E.g. [-0.3, 0, 0.7] => [0, 0, 0.7 - (-0.3)]/2 => [0, 0, 0.5]
        reflect = lambda a, b: max(0, (a - min(0, b))) / 2
        inputs = [(-(v * v) if v < 0 else v) for v in inputs]

        return [reflect(inputs[0], inputs[2]), 0, reflect(inputs[2], inputs[0])]

    def prepare_output(self, output):
        idx = np.where(output == max(output))[0]

        return ALL_SNAKE_ACTIONS[random.choice(idx)]

    def act(self, observation, reward):
        reward = self.set_reward(reward)

        observation = self.prepare_input(observation)
        self.snn.set_input(observation)

        self.snn.reset_neurons()
        ai.nest_simulate()

        output, self.w = self.snn.get_results()
        self.stats.append(self.w)

        if self.verbose > 0:
            print('Rew: %s\n---' % (print_me(reward, '+.1f')))
            print('Inp: %s, Out: %s' % (print_me(observation, '.2f'), print_me(output, '.2f')))
            print('W_L-F-R: %s-%s-%s' % (print_me(self.w[0], '4.0f'), print_me(self.w[1], '4.0f'), print_me(self.w[2], '4.0f')))

        output = self.prepare_output(output)
        return output

    def end_episode(self, observation, reward):
        reward = self.set_reward(reward)

        if self.verbose > 0:
            print('Rew: %s' % (print_me(reward, '+.1f')))

        self.snn.save_model(self.w)
        return self.w


class SNNAgent2D(SNNAgent):
    """ Represents a snake agent in 2D which actions come from a SNN. """
    def prepare_input(self, inputs):
        # Reflect negative values into positive ones onto the oposite sensor side and normalize them
        # E.g. [-0.3, 0.7] => [0, 0.7 - (-0.3)]/2 => [0, 0.5]
        reflect = lambda a, b: max(0, (a - min(0, b))) / 2
        reflect2 = lambda a, b, c: max(0, (2 * a - (min(0, b) + min(0, c)))) / 4

        return [reflect(inputs[0], inputs[1]), inputs[3], reflect2(inputs[1], inputs[0], inputs[2]), inputs[4], reflect(inputs[2], inputs[1]), inputs[5]]
