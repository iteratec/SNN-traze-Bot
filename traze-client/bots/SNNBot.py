import time
import random
import os, sys
import snn.parameters as params

from traze.bot import Action, BotBase
from traze.client import World
from snn.agent import SNNAgent


class SNNBot(BotBase):
    def __init__(self, game, name="SLab-ML Muenchen"):
        super(SNNBot, self).__init__(game, name)
        self.agent = SNNAgent(verbose=0)
        self.reset_bot()

    def reset_bot(self):
        self._lastAction = None
        self._nextAction = None
        self._reward = [0, 0, 0]
        self._last_position = [0, 0]

    def next_action(self, actions):
        def raycast(field, initial, action, value=1.0):
            next_point = [initial[0] + action.dX, initial[1] + action.dY]

            if not self.game.grid.valid(next_point[0], next_point[1]):
                return -value
            else:
                value = max(0.1, value - 0.05)
                return raycast(field, next_point, action, value)

        # this method is called more than once per step, therefore it needs to be returned after first call
        if self.x == self._last_position[0] and self.y == self._last_position[1]:
            return self._lastAction

        self._last_position = [self.x, self.y]
        output = None
        if self._lastAction in list(Action):
            direction_idx = self._lastAction.index
            left_direction_idx = (direction_idx - 1) % len(list(Action))
            right_direction_idx = (direction_idx + 1) % len(list(Action))

            front = raycast(self.game.grid.tiles, (self.x, self.y), self._lastAction)
            right = raycast(self.game.grid.tiles, (self.x, self.y), list(Action)[right_direction_idx])
            left = raycast(self.game.grid.tiles, (self.x, self.y), list(Action)[left_direction_idx])

            output = self.agent.act([left, front, right], self._reward)
            new_direction_idx = (direction_idx + output) % len(list(Action))
            self._nextAction = list(Action)[new_direction_idx]
            self._reward = [0, 0, 0]

        if not actions:
            self._lastAction = self._nextAction
            return self._nextAction

        if self._nextAction not in actions:
            if output is not None:
                # Penalize for wrong decision
                self._reward = [0.5, 0.5, 0.5]
                self._reward[output + 1] = -1
            else:
                self._nextAction = random.choice(tuple(actions))
        else:
            new_front = raycast(self.game.grid.tiles, [self.x, self.y], self._nextAction)
            # Reward if first object forward after action is further away than before or if the action was maintain direction reward if the object forward is furtheraway than objects left and right
            if new_front > front:
                self._reward = [-0.1, -0.1, -0.1]
                self._reward[output + 1] = 0.2
            elif front > left and front > right and output == 0:
                self._reward = [-0.05, -0.05, -0.05]
                self._reward[output + 1] = 0.1

        self._lastAction = self._nextAction
        return self._nextAction

    def play(self, max_time):
        start_time = time.time()
        i = 1
        while time.time() - start_time < max_time:
            self.agent.begin_episode()
            self.join()
            print("start game", i)

            # wait for death
            while self.alive:
                time.sleep(0.5)
            self.agent.end_episode(self._reward)
            self.reset_bot()
            print("end game", i)
            i += 1

        self.agent.reset_agent()
        return self

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python %s <minutes_until_reset> [bot_name]" % sys.argv[0])
        exit(1)
    weights_path = params.default_dir + params.weights_file
    while True:
        if os.path.exists(weights_path):
            os.remove(weights_path)
            print("Deleted weights file.")
        SNNBot(World().games[0], *sys.argv[2:3]).play(int(sys.argv[1]) * 60)
