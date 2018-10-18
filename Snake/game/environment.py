import time, random
import numpy as np

from .entities import Snake, Field, CellType, SnakeDirection, SnakeAction, ALL_SNAKE_DIRECTIONS, ALL_SNAKE_ACTIONS

class Environment(object):
    """
    Represents the environment for Snake that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """
    def __init__(self, level_map=["#############","#.....S.....#","#############"]):
        self.field = Field(level_map=level_map)
        self.snake = None
        self.last_action = None
        self.fruit = None
        self.initial_snake_length = 1
        self.is_game_over = False
        self.stats = EpisodeStatistics()

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.stats.reset()
        self.field.create_level()

        self.snake = Snake(self.field.find_snake_head(), length=self.initial_snake_length)
        self.field.place_snake(self.snake)
        self.generate_fruit()
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=[0, 0, 0],
            is_episode_end=self.is_game_over
        )

        return result

    def get_observation(self):
        """ Observe the state of the environment. """

        def raycast(field, initial, increment, value=1.0):
            next_point = initial + increment
            try:
                as_next = field[next_point]
            except:
                return -value

            if as_next == CellType.WALL:
                return -value
            elif as_next == CellType.FRUIT:
                return value
            else:
                value = max(0.2, value - 0.1)
                return raycast(field, next_point, increment, value)

        right = raycast(self.field, self.snake.head, SnakeDirection.EAST)
        left = raycast(self.field, self.snake.head, SnakeDirection.WEST)

        return np.array([left, 0, right])

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """
        if action == SnakeAction.GO_LEFT and self.snake.direction != SnakeDirection.WEST:
            self.snake.change_direction()
        elif action == SnakeAction.GO_RIGHT and self.snake.direction != SnakeDirection.EAST:
            self.snake.change_direction()

    def timestep(self):
        """ Execute the timestep and return the new observable state. """
        self.stats.increment_timestep()

        old_head = self.snake.head
        old_tail = self.snake.tail
        is_going_east = 1 if self.snake.direction == SnakeDirection.EAST else -1

        # Get reward based on approximation from positive and negative objects
        reward = 0
        '''
        idxs = [i for i, x in enumerate(self.field._cells[0]) if x == CellType.WALL]
        closer_wall = idxs[min(range(len(idxs)), key=lambda i: abs(idxs[i] - self.snake.head.x))]
        dist_sw = abs(self.snake.head.x - closer_wall)
        if dist_sw <= 3:
            reward = -1*(dist_sw - abs(self.snake.peek_next_move().x - closer_wall))

        dist = lambda a, b: abs((a - b).x)
        dist_sf = dist(self.snake.head, self.fruit)
        if dist_sf <= 3:
            reward = .2*(dist_sf - dist(self.snake.peek_next_move(), self.fruit))
        '''

        # Are we about to eat the fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.stats.fruit_eaten()
            self.snake.move() #self.snake.grow()
            self.generate_fruit()
            reward = 1

        # If not, just move forward.
        else:
            self.snake.move()

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward = -1

        # Terminate episode after limit timestep reached.
        if self.stats.timesteps >= 1000:
            self.is_game_over = True

        result = TimestepResult(
            observation=self.get_observation(),
            reward=[-reward*is_going_east, 0, reward*is_going_east],
            is_episode_end=self.is_game_over
        )

        return result

    def generate_fruit(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snake.head] == CellType.WALL

    def has_hit_own_body(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.snake.head] == CellType.SNAKE_BODY

    def is_alive(self):
        """ True if the snake is still alive, False otherwise. """
        return not self.has_hit_wall() and not self.has_hit_own_body()


class Environment2D(Environment):
    """
    Represents the environment for Snake in 2D that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """
    def get_observation(self):
        """ Observe the state of the environment. """
        def raycast(field, initial, increment, value=1.0):
            next_point = initial + increment
            try:
                as_next = field[next_point]
            except:
                return -value

            if as_next == CellType.WALL or as_next == CellType.SNAKE_BODY:
                return -value
            elif as_next == CellType.FRUIT:
                return value
            else:
                value = max(0.2, value - 0.1)
                return raycast(field, next_point, increment, value)

        direction_idx = ALL_SNAKE_DIRECTIONS.index(self.snake.direction)
        left_direction = ALL_SNAKE_DIRECTIONS[(direction_idx - 1) % len(ALL_SNAKE_DIRECTIONS)]
        right_direction = ALL_SNAKE_DIRECTIONS[(direction_idx + 1) % len(ALL_SNAKE_DIRECTIONS)]

        front = raycast(self.field, self.snake.head, self.snake.direction)
        right = raycast(self.field, self.snake.head, right_direction)
        left = raycast(self.field, self.snake.head, left_direction)

        # Calculate distance from the snake head to the fruit
        head = self.snake.head
        fruit = self.fruit
        diff = fruit - head
        dir = self.snake.direction
        front_smell = max(0, diff.y * dir.y) + max(0, diff.x * dir.x)
        left_smell = max(0, diff.x * dir.y) + max(0, diff.y * -dir.x)
        right_smell = max(0, diff.x * -dir.y) + max(0, diff.y * dir.x)
        max_val = np.amax(np.array([left_smell, front_smell, right_smell]))

        # Scale distances so that the nearest has the highest value
        smells = [s if s == 0 else max_val - s + 1 for s in [left_smell, front_smell, right_smell]]
        p1norm = sum([left_smell, front_smell, right_smell]) + 0.0000001
        smells = np.array(smells) / (4 * p1norm)

        return np.concatenate([np.array([left, front, right]), smells])

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """
        self.last_action = action

        if action == SnakeAction.GO_LEFT:
            self.snake.change_direction(-1)
        elif action == SnakeAction.GO_RIGHT:
            self.snake.change_direction(1)

    def timestep(self):
        """ Execute the timestep and return the new observable state. """
        self.stats.increment_timestep()

        old_head = self.snake.head
        old_tail = self.snake.tail

        # Get reward based on approximation from positive and negative objects
        reward_value = 0

        # Are we about to eat the fruit?
        if self.snake.peek_next_move() == self.fruit:
            self.stats.fruit_eaten()
            self.snake.grow() #self.snake.move()
            self.generate_fruit()
            reward_value = 1

        # If not, just move forward.
        else:
            self.snake.move()

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            reward_value = -2

        # Terminate episode after limit timestep reached.
        if self.stats.timesteps >= 1000:
            self.is_game_over = True

        # apply award to last action and the negative reward divided by 2 to the other actions
        reward = np.full(3, -reward_value / 2.0)
        action_idx = ALL_SNAKE_ACTIONS.index(self.last_action)
        reward[action_idx] = reward_value

        # give reward to last action when fruit is in front of the snake head
        o = self.get_observation()
        if o[1] > 0:
            reward = [o[1], -o[1], -o[1] / 2.0] if self.last_action == SnakeAction.GO_LEFT else (
                [-o[1] / 2.0, -o[1], o[1]] if self.last_action == SnakeAction.GO_RIGHT else
                [-o[1] / 8.0, o[1] / 4.0, -o[1] / 8.0])
            '''
        # negative reward for last action if no fruit is collected in the last 100 timesteps
        if reward == 0 and (self.stats.timesteps - self.last_fruit) > 100:
            value = ((self.stats.timesteps - self.last_fruit) - 100) / 50
            r = [-value, value / 2.0, value / 2.0] if self.last_action == "left" else (
                [value / 2.0, value / 2.0, -value] if self.last_action == "right" else [0, 0, 0])
        '''
        # noise reward in 1% of the cases
        if random.randint(1, 100) == 1:
            reward = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]

        return TimestepResult(observation=self.get_observation(), reward=reward, is_episode_end=self.is_game_over)


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return '%s\nR = %d  end=%d\n' % (field_map, self.reward, self.is_episode_end)


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.timesteps = 0
        self.fruits = 0

    def increment_timestep(self):
        self.timesteps += 1
        
    def fruit_eaten(self):
        self.fruits += 1
