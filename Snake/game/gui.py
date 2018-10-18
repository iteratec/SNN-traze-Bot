import numpy as np
import pygame

from .agent import HumanAgent
from .entities import CellType, SnakeDirection, SnakeAction, ALL_SNAKE_ACTIONS

class PyGameGUI:
    """ Provides a Snake GUI powered by Pygame. """

    FPS_LIMIT = 15
    TIMESTEP_MACHINE = 120
    TIMESTEP_HUMAN = 800
    CELL_SIZE = 50
    CAPTION = "Snake meets SNN!"

    SNAKE_CONTROL_KEYS = [
        pygame.K_SPACE,
        pygame.K_LEFT,
        pygame.K_RIGHT
    ]

    def __init__(self):
        pygame.init()
        self.agent = None
        self.env = None
        self.screen = None
        self.running = True
        self.fps_clock = None
        self.timestep_watch = Stopwatch()

    def load_environment(self, env):
        """ Load the environment into the GUI. """
        self.env = env
        screen_size = (self.env.field.width * self.CELL_SIZE, self.env.field.height * self.CELL_SIZE)
        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        pygame.display.set_caption(self.CAPTION)

    def load_agent(self, agent):
        """ Load the RL agent into the GUI. """
        self.agent = agent

    def render_cell(self, x, y):
        """ Draw the cell specified by the field coordinates. """
        cell_coords = pygame.Rect(
            x * self.CELL_SIZE,
            y * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        if self.env.field[x, y] == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        else:
            color = Colors.CELL_TYPE[self.env.field[x, y]]
            pygame.draw.rect(self.screen, color, cell_coords, 1)

            internal_padding = self.CELL_SIZE // 6 * 2
            internal_square_coords = cell_coords.inflate((-internal_padding, -internal_padding))
            pygame.draw.rect(self.screen, color, internal_square_coords)

    def map_key_to_snake_action(self, key):
        """ Convert a keystroke to an environment action. """
        if key == self.SNAKE_CONTROL_KEYS[0]:
            self.env.snake.change_direction()
        elif key == self.SNAKE_CONTROL_KEYS[1] and self.env.snake.direction != SnakeDirection.WEST:
            self.env.snake.change_direction()
        elif key == self.SNAKE_CONTROL_KEYS[2] and self.env.snake.direction != SnakeDirection.EAST:
            self.env.snake.change_direction()

    def render(self):
        """ Draw the entire game frame. """
        for x in range(self.env.field.width):
            for y in range(self.env.field.height):
                self.render_cell(x, y)

    def run(self, num_episodes=1):
        """ Run the GUI player for the specified number of episodes. """
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()
        self.running = True

        for episode in range(num_episodes):
            if not self.running:
                break

            self.run_episode()
            print('Episode [%d/%d] - Fruits: %d' % (episode + 1, num_episodes, self.env.stats.fruits))
            pygame.time.wait(1500)

    def run_episode(self):
        """ Run the GUI player for a single episode. """

        # Initialize the environment.
        self.timestep_watch.reset()
        timestep_result = self.env.new_episode()
        self.agent.begin_episode()
        is_human_agent = isinstance(self.agent, HumanAgent)

        # Main game loop.
        game_over = False
        while self.running and not game_over:
            action = SnakeAction.MAINTAIN_DIRECTION

            # Handle events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if is_human_agent and event.key in self.SNAKE_CONTROL_KEYS:
                        self.map_key_to_snake_action(event.key)
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                if event.type == pygame.QUIT:
                    self.running = False

            # Update game state.
            timestep_timed_out = self.timestep_watch.time() >= (self.TIMESTEP_HUMAN if is_human_agent else self.TIMESTEP_MACHINE)
            human_made_move = is_human_agent and action != SnakeAction.MAINTAIN_DIRECTION

            if timestep_timed_out or human_made_move:
                self.timestep_watch.reset()

                if not is_human_agent:
                    action = self.agent.act(timestep_result.observation, timestep_result.reward)

                self.env.choose_action(action)
                timestep_result = self.env.timestep()

                if timestep_result.is_episode_end:
                    self.agent.end_episode(timestep_result.observation, timestep_result.reward)
                    game_over = True

            # Render.
            self.render()
            pygame.display.set_caption(self.CAPTION + '  [Score: %d]' % self.env.stats.fruits)
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)

class Stopwatch(object):
    """ Measures the time elapsed since the last checkpoint. """

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        """ Set a new checkpoint. """
        self.start_time = pygame.time.get_ticks()

    def time(self):
        """ Get time (in milliseconds) since the last checkpoint. """
        return pygame.time.get_ticks() - self.start_time

class Colors:

    SCREEN_BACKGROUND = (170, 204, 153)
    CELL_TYPE = {
        CellType.WALL: (56, 56, 56),
        CellType.SNAKE_BODY: (105, 132, 164),
        CellType.SNAKE_HEAD: (122, 154, 191),
        CellType.FRUIT: (173, 52, 80),
    }