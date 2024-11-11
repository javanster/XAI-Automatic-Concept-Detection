from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from .entity import Entity
import pygame
import random


class MangoRun(Env):

    metadata = {"render_modes": [
        "human", "rgb_array"], "render_fps": 4}

    def __init__(
            self,
            render_mode=None,
            render_fps=4,
    ):
        super().__init__()
        self.STEP_PENALTY = -0.005
        self.UNRIPE_MANGO_REWARD = 1
        self.RIPE_MANGO_REWARD = 1.025
        self.reward_range = (-1, 1)

        self.RIPE_MANGO_COLOR = (255, 215, 0)  # Gold
        self.AGENT_COLOR = (78, 172, 248)    # Blue
        self.UNRIPE_MANGO_COLOR = (255, 0, 0)   # Red
        self.WALL_COLOR = (0, 0, 0)          # Black
        self.FLOOR_COLOR = (200, 200, 200)   # Light gray

        self.render_fps = render_fps
        self.grid_side_length = 6
        self.action_space = Discrete(4)
        self.action_dict = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }
        self.observation_space = Box(
            low=0, high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.window_size = 600
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.wall_map = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ])

        self.agent_spawn_locations = [(0, 0), (0, 5), (5, 0), (5, 5)]

        self.walls = []

        for y in range(self.grid_side_length):
            for x in range(self.grid_side_length):
                map_value = self.wall_map[y, x]
                if map_value == 1:
                    self.walls.append(Entity(
                        grid_side_length=self.grid_side_length,
                        starting_position=(x, y)
                    ))

    def reset(
            self,
            seed=None,
            options=None,
            agent_starting_position=None,
    ):
        super().reset(seed=seed)

        agent_start_position = agent_starting_position if agent_starting_position else self.agent_spawn_locations[random.randint(
            0, 3)]

        self.agent = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=agent_start_position
        )

        self.ripe_mango_1 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(2, 3)
        )

        self.ripe_mango_2 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(3, 3)
        )

        self.unripe_mango_1 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(2, 1)
        )

        self.unripe_mango_2 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(3, 4)
        )

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self):
        obs = np.full(
            (self.grid_side_length, self.grid_side_length, 3),
            self.FLOOR_COLOR,
            dtype=np.uint8
        )

        for wall in self.walls:
            obs[wall.y, wall.x] = self.WALL_COLOR

        obs[self.ripe_mango_1.y, self.ripe_mango_1.x] = self.RIPE_MANGO_COLOR
        obs[self.ripe_mango_2.y, self.ripe_mango_2.x] = self.RIPE_MANGO_COLOR
        obs[self.unripe_mango_1.y, self.unripe_mango_1.x] = self.UNRIPE_MANGO_COLOR
        obs[self.unripe_mango_2.y, self.unripe_mango_2.x] = self.UNRIPE_MANGO_COLOR

        obs[self.agent.y, self.agent.x] = self.AGENT_COLOR

        return obs

    def step(self, action):
        self.episode_step += 1

        # Performs action ONLY if wall not in the way
        agent_x, agent_y = self.agent.x, self.agent.y
        self.agent.action(action)
        if any(self.agent == wall for wall in self.walls):
            self.agent.x, self.agent.y = agent_x, agent_y

        new_observation = self._get_obs()

        terminated = False
        truncated = False

        reward = self.STEP_PENALTY

        if self.agent == self.unripe_mango_1 or self.agent == self.unripe_mango_2:
            reward += self.UNRIPE_MANGO_REWARD
            terminated = True

        elif self.agent == self.ripe_mango_1 or self.agent == self.ripe_mango_2:
            reward += self.RIPE_MANGO_REWARD
            terminated = True

        if self.episode_step >= 200:
            truncated = True

        info = {}

        return new_observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._draw_entities()
        elif self.render_mode == "rgb_array":
            pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _draw_entities(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.FLOOR_COLOR)
        pix_square_size = self.window_size / self.grid_side_length

        # Drawing the walls
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                self.WALL_COLOR,
                pygame.Rect(
                    pix_square_size * wall.x,
                    pix_square_size * wall.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        # Drawing the unripe mangos
        for unripe_mango in [self.unripe_mango_1, self.unripe_mango_2]:
            pygame.draw.rect(
                canvas,
                self.UNRIPE_MANGO_COLOR,
                pygame.Rect(
                    pix_square_size * unripe_mango.x,
                    pix_square_size * unripe_mango.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        # Drawing the ripe mangos
        for ripe_mango in [self.ripe_mango_1, self.ripe_mango_2]:
            pygame.draw.rect(
                canvas,
                self.RIPE_MANGO_COLOR,
                pygame.Rect(
                    pix_square_size * ripe_mango.x,
                    pix_square_size * ripe_mango.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        # Drawing the agent
        pygame.draw.rect(
            canvas,
            self.AGENT_COLOR,
            pygame.Rect(
                pix_square_size * self.agent.x,
                pix_square_size * self.agent.y,
                pix_square_size,
                pix_square_size
            )
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
