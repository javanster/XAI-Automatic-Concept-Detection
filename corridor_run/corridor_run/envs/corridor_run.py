from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from .entity import Entity
import pygame
import random


class CorridorRun(Env):

    metadata = {"render_modes": [
        "human", "rgb_array"], "render_fps": 4}

    def __init__(
            self,
            render_mode=None,
            render_fps=4,
    ):
        super().__init__()
        self.STEP_PENALTY = -0.00245
        self.SPIKE_PENALTY = -0.00255
        self.TREASURE_REWARD = 1.049
        self.reward_range = (-0.99235, 1)

        self.TREASURE_COLOR = (255, 215, 0)  # Gold
        self.AGENT_COLOR = (78, 172, 248)    # Blue
        self.SPIKE_COLOR = (255, 0, 0)   # Red
        self.WALL_COLOR = (0, 0, 0)          # Black
        self.FLOOR_COLOR = (200, 200, 200)   # Light gray

        self.render_fps = render_fps
        self.grid_side_length = 21
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
        self.window_size = 609
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.wall_map = np.array([
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        ])

        self.agent_spawn_loactions = [(0, 0), (0, 20), (20, 0), (20, 20)]

        self.walls = []
        self.spikes = []

        for y in range(self.grid_side_length):
            for x in range(self.grid_side_length):
                map_value = self.wall_map[y, x]
                if map_value == 1:
                    self.walls.append(Entity(
                        grid_side_length=self.grid_side_length,
                        starting_position=(x, y)
                    ))
                elif map_value == 2:
                    self.spikes.append(Entity(
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

        agent_start_position = agent_starting_position if agent_starting_position else self.agent_spawn_loactions[random.randint(
            0, 3)]

        self.agent = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=agent_start_position
        )

        self.treasure = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(10, 10)
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

        for spike in self.spikes:
            obs[spike.y, spike.x] = self.SPIKE_COLOR

        obs[self.treasure.y, self.treasure.x] = self.TREASURE_COLOR

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

        if any(self.agent == spike for spike in self.spikes):
            reward += self.SPIKE_PENALTY

        elif self.agent == self.treasure:
            reward += self.TREASURE_REWARD
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

        # Drawing the spikes
        for spike in self.spikes:
            pygame.draw.rect(
                canvas,
                self.SPIKE_COLOR,
                pygame.Rect(
                    pix_square_size * spike.x,
                    pix_square_size * spike.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        # Drawing the treasure
        pygame.draw.rect(
            canvas,
            self.TREASURE_COLOR,
            pygame.Rect(
                pix_square_size * self.treasure.x,
                pix_square_size * self.treasure.y,
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
