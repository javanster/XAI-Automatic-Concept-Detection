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
            agent_spawn_all_legal_locations=False,
            spawn_unripe_mangoes=True,
    ):
        super().__init__()
        self.STEP_PENALTY = -0.02
        self.UNRIPE_MANGO_REWARD = 0.1
        self.RIPE_MANGO_REWARD = 1.02
        self.reward_range = (-1, 1)

        self.RIPE_MANGO_COLOR = (255, 215, 0)  # Gold
        self.AGENT_COLOR = (78, 172, 248)    # Blue
        self.UNRIPE_MANGO_COLOR = (255, 0, 0)   # Red
        self.WALL_COLOR = (0, 100, 0)         # Dark green
        self.FLOOR_COLOR = (245, 222, 179)  # Light beige

        self.render_fps = render_fps
        self.grid_side_width = 7
        self.grid_side_height = 7
        self.action_space = Discrete(4)
        self.action_dict = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }
        self.observation_space = Box(
            low=0, high=255,
            shape=(self.grid_side_height, self.grid_side_width, 3),
            dtype=np.uint8,
        )
        self.window_width = 630
        self.window_height = 630
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.spawn_unripe_mangoes = spawn_unripe_mangoes

        self.wall_map = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])

        self.agent_spawn_locations = []
        if agent_spawn_all_legal_locations:
            for y in range(self.grid_side_height):
                for x in range(self.grid_side_width):
                    if self.wall_map[y, x] == 0:
                        self.agent_spawn_locations.append((x, y))
        else:
            self.agent_spawn_locations = [
                (0, 0),
                (0, self.grid_side_height - 1),
                (self.grid_side_width - 1, 0),
                (self.grid_side_width - 1, self.grid_side_height - 1)
            ]

        self.unripe_mango_spawn_positions = [(3, 1), (1, 3), (3, 5), (5, 3)]
        self.walls = []
        self.unripe_mangoes = []

    def _get_possible_ripe_mango_spawn_locations(self, wall_map):
        rows, cols = wall_map.shape
        surrounded_coordinates = []

        # First, find all cells that meet the "surrounded by one zero" condition
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if wall_map[y, x] == 1:
                    # Check the values in the four cardinal directions
                    neighbors = [
                        wall_map[y - 1, x],  # above
                        wall_map[y + 1, x],  # below
                        wall_map[y, x - 1],  # left
                        wall_map[y, x + 1]   # right
                    ]

                    # Count the number of zeros in the neighboring cells
                    zero_count = neighbors.count(0)

                    # If exactly one zero is found, mark this coordinate as a valid mango location
                    if zero_count == 1:
                        surrounded_coordinates.append((x, y))

        # Find all pairs of adjacent coordinates where both satisfy the "one zero" condition
        possible_pairs = []
        for coord in surrounded_coordinates:
            x, y = coord

            # Check the cell to the right and below to see if they are also valid spawn locations
            right_neighbor = (x + 1, y)
            below_neighbor = (x, y + 1)

            # Check if the right neighbor is also in the valid locations and forms a pair
            if right_neighbor in surrounded_coordinates:
                possible_pairs.append([coord, right_neighbor])

            # Check if the below neighbor is also in the valid locations and forms a pair
            if below_neighbor in surrounded_coordinates:
                possible_pairs.append([coord, below_neighbor])

        return possible_pairs

    def reset(
            self,
            seed=None,
            options=None,
            agent_starting_position=None,
    ):
        super().reset(seed=seed)

        agent_start_position = agent_starting_position if agent_starting_position else self.agent_spawn_locations[random.randint(
            0, len(self.agent_spawn_locations) - 1)]

        possible_ripe_mango_starting_positions = self._get_possible_ripe_mango_spawn_locations(
            self.wall_map)
        ripe_mango_starting_positions = possible_ripe_mango_starting_positions[random.randint(
            0, len(possible_ripe_mango_starting_positions) - 1)]

        for y in range(self.grid_side_height):
            for x in range(self.grid_side_width):
                map_value = self.wall_map[y, x]
                if map_value == 1:
                    self.walls.append(Entity(
                        grid_side_width=self.grid_side_width,
                        grid_side_height=self.grid_side_height,
                        starting_position=(x, y)
                    ))

        self.agent = Entity(
            grid_side_width=self.grid_side_width,
            grid_side_height=self.grid_side_height,
            starting_position=agent_start_position
        )

        self.ripe_mango_1 = Entity(
            grid_side_width=self.grid_side_width,
            grid_side_height=self.grid_side_height,
            starting_position=ripe_mango_starting_positions[0]
        )

        self.ripe_mango_2 = Entity(
            grid_side_width=self.grid_side_width,
            grid_side_height=self.grid_side_height,
            starting_position=ripe_mango_starting_positions[1]
        )
        if self.spawn_unripe_mangoes:
            for unripe_mango_spawn_position in self.unripe_mango_spawn_positions:
                self.unripe_mangoes.append(
                    Entity(
                        grid_side_height=self.grid_side_height,
                        grid_side_width=self.grid_side_width,
                        starting_position=unripe_mango_spawn_position
                    )
                )

        for ripe_mango in [self.ripe_mango_1, self.ripe_mango_2]:
            for unripe_mango in self.unripe_mangoes:
                if ripe_mango == unripe_mango:
                    self.unripe_mangoes.remove(unripe_mango)
            for wall in self.walls:
                if ripe_mango == wall:
                    self.walls.remove(wall)

        for unripe_mango in self.unripe_mangoes:
            for wall in self.walls:
                if unripe_mango == wall:
                    self.walls.remove(wall)

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self):
        obs = np.full(
            (self.grid_side_height, self.grid_side_width, 3),
            self.FLOOR_COLOR,
            dtype=np.uint8
        )

        for wall in self.walls:
            obs[wall.y, wall.x] = self.WALL_COLOR

        obs[self.ripe_mango_1.y, self.ripe_mango_1.x] = self.RIPE_MANGO_COLOR
        obs[self.ripe_mango_2.y, self.ripe_mango_2.x] = self.RIPE_MANGO_COLOR

        for unripe_mango in self.unripe_mangoes:
            obs[unripe_mango.y, unripe_mango.x] = self.UNRIPE_MANGO_COLOR

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

        if self.agent == self.ripe_mango_1 or self.agent == self.ripe_mango_2:
            reward += self.RIPE_MANGO_REWARD
            terminated = True

        elif any(self.agent == unripe_mango for unripe_mango in self.unripe_mangoes):
            reward += self.UNRIPE_MANGO_REWARD
            terminated = True

        if self.episode_step >= 50:
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
                (self.window_width, self.window_height)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(self.FLOOR_COLOR)
        pix_square_size_width = self.window_width / self.grid_side_width
        pix_square_size_height = self.window_height / self.grid_side_height

        # Drawing the walls
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                self.WALL_COLOR,
                pygame.Rect(
                    pix_square_size_width * wall.x,
                    pix_square_size_height * wall.y,
                    pix_square_size_width,
                    pix_square_size_height
                )
            )

        # Drawing the unripe mangos
        for unripe_mango in self.unripe_mangoes:
            pygame.draw.rect(
                canvas,
                self.UNRIPE_MANGO_COLOR,
                pygame.Rect(
                    pix_square_size_width * unripe_mango.x,
                    pix_square_size_height * unripe_mango.y,
                    pix_square_size_width,
                    pix_square_size_height
                )
            )

        # Drawing the ripe mangos
        for ripe_mango in [self.ripe_mango_1, self.ripe_mango_2]:
            pygame.draw.rect(
                canvas,
                self.RIPE_MANGO_COLOR,
                pygame.Rect(
                    pix_square_size_width * ripe_mango.x,
                    pix_square_size_height * ripe_mango.y,
                    pix_square_size_width,
                    pix_square_size_height
                )
            )

        # Drawing the agent
        pygame.draw.rect(
            canvas,
            self.AGENT_COLOR,
            pygame.Rect(
                pix_square_size_width * self.agent.x,
                pix_square_size_height * self.agent.y,
                pix_square_size_width,
                pix_square_size_height
            )
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
