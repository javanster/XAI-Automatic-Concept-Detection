from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from Entity import Entity
import pygame


class AvocadoRunGymEnv(Env):
    """
    A custom Gymnasium environment where the goal of the agent is to avoid moving enemies (angry farmers that
    partly move towards the agent), and eat the avocado.
    """

    METADATA = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    STEP_PENALTY = 1
    ENEMY_HIT_PENALTY = 300
    AVOCADO_REWARD = 30

    def __init__(self, render_mode=None, moving_enemy=False, num_enemies=1):
        self.moving_enemy = moving_enemy
        self.num_enemies = num_enemies
        self.grid_side_length = 15
        self.action_space = Discrete(5)
        self.observation_space_shape = (
            self.grid_side_length, self.grid_side_length, 3)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space_shape, dtype=np.uint8
        )
        self.window_size = 600

        assert render_mode is None or render_mode in self.METADATA["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent = Entity(env_size=self.grid_side_length)

        self.avocado = Entity(env_size=self.grid_side_length)
        while self.avocado == self.agent:
            self.avocado = Entity(self.grid_side_length)

        self.enemies = []
        for _ in range(self.num_enemies):
            enemy = Entity(self.grid_side_length)
            while enemy == self.agent or enemy == self.avocado:
                enemy = Entity(self.grid_side_length)
            self.enemies.append(enemy)

        self.episode_step = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def _get_obs(self):
        obs = np.zeros(
            (self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        obs[self.avocado.x, self.avocado.y] = (0, 255, 0)
        obs[self.agent.x, self.agent.y] = (0, 255, 175)
        for enemy in self.enemies:
            obs[enemy.x, enemy.y] = (255, 0, 0)

        return obs

    def step(self, action, step_limit=True):
        self.episode_step += 1
        self.agent.action(action)

        if self.moving_enemy:
            for enemy in self.enemies:
                if self.episode_step % 3 == 0:
                    enemy.move_towards_target(self.agent)
                else:
                    enemy.random_action()

        new_observation = self._get_obs()

        terminated = False
        reward = -self.STEP_PENALTY
        if any(self.agent == enemy for enemy in self.enemies):
            reward = -self.ENEMY_HIT_PENALTY
            terminated = True
        elif self.agent == self.avocado:
            reward = self.AVOCADO_REWARD
            terminated = True
        elif step_limit and self.episode_step >= 200:
            terminated = True

        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return new_observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.grid_side_length
        )

        # Drawing the avocado
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self.avocado.x,
                pix_square_size * self.avocado.y,
                pix_square_size,
                pix_square_size
            )
        )

        # Drawing the agent
        pygame.draw.rect(
            canvas,
            (78, 172, 248),
            pygame.Rect(
                pix_square_size * self.agent.x,
                pix_square_size * self.agent.y,
                pix_square_size,
                pix_square_size
            )
        )

        # Drawing the enemies
        for enemy in self.enemies:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * enemy.x,
                    pix_square_size * enemy.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.METADATA["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
