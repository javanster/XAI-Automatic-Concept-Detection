from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from Entity import Entity
import pygame


class AvocadoRunEnv(Env):
    """
    A custom Gymnasium environment where the goal of the agent is to avoid moving enemies (angry farmers that
    partly move towards the agent), and eat the avocado.
    """

    metadata = {"render_modes": [
        "human", "rgb_array"], "render_fps": 4}

    def __init__(
            self,
            render_mode=None,
            render_fps=4
    ):
        super().__init__()
        self.STEP_PENALTY = -1
        self.ENEMY_HIT_PENALTY = -100
        self.AVOCADO_REWARD = 100
        self.reward_range = (-300, 99)

        self.AVOCADO_COLOR = (0, 255, 0)
        self.AGENT_COLOR = (78, 172, 248)
        self.ENEMY_COLOR = (255, 0, 0)

        self.render_fps = render_fps
        self.num_enemies = 2
        self.grid_side_length = 10
        self.action_space = Discrete(5)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.avocado = Entity(
            env_size=self.grid_side_length)

        self.agent = Entity(env_size=self.grid_side_length)
        while self.agent == self.avocado:
            self.agent = Entity(self.grid_side_length)

        self.enemies = []
        for _ in range(self.num_enemies):
            enemy = Entity(self.grid_side_length)
            while enemy == self.agent or enemy == self.avocado:
                enemy = Entity(self.grid_side_length)
            self.enemies.append(enemy)

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self):
        obs = np.zeros(
            (self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8)
        obs[self.avocado.x, self.avocado.y] = self.AVOCADO_COLOR
        obs[self.agent.x, self.agent.y] = self.AGENT_COLOR
        for enemy in self.enemies:
            obs[enemy.x, enemy.y] = self.ENEMY_COLOR

        return obs

    def step(self, action):
        self.episode_step += 1
        self.agent.action(action)

        for enemy in self.enemies:
            if self.episode_step % 3 == 0:
                enemy.move_towards_target(self.agent)
            else:
                enemy.random_action()

        new_observation = self._get_obs()

        terminated = False
        truncated = False

        reward = self.STEP_PENALTY

        if any(self.agent == enemy for enemy in self.enemies):
            reward += self.ENEMY_HIT_PENALTY
            terminated = True

        elif self.agent == self.avocado:
            reward += self.AVOCADO_REWARD
            terminated = True

        elif self.episode_step >= 200:
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
        canvas.fill((0, 0, 0))
        pix_square_size = (self.window_size / self.grid_side_length)

        # Drawing the avocado
        pygame.draw.rect(
            canvas,
            self.AVOCADO_COLOR,
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
            self.AGENT_COLOR,
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
                self.ENEMY_COLOR,
                pygame.Rect(
                    pix_square_size * enemy.x,
                    pix_square_size * enemy.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
