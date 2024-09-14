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

    METADATA = {"render_modes": ["human", "rgb_array", "q_values"]}

    def __init__(self, config, render_mode=None, render_fps=4, show_every=1, agent_starting_position=None):
        self.STEP_PENALTY = config["step_penalty"]
        self.ENEMY_HIT_PENALTY = config["enemy_hit_penalty"]
        self.AVOCADO_REWARD = config["avocado_reward"]

        self.render_fps = render_fps
        self.show_every = show_every
        self.moving_enemy = config["moving_enemy"]
        self.num_enemies = config["num_enemies"]
        self.grid_side_length = 10
        self.action_space = Discrete(5)
        self.observation_space_shape = (
            self.grid_side_length, self.grid_side_length, 3)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space_shape, dtype=np.uint8
        )
        self.window_size = 600

        self.set_render_mode(render_mode=render_mode)

        self.window = None
        self.clock = None

        self.agent_starting_position = agent_starting_position

    def set_render_mode(self, render_mode=None):
        assert render_mode is None or render_mode in self.METADATA["render_modes"]
        self.render_mode = render_mode

    def reset(self, episode, model=None, seed=None, options=None):
        super().reset(seed=seed)

        self.avocado = Entity(
            env_size=self.grid_side_length)

        self.agent = Entity(env_size=self.grid_side_length,
                            starting_position=self.agent_starting_position)
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

        if self.show_every and episode % self.show_every == 0:
            self._render(model)

        return observation

    def _get_obs(self):
        obs = np.zeros(
            (self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        obs[self.avocado.x, self.avocado.y] = (0, 255, 0)
        obs[self.agent.x, self.agent.y] = (0, 255, 175)
        for enemy in self.enemies:
            obs[enemy.x, enemy.y] = (255, 0, 0)

        return obs

    def step(self, action, episode, step_limit=True, model=None):
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
        reward = self.STEP_PENALTY
        if any(self.agent == enemy for enemy in self.enemies):
            reward = self.ENEMY_HIT_PENALTY
            terminated = True
        elif self.agent == self.avocado:
            reward = self.AVOCADO_REWARD
            terminated = True
        elif step_limit and self.episode_step >= 200:
            terminated = True

        info = {}

        if self.show_every and episode % self.show_every == 0:
            self._render(model=model)

        return new_observation, reward, terminated, False, info

    def _render(self, model):
        if self.render_mode == "q_values":
            if not model:
                raise ValueError(
                    "When in render mode \"q_values\", a model needs to be provided")
            q_values_dict, best_action_for_agent = self._get_q_values_dict(
                model=model)
            self._draw_entities(q_value_dict=q_values_dict,
                                best_action_for_agent=best_action_for_agent)
        elif self.render_mode == "human":
            self._draw_entities()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_q_values_dict(self, model):
        obs = np.zeros(
            (self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        obs[self.avocado.x, self.avocado.y] = (0, 255, 0)
        for enemy in self.enemies:
            obs[enemy.x, enemy.y] = (255, 0, 0)

        q_value_dict = {}
        batch_obs = []
        agent_positions = []

        for x in range(self.grid_side_length):
            for y in range(self.grid_side_length):
                # Checks that the coordinates are of an empty cell
                if not (any((x == enemy.x and y == enemy.y) for enemy in self.enemies) or (x == self.avocado.x and y == self.avocado.y) or (x == self.agent.x and y == self.agent.y)):
                    obs_copy = obs.copy()
                    obs_copy[x, y] = (0, 255, 175)  # Simulate agent's position
                    observation_reshaped = obs_copy.reshape(
                        -1, *self.observation_space_shape) / 255.0

                    batch_obs.append(observation_reshaped)
                    agent_positions.append((x, y))

        if batch_obs:
            batch_obs = np.vstack(batch_obs)
            q_values_batch = model.predict(batch_obs)

            for i, (x, y) in enumerate(agent_positions):
                max_q_value = np.max(q_values_batch[i])
                q_value_dict[(x, y)] = max_q_value

        # Normalize the Q-values to be values between 0 and 255
        if q_value_dict:
            max_q_value_in_dict = max(q_value_dict.values())
            min_q_value_in_dict = min(q_value_dict.values())

            for (x, y) in q_value_dict.keys():
                value = q_value_dict[(x, y)]
                q_value_dict[(x, y)] = (
                    (value - min_q_value_in_dict) / (max_q_value_in_dict - min_q_value_in_dict)) * 255

        obs_copy = obs.copy()
        obs_copy[self.agent.x, self.agent.y] = (0, 255, 175)
        observation_reshaped = obs_copy.reshape(
            -1, *self.observation_space_shape) / 255.0
        best_action_for_agent = np.argmax(model.predict(observation_reshaped))

        return q_value_dict, best_action_for_agent

    def _draw_entities(self, q_value_dict=None, best_action_for_agent=None):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if not hasattr(self, 'arrow_image'):
            self.arrow_image = pygame.image.load(
                'avocado_run/sprites/arrow.png').convert_alpha()
            self.rectangle_image = pygame.image.load(
                'avocado_run/sprites/rectangle.png').convert_alpha()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (self.window_size / self.grid_side_length)

        # Draw the Q-value heatmap
        if q_value_dict:
            for (x, y) in q_value_dict.keys():
                pygame.draw.rect(
                    canvas,
                    (q_value_dict[(x, y)], 0, q_value_dict[(x, y)]),
                    pygame.Rect(
                        pix_square_size * x,
                        pix_square_size * y,
                        pix_square_size,
                        pix_square_size
                    )
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

        # Render the best action arrow/rectangle if applicable
        if best_action_for_agent is not None:
            if best_action_for_agent == 0:
                rotated_arrow = self.arrow_image  # No rotation needed
            elif best_action_for_agent == 1:
                rotated_arrow = pygame.transform.rotate(
                    self.arrow_image, -90)  # 90 degrees clockwise
            elif best_action_for_agent == 2:
                rotated_arrow = pygame.transform.rotate(
                    self.arrow_image, 180)  # 180 degrees clockwise
            elif best_action_for_agent == 3:
                rotated_arrow = pygame.transform.rotate(
                    self.arrow_image, -270)  # 270 degrees clockwise
            elif best_action_for_agent == 4:
                rotated_arrow = self.rectangle_image  # Use the rectangle image

            # Blit the rotated arrow or rectangle on the agent's position
            arrow_rect = rotated_arrow.get_rect(center=(
                pix_square_size * self.agent.x + pix_square_size / 2,
                pix_square_size * self.agent.y + pix_square_size / 2
            ))
            canvas.blit(rotated_arrow, arrow_rect)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
