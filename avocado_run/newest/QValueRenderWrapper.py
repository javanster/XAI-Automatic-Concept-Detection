import gym
from gymnasium import Wrapper
import numpy as np
import pygame


class QValueRenderWrapper(Wrapper):
    """
    A Gymnasium wrapper to render Q-values on top of the Avocado Run environment
    """

    def __init__(self, env, model, render_fps=4, window_size=600):
        super().__init__(env)
        self.model = model
        self.render_fps = render_fps
        self.window_size = window_size

        self.window = None
        self.clock = None
        self.arrow_image = None
        self.rectangle_image = None

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(
            (self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        try:
            self.arrow_image = pygame.image.load(
                'avocado_run/newest/sprites/arrow.png').convert_alpha()
            self.rectangle_image = pygame.image.load(
                'avocado_run/newest/sprites/rectangle.png').convert_alpha()
        except Exception as e:
            raise FileNotFoundError(f"Error loading sprites: {e}")

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.current_observation = observation
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self.current_observation = observation
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.env.render_mode != "human":
            return self.env.render()

        if self.model is not None:
            q_values_dict, best_action = self._compute_q_values()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((0, 0, 0))
            pix_square_size = (self.window_size / self.env.grid_side_length)

            self._draw_entities(canvas, pix_square_size)

            self._draw_q_values(q_values_dict)
            self._draw_best_action(best_action)

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)

    def _compute_q_values(self):
        """
        Compute Q-values for all possible agent positions using the model.
        Returns a dictionary mapping positions to Q-values and the best action.
        """
        obs_without_agent = np.zeros(
            (self.env.grid_side_length, self.env.grid_side_length, 3), dtype=np.uint8)
        obs_without_agent[self.env.avocado.x,
                          self.env.avocado.y] = self.env.AVOCADO_COLOR
        for enemy in self.env.enemies:
            obs_without_agent[enemy.x, enemy.y] = self.env.ENEMY_COLOR

        q_value_dict = {}
        batch_obs = []
        agent_positions = []

        for x in range(self.env.grid_side_length):
            for y in range(self.env.grid_side_length):
                # Checks that the coordinates are of an empty cell
                if not (any((x == enemy.x and y == enemy.y) for enemy in self.env.enemies) or (x == self.env.avocado.x and y == self.env.avocado.y) or (x == self.env.agent.x and y == self.env.agent.y)):
                    obs_copy = obs_without_agent.copy()
                    obs_copy[x, y] = self.env.AGENT_COLOR
                    observation_reshaped = obs_copy.reshape(
                        -1, *self.env.observation_space.shape) / 255.0

                    batch_obs.append(observation_reshaped)
                    agent_positions.append((x, y))

        if batch_obs:
            batch_obs = np.vstack(batch_obs)
            q_values_batch = self.model.predict(batch_obs)

            for i, (x, y) in enumerate(agent_positions):
                max_q_value = np.max(q_values_batch[i])
                q_value_dict[(x, y)] = max_q_value

        observation_reshaped = self.current_observation.reshape(
            -1, *self.env.observation_space.shape) / 255.0
        best_action_for_agent = np.argmax(
            self.model.predict(observation_reshaped))

        return q_value_dict, best_action_for_agent

    def _draw_q_values(self, q_value_dict):
        """
        Draws the Q-value heatmap on the environment's grid.
        """
        grid_size = self.env.grid_side_length
        pix_square_size = self.window_size / grid_size

        if not q_value_dict:
            return

        max_q = max(q_value_dict.values())
        min_q = min(q_value_dict.values())
        range_q = max_q - min_q if max_q != min_q else 1.0

        for (x, y), q in q_value_dict.items():
            normalized_q = int(((q - min_q) / range_q) * 255)
            # Purple hue based on Q-value
            color = (normalized_q, 0, normalized_q)
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(
                    pix_square_size * x,
                    pix_square_size * y,
                    pix_square_size,
                    pix_square_size
                )
            )

    def _draw_best_action(self, best_action):
        """
        Draws an arrow or rectangle indicating the learned best action at the agent's current position.
        """
        if best_action is None:
            return

        grid_size = self.env.grid_side_length
        pix_square_size = self.window_size / grid_size

        agent_x = self.env.agent.x
        agent_y = self.env.agent.y

        if best_action == 0:
            sprite_to_show = self.arrow_image  # No rotation needed
        elif best_action == 1:
            sprite_to_show = pygame.transform.rotate(
                self.arrow_image, -90)  # 90 degrees clockwise
        elif best_action == 2:
            sprite_to_show = pygame.transform.rotate(
                self.arrow_image, 180)  # 180 degrees clockwise
        elif best_action == 3:
            sprite_to_show = pygame.transform.rotate(
                self.arrow_image, 90)  # 90 degrees counter clockwise (left)
        else:
            sprite_to_show = self.rectangle_image  # Use the rectangle image

        # Blit the rotated arrow or rectangle on the agent's position
        arrow_rect = sprite_to_show.get_rect(center=(
            pix_square_size * agent_x + pix_square_size / 2,
            pix_square_size * agent_y + pix_square_size / 2
        ))
        self.window.blit(sprite_to_show, arrow_rect)

    def _draw_entities(self, canvas, pix_square_size):
        # Drawing the avocado
        pygame.draw.rect(
            canvas,
            self.env.AVOCADO_COLOR,
            pygame.Rect(
                pix_square_size * self.env.avocado.x,
                pix_square_size * self.env.avocado.y,
                pix_square_size,
                pix_square_size
            )
        )

        # Drawing the agent
        pygame.draw.rect(
            canvas,
            self.env.AGENT_COLOR,
            pygame.Rect(
                pix_square_size * self.env.agent.x,
                pix_square_size * self.env.agent.y,
                pix_square_size,
                pix_square_size
            )
        )

        # Drawing the enemies
        for enemy in self.env.enemies:
            pygame.draw.rect(
                canvas,
                self.env.ENEMY_COLOR,
                pygame.Rect(
                    pix_square_size * enemy.x,
                    pix_square_size * enemy.y,
                    pix_square_size,
                    pix_square_size
                )
            )

        self.window.blit(canvas, canvas.get_rect())

    def close(self):
        self.env.close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
