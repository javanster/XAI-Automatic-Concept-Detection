from Entity import Entity
import numpy as np
import cv2
from PIL import Image


class AvocadoRunEnv:
    """
    A custom environment where the goal of the agent is to avoid moving enemies (angry farmers that
    partly move towards the agent), and eat the avocado.

    Inspired by code from pythonprogramming.net: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
    """

    def __init__(self, observations_as_images=False, moving_enemy=False, num_enemies=1) -> None:
        self.OBSERVATIONS_AS_IMAGES = observations_as_images
        self.MOVING_ENEMY = moving_enemy
        self.episode_step = 0
        self.num_enemies = num_enemies

    SIZE = 10
    MOVE_PENALTY = 1
    ENEMY_HIT_PENALTY = 300
    AVOCADO_REWARD = 30
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9

    def reset(self):
        self.player = Entity(env_size=self.SIZE)

        self.avocado = Entity(env_size=self.SIZE)
        while self.avocado == self.player:
            self.avocado = Entity(self.SIZE)

        self.enemies = []
        for _ in range(self.num_enemies):
            enemy = Entity(self.SIZE)
            while enemy == self.player or enemy == self.avocado:
                enemy = Entity(self.SIZE)
            self.enemies.append(enemy)

        self.episode_step = 0

        if self.OBSERVATIONS_AS_IMAGES:
            observation = np.array(self.get_image())
        else:
            # Limited to checking the distance from only one enemy
            observation = (self.player-self.avocado) + \
                (self.player-self.enemies[0])
        return observation

    def step(self, action, step_limit=True):
        self.episode_step += 1
        self.player.action(action)

        if self.MOVING_ENEMY:
            for enemy in self.enemies:
                if self.episode_step % 3 == 0:
                    enemy.move_towards_target(self.player)
                else:
                    enemy.random_action()

        if self.OBSERVATIONS_AS_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            # Limited to checking the distance from only one enemy
            new_observation = (self.player-self.avocado) + \
                (self.player-self.enemies[0])

        terminated = False
        reward = -self.MOVE_PENALTY
        if any(self.player == enemy for enemy in self.enemies):
            reward = -self.ENEMY_HIT_PENALTY
            terminated = True
        elif self.player == self.avocado:
            reward = self.AVOCADO_REWARD
            terminated = True
        elif step_limit and self.episode_step >= 200:
            terminated = True

        return new_observation, reward, terminated

    def render(self):
        img = self.get_image()
        img = img.resize((600, 600), resample=Image.BOX)
        cv2.imshow("image", np.array(img))
        cv2.waitKey(150)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.avocado.x][self.avocado.y] = (0, 255, 0)
        env[self.player.x][self.player.y] = (255, 175, 0)
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = (0, 0, 255)

        img = Image.fromarray(env, 'RGB')
        return img
