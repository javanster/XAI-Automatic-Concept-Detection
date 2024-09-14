import pygame
import time


class HumanPlayer:
    """
    Allows a human to play as the agent in the AvocadoRunGymEnv
    """

    def __init__(self, env):
        self.env = env

    def _get_human_action(self):
        """
        Maps key presses to agent actions.
        Returns the action corresponding to the key pressed by the user.
        """
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0  # Up
                elif event.key == pygame.K_d:
                    action = 1  # Right
                elif event.key == pygame.K_s:
                    action = 2  # Down
                elif event.key == pygame.K_a:
                    action = 3  # Left
                elif event.key == pygame.K_SPACE:
                    action = 4  # Stay in place (no movement)

        return action

    def play(self, timeout=30, episodes=10):
        """
        Allows a human player to control the agent via the keyboard.
        The agent performs a 'stay in place' action if no key is pressed within a specified timeout.
        """
        for _ in range(episodes):

            self.env.reset()

            self.env.render()

            terminated = False
            default_action = 4  # Stay in place action

            while not terminated:
                action = None
                start_ticks = pygame.time.get_ticks()

                while action is None:
                    action = self._get_human_action()

                    if pygame.time.get_ticks() - start_ticks > timeout:
                        action = default_action

                _, _, terminated, _, _ = self.env.step(
                    action=action)

                self.env.render()

            time.sleep(2)

        self.env.close()
