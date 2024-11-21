import gymnasium as gym
import corridor_run
from agents import HumanPlayer

env = gym.make(id="CorridorRun-v0", render_mode="human")

human_player = HumanPlayer(env=env)

human_player.play(timeout=20, episodes=20)
