import gymnasium as gym
import avocado_run
from HumanPlayer import HumanPlayer

env = gym.make(id="AvocadoRun-v0", render_mode="human")

human_player = HumanPlayer(env=env)

human_player.play(timeout=20, episodes=5)
