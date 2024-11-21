import gymnasium as gym
import mango_run
from agents import HumanPlayer

env = gym.make(id="MangoRun-v0", render_mode="human",
               agent_spawn_all_legal_locations=False, spawn_unripe_mangoes=True)

human_player = HumanPlayer(env=env)

human_player.play(timeout=20, episodes=20)
