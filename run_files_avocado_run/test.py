import gymnasium as gym
import mango_run
from avocado_run.wrappers import QValueRenderWrapper
from agents import DoubleDQNAgent
from keras.api.saving import load_model
import numpy as np

MODEL_PATH = "models/mango_run/polar-paper-25/1732215466_model____0.9060avg____1.0000max____0.6800min.keras"

env = gym.make(id="MangoRun-v0", render_mode="human")

agent = DoubleDQNAgent(
    env=env,
)

""" model = load_model(MODEL_PATH)

wrapped_env = QValueRenderWrapper(
    env=env, model=model, render_fps=1) """

agent.test(model_path=MODEL_PATH,
           episodes=20, env=env)
# agent.test(model_path=MODEL_PATH, episodes=10, env=wrapped_env)

""" agent_not_reaching_avo_n = sum(reward < 0 for reward in episode_rewards)
agent_reaching_avo_n = sum(reward > 0 for reward in episode_rewards)
terminated_without_reaching_avo = terminations - agent_reaching_avo_n
avg_reward = np.mean(episode_rewards)
median_reward = np.median(episode_rewards)


print(f"Number of times agent never reaches avo: {agent_not_reaching_avo_n}")
print(f"Number of times agent reaches avo: {agent_reaching_avo_n}")
print(f"Player caught n: {terminated_without_reaching_avo}")
print(
    f"Truncations (agent not reaching avo and not being caugt): {truncations}")
print(f"Average reward: {avg_reward}")
print(f"Median reward: {median_reward}") """
