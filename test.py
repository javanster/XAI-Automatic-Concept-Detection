import gymnasium as gym
import avocado_run
from avocado_run.wrappers import QValueRenderWrapper
from DoubleDQNAgent import DoubleDQNAgent

env = gym.make(id="AvocadoRun-v0", render_mode="human", num_avocados=1)

agent = DoubleDQNAgent(
    env=env,
    model_path="models/wild-glitter-14/model___99.23avg___99.90max___98.10min__1726876253.keras"
)

wrapped_env = QValueRenderWrapper(
    env=env, model=agent.online_model, render_fps=1)

agent.test(episodes=10, env=env)
# agent.test(episodes=10, env=wrapped_env)
