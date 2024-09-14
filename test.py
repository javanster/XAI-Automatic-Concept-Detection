import gymnasium as gym
import avocado_run
from avocado_run.wrappers import QValueRenderWrapper
from DoubleDQNAgent import DoubleDQNAgent

env = gym.make(id="AvocadoRun-v0", render_mode="human")

agent = DoubleDQNAgent(
    env=env,
    model_path="models/64x2___99.00max__-91.65avg_-160.00min__1726327515.keras"
)

wrapped_env = QValueRenderWrapper(env=env, model=agent.online_model)

# agent.test(episodes=5, env=env)
agent.test(episodes=5, env=wrapped_env)
