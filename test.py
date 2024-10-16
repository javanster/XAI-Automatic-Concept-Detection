import gymnasium as gym
import avocado_run
from avocado_run.wrappers import QValueRenderWrapper
from DoubleDQNAgent import DoubleDQNAgent
from keras.api.saving import load_model

MODEL_PATH = "models/eager_disco_16/best_model.keras"

env = gym.make(id="AvocadoRun-v0", render_mode="human",
               num_avocados=1, num_enemies=2, aggressive_enemies=False)

agent = DoubleDQNAgent(
    env=env,
)

model = load_model(MODEL_PATH)

wrapped_env = QValueRenderWrapper(
    env=env, model=model, render_fps=1)

agent.test(model_path=MODEL_PATH,
           episodes=10, env=env)
# agent.test(model_path=MODEL_PATH, episodes=10, env=wrapped_env)
