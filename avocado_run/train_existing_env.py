from AvocadoRunEnv import AvocadoRunEnv
from DoubleDQNAgent import DoubleDQNAgent
from QValueRenderWrapper import QValueRenderWrapper
import gymnasium as gym
from HumanPlayer import HumanPlayer

config = {

    # For agent
    "replay_buffer_size": 100_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 64,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 2_000,
    "learning_rate": 0.001,
    "starting_epsilon": 1,
    "epsilon_decay": 0.99985,
    "min_epsilon": 0.01,
    "average_window": 100,
    "episodes_to_train": 1000,
}

env = AvocadoRunEnv(render_mode="human")

ddqn_agent = DoubleDQNAgent(
    env=env,
    model_path="models/64x2___99.00max__-96.71avg_-140.00min__1726317237.keras",
    config=config
)

# wrapper_env = QValueRenderWrapper(env, ddqn_agent.online_model, render_fps=4)


""" ddqn_agent.train(
    episodes=config["episodes_to_train"],
    average_window=config["average_window"],
    track_metrics=False
) """

# ddqn_agent.test(episodes=5)

human_player = HumanPlayer(env)
human_player.play()
