import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent

config = {
    # Not relevant for testing
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
    "episodes_to_train": 3_000,
    "average_window": 100,
}

env = gym.make(id="AvocadoRun-v0", render_mode="human")

agent = DoubleDQNAgent(
    env=env,
    model_path=None,
    learning_rate=config["learning_rate"]
)

agent.train(config=config, track_metrics=False)
