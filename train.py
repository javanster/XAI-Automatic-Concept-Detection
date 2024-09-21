import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent

config = {
    "project_name": "AvocadoRun",
    "step_penalty": -0.1,
    "enemy_hit_penalty": -20,
    "avocado_reward": 100,

    "replay_buffer_size": 100_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 64,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 2000,
    "learning_rate": 0.001,
    "prop_steps_epsilon_decay": 0.9,  # The proportion of steps epsilon should decay
    "starting_epsilon": 1,
    "min_epsilon": 0.01,
    "steps_to_train": 2_000_000,
    "episode_metrics_window": 100,  # Number of episodes to take into account for metrics
}

env = gym.make(id="AvocadoRun-v0", render_mode="human")

agent = DoubleDQNAgent(
    env=env,
    model_path=None,
    learning_rate=config["learning_rate"]
)

agent.train(config=config, track_metrics=True)
