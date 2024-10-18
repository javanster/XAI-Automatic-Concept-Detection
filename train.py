import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent

# Based on best hyperparams found using Bayesian Hyperparameter Optimization - See wandb sandy-sweep-16
config = {
    "project_name": "AvocadoRun",
    "replay_buffer_size": 50_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 64,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 1000,
    "learning_rate": 0.001,
    "prop_steps_epsilon_decay": 0.9,  # The proportion of steps epsilon should decay
    "starting_epsilon": 1,
    "min_epsilon": 0.05,
    "steps_to_train": 2_000_000,
    "episode_metrics_window": 100,  # Number of episodes to take into account for metrics
}

env = gym.make(id="AvocadoRun-v0")

agent = DoubleDQNAgent(
    env=env,
)

agent.train(config=config, use_wandb=False)
