import gymnasium as gym
import mango_run
from agents import DoubleDQNAgent
import time
from datetime import timedelta

# Based on best hyperparams found using Bayesian Hyperparameter Optimization - See wandb sandy-sweep-16
config = {
    "project_name": "MangoRun",
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
    "steps_to_train": 50_000,
    "episode_metrics_window": 50,  # Number of episodes to take into account for metrics

    "env": {
        "agent_spawn_all_legal_locations": True,
        "spawn_unripe_mangoes": True,
        "STEP_PENALTY": -0.02,
        "UNRIPE_MANGO_REWARD": 0.1,
        "RIPE_MANGO_REWARD": 1.02,
        "reward_range": (-1, 1),
    }
}

env = gym.make(
    id="MangoRun-v0",
    agent_spawn_all_legal_locations=True,
    spawn_unripe_mangoes=True
)

agent = DoubleDQNAgent(
    env=env,
)

start_time = time.time()


agent.train(
    config=config,
    use_wandb=False
)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Format elapsed time into hours, minutes, and seconds
formatted_time = str(timedelta(seconds=int(elapsed_time)))
print(f"Training took {formatted_time}.")
