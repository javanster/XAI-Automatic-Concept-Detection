import wandb
import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "tumbling_window_average_reward"},
    "parameters": {
        "replay_buffer_size": {"values": [50_000, 100_000, 500_000]},
        "min_replay_buffer_size": {"values": [10_000]},
        "minibatch_size": {"values": [32, 64, 128]},
        "discount": {"values": [0.9, 0.95, 0.99]},
        "training_frequency": {"values": [i for i in range(4, 17)]},
        "update_target_every": {"values": [500, 750, 1000, 1250, 1500, 1750, 2_000]},
        "learning_rate": {"values": [0.0001, 0.001, 0.01]},
        "prop_steps_epsilon_decay": {"values": [0.9]},
        "starting_epsilon": {"values": [1]},
        "min_epsilon": {"values": [0.01, 0.05]},
        "steps_to_train": {"values": [100000]},
        "episode_metrics_window": {"values": [100]},
    },
}


def train():
    env = gym.make(id="AvocadoRun-v0", render_mode="human")

    agent = DoubleDQNAgent(
        env=env,
        model_path=None,
        learning_rate=0.001
    )

    agent.train(config=None, use_wandb=True, use_sweep=True)


wandb.login()

sweep_id = wandb.sweep(sweep=sweep_configuration, project="AvocadoRun")

wandb.agent(sweep_id=sweep_id, function=train, count=4)
