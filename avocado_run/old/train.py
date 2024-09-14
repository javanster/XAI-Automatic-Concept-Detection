from AvocadoRunGymEnv import AvocadoRunGymEnv
from DoubleDQNAgent import DoubleDQNAgent

config = {
    # For env
    "moving_enemy": True,
    "num_enemies": 2,
    "step_penalty": -1,
    "enemy_hit_penalty": -20,
    "avocado_reward": 50,

    # For agent
    "conv_filters": 128,
    "dense_units": 64,
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
    "episodes_to_train": 80_000,
}

env = AvocadoRunGymEnv(
    config=config,
    render_mode="q_values",  # Can be None, "human" or "q_values"
    show_every=1000,
    render_fps=4
)


ddqn_agent = DoubleDQNAgent(
    env=env,
    model_path=None,
    config=config
)

ddqn_agent.train(
    episodes=config["episodes_to_train"],
    average_window=config["average_window"],
)
