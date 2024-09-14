from AvocadoRunGymEnv import AvocadoRunGymEnv
from DoubleDQNAgent import DoubleDQNAgent

config = {
    # For env
    "moving_enemy": True,
    "num_enemies": 2,
    "step_penalty": -0.1,
    "enemy_hit_penalty": -10,
    "avocado_reward": 20,

    # For agent
    "conv_filters": 256,
    "dense_units": 64,
    "replay_buffer_size": 100_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 64,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 2_000,
    "learning_rate": 0.001,
    "starting_epsilon": 1,
    "epsilon_decay": 0.99995,
    "min_epsilon": 0.005,
    "average_window": 100,
    "episodes_to_train": 40_000,
}

env = AvocadoRunGymEnv(
    config=config,
    render_mode="q_values",  # Can be None, "human" or "q_values"
    show_every=1,
    render_fps=4,
    agent_starting_position=(0, 0)
)


ddqn_agent = DoubleDQNAgent(
    env=env,
    model_path="models/256x2__64x1___30.00max___-9.18avg_-314.00min__1725863125.keras",
    config=config
)


ddqn_agent.test(episodes=20)
