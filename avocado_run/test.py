from AvocadoRunGymEnv import AvocadoRunGymEnv
from DQLAgent import DQLAgent

env = AvocadoRunGymEnv(
    moving_enemy=True,
    num_enemies=2,

    # The render mode. Can be None, "human" or "q_values"
    render_mode="q_values",

    # If render mode is either "human" or "q_values", denotes how many episodes should pass before showing a game
    show_every=1,

    # Frames per second of the rendering of the game, If render mode is either "human" or "q_values"
    render_fps=4
)

dql_agent = DQLAgent(
    env=env,
    replay_buffer_size=50_000,
    min_replay_buffer_size=1_000,
    minibatch_size=64,
    discount=0.99,
    update_target_every=5,
    epsilon=1,
    epsilon_decay=0.99975,
    min_epsilon=0.001,
    model_path="models/256x2___30.00max___-9.18avg_-314.00min__1725863125.keras"
)

dql_agent.test()
