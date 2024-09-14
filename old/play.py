from AvocadoRunGymEnv import AvocadoRunGymEnv
from HumanPlayer import HumanPlayer

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

human_player = HumanPlayer(env=env)
human_player.play(timeout=30, episodes=10)
