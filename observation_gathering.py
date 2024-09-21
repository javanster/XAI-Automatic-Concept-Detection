import gymnasium as gym
import avocado_run
from ObservationHandler import ObservationHandler
from data.observations.custom_entity_positions import custom_positions

env = gym.make(id="AvocadoRun-v0", num_avocados=1)

observation_handler = ObservationHandler(env=env)

observation_handler.save_random_observations(
    num_observations=300,
    file_path="data/observations/random_observations.npy"
)

for key in custom_positions.keys():
    observation_dict = custom_positions[key]
    observation_handler.save_custom_observations(
        file_path=f"data/observations/{key}_observations.npy",
        agent_position_list=observation_dict["agent_position_list"],
        avocado_positions_list=observation_dict["avocado_positions_list"],
        enemy_positions_list=observation_dict["enemy_positions_list"],
    )

for i in range(3):
    observation_handler.show_observation(
        file_path="data/observations/do_nothing_good_move_enemy_focused_observations.npy",
        observation_index=i
    )
