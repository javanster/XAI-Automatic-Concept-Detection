import gymnasium as gym
import avocado_run
from ObservationHandler import ObservationHandler
from data.observations.normal_environment.normal_env_entity_positions import normal_env_entity_positions
import time

env = gym.make(id="AvocadoRun-v0", num_avocados=1)

observation_handler = ObservationHandler()

observation_handler.save_random_observations(
    envs=[env],
    num_total_observations=1000,
    file_path=f"data/observations/normal_environment/random_observations_normal_env_{int(time.time())}.npy"
)

for key in normal_env_entity_positions.keys():
    observation_dict = normal_env_entity_positions[key]
    observation_handler.save_custom_observations(
        envs=[env for _ in range(3)],
        file_path=f"data/observations/normal_environment/{key}_observations.npy",
        agent_position_list=observation_dict["agent_position_list"],
        avocado_positions_list=observation_dict["avocado_positions_list"],
        enemy_positions_list=observation_dict["enemy_positions_list"],
    )

for i in range(3):
    observation_handler.show_observation(
        file_path="data/observations/normal_environment/do_nothing_good_action_enemy_focused_observations.npy",
        observation_index=i
    )
