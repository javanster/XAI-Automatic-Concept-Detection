import gymnasium as gym
import avocado_run
from ObservationHandler import ObservationHandler
from data.observations.ood.no_enemies.ood_no_enemies_entity_positions import ood_no_enemies_positions
import time

env_0_enemies = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=0)
env_2_enemies = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)

observation_handler = ObservationHandler()

time_stamp = int(time.time())

observation_handler.save_random_observations(
    envs=[env_0_enemies, env_2_enemies],
    num_total_observations=1000,
    file_path=f"data/observations/ood/no_enemies/random_observations_0_and_2_enemies_{time_stamp}.npy"
)

for key in ood_no_enemies_positions.keys():
    observation_dict = ood_no_enemies_positions[key]
    observation_handler.save_custom_observations(
        envs=[env_0_enemies, env_2_enemies],
        file_path=f"data/observations/ood/no_enemies/{key}_observations.npy",
        agent_position_list=observation_dict["agent_position_list"],
        avocado_positions_list=observation_dict["avocado_positions_list"],
        enemy_positions_list=observation_dict["enemy_positions_list"],
    )

for key in ood_no_enemies_positions.keys():
    observation_dict = ood_no_enemies_positions[key]
    for i in range(len(observation_dict["agent_position_list"])):
        observation_handler.show_observation(
            file_path=f"data/observations/ood/no_enemies/{key}_observations.npy",
            observation_index=i
        )
