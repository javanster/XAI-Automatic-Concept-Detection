import gymnasium as gym
import avocado_run
from ObservationHandler import ObservationHandler
from data.observations.ood.no_enemies.ood_1_avo_stuck_positions import ood_1_avo_stuck_positions
import time

env_0_enemies_1_avocado = gym.make(
    id="AvocadoRun-v0", num_avocados=1, num_enemies=0)
env_1_enemy_1_avocado = gym.make(
    id="AvocadoRun-v0", num_avocados=1, num_enemies=1)
env_0_enemies_2_avocados = gym.make(
    id="AvocadoRun-v0", num_avocados=2, num_enemies=0)

observation_handler = ObservationHandler()

time_stamp = int(time.time())

observation_handler.save_random_observations(
    envs=[env_0_enemies_1_avocado, env_1_enemy_1_avocado,
          env_0_enemies_2_avocados],
    num_total_observations=1000,
    file_path=f"data/observations/ood/no_enemies/random_observations_{time_stamp}.npy"
)

for key in ood_1_avo_stuck_positions.keys():
    observation_dict = ood_1_avo_stuck_positions[key]
    observation_handler.save_custom_observations(
        envs=[env_0_enemies_1_avocado, env_1_enemy_1_avocado,
              env_0_enemies_2_avocados],
        file_path=f"data/observations/ood/no_enemies/{key}_observations.npy",
        agent_position_list=observation_dict["agent_position_list"],
        avocado_positions_list=observation_dict["avocado_positions_list"],
        enemy_positions_list=observation_dict["enemy_positions_list"],
    )

for key in ood_1_avo_stuck_positions.keys():
    observation_dict = ood_1_avo_stuck_positions[key]
    for i in range(len(observation_dict["agent_position_list"])):
        observation_handler.show_observation(
            file_path=f"data/observations/ood/no_enemies/{key}_observations.npy",
            observation_index=i
        )
