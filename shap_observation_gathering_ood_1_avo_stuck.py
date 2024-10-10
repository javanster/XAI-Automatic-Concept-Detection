import gymnasium as gym
import avocado_run
from ObservationHandler import ObservationHandler
import time

#################### CUSTOM ENTITY POSITIONS FOR OOD OBSERVATIONS ####################

# Positions for entities in observations (x, y)

# Index 0 in each list represents observations with no enemies and a single avocado, where the agent got stuck
# Index 1 in each list introduces 1 enemy to the environment
# Index 2 in each list introduces an additional avocado to the environment, in the same location as the enemy


wild_glitter_14_best_model_ood_1_avo_stuck_0 = {
    "agent_position_list": [(2, 7), (2, 7), (2, 7)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)], [(6, 5), (3, 9)]],
    "enemy_positions_list": [[], [(3, 9)], []]
}

wild_glitter_14_best_model_ood_1_avo_stuck_1 = {
    "agent_position_list": [(6, 3), (6, 3), (6, 3)],
    "avocado_positions_list": [[(6, 5)], [(6, 5)], [(6, 5), (7, 3)]],
    "enemy_positions_list": [[], [(7, 3)], []]
}

wild_glitter_14_best_model_ood_1_avo_stuck_2 = {
    "agent_position_list": [(6, 9), (6, 9), (6, 9)],
    "avocado_positions_list": [[(0, 6)], [(0, 6)], [(0, 6), (7, 7)]],
    "enemy_positions_list": [[], [(7, 7)], []]
}


wild_glitter_14_best_model_ood_1_avo_stuck_positions = {
    "ood_1_avo_stuck_0": wild_glitter_14_best_model_ood_1_avo_stuck_0,
    "ood_1_avo_stuck_1": wild_glitter_14_best_model_ood_1_avo_stuck_1,
    "ood_1_avo_stuck_2": wild_glitter_14_best_model_ood_1_avo_stuck_2,
}

eager_disco_16_best_model_ood_1_avo_stuck_0 = {
    "agent_position_list": [(6, 9), (6, 9), (6, 9)],
    "avocado_positions_list": [[(5, 9)], [(5, 9)], [(5, 9), (9, 9)]],
    "enemy_positions_list": [[], [(9, 9)], []]
}

eager_disco_16_best_model_ood_1_avo_stuck_1 = {
    "agent_position_list": [(7, 7), (7, 7), (7, 7)],
    "avocado_positions_list": [[(7, 8)], [(7, 8)], [(7, 8), (7, 5)]],
    "enemy_positions_list": [[], [(7, 5)], []]
}

eager_disco_16_best_model_ood_1_avo_stuck_2 = {
    "agent_position_list": [(9, 9), (9, 9), (9, 9)],
    "avocado_positions_list": [[(4, 8)], [(4, 8)], [(4, 8), (9, 7)]],
    "enemy_positions_list": [[], [(9, 7)], []]
}

eager_disco_16_best_model_ood_1_avo_stuck_positions = {
    "ood_1_avo_stuck_0": eager_disco_16_best_model_ood_1_avo_stuck_0,
    "ood_1_avo_stuck_1": eager_disco_16_best_model_ood_1_avo_stuck_1,
    "ood_1_avo_stuck_2": eager_disco_16_best_model_ood_1_avo_stuck_2,
}


#################### OBSERVATION GATHERING SCRIPT ####################

env_0_enemies_1_avocado = gym.make(
    id="AvocadoRun-v0", num_avocados=1, num_enemies=0)
env_1_enemy_1_avocado = gym.make(
    id="AvocadoRun-v0", num_avocados=1, num_enemies=1)
env_0_enemies_2_avocados = gym.make(
    id="AvocadoRun-v0", num_avocados=2, num_enemies=0)


configs = [
    {
        "train_run_name": "wild_glitter_14",
        "model_name": "best_model",
        "envs": [env_0_enemies_1_avocado, env_1_enemy_1_avocado, env_0_enemies_2_avocados],
        "ood_observations_dict": wild_glitter_14_best_model_ood_1_avo_stuck_positions,
    },
    {
        "train_run_name": "eager_disco_16",
        "model_name": "best_model",
        "envs": [env_0_enemies_1_avocado, env_1_enemy_1_avocado, env_0_enemies_2_avocados],
        "ood_observations_dict": eager_disco_16_best_model_ood_1_avo_stuck_positions,
    }
]

for config in configs:

    train_run_name = config["train_run_name"]
    model_name = config["model_name"]

    ObservationHandler.save_random_observations(
        envs=[env_0_enemies_1_avocado, env_1_enemy_1_avocado,
              env_0_enemies_2_avocados],
        num_total_observations=1000,
        file_path=f"shap_data/observations/ood/1_avo_stuck/{train_run_name}/{model_name}/random_observations.npy"
    )

    for key in config["ood_observations_dict"].keys():
        observation_dict = config["ood_observations_dict"][key]
        ObservationHandler.save_custom_observations(
            envs=[env_0_enemies_1_avocado, env_1_enemy_1_avocado,
                  env_0_enemies_2_avocados],
            file_path=f"shap_data/observations/ood/1_avo_stuck/{train_run_name}/{model_name}/{key}_observations.npy",
            agent_position_list=observation_dict["agent_position_list"],
            avocado_positions_list=observation_dict["avocado_positions_list"],
            enemy_positions_list=observation_dict["enemy_positions_list"],
        )

    for key in config["ood_observations_dict"].keys():
        observation_dict = config["ood_observations_dict"][key]
        for i in range(len(observation_dict["agent_position_list"])):
            ObservationHandler.show_observation(
                file_path=f"shap_data/observations/ood/1_avo_stuck/{train_run_name}/{model_name}/{key}_observations.npy",
                observation_index=i
            )
