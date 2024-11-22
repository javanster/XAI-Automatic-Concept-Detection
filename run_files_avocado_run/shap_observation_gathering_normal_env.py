import gymnasium as gym
import avocado_run
from utils import AvocadoRunObservationHandler
import time


#################### CUSTOM ENTITY POSITIONS FOR OOD OBSERVATIONS ####################

# Positions for entities in observations (x, y), for use in SHAP

up_good_action_avocado_focused = {
    "agent_position_list": [(2, 9), (2, 4), (6, 9)],
    "avocado_positions_list": [[(1, 5)], [(2, 1)], [(6, 7)]],
    "enemy_positions_list": [[(9, 8), (7, 9)], [(7, 6), (2, 8)], [(1, 3), (5, 2)]]
}


up_good_action_enemy_focused = {
    "agent_position_list": [(0, 6), (6, 3), (1, 5),],
    "avocado_positions_list": [[(6, 8)], [(6, 8)], [(6, 7)]],
    "enemy_positions_list": [[(0, 8), (2, 6)], [(5, 4), (7, 4)], [(1, 6), (4, 5)]]
}

right_good_action_avocado_focused = {
    "agent_position_list": [(1, 4), (0, 1), (1, 9),],
    "avocado_positions_list": [[(7, 4)], [(3, 1)], [(8, 8)],],
    "enemy_positions_list": [[(1, 0), (2, 9)], [(0, 6), (8, 3)], [(1, 1), (3, 2)]]
}

right_good_action_enemy_focused = {
    "agent_position_list": [(2, 8), (2, 0), (5, 1),],
    "avocado_positions_list": [[(0, 6)], [(0, 5)], [(1, 8)],],
    "enemy_positions_list": [[(1, 7), (1, 9)], [(0, 0), (2, 2)], [(3, 1), (5, 2)],]
}

down_good_action_avocado_focused = {
    "agent_position_list": [(5, 7), (7, 1), (1, 1),],
    "avocado_positions_list": [[(5, 8)], [(7, 3)], [(0, 6)],],
    "enemy_positions_list": [[(7, 1), (1, 2)], [(2, 7), (7, 7)], [(6, 1), (8, 4)],]
}


down_good_action_enemy_focused = {
    "agent_position_list": [(9, 2), (2, 0), (8, 5),],
    "avocado_positions_list": [[(1, 1)], [(9, 0)], [(1, 1)],],
    "enemy_positions_list": [[(7, 2), (9, 0)], [(0, 0), (4, 0)], [(6, 5), (8, 4)],]
}

left_good_action_avocado_focused = {
    "agent_position_list": [(8, 2), (8, 4), (9, 8),],
    "avocado_positions_list": [[(1, 2)], [(7, 4)], [(3, 9)],],
    "enemy_positions_list": [[(2, 8), (8, 6)], [(2, 1), (4, 6)], [(5, 2), (9, 1)],]
}

left_good_action_enemy_focused = {
    "agent_position_list": [(6, 0), (8, 3), (4, 8),],
    "avocado_positions_list": [[(9, 2)], [(9, 8)], [(8, 4)],],
    "enemy_positions_list": [[(7, 1), (8, 0)], [(9, 2), (9, 4)], [(3, 6), (5, 8)],]
}

do_nothing_good_action_enemy_focused = {
    "agent_position_list": [(5, 4), (9, 0), (0, 9),],
    "avocado_positions_list": [[(1, 1)], [(3, 4)], [(8, 9)],],
    "enemy_positions_list": [[(1, 1), (8, 8)], [(7, 0), (9, 2)], [(0, 7), (2, 9)],]
}

normal_env_entity_positions = {
    "up_good_action_avocado_focused": up_good_action_avocado_focused,
    "up_good_action_enemy_focused": up_good_action_enemy_focused,
    "right_good_action_avocado_focused": right_good_action_avocado_focused,
    "right_good_action_enemy_focused": right_good_action_enemy_focused,
    "down_good_action_avocado_focused": down_good_action_avocado_focused,
    "down_good_action_enemy_focused": down_good_action_enemy_focused,
    "left_good_action_avocado_focused": left_good_action_avocado_focused,
    "left_good_action_enemy_focused": left_good_action_enemy_focused,
    "do_nothing_good_action_enemy_focused": do_nothing_good_action_enemy_focused,
}


#################### OBSERVATION GATHERING SCRIPT ####################


env = gym.make(id="AvocadoRun-v0", num_avocados=1)


AvocadoRunObservationHandler.save_random_observations(
    envs=[env],
    num_total_observations=1000,
    file_path=f"shap_data/observations/normal_environment/random_observations.npy"
)

for key in normal_env_entity_positions.keys():
    observation_dict = normal_env_entity_positions[key]
    AvocadoRunObservationHandler.save_custom_observations(
        envs=[env for _ in range(3)],
        file_path=f"shap_data/observations/normal_environment/{key}_observations.npy",
        agent_position_list=observation_dict["agent_position_list"],
        avocado_positions_list=observation_dict["avocado_positions_list"],
        enemy_positions_list=observation_dict["enemy_positions_list"],
    )

for i in range(3):
    AvocadoRunObservationHandler.show_observation(
        file_path=f"shap_data/observations/normal_environment/do_nothing_good_action_enemy_focused_observations.npy",
        observation_index=i
    )
