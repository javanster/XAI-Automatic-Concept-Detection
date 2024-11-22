from utils import AvocadoRunObservationHandler
import gymnasium as gym
import avocado_run
import numpy as np
from agents import ArValueIterationAgent
import os

######################## GATHERING OF OBSERVATIONS WHERE AN OPTIMAL POLICY OUTPUTS A SPECIFIC CLASS AS THE CLASS OF HIGHEST VALUE ########################


def obtain_policy_specific_observations(env, env_name, output_classes, observations_n):

    directory_path = f"tcav_data/observations/policy_specific_output_class/{env_name}"

    file_path_base = f"{directory_path}/observations_where_best_class_"

    AvocadoRunObservationHandler.save_observations_optimal_policy_specific_output_classes(
        env=env,
        policy=policy,
        state_to_index=state_to_index,
        output_classes=output_classes,
        num_observations=observations_n,
        file_path_base=file_path_base,
    )

    file_path = os.path.join(directory_path, "info.txt")
    with open(file_path, "w") as f:
        f.write(f"Observation count in each file: {observations_n}\n")


if __name__ == "__main__":
    env_1_enemy = gym.make(id="AvocadoRun-v0", num_enemies=1)
    env_name = "env_1_enemies_1_avocados"

    agent = ArValueIterationAgent(env=env_1_enemy)

    output_classes = [action for action in range(env_1_enemy.action_space.n)]

    policy = np.load("optimal_policy/optimal_policy.npy")
    state_to_index = agent.state_to_index
    index_to_state = agent.index_to_state

    obtain_policy_specific_observations(
        env=env_1_enemy,
        env_name=env_name,
        output_classes=output_classes,
        observations_n=5000
    )
