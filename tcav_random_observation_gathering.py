from utils import ObservationHandler
import gymnasium as gym
import avocado_run
import os


def obtain_policy_specific_observations(envs, env_names, observations_n):

    if len(envs) != len(env_names):
        raise ValueError("Provided number of envs and env_names must be equal")

    for i in range(len(envs)):
        env = envs[i]
        env_name = env_names[i]

        directory_path = "tcav_data/observations/random_observations"
        file_path = f"{directory_path}/{env_name}.npy"

        ObservationHandler.save_random_observations(
            envs=[env],
            num_total_observations=observations_n,
            file_path=file_path
        )

    file_path = os.path.join(directory_path, "info.txt")
    with open(file_path, "w") as f:
        f.write(
            f"Observation count in each file: {observations_n}\n")


if __name__ == "__main__":
    env_2_enemies_1_avocados = gym.make(id="AvocadoRun-v0",
                                        num_avocados=1, num_enemies=2
                                        )
    env_1_enemies_1_avocados = gym.make(id="AvocadoRun-v0",
                                        num_avocados=1, num_enemies=1)

    envs = [env_2_enemies_1_avocados, env_1_enemies_1_avocados]
    env_names = ["env_2_enemies_1_avocados", "env_1_enemies_1_avocados"]

    obtain_policy_specific_observations(
        envs=envs,
        env_names=env_names,
        observations_n=5000,
    )
