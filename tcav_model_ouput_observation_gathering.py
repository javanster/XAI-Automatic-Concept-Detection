from tcav import ConceptDetector
from utils import ObservationHandler
from keras.api.saving import load_model
import gymnasium as gym
import avocado_run
import os


######################## GATHERING OF OBSERVATIONS WHERE THE MODEL OUTPUTS A SPECIFIC CLASS AS THE CLASS OF HIGHEST VALUE ########################


def obtain_model_specific_observations(train_run_name, model_name, env, env_name, output_classes, observations_n):
    model = load_model(f"models/{train_run_name}/{model_name}.keras")

    directory_path = f"tcav_data/observations/model_specific_output_class/{train_run_name}/{model_name}/{env_name}"

    file_path_base = f"{directory_path}/observations_where_best_class_"

    ObservationHandler.save_observations_specific_output_classes(
        env=env,
        model=model,
        output_classes=output_classes,
        num_observations=observations_n,
        file_path_base=file_path_base
    )

    file_path = os.path.join(directory_path, "info.txt")
    with open(file_path, "w") as f:
        f.write(f"Observation count in each file: {observations_n}\n")


if __name__ == "__main__":
    env_1_enemy = gym.make(id="AvocadoRun-v0", num_enemies=1)
    env_name = "env_1_enemies_1_avocados"

    output_classes = [action for action in range(env_1_enemy.action_space.n)]
    train_run_name = "dutiful_frog_68"
    model_name = "best_model"

    obtain_model_specific_observations(
        train_run_name=train_run_name,
        model_name=model_name,
        env=env_1_enemy,
        env_name=env_name,
        output_classes=output_classes,
        observations_n=5000
    )
