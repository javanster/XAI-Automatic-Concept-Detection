from utils import ObservationHandler
from concept_observation_args import concept_observations_args
from tqdm import tqdm
import os


def save_batch_info(batch, observation_count, env):
    """
    Saves a text file in the batch directory with the information about observation count.
    """
    batch_dir = f"tcav_data/observations/concept_observations/{get_env_name(env)}/batch_{batch}"
    os.makedirs(batch_dir, exist_ok=True)

    file_path = os.path.join(batch_dir, "batch_info.txt")
    with open(file_path, "w") as f:
        f.write(f"Observation count in each file: {observation_count}\n")


def get_env_name(env):
    enemies_n = env.unwrapped.num_enemies
    avocados_n = env.unwrapped.num_avocados

    return f"env_{enemies_n}_enemies_{avocados_n}_avocados"


def get_concept_file_path(batch, concept_index, env):
    return f"tcav_data/observations/concept_observations/{get_env_name(env)}/batch_{batch}/observations_containing_concept_{concept_index}.npy"


def get_other_file_path(batch, concept_index, env):
    return f"tcav_data/observations/concept_observations/{get_env_name(env)}/batch_{batch}/observations_not_containing_concept_{concept_index}.npy"


def gather_concept_observation_bathes(batches_n, observations_n, concept_observations_args):
    with tqdm(total=batches_n, unit="batch") as pbar:
        for batch in range(BATCHES):

            for args in concept_observations_args:

                concept_index = args["concept_index"]

                ObservationHandler.save_observations_given_concept(
                    **args,
                    num_observations_for_each=observations_n,
                    file_path_concept=get_concept_file_path(
                        batch, concept_index, args["env"]),
                    file_path_other=get_other_file_path(
                        batch, concept_index, args["env"])
                )

                save_batch_info(batch, observations_n, args["env"])

            pbar.update(1)


if __name__ == "__main__":

    # For each concept, how many observations should be saved containing the concept, and how many random observations should be saved not containing the concept
    NUM_OBSERVATIONS_FOR_EACH = 1500
    BATCHES = 1

    # -------- REMEMBER TO MODIFY concept_observations_args BEFORE RUNNING

    c_a_args = concept_observations_args

    gather_concept_observation_bathes(
        batches_n=BATCHES, observations_n=NUM_OBSERVATIONS_FOR_EACH, concept_observations_args=c_a_args)
