from ConceptDetector import ConceptDetector
from ObservationHandler import ObservationHandler
import gymnasium as gym
import avocado_run

OBSERVATION_COUNT = 1000  # For each concept, how many observations should be saved containing the concept, and how many random observations should be saved not containing the concept
CONCEPT_FILE_PATH_BASE = "tcav_data/observations/observations_containing_concept_"
OTHER_FILE_PATH_BASE = "tcav_data/observations/observations_not_containing_concept_"


def get_concept_file_path(concept_num):
    return f"{CONCEPT_FILE_PATH_BASE}{concept_num}.npy"


def get_other_file_path(concept_num):
    return f"{OTHER_FILE_PATH_BASE}{concept_num}.npy"


normal_env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)

concept_observations_args = [
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_1_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_2_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_3_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_4_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_5_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_6_present,
        "avocado_starting_positions": [(9, 9)],
        "agent_starting_position": (8, 9)
    },
]

for i, args in enumerate(concept_observations_args):
    concept_num = i + 1
    ObservationHandler.save_observations_for_tcav(
        **args,
        num_observations_for_each=OBSERVATION_COUNT,
        file_path_concept=get_concept_file_path(concept_num),
        file_path_other=get_other_file_path(concept_num)
    )

# View 4 observations of concept 6
for i in range(4):
    ObservationHandler.show_observation(
        file_path=get_concept_file_path(6),
        observation_index=i
    )
