from ConceptDetector import ConceptDetector
from ObservationHandler import ObservationHandler
import gymnasium as gym
import avocado_run
from keras.api.saving import load_model


######################## GATHERING OF OBSERVATIONS WHERE CERTAIN CONCEPTS EITHER ARE OR ARE NOT PRESENT IN THE OBSERVATION ########################

OBSERVATION_COUNT = 1000  # For each concept, how many observations should be saved containing the concept, and how many random observations should be saved not containing the concept
CONCEPT_FILE_PATH_BASE = "tcav_data/observations/concept_observations/observations_containing_concept_"
OTHER_FILE_PATH_BASE = "tcav_data/observations/concept_observations/observations_not_containing_concept_"


def get_concept_file_path(concept_num):
    return f"{CONCEPT_FILE_PATH_BASE}{concept_num}.npy"


def get_other_file_path(concept_num):
    return f"{OTHER_FILE_PATH_BASE}{concept_num}.npy"


normal_env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)

concept_observations_args = [
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_0_present
    },
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
]

for i, args in enumerate(concept_observations_args):
    ObservationHandler.save_observations_given_concept(
        **args,
        num_observations_for_each=OBSERVATION_COUNT,
        file_path_concept=get_concept_file_path(i),
        file_path_other=get_other_file_path(i)
    )

# View 4 observations of each concept
for i in range(len(concept_observations_args)):
    for j in range(4):
        ObservationHandler.show_observation(
            file_path=get_concept_file_path(i),
            observation_index=j
        )


######################## GATHERING OF OBSERVATIONS WHERE THE MODEL OUTPUTS A SPECIFIC CLASS AS THE CLASS OF HIGHEST VALUE ########################

OUTPUT_CLASSES = [action for action in range(normal_env.action_space.n)]
TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"

model = load_model(f"models/{TRAIN_RUN_NAME}/{MODEL_NAME}.keras")


file_path_base = f"tcav_data/observations/model_specific/{TRAIN_RUN_NAME}/{MODEL_NAME}/observations_where_best_class_"

ObservationHandler.save_observations_specific_output_classes(
    env=normal_env,
    model=model,
    output_classes=OUTPUT_CLASSES,
    num_observations=1000,
    file_path_base=file_path_base
)
