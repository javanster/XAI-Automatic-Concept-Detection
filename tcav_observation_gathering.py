from ConceptDetector import ConceptDetector
from ObservationHandler import ObservationHandler
from keras.api.saving import load_model
from concept_observation_args import concept_observations_args
import gymnasium as gym
import avocado_run


GATHER_CONCEPT_OBSERVATIONS = True
GATHER_MODEL_SPECIFIC_OBSERVATIONS = False


######################## GATHERING OF OBSERVATIONS WHERE CERTAIN CONCEPTS EITHER ARE OR ARE NOT PRESENT IN THE OBSERVATION ########################

if GATHER_CONCEPT_OBSERVATIONS:

    OBSERVATION_COUNT = 1500  # For each concept, how many observations should be saved containing the concept, and how many random observations should be saved not containing the concept
    CONCEPT_FILE_PATH_BASE = "tcav_data/observations/concept_observations/observations_containing_concept_"
    OTHER_FILE_PATH_BASE = "tcav_data/observations/concept_observations/observations_not_containing_concept_"

    def get_concept_file_path(concept_num):
        return f"{CONCEPT_FILE_PATH_BASE}{concept_num}.npy"

    def get_other_file_path(concept_num):
        return f"{OTHER_FILE_PATH_BASE}{concept_num}.npy"

    for i, args in enumerate(concept_observations_args):
        ObservationHandler.save_observations_given_concept(
            **args,
            num_observations_for_each=OBSERVATION_COUNT,
            file_path_concept=get_concept_file_path(i),
            file_path_other=get_other_file_path(i)
        )

    # View 3 observations of each concept
    for i in range(len(concept_observations_args)):
        for j in range(3):
            ObservationHandler.show_observation(
                file_path=get_concept_file_path(i),
                observation_index=j,
                title=f"Concept {i}: {ConceptDetector.concept_name_dict.get(i)}"
            )


######################## GATHERING OF OBSERVATIONS WHERE THE MODEL OUTPUTS A SPECIFIC CLASS AS THE CLASS OF HIGHEST VALUE ########################

if GATHER_MODEL_SPECIFIC_OBSERVATIONS:

    normal_env = gym.make(id="AvocadoRun-v0")

    OUTPUT_CLASSES = [action for action in range(normal_env.action_space.n)]
    TRAIN_RUN_NAME = "mild_cosmos_59"
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
