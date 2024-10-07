from ConceptDetector import ConceptDetector
from ObservationHandler import ObservationHandler
import gymnasium as gym
import avocado_run


env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)

ObservationHandler.save_observations_for_tcav(
    env=env,
    num_observations_for_each=50,
    file_path_concept="tcav_data/observations/concept_2_data.npy",
    file_path_other="tcav_data/observations/random_data_not_containing_concept_2.npy",
    is_concept_in_observation=ConceptDetector.is_concept_2_present
)

for i in range(10):
    ObservationHandler.show_observation(
        file_path="tcav_data/observations/concept_2_data.npy",
        observation_index=i
    )
