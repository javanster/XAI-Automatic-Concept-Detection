from ConceptDetector import ConceptDetector
import gymnasium as gym
import avocado_run

normal_env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)
env_1_enemy = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=1)


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
        "is_concept_in_observation": ConceptDetector.is_concept_6_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_7_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_8_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_9_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_10_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_11_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_12_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_13_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_14_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_15_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_16_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_17_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_18_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_19_present
    },
    {
        "env": env_1_enemy,
        "is_concept_in_observation": ConceptDetector.is_concept_20_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_21_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_22_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_23_present
    },
    {
        "env": normal_env,
        "is_concept_in_observation": ConceptDetector.is_concept_24_present
    },
]
