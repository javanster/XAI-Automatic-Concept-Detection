import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent
from ConceptDetector import ConceptDetector

concept_observations_dict = {
    c: {
        "concept_obs_filepath": f"tcav_data/observations/concept_observations/observations_containing_concept_{c}.npy",
        "not_concept_obs_filepath": f"tcav_data/observations/concept_observations/observations_not_containing_concept_{c}.npy",
    } for c in ConceptDetector.concept_name_dict.keys()
}

# Based on best hyperparams found using Bayesian Hyperparameter Optimization - See wandb sandy-sweep-16
config = {
    "project_name": "AvocadoRun",
    "replay_buffer_size": 50_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 64,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 1000,
    "learning_rate": 0.001,
    "prop_steps_epsilon_decay": 0.9,  # The proportion of steps epsilon should decay
    "starting_epsilon": 1,
    "min_epsilon": 0.05,
    "steps_to_train": 2_000_000,
    "episode_metrics_window": 100,  # Number of episodes to take into account for metrics
    "obtain_cav_sensitivities_every": 50_000  # Steps
}

env = gym.make(id="AvocadoRun-v0")

agent = DoubleDQNAgent(
    env=env,
)

agent.train_with_cav_checkpoints(
    config=config,
    concept_observations_dict=concept_observations_dict,
    cav_file_path="tcav_data/cav_sensitivities_during_training/newest_model_sensitivities.csv",
    use_wandb=True
)
