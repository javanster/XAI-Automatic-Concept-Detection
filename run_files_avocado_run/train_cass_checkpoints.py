import gymnasium as gym
import avocado_run
from agents import DoubleDQNAgent
from tcav import ConceptDetector


def train_with_cass_checkpoints(env, env_name, batch_n, config, concept_indexes, classifier_scores_file_path, use_wandb):

    if len(concept_indexes) < 1:
        raise ValueError(
            "concept_indexes must contain at least one concept index")

    for ci in concept_indexes:
        if ci not in ConceptDetector.concept_name_dict.keys():
            raise ValueError(
                "All given concept indexes must be defined in ConceptDetector")

    concept_observations_dict = {
        c: {
            "concept_obs_filepath": f"tcav_data/observations/concept_observations/{env_name}/batch_{batch_n}/observations_containing_concept_{c}.npy",
            "not_concept_obs_filepath": f"tcav_data/observations/concept_observations/{env_name}/batch_{batch_n}/observations_not_containing_concept_{c}.npy",
        } for c in concept_indexes
    }

    print(concept_observations_dict)

    agent = DoubleDQNAgent(
        env=env,
    )

    agent.train_with_concept_classifier_checkpoints(
        config=config,
        concept_observations_dict=concept_observations_dict,
        classifier_scores_file_path=classifier_scores_file_path,
        use_wandb=use_wandb
    )


if __name__ == "__main__":
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
        "steps_to_train": 300_000,
        "episode_metrics_window": 100,  # Number of episodes to take into account for metrics

        # Number of steps to pass before obtaining concept activation separation scores
        "obtain_cass_every": 50_000
    }

    env = gym.make(
        id="AvocadoRun-v0",
        num_avocados=1,
        num_enemies=1
    )

    train_with_cass_checkpoints(
        env=env,
        env_name="env_1_enemies_1_avocados",
        batch_n=0,
        config=config,
        concept_indexes=[0, 1, 2, 3, 9, 10, 11, 12],
        classifier_scores_file_path="tcav_data/cassat/newest_model_cass_scores.csv",
        use_wandb=False
    )
