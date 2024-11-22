from keras.api.saving import load_model
from utils import AvocadoRunObservationHandler
from keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tcav import ConceptDetector
import pandas as pd
import os


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)


def get_activations(activation_model, observations):
    activations = activation_model.predict(observations)

    # If the layer is a Conv2D layer, flatten the output
    if len(activations.shape) == 4:
        batch_size, height, width, channels = activations.shape
        activations = activations.reshape(
            batch_size, height * width * channels)

    return activations


def obtain_cavs(layer_indexes, concept_indexes, observation_batches, model, env_type, train_run_name, model_name):

    for ci in concept_indexes:
        if ci not in ConceptDetector.concept_name_dict.keys():
            raise ValueError(
                "Provided concept indexes must be defined in ConceptDetector")

    for observation_batch in observation_batches:

        classifier_results = []

        for layer_index in layer_indexes:

            activation_model_for_layer = Model(
                inputs=model.layers[0].input,
                outputs=model.layers[layer_index].output
            )

            for concept_index in concept_indexes:

                observations_with_concept = AvocadoRunObservationHandler.load_observations(
                    file_path=f"tcav_data/observations/concept_observations/{env_type}/batch_{observation_batch}/observations_containing_concept_{concept_index}.npy",
                    normalize=True,
                )

                random_observations_without_concept = AvocadoRunObservationHandler.load_observations(
                    file_path=f"tcav_data/observations/concept_observations/{env_type}/batch_{observation_batch}/observations_not_containing_concept_{concept_index}.npy",
                    normalize=True,
                )

                concept_activations_for_layer = get_activations(
                    activation_model=activation_model_for_layer,
                    observations=observations_with_concept
                )

                non_concept_activations_for_layer = get_activations(
                    activation_model=activation_model_for_layer,
                    observations=random_observations_without_concept
                )

                labels_with_concept = np.ones(
                    concept_activations_for_layer.shape[0])
                labels_without_concept = np.zeros(
                    non_concept_activations_for_layer.shape[0])

                X = np.vstack([concept_activations_for_layer,
                               non_concept_activations_for_layer])
                y = np.concatenate(
                    [labels_with_concept, labels_without_concept])

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                clf = LogisticRegression(
                    max_iter=1000, random_state=42)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_val)
                original_accuracy = accuracy_score(y_val, y_pred)

                # Ensures score is between 0 and 1, where 0 is equal to random guess or worse
                classifier_score = max(2 * (original_accuracy - 0.5), 0)

                classifier_results.append(
                    {
                        "concept_index": concept_index,
                        "concept_name": ConceptDetector.concept_name_dict[concept_index],
                        "layer_index": layer_index,
                        "layer_name": model.layers[layer_index].name,
                        "observations_with_concept_n": len(observations_with_concept),
                        "observations_without_concept_n": len(random_observations_without_concept),
                        "classifier_score": classifier_score
                    }
                )

                cav = clf.coef_[0]

                cav_file_path = f"tcav_data/cavs/{train_run_name}/{model_name}/observation_batch_{observation_batch}/concept_{concept_index}_layer_{layer_index}_cav.npy"
                ensure_dir_exists(cav_file_path)
                np.save(cav_file_path, cav)

                print(
                    f"CAV obtained for concept {concept_index} in layer {layer_index}, with classifier score {classifier_score} - Batch: {observation_batch}")

        classifier_results_df = pd.DataFrame(classifier_results)

        results_file_path = f"tcav_data/cavs/{train_run_name}/{model_name}/observation_batch_{observation_batch}/classifier_scores.csv"
        ensure_dir_exists(results_file_path)
        classifier_results_df.to_csv(results_file_path, index=False)

        print(f"Batch {observation_batch} complete")


if __name__ == "__main__":
    train_run_name = "dutiful_frog_68"
    model_name = "best_model"

    model = load_model(f"models/{train_run_name}/{model_name}.keras")

    layer_indexes = [l for l in range(len(model.layers))]
    concept_indexes = [0, 1, 2, 3, 9, 10, 11, 12]
    observation_batches = [b for b in range(500)]
    env_type = "env_1_enemies_1_avocados"

    obtain_cavs(
        layer_indexes=layer_indexes,
        concept_indexes=concept_indexes,
        observation_batches=observation_batches,
        env_type=env_type,
        model=model,
        train_run_name=train_run_name,
        model_name=model_name,
    )
