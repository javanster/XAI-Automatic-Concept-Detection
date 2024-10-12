from keras.api.saving import load_model
from ObservationHandler import ObservationHandler
from keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ConceptDetector import ConceptDetector
import pandas as pd

TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"

model = load_model(f"models/{TRAIN_RUN_NAME}/{MODEL_NAME}.keras")

LAYER_INDEXES = [l for l in range(len(model.layers))]
CONCEPTS = [c for c in ConceptDetector.concept_name_dict.keys()]


def get_activations(activation_model, observations):
    activations = activation_model.predict(observations)

    # If the layer is a Conv2D layer, flatten the output
    if len(activations.shape) == 4:
        batch_size, height, width, channels = activations.shape
        activations = activations.reshape(
            batch_size, height * width * channels)

    return activations


accuracy_results = []

for layer_index in LAYER_INDEXES:

    activation_model_for_layer = Model(
        inputs=model.layers[0].input,
        outputs=model.layers[layer_index].output
    )

    for concept in CONCEPTS:

        observations_with_concept = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/concept_observations/observations_containing_concept_{concept}.npy",
            normalize=True,
        )

        random_observations_without_concept = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/concept_observations/observations_not_containing_concept_{concept}.npy",
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

        labels_with_concept = np.ones(concept_activations_for_layer.shape[0])
        labels_without_concept = np.zeros(
            non_concept_activations_for_layer.shape[0])

        X = np.vstack([concept_activations_for_layer,
                       non_concept_activations_for_layer])
        y = np.concatenate([labels_with_concept, labels_without_concept])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_results.append(
            {
                "concept_index": concept,
                "concept_name": ConceptDetector.concept_name_dict[concept],
                "layer_index": layer_index,
                "layer_name": model.layers[layer_index].name,
                "classifier_accuracy": accuracy
            }
        )

        cav = clf.coef_[0]

        np.save(
            f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/concept_{concept}_layer_{layer_index}_cav.npy", cav)

accuracy_results_df = pd.DataFrame(accuracy_results)

print(accuracy_results_df)

accuracy_results_df.to_csv(
    f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/cav_accuracies.csv", index=False)
