from ObservationHandler import ObservationHandler
from keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from ConceptDetector import ConceptDetector
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Input, Dense
from keras import regularizers
from keras.api.losses import MeanSquaredError
from keras.api.optimizers import Adam
from tqdm import tqdm


class CavSensitivityObtainer:

    def __init__(self, concept_observations_dict):
        self.concept_observations_dict = concept_observations_dict
        self.concepts = [c for c in ConceptDetector.concept_name_dict.keys()]
        self.cav_sensitivities_df = pd.DataFrame(
            columns=["concept_index", "concept_name", "layer_index",
                     "layer_name", "cav_sensitivity", "training_step"]
        )

    @staticmethod
    def _get_activations(model, observations):
        print(f"Observations shape: {observations.shape}")
        activations = model.predict(observations)

        # If the layer is a Conv2D layer, flatten the output
        if len(activations.shape) == 4:
            batch_size, height, width, channels = activations.shape
            activations = activations.reshape(
                batch_size, height * width * channels)

        print(f"Activations shape: {activations.shape}")
        return activations

    def _create_logistic_regression_model(self, input_dim):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(
            Dense(
                units=1,
                activation='sigmoid',
                kernel_regularizer=regularizers.L1(0.01),
                bias_regularizer=regularizers.L1(0.01),
            )
        )

        return model

    def _calculate_concept_score(self, predictions, labels):
        # Convert predictions to binary using the threshold of 0.5
        binary_predictions = np.round(predictions)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(binary_predictions == labels)

        # Calculate the proportion of correct predictions
        concept_score = correct_predictions / len(labels)

        # Subtract 0.5 and apply ReLU (max with 0)
        concept_score_adjusted = max(concept_score - 0.5, 0)

        return concept_score_adjusted

    def calculate_cav_sensitivity(self, model, training_step, file_path):
        layer_indexes = [l for l in range(len(model.layers))]

        cav_sensitivities = []

        total_len = len(layer_indexes) * len(self.concepts)
        with tqdm(total=total_len, unit="score") as pbar:
            for layer_index in layer_indexes:

                activation_model_for_layer = Model(
                    inputs=model.layers[0].input,
                    outputs=model.layers[layer_index].output
                )

                for concept in self.concepts:

                    print(
                        f"Sensitivity for concept {concept} in layer {layer_index} at step {training_step}")

                    observations_with_concept = ObservationHandler.load_observations(
                        file_path=self.concept_observations_dict.get(
                            concept)["concept_obs_filepath"],
                        normalize=True,
                    )

                    random_observations_without_concept = ObservationHandler.load_observations(
                        file_path=self.concept_observations_dict.get(
                            concept)["not_concept_obs_filepath"],
                        normalize=True,
                    )

                    concept_activations_for_layer = CavSensitivityObtainer._get_activations(
                        model=activation_model_for_layer,
                        observations=observations_with_concept
                    )

                    non_concept_activations_for_layer = CavSensitivityObtainer._get_activations(
                        model=activation_model_for_layer,
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

                    print(f"X shape: {X.shape}, y shape: {y.shape}")

                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    print(
                        f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                    print(
                        f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

                    logistic_model = self._create_logistic_regression_model(
                        input_dim=X_train.shape[1])
                    logistic_model.compile(
                        optimizer=Adam(),
                        loss=MeanSquaredError(),
                        metrics=['accuracy']
                    )

                    logistic_model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=0
                    )

                    predictions = logistic_model.predict(
                        X_val).flatten()

                    print(f"Predictions shape: {predictions.shape}")
                    print(f"y_val shape: {y_val.shape}")

                    cav_sensitivity = self._calculate_concept_score(
                        predictions, y_val)

                    cav_sensitivities.append(
                        {
                            "concept_index": concept,
                            "concept_name": ConceptDetector.concept_name_dict[concept],
                            "layer_index": layer_index,
                            "layer_name": model.layers[layer_index].name,
                            "cav_sensitivity": cav_sensitivity,
                            "training_step": training_step
                        }
                    )

                    pbar.update(1)

        self.cav_sensitivities_df = pd.concat(
            [self.cav_sensitivities_df, pd.DataFrame(cav_sensitivities)],
            ignore_index=True
        )

        self.cav_sensitivities_df.to_csv(file_path, index=False)
