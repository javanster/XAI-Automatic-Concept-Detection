from sklearn.linear_model import LogisticRegression
from utils import AvocadoRunObservationHandler
from keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from tcav.concept_detector import ConceptDetector
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class CASSObtainer:
    """
    A class to obtain Concept Activation Separation Score (CASS) for various concepts across layers in a neural network.

    This class computes the Concept Activation Separation Score (CASS) by training a logistic regression classifier 
    to distinguish between activations of concepts and non-concepts across different layers of a neural network. The 
    CASS measures how well the network can separate concept activations from non-concept activations at different layers 
    and training steps.

    Parameters
    ----------
    concept_observations_dict : dict
        A dictionary containing file paths to the observations for each concept and non-concept. It should map each 
        concept to a dictionary with keys "concept_obs_filepath" and "not_concept_obs_filepath".

    """

    def __init__(self, concept_observations_dict):
        self.concept_observations_dict = concept_observations_dict
        self.concepts = [c for c in concept_observations_dict.keys()]
        self.classifier_scores_df = pd.DataFrame(
            columns=["concept_index", "concept_name", "layer_index",
                     "layer_name", "classifier_score", "training_step"]
        )

    @staticmethod
    def _get_activations(model, observations):
        activations = model.predict(observations)

        # If the layer is a Conv2D layer, flatten the output
        if len(activations.shape) == 4:
            batch_size, height, width, channels = activations.shape
            activations = activations.reshape(
                batch_size, height * width * channels)

        return activations

    def calculate_cass(self, model, training_step, file_path):
        layer_indexes = [l for l in range(len(model.layers))]

        classifier_scores = []

        total_len = len(layer_indexes) * len(self.concepts)
        with tqdm(total=total_len, unit="score") as pbar:
            for layer_index in layer_indexes:

                activation_model_for_layer = Model(
                    inputs=model.layers[0].input,
                    outputs=model.layers[layer_index].output
                )

                for concept in self.concepts:

                    observations_with_concept = AvocadoRunObservationHandler.load_observations(
                        file_path=self.concept_observations_dict.get(
                            concept)["concept_obs_filepath"],
                        normalize=True,
                    )

                    random_observations_without_concept = AvocadoRunObservationHandler.load_observations(
                        file_path=self.concept_observations_dict.get(
                            concept)["not_concept_obs_filepath"],
                        normalize=True,
                    )

                    concept_activations_for_layer = CASSObtainer._get_activations(
                        model=activation_model_for_layer,
                        observations=observations_with_concept
                    )

                    non_concept_activations_for_layer = CASSObtainer._get_activations(
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

                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_val)
                    original_accuracy = accuracy_score(y_val, y_pred)

                    # Ensures score is between 0 and 1, where 0 is equal to random guess or worse
                    classifier_score = max(2 * (original_accuracy - 0.5), 0)

                    classifier_scores.append(
                        {
                            "concept_index": concept,
                            "concept_name": ConceptDetector.concept_name_dict[concept],
                            "layer_index": layer_index,
                            "layer_name": model.layers[layer_index].name,
                            "classifier_score": classifier_score,
                            "training_step": training_step
                        }
                    )

                    pbar.update(1)
                    print(
                        f"Classifier score for concept {concept} in layer {layer_index} at training step {training_step}: {classifier_score}")

        self.classifier_scores_df = pd.concat(
            [self.classifier_scores_df, pd.DataFrame(classifier_scores)],
            ignore_index=True
        )

        self.classifier_scores_df.to_csv(file_path, index=False)
