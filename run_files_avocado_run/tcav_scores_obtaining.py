from keras.api.saving import load_model
import numpy as np
from utils import AvocadoRunObservationHandler
from tcav import ConceptDetector
import gymnasium as gym
import avocado_run
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Input
import os

OBSERVATION_TYPES = ["model_specific", "policy_specific", "random"]


def get_observations(observation_type, train_run_name, model_name, target_class, env_name):
    if observation_type not in OBSERVATION_TYPES:
        raise ValueError(
            "Given observation type must be one of the types defined in OBSERVATION_TYPES")

    if observation_type == "model_specific":
        observation_file_path = f"tcav_data/observations/model_specific_output_class/{train_run_name}/{model_name}/{env_name}/observations_where_best_class_{target_class}.npy"
    elif observation_type == "policy_specific":
        observation_file_path = f"tcav_data/observations/policy_specific_output_class/{env_name}/observations_where_best_class_{target_class}.npy"
    elif observation_type == "random":
        observation_file_path = f"tcav_data/observations/random_observations/{env_name}.npy"

    return AvocadoRunObservationHandler.load_observations(
        file_path=observation_file_path,
        normalize=True,
    )


def calculate_gradients(model, layer_index, observations, target_class):
    # Defining activation model, consisting of every layer of the original model up until layer_index, inclusive
    activation_model_for_layer = Model(
        inputs=model.layers[0].input,
        outputs=model.layers[layer_index].output
    )

    activations = activation_model_for_layer.predict(observations)

    # Difining output model, consisting of every layer above layer_index, if any
    if layer_index < len(model.layers) - 1:
        new_input = Input(
            shape=model.layers[layer_index].output.shape[1:])

        x = new_input
        for layer in model.layers[layer_index + 1:]:
            x = layer(x)

        output_model = Model(inputs=new_input, outputs=x)

    else:
        # If layer_index equals the last layer index of the original model, the activations of the activation model are the outputs
        output_model = None

    activations_tensor = tf.convert_to_tensor(activations)

    with tf.GradientTape() as tape:
        tape.watch(activations_tensor)
        output = output_model(
            activations_tensor,
            training=False
        ) if output_model else activations_tensor
        class_output = output[:, target_class]

    gradient = tape.gradient(
        target=class_output,
        sources=activations_tensor
    )

    gradients_flat = tf.reshape(gradient, (gradient.shape[0], -1)).numpy()

    return gradients_flat


def obtain_tcav_scores(
        train_run_name,
        model_name,
        model,
        action_dict,
        layer_indexes,
        concept_indexes,
        target_classes,
        observation_type,
        env_name,
        batches,
):

    for ci in concept_indexes:
        if ci not in ConceptDetector.concept_name_dict.keys():
            raise ValueError(
                "All concept indexes provided must be defined in ConceptDetector")

    total_iterations = len(target_classes) * \
        len(concept_indexes) * len(layer_indexes) * len(batches)

    with tqdm(total=total_iterations, desc="Calculating TCAV scores", unit="score") as pbar:

        for batch in batches:

            tcav_results = []

            for target_class in target_classes:

                observations = get_observations(
                    observation_type=observation_type,
                    train_run_name=train_run_name,
                    model_name=model_name,
                    target_class=target_class,
                    env_name=env_name,
                )

                for layer_index in layer_indexes:

                    gradients = calculate_gradients(
                        model=model,
                        layer_index=layer_index,
                        observations=observations,
                        target_class=target_class
                    )

                    for concept_index in concept_indexes:

                        cav_path = f"tcav_data/cavs/{train_run_name}/{model_name}/observation_batch_{batch}/concept_{concept_index}_layer_{layer_index}_cav.npy"
                        cav = np.load(cav_path)

                        directional_derivatives = np.dot(gradients, cav)
                        tcav_score = np.mean(directional_derivatives > 0)

                        tcav_results.append({
                            'target_class': target_class,
                            'action': action_dict[target_class],
                            'concept_index': concept_index,
                            'concept_name': ConceptDetector.concept_name_dict[concept_index],
                            'layer_index': layer_index,
                            'layer_name': model.layers[layer_index].name,
                            'tcav_score': tcav_score
                        })

                        pbar.update(1)

            tcav_df = pd.DataFrame(tcav_results)

            print(tcav_df)

            file_path = f"tcav_explanations/tcav_scores/{train_run_name}/{model_name}/{observation_type}_observations/tcav_scores_cav_batch_{batch}.csv"

            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            tcav_df.to_csv(
                file_path, index=False)


if __name__ == "__main__":
    train_run_name = "dutiful_frog_68"
    model_name = "best_model"

    model = load_model(f"models/{train_run_name}/{model_name}.keras")
    env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=1)
    env_name = "env_1_enemies_1_avocados"
    action_dict = env.unwrapped.action_dict

    # Not using the two last layers
    layer_indexes = [l for l in range(len(model.layers))]
    concept_indexes = [0, 1, 2, 3, 9, 10, 11, 12]
    target_classes = [action for action in range(env.action_space.n)]
    batches = [b for b in range(500)]

    for observation_type in OBSERVATION_TYPES:

        obtain_tcav_scores(
            train_run_name=train_run_name,
            model_name=model_name,
            model=model,
            action_dict=action_dict,
            layer_indexes=layer_indexes,
            concept_indexes=concept_indexes,
            target_classes=target_classes,
            observation_type=observation_type,
            env_name=env_name,
            batches=batches,
        )
