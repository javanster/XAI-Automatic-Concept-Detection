from keras.api.saving import load_model
import numpy as np
from ObservationHandler import ObservationHandler
from ConceptDetector import ConceptDetector
import gymnasium as gym
import avocado_run
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Input


TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"

model = load_model(f"models/{TRAIN_RUN_NAME}/{MODEL_NAME}.keras")
env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)
action_dict = env.unwrapped.action_dict

LAYER_INDEXES = [l for l in range(len(model.layers))]
CONCEPTS = [concept for concept in ConceptDetector.concept_name_dict.keys()]
TARGET_CLASSES = [action for action in range(env.action_space.n)]

tcav_results = []


total_iterations = len(TARGET_CLASSES) * len(CONCEPTS) * len(LAYER_INDEXES)


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

    if output_model:
        with tf.GradientTape() as tape:
            tape.watch(activations_tensor)
            output = output_model(
                activations_tensor,
                training=False
            )
            class_output = output[:, target_class]

        gradient = tape.gradient(
            target=class_output,
            sources=activations_tensor
        )
    else:
        # If no output_model exists (i.e., last layer), derivatives are ones
        gradient = tf.ones_like(activations_tensor)

    gradients_flat = tf.reshape(gradient, (gradient.shape[0], -1)).numpy()

    return gradients_flat


with tqdm(total=total_iterations, desc="Calculating TCAV scores", unit="score") as pbar:
    for target_class in TARGET_CLASSES:

        observations_classified_as_target_class = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/model_specific/{TRAIN_RUN_NAME}/{MODEL_NAME}/observations_where_best_class_{target_class}.npy",
            normalize=True,
        )

        for layer_index in LAYER_INDEXES:

            target_layer = model.layers[layer_index]

            gradients = calculate_gradients(
                model=model,
                layer_index=layer_index,
                observations=observations_classified_as_target_class,
                target_class=target_class
            )

            for concept in CONCEPTS:

                cav_path = f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/concept_{concept}_layer_{layer_index}_cav.npy"
                cav = np.load(cav_path)

                directional_derivatives = np.dot(gradients, cav)
                tcav_score = np.mean(directional_derivatives > 0)

                tcav_results.append({
                    'target_class': target_class,
                    'action': action_dict[target_class],
                    'concept_id': concept,
                    'concept_name': ConceptDetector.concept_name_dict[concept],
                    'layer_index': layer_index,
                    'layer_name': model.layers[layer_index].name,
                    'tcav_score': tcav_score
                })

                pbar.update(1)

tcav_df = pd.DataFrame(tcav_results)

print(tcav_df)

tcav_df.to_csv(
    f'tcav_explanations/{TRAIN_RUN_NAME}/{MODEL_NAME}/tcav_scores.csv', index=False)
