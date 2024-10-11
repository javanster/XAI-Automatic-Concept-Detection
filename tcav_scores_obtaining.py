from keras.api.saving import load_model
import numpy as np
from keras import Model
from ObservationHandler import ObservationHandler
from ConceptDetector import ConceptDetector
import gymnasium as gym
import avocado_run
import pandas as pd
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

action_dict = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
    4: "do_nothing",
}

TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"

model = load_model(f"models/{TRAIN_RUN_NAME}/{MODEL_NAME}.keras")
env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)

LAYER_INDEXES = [l for l in range(len(model.layers))]
CONCEPTS = [concept for concept in ConceptDetector.concept_name_dict.keys()]
TARGET_CLASSES = [action for action in range(env.action_space.n)]

tcav_results = []


def get_activations(activation_model, observations):
    activations = activation_model.predict(observations)

    # If the layer is a Conv2D layer, flatten the output
    if len(activations.shape) == 4:
        batch_size, height, width, channels = activations.shape
        activations = activations.reshape(
            batch_size, height * width * channels)

    return activations


total_iterations = len(TARGET_CLASSES) * len(CONCEPTS) * len(LAYER_INDEXES)

with tqdm(total=total_iterations, desc="Calculating TCAV scores", unit="score") as pbar:
    for target_class in TARGET_CLASSES:

        observations_classified_as_target_class = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/model_specific/{TRAIN_RUN_NAME}/{MODEL_NAME}/observations_where_best_class_{target_class}.npy",
            normalize=True,
        )

        for concept in CONCEPTS:
            for layer_index in LAYER_INDEXES:

                cav_path = f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/concept_{concept}_layer_{layer_index}_cav.npy"
                cav = np.load(cav_path)

                target_layer = model.layers[layer_index]

                activation_model = Model(
                    inputs=model.layers[0].input,
                    outputs=model.layers[layer_index].output
                )

                # CALCULATING TCAV SCORE

                activations = get_activations(
                    activation_model, observations_classified_as_target_class)
                directional_derivatives = np.dot(activations, cav)
                tcav_score = np.mean(directional_derivatives > 0)

                tcav_results.append({
                    'Target_Class': target_class,
                    'Action': action_dict[target_class],
                    'Concept_ID': concept,
                    'Concept_Name': ConceptDetector.concept_name_dict[concept],
                    'Layer_Index': layer_index,
                    'TCAV_Score': tcav_score
                })

                pbar.update(1)

tcav_df = pd.DataFrame(tcav_results)

print(tcav_df.head())

tcav_df.to_csv('tcav_data/tcav_scores/tcav_scores.csv', index=False)
