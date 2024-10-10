import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent
from ObservationHandler import ObservationHandler
from keras import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv


LAYER_INDEXES = [0, 1, 3]
CONCEPTS = [c for c in range(1, 7)]
TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"


env = gym.make(id="AvocadoRun-v0", render_mode="human",
               num_avocados=1, num_enemies=2, aggressive_enemies=False)

agent = DoubleDQNAgent(
    env=env,
    model_path=f"models/{TRAIN_RUN_NAME}/{MODEL_NAME}.keras"
)


def get_activations_of_layer(model, layer_index, observations):
    activation_model = Model(
        inputs=model.layers[0].input, outputs=model.layers[layer_index].output)
    activations = activation_model.predict(observations)

    # If the layer is a Conv2D layer, flatten the output
    if len(activations.shape) == 4:  # This indicates a 4D tensor: (batch_size, height, width, channels)
        batch_size, height, width, channels = activations.shape
        # Flatten the activations to 2D shape: (batch_size, height * width * channels)
        activations = activations.reshape(
            batch_size, height * width * channels)

    return activations


accuracy_dict = {}

for layer_index in LAYER_INDEXES:
    for concept in CONCEPTS:

        observations_with_concept = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/observations_containing_concept_{concept}.npy",
            normalize=True,
        )

        random_observations_without_concept = ObservationHandler.load_observations(
            file_path=f"tcav_data/observations/observations_not_containing_concept_{concept}.npy",
            normalize=True,
        )

        concept_activations_for_layer = get_activations_of_layer(
            model=agent.online_model,
            layer_index=layer_index,
            observations=observations_with_concept
        )
        non_concept_activations_for_layer = get_activations_of_layer(
            model=agent.online_model,
            layer_index=layer_index,
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

        accuracy_dict[f"Concept {concept} - layer {layer_index}"] = accuracy

        cav = clf.coef_[0]

        np.save(
            f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/concept_{concept}_layer_{layer_index}_cav.npy", cav)

print(f"{'Concept':<20} | {'Classifier Accuracy':>18}")
print('-' * 32)
for concept in accuracy_dict.keys():
    print(f"{concept:<20} | {accuracy_dict[concept]}")

with open(
    f"tcav_data/cavs/{TRAIN_RUN_NAME}/{MODEL_NAME}/accuracy_table.csv",
    mode='w',
    newline=''
) as csv_file:
    fieldnames = ['Concept', 'Classifier Accuracy']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for concept in accuracy_dict:
        writer.writerow({
            'Concept': concept,
            'Classifier Accuracy': accuracy_dict[concept]
        })
