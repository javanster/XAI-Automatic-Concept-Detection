from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class ObservationHandler:
    """
    A class for saving random and custom observations from environments, and showing
    them.
    """
    @classmethod
    def _save_observations(cls, observations, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        observations_array = np.array(observations, dtype=np.uint8)
        np.save(file_path, observations_array)

    @classmethod
    def save_random_observations(cls, envs, num_total_observations, file_path):
        """
        Parameters:
        - envs ([Env]): A list of environment instances to gather observations from.
        - num_observations (int): Number of random observations to save.
        - file_path (str): Path to save the NumPy array file.
        """
        observations = []

        num_observations_from_each_env = num_total_observations // len(envs)

        with tqdm(total=num_total_observations, unit="observation") as pbar:
            for env in envs:
                for _ in range(num_observations_from_each_env):
                    observation, _ = env.reset()
                    observations.append(observation)
                    pbar.update(1)

        random.shuffle(observations)

        cls._save_observations(
            observations=observations, file_path=file_path)

    @classmethod
    def show_observation(cls, file_path, observation_index, title=None):
        """
        Parameters:
        - file_path (str): Path to save the NumPy array file.
        - observation_index (int): The index of the observation to show.
        """
        observations = np.load(file_path)
        observation = observations[observation_index]

        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(observation, interpolation='nearest')
        ax.axis('off')
        if title:
            plt.title(title)
        plt.show()

    @classmethod
    def load_observations(cls, file_path, normalize=False):
        try:
            observations = np.load(file_path)
            return observations / 255 if normalize else observations
        except:
            return []

    @classmethod
    def save_observations_specific_output_classes(
        cls,
        env,
        model,
        output_classes,
        num_observations,
        file_path_base

    ):
        """
        Saves observations corresponding to specific output classes predicted by a model.

        Parameters:
        - env (Env): The environment instance to gather observations from.
        - model: The trained Keras model used to predict output classes.
        - output_classes (list of int): A list of target output class indices for which observations
            should be collected.
        - num_observations (int): Number of observations to save for each specified output class.
        - file_path_base (str): Base file path to save the NumPy array files. The output class index
            will be appended to this base to form the complete file path for each class
        """
        observation_dict = {ocls: []
                            for ocls in output_classes}

        with tqdm(total=num_observations * len(output_classes), desc="Collecting Observations") as pbar:
            while any(len(observation_dict[ocls]) < num_observations for ocls in observation_dict.keys()):
                batch_size = 256
                observation_batch = np.array([env.reset()[0]
                                              for _ in range(batch_size)])
                observation_batch_normalized = observation_batch / 255
                model_output_batch = model.predict(
                    observation_batch_normalized, batch_size=batch_size)

                for i, model_output in enumerate(model_output_batch):
                    predicted_class = np.argmax(model_output)
                    class_observations = observation_dict[predicted_class]

                    if len(class_observations) < num_observations:
                        class_observations.append(observation_batch[i])

                        pbar.update(1)
                        if all(len(observation_dict[ocls]) >= num_observations for ocls in observation_dict.keys()):
                            break

        for ocls in output_classes:
            observations = observation_dict[ocls]
            file_path = f"{file_path_base}{ocls}.npy"

            cls._save_observations(
                observations=observations, file_path=file_path)

            print(
                f"Collected {len(observations)} observations for output class {ocls}")
