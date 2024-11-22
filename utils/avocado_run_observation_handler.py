from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class AvocadoRunObservationHandler:
    """
    A class for saving random and custom observations from the AvocadoRun environment, and showing
    them.
    """
    @staticmethod
    def _save_observations(observations, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        observations_array = np.array(observations, dtype=np.uint8)
        np.save(file_path, observations_array)

    @staticmethod
    def save_random_observations(envs, num_total_observations, file_path):
        """
        Parameters:
        - envs ([Env]): A list of AvocadoRun environment instances to gather observations from.
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

        AvocadoRunObservationHandler._save_observations(
            observations=observations, file_path=file_path)

    @staticmethod
    def show_observation(file_path, observation_index, title=None):
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

    @staticmethod
    def save_custom_observations(
        envs,
        file_path,
        agent_position_list,
        avocado_positions_list,
        enemy_positions_list,
    ):
        """
        Saves multiple custom observations with specified positions for the agent, avocados, and enemies.

        Parameters:
        - envs ([Env]): A list of AvocadoRun environment instances to gather observations from. The length of the list must equal the lenght of the lists of entity positions. Each index in the positions lists will use the env at the corresponding index in the env list. 
        - file_path (str): Path to save the NumPy array file.
        - agent_position_list (list of tuples): A list where each tuple (x, y) denotes the cell position of the agent for each observation.
        - avocado_positions_list (list of lists of tuples): A list of lists where each inner list contains tuples (x, y) representing the positions of avocados for each observation.
        - enemy_positions_list (list of lists of tuples): A list of lists where each inner list contains tuples (x, y) representing the positions of enemies for each observation.

        The method generates multiple observations by resetting the environment for each set of positions provided
        for the agent, avocados, and enemies. It saves the resulting observations as a NumPy array.

        Raises:
        - ValueError: If the lengths of `agent_position_list`, `avocado_positions_list`, and `enemy_positions_list` are not equal.
        """
        if not (len(agent_position_list) == len(avocado_positions_list)
                and len(agent_position_list) == len(enemy_positions_list)):
            raise ValueError("The length of each position list must be equal")

        observations = []

        for i in range(len(agent_position_list)):
            agent_position = agent_position_list[i]
            avocado_positions = avocado_positions_list[i]
            enemy_positions = enemy_positions_list[i]

            env = envs[i]

            observation, _ = env.reset(
                agent_starting_position=agent_position,
                avocado_starting_positions=avocado_positions,
                enemy_starting_positions=enemy_positions,
            )

            observations.append(observation)

        AvocadoRunObservationHandler._save_observations(
            observations=observations, file_path=file_path)

    @staticmethod
    def load_observations(file_path, normalize=False):
        try:
            observations = np.load(file_path)
            return observations / 255 if normalize else observations
        except:
            return []

    @staticmethod
    def save_observations_given_concept(
        env,
        concept_index,
        num_observations_for_each,
        file_path_concept,
        file_path_other,
        is_concept_in_observation,
        agent_starting_position=None,
        avocado_starting_positions=None,
        enemy_starting_positions=None
    ):
        """
        Saves observations categorized based on the presence or absence of a specific concept.

        Parameters:
        - env (Env): The AvocadoRun environment instance to gather observations from.
        - num_observations_for_each (int): Number of observations to save for each category
            (with and without the concept).
        - file_path_concept (str): Path to save the NumPy array file containing random observations
            where the concept is present.
        - file_path_other (str): Path to save the NumPy array file containing random observations
            where the concept is absent.
        - is_concept_in_observation (callable): A function that takes the environment as input and
            returns True if the concept is present in the current observation, otherwise False.
        - agent_starting_position (tuple, optional): Starting position (x, y) of the agent.
            Defaults to None.
        - avocado_starting_positions (list of tuple, optional): Starting positions of avocados.
            Defaults to None.
        - enemy_starting_positions (list of tuple, optional): Starting positions of enemies.
            Defaults to None.
        """
        observations_with_concept = []
        observations_without_concept = []

        while len(observations_with_concept) < num_observations_for_each or len(observations_without_concept) < num_observations_for_each:
            observation, _ = env.reset(agent_starting_position=agent_starting_position,
                                       avocado_starting_positions=avocado_starting_positions,
                                       enemy_starting_positions=enemy_starting_positions)
            if is_concept_in_observation(env) and len(observations_with_concept) < num_observations_for_each:
                observations_with_concept.append(observation)
            elif not is_concept_in_observation(env) and len(observations_without_concept) < num_observations_for_each:
                observations_without_concept.append(observation)

        random.shuffle(observations_with_concept)
        random.shuffle(observations_without_concept)

        AvocadoRunObservationHandler._save_observations(
            observations=observations_with_concept, file_path=file_path_concept)
        AvocadoRunObservationHandler._save_observations(
            observations=observations_without_concept, file_path=file_path_other)

    @staticmethod
    def save_observations_specific_output_classes(
        env,
        model,
        output_classes,
        num_observations,
        file_path_base

    ):
        """
        Saves observations corresponding to specific output classes predicted by a model.

        Parameters:
        - env (Env): The AvocadoRun environment instance to gather observations from.
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
                model_output_batch = model.predict(
                    observation_batch, batch_size=batch_size)

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

            AvocadoRunObservationHandler._save_observations(
                observations=observations, file_path=file_path)

            print(
                f"Collected {len(observations)} observations for output class {ocls}")

    @staticmethod
    def save_observations_optimal_policy_specific_output_classes(
        env,
        policy,
        state_to_index,
        output_classes,
        num_observations,
        file_path_base
    ):
        """
        Saves observations where the optimal policy outputs specific actions by relying on environment resets.
        This version removes the maximum resets limit to ensure all required observations are collected.

        Parameters:
        - env (gym.Env): The AvocadoRun environment instance to gather observations from.
        - policy (np.ndarray): The optimal policy array, indexed by state index.
        - state_to_index (dict): Mapping from state tuples to state indices.
        - output_classes (list of int): A list of target action indices for which observations
            should be collected.
        - num_observations (int): Number of observations to save for each specified action.
        - file_path_base (str): Base file path to save the NumPy array files. The action index
            will be appended to this base to form the complete file path for each action.
        """
        observation_dict = {action: [] for action in output_classes}
        total_required = num_observations * len(output_classes)
        collected = 0

        with tqdm(total=total_required, desc="Collecting Observations") as pbar:
            while any(len(obs) < num_observations for obs in observation_dict.values()):
                observation, _ = env.reset()

                agent_entity = env.unwrapped.agent
                enemy_entities = env.unwrapped.enemies
                avocado_entities = env.unwrapped.avocados

                agent_pos = (agent_entity.x, agent_entity.y)
                avocado_pos = (avocado_entities[0].x, avocado_entities[0].y)
                enemy_pos = (enemy_entities[0].x, enemy_entities[0].y)

                state = (agent_pos, avocado_pos, enemy_pos)
                state_idx = state_to_index.get(state, None)

                if state_idx is None:
                    # Invalid state; skip to the next reset
                    continue

                action_idx = policy[state_idx]

                if action_idx in output_classes and len(observation_dict[action_idx]) < num_observations:
                    observation_dict[action_idx].append(observation)
                    pbar.update(1)
                    collected += 1

        for action in output_classes:
            observations = observation_dict[action]
            file_path = f"{file_path_base}{action}.npy"

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            AvocadoRunObservationHandler._save_observations(
                observations=observations,
                file_path=file_path
            )

            print(
                f"Collected {len(observations)} observations for action {action}")
