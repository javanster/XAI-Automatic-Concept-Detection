from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


class ObservationHandler:
    """
    A class for saving random and custom observations from the AvocadoRun environment, and showing
    them.
    """
    @staticmethod
    def _save_observations(observations, file_path):
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

        ObservationHandler._save_observations(
            observations=observations, file_path=file_path)

    @staticmethod
    def show_observation(file_path, observation_index):
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

        ObservationHandler._save_observations(
            observations=observations, file_path=file_path)

    @staticmethod
    def load_observations(file_path, normalize=False):
        try:
            observations = np.load(file_path)
            return observations / 255 if normalize else observations
        except:
            return []

    @staticmethod
    def save_observations_for_tcav(env, num_observations_for_each, file_path_concept, file_path_other, is_concept_in_observation, agent_starting_position=None, avocado_starting_positions=None, enemy_starting_positions=None):
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

        print(
            f"Observations with concept gathered: {len(observations_with_concept)}")
        print(
            f"Observations without concept gathered: {len(observations_without_concept)}")

        random.shuffle(observations_with_concept)
        random.shuffle(observations_without_concept)

        ObservationHandler._save_observations(
            observations=observations_with_concept, file_path=file_path_concept)
        ObservationHandler._save_observations(
            observations=observations_without_concept, file_path=file_path_other)
