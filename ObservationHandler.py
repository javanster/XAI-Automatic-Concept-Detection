from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


class ObservationHandler:
    """
    A class for saving random and custom observations from the AvocadoRun environment, and showing
    them.
    """

    def save_random_observations(self, envs, num_total_observations, file_path):
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

        observations_array = np.array(observations, dtype=np.uint8)
        np.save(file_path, observations_array)

    def show_observation(self, file_path, observation_index):
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

    def save_custom_observations(
        self,
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

        observation_array = np.array(observations, dtype=np.uint8)
        np.save(file_path, observation_array)

    def load_observations(self, file_path, normalize=False):
        try:
            observations = np.load(file_path)
            return observations / 255 if normalize else observations
        except:
            return []
