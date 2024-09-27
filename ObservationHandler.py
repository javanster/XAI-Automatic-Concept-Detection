from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ObservationHandler:
    """
    A class for saving random and custom observations from the AvocadoRun environment, and showing
    them.
    """

    def __init__(self, env):
        self.env = env

    def save_random_observations(self, num_observations, file_path):
        """
        Parameters:
        - num_observations (int): Number of random observations to save.
        - env (Env): The Gymnasium environment instance.
        - file_path (str): Path to save the NumPy array file.
        """
        observations = []

        with tqdm(total=num_observations, unit="observation") as pbar:
            while len(observations) < num_observations:
                observation, _ = self.env.reset()
                observations.append(observation)
                pbar.update(1)

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
        file_path,
        agent_position_list,
        avocado_positions_list,
        enemy_positions_list,
    ):
        """
        Saves multiple custom observations with specified positions for the agent, avocados, and enemies.

        Parameters:
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

            observation, _ = self.env.reset(
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
