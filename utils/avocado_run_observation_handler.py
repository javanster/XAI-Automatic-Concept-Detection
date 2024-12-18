from tqdm import tqdm
import random
from .observation_handler import ObservationHandler


class AvocadoRunObservationHandler(ObservationHandler):
    """
    A class for saving random and custom observations from the AvocadoRun environment, and showing
    them.
    """
    @classmethod
    def save_custom_observations(
        cls,
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

        cls._save_observations(
            observations=observations, file_path=file_path)

    @classmethod
    def save_observations_given_concept(
        cls,
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

        cls._save_observations(
            observations=observations_with_concept, file_path=file_path_concept)
        cls._save_observations(
            observations=observations_without_concept, file_path=file_path_other)

    @classmethod
    def save_observations_optimal_policy_specific_output_classes(
        cls,
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

            cls._save_observations(
                observations=observations,
                file_path=file_path
            )

            print(
                f"Collected {len(observations)} observations for action {action}")
