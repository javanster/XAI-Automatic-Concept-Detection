from tqdm import tqdm
import random
from .observation_handler import ObservationHandler


class MangoRunObservationHandler(ObservationHandler):
    """
    A class for saving random and custom observations from the MangoRun environment, and showing
    them.
    """
    @classmethod
    def save_observations_given_concept(
        cls,
        env,
        num_observations_for_each,
        file_path_concept,
        file_path_other,
        is_concept_in_observation,
        agent_starting_position=None,
    ):
        """
        Saves observations categorized based on the presence or absence of a specific concept.

        Parameters:
        - env (Env): The MangoRun environment instance to gather observations from.
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
        """
        observations_with_concept = []
        observations_without_concept = []

        while len(observations_with_concept) < num_observations_for_each or len(observations_without_concept) < num_observations_for_each:
            observation, _ = env.reset(
                agent_starting_position=agent_starting_position)
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
        optimal_policy,
        output_classes,
        num_observations,
        file_path_base,
    ):
        """
        Saves observations where the optimal policy outputs specific actions by relying on environment resets.
        This version is tailored for MangoRun's dictionary-based optimal policy. 
        The environment must have `agent_spawn_all_legal_locations` set to True to ensure variance in the observations.

        Parameters:
        - env (gym.Env): 
            The MangoRun environment instance to gather observations from. 
            **Important:** Must have `agent_spawn_all_legal_locations` set to True to ensure a variety of starting positions for the agent.
        - optimal_policy (dict): 
            A dictionary representing the optimal policy, mapping state tuples to action indices. 
            Each key is a tuple of the form `(agent_pos, ripe_mango1_pos, ripe_mango2_pos)`, and each value is an integer corresponding to an action index.
        - output_classes (List[int]): 
            A list of target action indices for which observations should be collected. 
            Only observations where the optimal policy selects actions in this list will be saved.
        - num_observations (int): 
            The number of observations to save for each specified action in `output_classes`.
        - file_path_base (str): 
            The base file path to save the NumPy array files. 
            The action index will be appended to this base to form the complete file path for each action. 
            For example, if `file_path_base` is `"data/mango_run_action_"` and the action index is `1`, the file will be saved as `"data/mango_run_action_1.npy"`.

        Raises:
        - ValueError: 
            If `env.unwrapped.agent_spawn_all_legal_locations` is set to `False`. 
            This ensures that the environment provides varied starting positions for the agent, which is crucial for collecting diverse observations.
        """
        if not getattr(env.unwrapped, 'agent_spawn_all_legal_locations', False):
            raise ValueError(
                "MangoRun environment must have `agent_spawn_all_legal_locations` set to True to ensure variance in the observations!"
            )

        observation_dict = {action: [] for action in output_classes}
        total_required = num_observations * len(output_classes)

        with tqdm(total=total_required, desc="Collecting Observations") as pbar:
            while any(len(obs) < num_observations for obs in observation_dict.values()):
                observation, _ = env.reset()

                agent_entity = env.unwrapped.agent
                ripe_mangoes = [env.unwrapped.ripe_mango_1,
                                env.unwrapped.ripe_mango_2]

                agent_pos = (agent_entity.x, agent_entity.y)
                ripe_positions = sorted([(mango.x, mango.y)
                                        for mango in ripe_mangoes])
                state = (agent_pos, ripe_positions[0], ripe_positions[1])

                action_idx = optimal_policy.get(state)

                if action_idx is None:
                    raise ValueError(f"No policy for state {state}")

                if action_idx in output_classes and len(observation_dict[action_idx]) < num_observations:
                    observation_dict[action_idx].append(observation)
                    pbar.update(1)

        for action in output_classes:
            observations = observation_dict[action]
            file_path = f"{file_path_base}{action}.npy"
            cls._save_observations(
                observations=observations, file_path=file_path
            )
            print(
                f"Collected {len(observations)} observations for action {action}")
