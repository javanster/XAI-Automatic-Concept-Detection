class ConceptDetector:
    """
    A class to detect the presence of specific directional concepts in the AvocadoRunEnv environment.
    Each concept is defined based on the relative position of the avocado to the agent.
    """

    concept_name_dict = {
        0: "avo_above_agent",
        1: "avo_right_of_agent",
        2: "avo_below_agent",
        3: "avo_left_of_agent",
    }

    @staticmethod
    def is_concept_0_present(env):
        """
        Checks whether Concept 0 is present in the current state of the environment.
        Concept 0: The avocado is above the agent (more above than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 0 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "This concept only allows for 1 avo in the environment")

        avocado = env.unwrapped.avocados[0]

        if any(enemy == avocado for enemy in env.unwrapped.enemies):
            return False

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        avocado_x, avocado_y = avocado.x, avocado.y

        # Calculate differences
        delta_x = avocado_x - agent_x
        delta_y = avocado_y - agent_y

        # Avocado is above if delta_y < 0 and |delta_y| > |delta_x|
        if delta_y < 0 and abs(delta_y) > abs(delta_x):
            return True

        # If no avocado meets the conditions, return False
        return False

    @staticmethod
    def is_concept_1_present(env):
        """
        Checks whether Concept 1 is present in the current state of the environment.
        Concept 1: The avocado is to the right of the agent (more to the right than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 1 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "This concept only allows for 1 avo in the environment")

        avocado = env.unwrapped.avocados[0]

        if any(enemy == avocado for enemy in env.unwrapped.enemies):
            return False

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        avocado_x, avocado_y = avocado.x, avocado.y

        # Calculate differences
        delta_x = avocado_x - agent_x
        delta_y = avocado_y - agent_y

        # Avocado is to the right if delta_x > 0 and |delta_x| > |delta_y|
        if delta_x > 0 and abs(delta_x) > abs(delta_y):
            return True

        # If no avocado meets the conditions, return False
        return False

    @staticmethod
    def is_concept_2_present(env):
        """
        Checks whether Concept 2 is present in the current state of the environment.
        Concept 2: The avocado is below the agent (more below than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 2 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "This concept only allows for 1 avo in the environment")

        avocado = env.unwrapped.avocados[0]

        if any(enemy == avocado for enemy in env.unwrapped.enemies):
            return False

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        avocado_x, avocado_y = avocado.x, avocado.y

        # Calculate differences
        delta_x = avocado_x - agent_x
        delta_y = avocado_y - agent_y

        # Avocado is below if delta_y > 0 and |delta_y| > |delta_x|
        if delta_y > 0 and abs(delta_y) > abs(delta_x):
            return True

        # If no avocado meets the conditions, return False
        return False

    @staticmethod
    def is_concept_3_present(env):
        """
        Checks whether Concept 3 is present in the current state of the environment.
        Concept 3: The avocado is to the left of the agent (more to the left than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 3 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "This concept only allows for 1 avo in the environment")

        avocado = env.unwrapped.avocados[0]

        if any(enemy == avocado for enemy in env.unwrapped.enemies):
            return False

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        avocado_x, avocado_y = avocado.x, avocado.y

        # Calculate differences
        delta_x = avocado_x - agent_x
        delta_y = avocado_y - agent_y

        # Avocado is to the left if delta_x < 0 and |delta_x| > |delta_y|
        if delta_x < 0 and abs(delta_x) > abs(delta_y):
            return True

        # If no avocado meets the conditions, return False
        return False
