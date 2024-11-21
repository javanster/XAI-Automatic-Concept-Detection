import numpy as np


class MangoRunConceptDetector:
    """
    A class to detect the presence of concepts in the MangoRunEnv environment.
    """

    concept_name_dict = {
        0: "agent_adjacent_wall_up",
        1: "agent_adjacent_wall_right",
        2: "agent_adjacent_wall_down",
        3: "agent_adjacent_wall_left",

        4: "agent_adjacent_unripe_mango_up",
        5: "agent_adjacent_unripe_mango_right",
        6: "agent_adjacent_unripe_mango_down",
        7: "agent_adjacent_unripe_mango_left",

        8: "closest_unripe_mango_above_agent",
        9: "closest_unripe_mango_right_of_agent",
        10: "closest_unripe_mango_below_agent",
        11: "closest_unripe_mango_left_of_agent",

        12: "agent_adjacent_ripe_mango_up",
        13: "agent_adjacent_ripe_mango_right",
        14: "agent_adjacent_ripe_mango_down",
        15: "agent_adjacent_ripe_mango_left",

        16: "closest_ripe_mango_above_agent",
        17: "closest_ripe_mango_right_of_agent",
        18: "closest_ripe_mango_below_agent",
        19: "closest_ripe_mango_left_of_agent",
    }

    @staticmethod
    def is_concept_0_present(env):
        """
        Checks whether Concept 0 is present in the current state of the environment.
        Concept 0: The agent is adjacent to a wall above it or at the upper edge.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 0 is present, otherwise False.
        """
        walls = {(wall.x, wall.y) for wall in env.unwrapped.walls}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x, agent_y - 1) in walls or agent_y == 0

    @staticmethod
    def is_concept_1_present(env):
        """
        Checks whether Concept 1 is present in the current state of the environment.
        Concept 1: The agent is adjacent to a wall on its right or at the right edge.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 1 is present, otherwise False.
        """
        walls = {(wall.x, wall.y) for wall in env.unwrapped.walls}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        grid_width = env.unwrapped.grid_side_width

        return (agent_x + 1, agent_y) in walls or agent_x == grid_width - 1

    @staticmethod
    def is_concept_2_present(env):
        """
        Checks whether Concept 2 is present in the current state of the environment.
        Concept 2: The agent is adjacent to a wall below it or at the bottom edge.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 2 is present, otherwise False.
        """
        walls = {(wall.x, wall.y) for wall in env.unwrapped.walls}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        grid_height = env.unwrapped.grid_side_height

        return (agent_x, agent_y + 1) in walls or agent_y == grid_height - 1

    @staticmethod
    def is_concept_3_present(env):
        """
        Checks whether Concept 3 is present in the current state of the environment.
        Concept 3: The agent is adjacent to a wall on its left or at the left edge.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 3 is present, otherwise False.
        """
        walls = {(wall.x, wall.y) for wall in env.unwrapped.walls}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x - 1, agent_y) in walls or agent_x == 0

    @staticmethod
    def is_concept_4_present(env):
        """
        Checks whether Concept 4 is present in the current state of the environment.
        Concept 4: The agent is adjacent to an unripe mango above it.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 4 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = {(unripe_mango.x, unripe_mango.y)
                          for unripe_mango in env.unwrapped.unripe_mangoes}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x, agent_y - 1) in unripe_mangoes

    @staticmethod
    def is_concept_5_present(env):
        """
        Checks whether Concept 5 is present in the current state of the environment.
        Concept 5: The agent is adjacent to an unripe mango to its right.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 5 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = {(unripe_mango.x, unripe_mango.y)
                          for unripe_mango in env.unwrapped.unripe_mangoes}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x + 1, agent_y) in unripe_mangoes

    @staticmethod
    def is_concept_6_present(env):
        """
        Checks whether Concept 6 is present in the current state of the environment.
        Concept 6: The agent is adjacent to an unripe mango below it.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 6 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = {(unripe_mango.x, unripe_mango.y)
                          for unripe_mango in env.unwrapped.unripe_mangoes}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x, agent_y + 1) in unripe_mangoes

    @staticmethod
    def is_concept_7_present(env):
        """
        Checks whether Concept 7 is present in the current state of the environment.
        Concept 7: The agent is adjacent to an unripe mango to its left.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 7 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = {(unripe_mango.x, unripe_mango.y)
                          for unripe_mango in env.unwrapped.unripe_mangoes}
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        return (agent_x - 1, agent_y) in unripe_mangoes

    @staticmethod
    def is_concept_8_present(env):
        """
        Checks whether Concept 8 is present in the current state of the environment.
        Concept 8: The closest unripe mango is above the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 8 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = env.unwrapped.unripe_mangoes
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(unripe_mango.x - agent_x) +
             abs(unripe_mango.y - agent_y), unripe_mango)
            for unripe_mango in unripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_unripe_mangoes = [
            unripe_mango for distance, unripe_mango in distances if distance == min_distance
        ]

        for unripe_mango in closest_unripe_mangoes:
            if unripe_mango.x == agent_x and unripe_mango.y < agent_y:
                return True

        return False

    @staticmethod
    def is_concept_9_present(env):
        """
        Checks whether Concept 9 is present in the current state of the environment.
        Concept 9: The closest unripe mango is to the right of the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 9 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = env.unwrapped.unripe_mangoes
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(unripe_mango.x - agent_x) +
             abs(unripe_mango.y - agent_y), unripe_mango)
            for unripe_mango in unripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_unripe_mangoes = [
            unripe_mango for distance, unripe_mango in distances if distance == min_distance
        ]

        for unripe_mango in closest_unripe_mangoes:
            if unripe_mango.y == agent_y and unripe_mango.x > agent_x:
                return True

        return False

    @staticmethod
    def is_concept_10_present(env):
        """
        Checks whether Concept 10 is present in the current state of the environment.
        Concept 10: The closest unripe mango is below the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 10 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = env.unwrapped.unripe_mangoes
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(unripe_mango.x - agent_x) +
             abs(unripe_mango.y - agent_y), unripe_mango)
            for unripe_mango in unripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_unripe_mangoes = [
            unripe_mango for distance, unripe_mango in distances if distance == min_distance
        ]

        for unripe_mango in closest_unripe_mangoes:
            if unripe_mango.x == agent_x and unripe_mango.y > agent_y:
                return True

        return False

    @staticmethod
    def is_concept_11_present(env):
        """
        Checks whether Concept 11 is present in the current state of the environment.
        Concept 11: The closest unripe mango is to the left of the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 11 is present, otherwise False.
        :raises ValueError: If unripe_mangoes is empty or not set.
        """
        if not hasattr(env.unwrapped, 'unripe_mangoes') or not env.unwrapped.unripe_mangoes:
            raise ValueError(
                "unripe_mangoes is empty or not set in the environment.")

        unripe_mangoes = env.unwrapped.unripe_mangoes
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(unripe_mango.x - agent_x) +
             abs(unripe_mango.y - agent_y), unripe_mango)
            for unripe_mango in unripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_unripe_mangoes = [
            unripe_mango for distance, unripe_mango in distances if distance == min_distance
        ]

        for unripe_mango in closest_unripe_mangoes:
            if unripe_mango.y == agent_y and unripe_mango.x < agent_x:
                return True

        return False

    @staticmethod
    def is_concept_12_present(env):
        """
        Checks whether Concept 12 is present in the current state of the environment.
        Concept 12: The agent is adjacent to a ripe mango above it.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 12 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for ripe_mango in ripe_mangoes:
            if (ripe_mango.x, ripe_mango.y) == (agent_x, agent_y - 1):
                return True

        return False

    @staticmethod
    def is_concept_13_present(env):
        """
        Checks whether Concept 13 is present in the current state of the environment.
        Concept 13: The agent is adjacent to a ripe mango to its right.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 13 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for ripe_mango in ripe_mangoes:
            if (ripe_mango.x, ripe_mango.y) == (agent_x + 1, agent_y):
                return True

        return False

    @staticmethod
    def is_concept_14_present(env):
        """
        Checks whether Concept 14 is present in the current state of the environment.
        Concept 14: The agent is adjacent to a ripe mango below it.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 14 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for ripe_mango in ripe_mangoes:
            if (ripe_mango.x, ripe_mango.y) == (agent_x, agent_y + 1):
                return True

        return False

    @staticmethod
    def is_concept_15_present(env):
        """
        Checks whether Concept 15 is present in the current state of the environment.
        Concept 15: The agent is adjacent to a ripe mango to its left.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 15 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for ripe_mango in ripe_mangoes:
            if (ripe_mango.x, ripe_mango.y) == (agent_x - 1, agent_y):
                return True

        return False

    @staticmethod
    def is_concept_16_present(env):
        """
        Checks whether Concept 16 is present in the current state of the environment.
        Concept 16: The closest ripe mango is above the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 16 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(ripe_mango.x - agent_x) + abs(ripe_mango.y - agent_y), ripe_mango)
            for ripe_mango in ripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_ripe_mangoes = [
            ripe_mango for distance, ripe_mango in distances if distance == min_distance
        ]

        for ripe_mango in closest_ripe_mangoes:
            if ripe_mango.x == agent_x and ripe_mango.y < agent_y:
                return True

        return False

    @staticmethod
    def is_concept_17_present(env):
        """
        Checks whether Concept 17 is present in the current state of the environment.
        Concept 17: The closest ripe mango is to the right of the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 17 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(ripe_mango.x - agent_x) + abs(ripe_mango.y - agent_y), ripe_mango)
            for ripe_mango in ripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_ripe_mangoes = [
            ripe_mango for distance, ripe_mango in distances if distance == min_distance
        ]

        for ripe_mango in closest_ripe_mangoes:
            if ripe_mango.y == agent_y and ripe_mango.x > agent_x:
                return True

        return False

    @staticmethod
    def is_concept_18_present(env):
        """
        Checks whether Concept 18 is present in the current state of the environment.
        Concept 18: The closest ripe mango is below the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 18 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(ripe_mango.x - agent_x) + abs(ripe_mango.y - agent_y), ripe_mango)
            for ripe_mango in ripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_ripe_mangoes = [
            ripe_mango for distance, ripe_mango in distances if distance == min_distance
        ]

        for ripe_mango in closest_ripe_mangoes:
            if ripe_mango.x == agent_x and ripe_mango.y > agent_y:
                return True

        return False

    @staticmethod
    def is_concept_19_present(env):
        """
        Checks whether Concept 19 is present in the current state of the environment.
        Concept 19: The closest ripe mango is to the left of the agent.

        :param env: The current environment instance (MangoRun)
        :return: True if Concept 19 is present, otherwise False.
        :raises ValueError: If ripe_mango_1 or ripe_mango_2 is not set.
        """
        if not hasattr(env.unwrapped, 'ripe_mango_1') or not hasattr(env.unwrapped, 'ripe_mango_2'):
            raise ValueError(
                "ripe_mango_1 or ripe_mango_2 is not set in the environment.")

        ripe_mangoes = [env.unwrapped.ripe_mango_1, env.unwrapped.ripe_mango_2]
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        distances = [
            (abs(ripe_mango.x - agent_x) + abs(ripe_mango.y - agent_y), ripe_mango)
            for ripe_mango in ripe_mangoes
        ]

        min_distance = min(distance for distance, _ in distances)

        closest_ripe_mangoes = [
            ripe_mango for distance, ripe_mango in distances if distance == min_distance
        ]

        for ripe_mango in closest_ripe_mangoes:
            if ripe_mango.y == agent_y and ripe_mango.x < agent_x:
                return True

        return False
