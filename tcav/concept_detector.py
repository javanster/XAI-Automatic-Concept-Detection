import numpy as np


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
        4: "random_observation",
        5: "enemy_close_to_agent",
        6: "avocado_visible",
        7: "agent_close_to_avocado",
        8: "agent_next_to_wall",
        9: "enemy_within_3_left",
        10: "enemy_within_3_right",
        11: "enemy_within_3_up",
        12: "enemy_within_3_down",
        13: "enemy_2_steps_left",
        14: "enemy_2_steps_right",
        15: "enemy_2_steps_up",
        16: "enemy_2_steps_down",
        17: "enemy_1_step_left",
        18: "enemy_1_step_right",
        19: "enemy_1_step_up",
        20: "enemy_1_step_down",
        21: "agent_against_left_wall",
        22: "agent_against_right_wall",
        23: "agent_against_upper_wall",
        24: "agent_against_bottom_wall",
        25: "enemy_above_agent",
        26: "enemy_right_of_agent",
        27: "enemy_below_agent",
        28: "enemy_left_of_agent",
        29: "avo_above_enemy_below",
        30: "avo_right_enemy_left",
        31: "avo_below_enemy_above",
        32: "avo_left_enemy_right",

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

    def is_concept_4_present(env):
        """
        For sanity checking TCAV scores. Returns True roughly 50 % of the time.

        :param env: The current environment instance (AvocadoRunEnv), unused
        :return: True or False, roughly 50 % of the time each
        """
        if np.random.random() > 0.5:
            return True
        return False

    @staticmethod
    def is_concept_5_present(env):
        """
        Checks whether Concept 5 is present in the current state of the environment.
        Concept 5: At least one enemy is within a Manhattan distance of 3 from the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 5 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        enemies = env.unwrapped.enemies

        for enemy in enemies:
            distance = abs(agent.x - enemy.x) + abs(agent.y - enemy.y)
            if distance <= 3:
                return True
        return False

    @staticmethod
    def is_concept_6_present(env):
        """
        Checks whether Concept 6 is present in the current state of the environment.
        Concept 6: All avocados are visible, meaning no avocado shares its position with any enemy.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 6 is present, otherwise False.
        :raises ValueError: If there are more avocados in the environment than enemies.
        """

        if (len(env.unwrapped.avocados) > len(env.unwrapped.enemies)):
            raise ValueError(
                "Number of enemies in the environment must be greater than or equal to the number of avocados, in order for all avocados to be hidden."
            )

        avocados = env.unwrapped.avocados
        enemies = env.unwrapped.enemies

        for avocado in avocados:
            for enemy in enemies:
                if avocado == enemy:
                    return False
        return True

    @staticmethod
    def is_concept_7_present(env):
        """
        Checks whether Concept 7 is present in the current state of the environment.
        Concept 7: The agent is close to at least one avocado, defined as being within a Manhattan distance of 3.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 7 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        avocados = env.unwrapped.avocados

        for avocado in avocados:
            distance = abs(agent.x - avocado.x) + abs(agent.y - avocado.y)
            if distance <= 3:
                return True
        return False

    @staticmethod
    def is_concept_8_present(env):
        """
        Concept 8: Agent is next to any wall (top, bottom, left, or right).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 8 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        grid_side_length = env.unwrapped.grid_side_length

        if (agent.x == 0 or  # Left wall
            agent.x == grid_side_length - 1 or  # Right wall
            agent.y == 0 or  # Top wall
                agent.y == grid_side_length - 1):  # Bottom wall
            return True

        return False

    @staticmethod
    def is_concept_9_present(env):
        """
        Concept 9: Enemy within 3 steps directly to the left of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 9 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        # Check if enemy is directly to the left within 3 steps
        if enemy.y == agent.y and (agent.x - enemy.x) in range(1, 4):
            return True
        return False

    @staticmethod
    def is_concept_10_present(env):
        """
        Concept 10: Enemy within 3 steps directly to the right of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 10 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.y == agent.y and (enemy.x - agent.x) in range(1, 4):
            return True
        return False

    @staticmethod
    def is_concept_11_present(env):
        """
        Concept 11: Enemy within 3 steps directly above the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 11 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (agent.y - enemy.y) in range(1, 4):
            return True
        return False

    @staticmethod
    def is_concept_12_present(env):
        """
        Concept 12: Enemy within 3 steps directly below the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 12 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (enemy.y - agent.y) in range(1, 4):
            return True
        return False

    @staticmethod
    def is_concept_13_present(env):
        """
        Concept 13: Enemy is exactly 2 steps directly to the left of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 13 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.y == agent.y and (agent.x - enemy.x) == 2:
            return True
        return False

    @staticmethod
    def is_concept_14_present(env):
        """
        Concept 14: Enemy is exactly 2 steps directly to the right of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 14 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.y == agent.y and (enemy.x - agent.x) == 2:
            return True
        return False

    @staticmethod
    def is_concept_15_present(env):
        """
        Concept 15: Enemy is exactly 2 steps directly above the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 15 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (agent.y - enemy.y) == 2:
            return True
        return False

    @staticmethod
    def is_concept_16_present(env):
        """
        Concept 16: Enemy is exactly 2 steps directly below the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 16 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (enemy.y - agent.y) == 2:
            return True
        return False

    @staticmethod
    def is_concept_17_present(env):
        """
        Concept 17: Enemy is exactly 1 step directly to the left of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 17 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.y == agent.y and (agent.x - enemy.x) == 1:
            return True
        return False

    @staticmethod
    def is_concept_18_present(env):
        """
        Concept 18: Enemy is exactly 1 step directly to the right of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 18 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.y == agent.y and (enemy.x - agent.x) == 1:
            return True
        return False

    @staticmethod
    def is_concept_19_present(env):
        """
        Concept 19: Enemy is exactly 1 step directly above the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 19 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (agent.y - enemy.y) == 1:
            return True
        return False

    @staticmethod
    def is_concept_20_present(env):
        """
        Concept 20: Enemy is exactly 1 step directly below the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 20 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Only provide an env with 1 enemy, to avoid confusing results"
            )

        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        if enemy.x == agent.x and (enemy.y - agent.y) == 1:
            return True
        return False

    @staticmethod
    def is_concept_21_present(env):
        """
        Concept 21: Agent is against the left wall.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 21 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        if agent.x == 0:
            return True
        return False

    @staticmethod
    def is_concept_22_present(env):
        """
        Concept 22: Agent is against the right wall.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 22 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        if agent.x == env.unwrapped.grid_side_length - 1:
            return True
        return False

    @staticmethod
    def is_concept_23_present(env):
        """
        Concept 23: Agent is against the upper wall.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 23 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        if agent.y == 0:
            return True
        return False

    @staticmethod
    def is_concept_24_present(env):
        """
        Concept 24: Agent is against the bottom wall.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 24 is present, otherwise False.
        """
        agent = env.unwrapped.agent
        if agent.y == env.unwrapped.grid_side_length - 1:
            return True
        return False

    @staticmethod
    def is_concept_25_present(env):
        """
        Checks whether Concept 25 is present in the current state of the environment.
        Concept 25: The enemy is above the agent (more above than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 25 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "This concept only allows for 1 enemy in the environment")

        enemy = env.unwrapped.enemies[0]

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        enemy_x, enemy_y = enemy.x, enemy.y

        # Calculate differences
        delta_x = enemy_x - agent_x
        delta_y = enemy_y - agent_y

        # Enemy is above if delta_y < 0 and |delta_y| > |delta_x|
        if delta_y < 0 and abs(delta_y) > abs(delta_x):
            return True

        return False

    @staticmethod
    def is_concept_26_present(env):
        """
        Checks whether Concept 26 is present in the current state of the environment.
        Concept 26: The enemy is to the right of the agent (more to the right than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 26 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "This concept only allows for 1 enemy in the environment")

        enemy = env.unwrapped.enemies[0]

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        enemy_x, enemy_y = enemy.x, enemy.y

        # Calculate differences
        delta_x = enemy_x - agent_x
        delta_y = enemy_y - agent_y

        # Enemy is to the right if delta_x > 0 and |delta_x| > |delta_y|
        if delta_x > 0 and abs(delta_x) > abs(delta_y):
            return True

        return False

    @staticmethod
    def is_concept_27_present(env):
        """
        Checks whether Concept 27 is present in the current state of the environment.
        Concept 27: The enemy is below the agent (more below than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 27 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "This concept only allows for 1 enemy in the environment")

        enemy = env.unwrapped.enemies[0]

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        enemy_x, enemy_y = enemy.x, enemy.y

        # Calculate differences
        delta_x = enemy_x - agent_x
        delta_y = enemy_y - agent_y

        # Enemy is below if delta_y > 0 and |delta_y| > |delta_x|
        if delta_y > 0 and abs(delta_y) > abs(delta_x):
            return True

        return False

    @staticmethod
    def is_concept_28_present(env):
        """
        Checks whether Concept 28 is present in the current state of the environment.
        Concept 28: The enemy is to the left of the agent (more to the left than any other direction).

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 28 is present, otherwise False.
        :raises ValueError: If there is not exactly one enemy in the environment.
        """
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "This concept only allows for 1 enemy in the environment")

        enemy = env.unwrapped.enemies[0]

        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y
        enemy_x, enemy_y = enemy.x, enemy.y

        # Calculate differences
        delta_x = enemy_x - agent_x
        delta_y = enemy_y - agent_y

        # Enemy is to the left if delta_x < 0 and |delta_x| > |delta_y|
        if delta_x < 0 and abs(delta_x) > abs(delta_y):
            return True

        return False

    @staticmethod
    def is_concept_29_present(env):
        """
        Checks whether Concept 29 is present in the current state of the environment.
        Concept 29: Avocado is above the agent and enemy is below the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 29 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado and one enemy in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "Concept 29 requires exactly 1 avocado in the environment.")
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Concept 29 requires exactly 1 enemy in the environment.")

        avocado = env.unwrapped.avocados[0]
        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        # Check if avocado is above the agent
        delta_x_avo = avocado.x - agent.x
        delta_y_avo = avocado.y - agent.y
        avo_above = (delta_y_avo < 0) and (abs(delta_y_avo) > abs(delta_x_avo))

        # Check if enemy is below the agent
        delta_x_enemy = enemy.x - agent.x
        delta_y_enemy = enemy.y - agent.y
        enemy_below = (delta_y_enemy > 0) and (
            abs(delta_y_enemy) > abs(delta_x_enemy))

        return avo_above and enemy_below

    @staticmethod
    def is_concept_30_present(env):
        """
        Checks whether Concept 30 is present in the current state of the environment.
        Concept 30: Avocado is to the right of the agent and enemy is to the left of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 30 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado and one enemy in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "Concept 30 requires exactly 1 avocado in the environment.")
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Concept 30 requires exactly 1 enemy in the environment.")

        avocado = env.unwrapped.avocados[0]
        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        # Check if avocado is to the right of the agent
        delta_x_avo = avocado.x - agent.x
        delta_y_avo = avocado.y - agent.y
        avo_right = (delta_x_avo > 0) and (abs(delta_x_avo) > abs(delta_y_avo))

        # Check if enemy is to the left of the agent
        delta_x_enemy = enemy.x - agent.x
        delta_y_enemy = enemy.y - agent.y
        enemy_left = (delta_x_enemy < 0) and (
            abs(delta_x_enemy) > abs(delta_y_enemy))

        return avo_right and enemy_left

    @staticmethod
    def is_concept_31_present(env):
        """
        Checks whether Concept 31 is present in the current state of the environment.
        Concept 31: Avocado is below the agent and enemy is above the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 31 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado and one enemy in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "Concept 31 requires exactly 1 avocado in the environment.")
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Concept 31 requires exactly 1 enemy in the environment.")

        avocado = env.unwrapped.avocados[0]
        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        # Check if avocado is below the agent
        delta_x_avo = avocado.x - agent.x
        delta_y_avo = avocado.y - agent.y
        avo_below = (delta_y_avo > 0) and (abs(delta_y_avo) > abs(delta_x_avo))

        # Check if enemy is above the agent
        delta_x_enemy = enemy.x - agent.x
        delta_y_enemy = enemy.y - agent.y
        enemy_above = (delta_y_enemy < 0) and (
            abs(delta_y_enemy) > abs(delta_x_enemy))

        return avo_below and enemy_above

    @staticmethod
    def is_concept_32_present(env):
        """
        Checks whether Concept 32 is present in the current state of the environment.
        Concept 32: Avocado is to the left of the agent and enemy is to the right of the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if Concept 32 is present, otherwise False.
        :raises ValueError: If there is not exactly one avocado and one enemy in the environment.
        """
        if len(env.unwrapped.avocados) != 1:
            raise ValueError(
                "Concept 32 requires exactly 1 avocado in the environment.")
        if len(env.unwrapped.enemies) != 1:
            raise ValueError(
                "Concept 32 requires exactly 1 enemy in the environment.")

        avocado = env.unwrapped.avocados[0]
        enemy = env.unwrapped.enemies[0]
        agent = env.unwrapped.agent

        # Check if avocado is to the left of the agent
        delta_x_avo = avocado.x - agent.x
        delta_y_avo = avocado.y - agent.y
        avo_left = (delta_x_avo < 0) and (abs(delta_x_avo) > abs(delta_y_avo))

        # Check if enemy is to the right of the agent
        delta_x_enemy = enemy.x - agent.x
        delta_y_enemy = enemy.y - agent.y
        enemy_right = (delta_x_enemy > 0) and (
            abs(delta_x_enemy) > abs(delta_y_enemy))

        return avo_left and enemy_right
