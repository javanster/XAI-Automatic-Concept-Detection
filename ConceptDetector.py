class ConceptDetector:

    @staticmethod
    def is_concept_1_present(env):
        """
        Checks whether concept 1 is present in the current state of the environment.
        Concept 1: The agent is to the left of the avocado, with no enemies between the agent and
        the avocado, and no enemy directly on the avocado.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if concept 1 is present, otherwise False.
        """
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for avocado in env.unwrapped.avocados:
            avocado_x, avocado_y = avocado.x, avocado.y

            # Condition 1: The agent is to the left of the avocado (same row, x < avocado_x)
            if agent_y == avocado_y and agent_x < avocado_x:

                # Condition 2: Check that no enemies are between the agent and the avocado,
                # and also ensure no enemy is on the avocado's position
                enemies_in_path_or_on_avocado = [
                    enemy for enemy in env.unwrapped.enemies
                    if enemy.y == agent_y and agent_x < enemy.x <= avocado_x
                ]

                # If no enemies are in the path or on the avocado, return True
                if not enemies_in_path_or_on_avocado:
                    return True

        # If no avocados meet the conditions, return False
        return False

    @staticmethod
    def is_concept_2_present(env):
        """
        Checks whether the minimum shortest distance to any enemy is exactly two steps away.
        Concept 2: The agent's closest enemy is exactly two steps away. If any enemy is closer,
        the concept is considered absent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if the minimum shortest distance to any enemy is exactly two steps away, otherwise False.
        """
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        min_distance = float('inf')

        for enemy in env.unwrapped.enemies:
            enemy_x, enemy_y = enemy.x, enemy.y

            manhattan_distance = abs(agent_x - enemy_x) + \
                abs(agent_y - enemy_y)

            if manhattan_distance < min_distance:
                min_distance = manhattan_distance

        return min_distance == 2

    @staticmethod
    def is_concept_3_present(env):
        """
        Checks whether any avocado is located in the bottom right quartile of the environment.
        Concept 3: The avocado is located in the bottom right quartile of the grid.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if at least one avocado is in the bottom right quartile, otherwise False.
        """
        # Get the grid side length to determine the quartiles
        grid_size = env.unwrapped.grid_side_length

        # Define the bounds of the bottom right quartile
        # Bottom right quartile means the bottom-right quarter of the grid.
        # This is where x > grid_size / 2 and y > grid_size / 2
        x_bound = grid_size // 2
        y_bound = grid_size // 2

        # Check if any avocado is within the bottom right quartile
        for avocado in env.unwrapped.avocados:
            avocado_x, avocado_y = avocado.x, avocado.y

            if avocado_x >= x_bound and avocado_y >= y_bound:
                return True

        # If no avocado is in the bottom right quartile, return False
        return False

    @staticmethod
    def is_concept_4_present(env):
        """
        Checks whether the agent is within a 3x3 grid space (up to 1 step away in any direction,
        including diagonals) of at least one enemy.
        Concept 4: The agent is within a 3x3 grid space of at least one enemy.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if the agent is within a 3x3 grid space of at least one enemy, otherwise False.
        """
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for enemy in env.unwrapped.enemies:
            enemy_x, enemy_y = enemy.x, enemy.y

            # Calculate the Manhattan distance between the agent and the enemy
            manhattan_distance = abs(agent_x - enemy_x) + \
                abs(agent_y - enemy_y)

            # If the Manhattan distance is 2 or less, the agent is within the 3x3 grid space of this enemy
            if manhattan_distance <= 2:
                return True

        # If no enemies are within the 3x3 grid space, return False
        return False

    @staticmethod
    def is_concept_5_present(env):
        """
        Checks whether the agent is exactly one step to the left of the avocado,
        and no enemy is closer than 3 steps away from the agent.

        Concept 5: The agent is one step to the left of the avocado, and no enemy is closer than 3 steps away.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if concept is present, otherwise False.
        """
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for avocado in env.unwrapped.avocados:
            avocado_x, avocado_y = avocado.x, avocado.y

            # Condition 1: The agent is exactly one step to the left of the avocado (same row, x = avocado_x - 1)
            if agent_y == avocado_y and agent_x == avocado_x - 1:

                # Condition 2: Check that no enemies are closer than 3 steps away from the agent
                enemies_too_close = [
                    enemy for enemy in env.unwrapped.enemies
                    # Manhattan distance < 2
                    if abs(agent_x - enemy.x) + abs(agent_y - enemy.y) < 3
                ]

                # If no enemies are closer than 3 steps, return True
                if not enemies_too_close:
                    return True

        # If no avocados or enemies meet the conditions, return False
        return False

    @staticmethod
    def is_concept_6_present(env):
        """
        Checks whether the avocado is in the bottom-right corner of the grid,
        the agent is one step to the left of the avocado, and no enemy is closer than 3 steps away from the agent.

        Concept 6: The avocado is in the bottom-right corner, the agent is one step to the left, 
        and no enemy is closer than 3 steps away from the agent.

        :param env: The current environment instance (AvocadoRunEnv)
        :return: True if concept is present, otherwise False.
        """
        grid_size = env.unwrapped.grid_side_length
        agent_x, agent_y = env.unwrapped.agent.x, env.unwrapped.agent.y

        for avocado in env.unwrapped.avocados:
            avocado_x, avocado_y = avocado.x, avocado.y

            # Condition 1: Avocado is in the bottom-right corner (bottom-right corner is at (grid_size - 1, grid_size - 1))
            if avocado_x == grid_size - 1 and avocado_y == grid_size - 1:

                # Condition 2: The agent is exactly one step to the left of the avocado (same row, x = avocado_x - 1)
                if agent_y == avocado_y and agent_x == avocado_x - 1:

                    # Condition 3: No enemies are closer than 3 steps away from the agent
                    enemies_too_close = [
                        enemy for enemy in env.unwrapped.enemies
                        # Manhattan distance < 3
                        if abs(agent_x - enemy.x) + abs(agent_y - enemy.y) < 3
                    ]

                    # If no enemies are closer than 3 steps, return True
                    if not enemies_too_close:
                        return True

        return False
