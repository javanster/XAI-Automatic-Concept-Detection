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
