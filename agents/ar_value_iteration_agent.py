import numpy as np
import os
from tqdm import tqdm


class ArValueIterationAgent:
    """
    A Value iteration agent for the AvocadoRun environment. Only works for 1 enemy and 1 avocado.
    """

    def __init__(self, env):
        self.env = env

        self.grid_side_length = env.unwrapped.grid_side_length
        self.actions = env.unwrapped.action_dict
        self.actions_n = len(self.actions)
        self.all_positions = [(x, y) for x in range(self.grid_side_length)
                              for y in range(self.grid_side_length)]
        self.all_states = self._generate_all_states()
        self.states_n = len(self.all_states)
        self.state_to_index = {state: idx for idx,
                               state in enumerate(self.all_states)}
        self.index_to_state = {idx: state for idx,
                               state in enumerate(self.all_states)}
        self.enemy_random_move_p = 2 / 3
        self.enemy_directed_move_p = 1 / 3
        self.step_penalty = env.unwrapped.STEP_PENALTY
        self.enemy_hit_penalty = env.unwrapped.ENEMY_HIT_PENALTY
        self.avocado_reward = env.unwrapped.AVOCADO_REWARD

    def _get_possible_enemy_moves(self, position):
        x, y = position
        moves = []
        if y > 0:
            moves.append(0)  # Up
        if x < self.grid_side_length - 1:
            moves.append(1)  # Right
        if y < self.grid_side_length - 1:
            moves.append(2)  # Down
        if x > 0:
            moves.append(3)  # Left
        return moves

    def _apply_action(self, position, action):
        x, y = position
        if action == 0 and y > 0:  # Up
            return (x, y - 1)
        elif action == 1 and x < self.grid_side_length - 1:  # Right
            return (x + 1, y)
        elif action == 2 and y < self.grid_side_length - 1:  # Down
            return (x, y + 1)
        elif action == 3 and x > 0:  # Left
            return (x - 1, y)
        elif action == 4:  # Do nothing
            return (x, y)
        else:
            return (x, y)

    def _generate_all_states(self):
        """
        Generates all valid states where the agent, avocado, and enemy positions are defined.

        Returns:
        - List of tuples: Each tuple represents a state as (agent_pos, avocado_pos, enemy_pos).
        """
        states = []
        for agent in self.all_positions:
            for avocado in self.all_positions:
                if avocado == agent:
                    continue  # Avocado cannot be on the agent's position
                for enemy in self.all_positions:
                    if enemy == agent:
                        continue  # Enemy cannot be on the agent's position
                    # Allow enemy and avocado to be on the same position
                    states.append((agent, avocado, enemy))
        return states

    def _get_enemy_moves(self, enemy_pos, agent_pos):
        """
        Determines the enemy's possible moves and their associated probabilities.

        Parameters:
        - enemy_pos (tuple): Current (x, y) position of the enemy.
        - agent_pos (tuple): Current (x, y) position of the agent.
        - avocado_pos (tuple): Current (x, y) position of the avocado.

        Returns:
        - List of tuples: Each tuple contains a possible move (str) and its probability (float).
        """
        move_probabilities = []
        towards_moves = []

        # Determine moves that bring the enemy closer to the agent
        if enemy_pos[0] < agent_pos[0]:
            towards_moves.append(0)  # Up
        elif enemy_pos[0] > agent_pos[0]:
            towards_moves.append(1)  # Right
        if enemy_pos[1] < agent_pos[1]:
            towards_moves.append(2)  # Down
        elif enemy_pos[1] > agent_pos[1]:
            towards_moves.append(3)  # Left
        if not towards_moves:
            towards_moves = self._get_possible_enemy_moves(enemy_pos)

        # Possible random moves (excluding "do_nothing")
        random_moves = self._get_possible_enemy_moves(enemy_pos)

        # Enemy is not on avocado's cell
        # Assign probabilities to moves towards the agent and random moves
        num_towards = len(towards_moves)
        num_random = len(random_moves)

        # Probability assignments
        if num_towards > 0:
            prob_towards = self.enemy_directed_move_p / num_towards
            for move in towards_moves:
                move_probabilities.append((move, prob_towards))

        if num_random > 0:
            prob_random = self.enemy_random_move_p / num_random
            for move in random_moves:
                move_probabilities.append((move, prob_random))

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(prob for _, prob in move_probabilities)
        if not np.isclose(total_prob, 1.0):
            move_probabilities = [(move, prob / total_prob)
                                  for move, prob in move_probabilities]

        return move_probabilities

    def find_optimal_policy(self, gamma, theta, max_iterations, file_path):
        V = np.zeros(self.states_n)
        policy = np.zeros(self.states_n)

        print("Starting iterations")
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.copy(V)

            with tqdm(total=len(self.all_states), unit="state") as pbar:
                for idx, state in enumerate(self.all_states):
                    agent_pos, avocado_pos, enemy_pos = state

                    action_values = np.zeros(self.actions_n)

                    for action in range(self.actions_n):

                        new_agent_pos = self._apply_action(agent_pos, action)

                        enemy_moves = self._get_enemy_moves(
                            enemy_pos, new_agent_pos)
                        expected_value = 0

                        for move, prob in enemy_moves:
                            new_enemy_pos = self._apply_action(enemy_pos, move)

                            if new_agent_pos == new_enemy_pos:
                                reward = self.enemy_hit_penalty
                                value = reward
                            elif new_agent_pos == avocado_pos:
                                reward = self.avocado_reward
                                value = reward
                            else:
                                reward = self.step_penalty
                                next_state = (
                                    new_agent_pos, avocado_pos, new_enemy_pos)
                                if next_state in self.state_to_index:
                                    next_idx = self.state_to_index[next_state]
                                    value = reward + gamma * V[next_idx]
                                else:
                                    # Invalid state (overlapping positions), treat as collision
                                    value = self.enemy_hit_penalty

                            expected_value += prob * value

                        action_values[action] = expected_value

                    best_action_value = np.max(action_values)
                    delta = max(delta, abs(best_action_value - V[idx]))
                    V_new[idx] = best_action_value

                    pbar.update(1)

            V = V_new
            print(f"Iteration {iteration + 1}, delta={delta:.6f}")
            if delta < theta:
                print("Value iteration converged.")
                break

        else:
            print(
                "Value iteration did not converge within the maximum number of iterations.")

            # Extract the policy
        print("Extracting the optimal policy...")
        for idx, state in enumerate(self.all_states):
            agent_pos, avocado_pos, enemy_pos = state

            # If the state is terminal, no action is needed
            if agent_pos == avocado_pos or agent_pos == enemy_pos:
                policy[idx] = -1
                continue

            action_values = np.zeros(self.actions_n)

            for action in range(self.actions_n):
                new_agent_pos = self._apply_action(agent_pos, action)

                # Enemy movement
                enemy_moves = self._get_enemy_moves(
                    enemy_pos, new_agent_pos, avocado_pos)
                expected_value = 0

                for move, prob in enemy_moves:
                    new_enemy_pos = self._apply_action(enemy_pos, move)

                    # Determine the outcome
                    if new_agent_pos == new_enemy_pos:
                        reward = self.enemy_hit_penalty
                        value = reward
                    elif new_agent_pos == avocado_pos:
                        reward = self.avocado_reward
                        value = reward
                    else:
                        reward = self.step_penalty
                        next_state = (
                            new_agent_pos, avocado_pos, new_enemy_pos)
                        if next_state in self.state_to_index:
                            next_idx = self.state_to_index[next_state]
                            value = reward + gamma * V[next_idx]
                        else:
                            # Invalid state (overlapping positions), treat as collision
                            value = self.enemy_hit_penalty

                    expected_value += prob * value

                action_values[action] = expected_value

            # Choose the action with the highest expected value
            best_action = np.argmax(action_values)
            policy[idx] = best_action

        print("Policy extraction completed.")

        np.save(file_path, policy)
        print(f"Optimal policy saved to '{file_path}'")

    def test_policy(self, policy_file_path, episodes=10, render=False, verify_never_caught=False):
        if not os.path.exists(policy_file_path):
            print(f"Policy file '{policy_file_path}' does not exist.")
            return
        policy = np.load(policy_file_path)

        never_caught = True

        with tqdm(total=episodes, unit="episode") as pbar:
            for _ in range(episodes):

                _, _ = self.env.reset()

                if len(self.env.unwrapped.enemies) != 1 or len(self.env.unwrapped.avocados) != 1:
                    raise ValueError(
                        "This agent only works with an AvocadoRun env with 1 enemy and 1 avocado")

                if render:
                    self.env.render()

                total_reward = 0

                terminated = False
                truncated = False

                while not terminated and not truncated:
                    agent_entity = self.env.unwrapped.agent
                    enemy_entities = self.env.unwrapped.enemies
                    avocado_entities = self.env.unwrapped.avocados

                    if agent_entity == enemy_entities[0]:
                        never_caught = False

                    agent_pos = (agent_entity.x, agent_entity.y)
                    avocado_pos = (
                        avocado_entities[0].x, avocado_entities[0].y)
                    enemy_pos = (enemy_entities[0].x, enemy_entities[0].y)

                    state = (agent_pos, avocado_pos, enemy_pos)

                    # Retrieve the state index
                    state_idx = self.state_to_index.get(state, None)
                    if state_idx is None:
                        print(
                            f"Error: State {state} not found in STATE_TO_INDEX.")
                        break

                    # Retrieve the action from the policy
                    action_idx = policy[state_idx]

                    # Check for terminal action
                    if action_idx == -1:
                        break

                    action = action_idx
                    _, reward, terminated, truncated, _ = self.env.step(
                        action)
                    total_reward += reward

                    if render:
                        self.env.render()

                pbar.update(1)

        if verify_never_caught:
            print(f"Agent never caught: {never_caught}")
