import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0", render_mode=None)
env.reset()

LEARNING_RATE = 0.1  # By how much we tweak q-values
DISCOUNT = 0.95  # How important future rewards are
EPISODES = 20000
epsilon = 0.5  # The probability of which the agent chooses a random action
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
EPSILON_DECAY_RATE = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

SHOW_EVERY = 1000

# Defining each observation variable to be divided into 20 buckets
DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))  # 20 * 20 * 3, 20 intervals of position * 20 intevals of velocity * 3 possible actions

episode_rewards = []
aggregate_episode_rewards = {'episode': [],
                             'average': [], 'min': [], 'max': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


for episode in range(EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")
        env.close()
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env.close()
        env = gym.make("MountainCar-v0", render_mode=None)

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated:
        step += 1
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, info = env.step(
            action)
        next_discrete_state = get_discrete_state(new_state)
        episode_reward += reward

        if not terminated and not truncated:
            max_future_q = np.max(q_table[next_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
                (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif terminated:
            # print(f"Goal reached on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = next_discrete_state

    if START_EPSILON_DECAY <= episode <= END_EPSILON_DECAY:
        epsilon -= EPSILON_DECAY_RATE

    episode_rewards.append(episode_reward)

    if episode % SHOW_EVERY == 0:
        rewards_for_this_episode = episode_rewards[-SHOW_EVERY:]

        average_reward = sum(
            rewards_for_this_episode) / len(rewards_for_this_episode)
        min_reward = min(rewards_for_this_episode)
        max_reward = max(rewards_for_this_episode)

        aggregate_episode_rewards['episode'].append(episode)
        aggregate_episode_rewards['average'].append(average_reward)
        aggregate_episode_rewards['min'].append(min_reward)
        aggregate_episode_rewards['max'].append(max_reward)

        print(
            f"Episode: {episode} Avg: {average_reward} Min: {min_reward} Max {max_reward}")

env.close()

plt.plot(aggregate_episode_rewards['episode'],
         aggregate_episode_rewards['average'], label="avg")
plt.plot(aggregate_episode_rewards['episode'],
         aggregate_episode_rewards['min'], label="min")
plt.plot(aggregate_episode_rewards['episode'],
         aggregate_episode_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
