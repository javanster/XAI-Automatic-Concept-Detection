import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class QLAgent:
    """
    A class for a Q-Learning agent utilizing a q-table for learning
    """

    def __init__(self, env) -> None:
        self.SHOW_EVERY = 5000
        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.95
        self.EPSILON_DECAY = 0.9998
        self.epsilon = 0.9
        self.env = env
        self.q_table = np.random.uniform(
            low=-5, high=0, size=((self.env.SIZE, self.env.SIZE, self.env.SIZE, self.env.SIZE) + (self.env.ACTION_SPACE_SIZE,)))

    def train(self, episodes):
        episode_rewards = []

        for episode in range(episodes):
            if episode % self.SHOW_EVERY == 0:
                print(f"Episode: {episode}, Epsilon: {self.epsilon}")
                print(
                    f"{self.SHOW_EVERY} episode average: {np.average(episode_rewards[-self.SHOW_EVERY:])}")
                show = True
            else:
                show = False

            episode_reward = 0
            observation = self.env.reset()
            terminated = False

            while not terminated:
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.q_table[observation])
                else:
                    action = np.random.randint(0, self.env.ACTION_SPACE_SIZE)

                new_observation, reward, terminated = self.env.step(action)

                max_future_q = np.max(self.q_table[new_observation])
                current_q = self.q_table[observation][action]

                episode_reward += reward

                if terminated:
                    new_q = reward
                else:
                    new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * \
                            (reward + self.DISCOUNT * max_future_q)

                self.q_table[observation][action] = new_q

                if show:
                    self.env.render()

                observation = new_observation

            episode_rewards.append(episode_reward)
            self.epsilon *= self.EPSILON_DECAY

        moving_avg = np.convolve(episode_rewards, np.ones(
            (self.SHOW_EVERY,)) / self.SHOW_EVERY, mode="valid")

        with open(f"qtables/qtable-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(self.q_table, f)

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {self.SHOW_EVERY}")
        plt.xlabel(f"Episode num")
        plt.show()

    def test(self, qt_file_path):
        with open(qt_file_path, "rb") as f:
            self.q_table = pickle.load(f)

        while True:
            observation = self.env.reset()
            terminated = False

            while not terminated:
                action = np.argmax(self.q_table[observation])
                observation, _, terminated = self.env.step(
                    action, step_limit=False)
                self.env.render()

            time.sleep(2)
