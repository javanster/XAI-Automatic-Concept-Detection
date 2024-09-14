from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
from keras.layers import Input
from keras.saving import load_model
import wandb
import gymnasium as gym
import os


class DoubleDQNAgent:
    def __init__(self, env, config, model_path=None):

        self.config = config
        self.REPLAY_BUFFER_SIZE = config["replay_buffer_size"]
        self.MIN_REPLAY_BUFFER_SIZE = config["min_replay_buffer_size"]
        self.MINIBATCH_SIZE = config["minibatch_size"]
        self.MODEL_NAME = "64x2"
        self.DISCOUNT = config["discount"]
        self.TRAINING_FREQUENCY = config["training_frequency"]
        self.UPDATE_TARGET_EVERY = config["update_target_every"]
        self.LEARNING_RATE = config["learning_rate"]
        self.env = env

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "DoubleDQNAgent only supports discrete action spaces.")

        online_model = load_model(
            model_path) if model_path else self._create_model()
        self.online_model = online_model
        self.target_model = self._create_model()
        self.target_model.set_weights(self.online_model.get_weights())

        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)

        self.training_counter = 0
        self.target_update_counter = 0

        self.epsilon = config["starting_epsilon"]
        self.EPSILON_DECAY = config["epsilon_decay"]
        self.MIN_EPSILON = config["min_epsilon"]

        self.best_static_average = float('-inf')

    def _create_model(self):
        model = Sequential()
        model.add(Input(shape=self.env.observation_space.shape))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.LEARNING_RATE),
            metrics=["accuracy"]
        )

        return model

    def _update_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def _get_qs(self, state):
        return self.online_model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def _train_network(self, terminal_state):
        if len(self.replay_buffer) < self.MIN_REPLAY_BUFFER_SIZE:
            return

        minibatch = random.sample(self.replay_buffer, self.MINIBATCH_SIZE)

        current_states = np.array([transition[0]
                                  for transition in minibatch]) / 255
        current_qs_list = self.online_model.predict(current_states)

        next_states = np.array(
            [transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(next_states)

        X = []
        y = []

        for index, (current_state, action, reward, _, terminated) in enumerate(minibatch):
            if not terminated:
                # In accordance with double dqn
                max_future_action = np.argmax(current_qs_list[index])
                max_future_q = future_qs_list[index][max_future_action]
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.online_model.fit(
            np.array(X) / 255, np.array(y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False if terminal_state else None,
        )

    def train(self, episodes, average_window=100, track_metrics=False):
        if track_metrics:
            wandb.init(
                project=self.config["project_name"],
                config=self.config,
                mode="online"
            )

        if not os.path.exists('models'):
            os.makedirs('models')

        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        rewards_queue = deque(maxlen=average_window)

        for episode in tqdm(range(1, episodes + 1), unit="episode"):

            episode_reward = 0
            current_state, _ = self.env.reset()

            terminated = False
            truncated = False

            while not terminated and not truncated:
                if np.random.random() > self.epsilon:
                    action = np.argmax(self._get_qs(current_state))
                else:
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, terminated, truncated, _ = self.env.step(
                    action=action)

                episode_reward += reward

                self._update_replay_buffer(
                    (current_state, action, reward, new_state, terminated))

                # Updates every step
                self.training_counter += 1
                self.target_update_counter += 1

                if (self.training_counter >= self.TRAINING_FREQUENCY):
                    self._train_network(terminated)
                    self.training_counter = 0

                if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
                    self.target_model.set_weights(
                        self.online_model.get_weights())
                    self.target_update_counter = 0

                current_state = new_state

            rewards_queue.append(episode_reward)

            log_data = {
                "epsilon": self.epsilon,
                "episode_reward": episode_reward
            }

            if len(rewards_queue) >= average_window:
                average_reward = sum(rewards_queue) / average_window
                min_reward = min(rewards_queue)
                max_reward = max(rewards_queue)

                log_data["rolling_average_reward"] = average_reward
                log_data["min_reward_over_rolling_window"] = min_reward
                log_data["max_reward_over_rolling_window"] = max_reward

                if episode % average_window == 0:
                    log_data["static_average_reward"] = average_reward

                    # Checks every average window episode whether a new best static average is reached
                    if average_reward > self.best_static_average:
                        self.best_static_average = average_reward
                        self.online_model.save(
                            f'models/{self.MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

            if track_metrics:
                wandb.log(log_data)

            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon = max(self.MIN_EPSILON, self.epsilon)

        self.env.close()

    def test(self, episodes=10, env=None):
        if env:
            self.env = env

        for _ in range(episodes):

            observation, _ = self.env.reset()
            terminated = False

            self.env.render()

            while not terminated:
                observation_reshaped = np.array(
                    observation).reshape(-1, *self.env.observation_space.shape) / 255.0
                action = np.argmax(
                    self.online_model.predict(observation_reshaped))

                observation, _, terminated, _, _ = self.env.step(
                    action=action)

                self.env.render()

            time.sleep(2)

        self.env.close()
