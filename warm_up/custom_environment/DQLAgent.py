from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np
import random
import tensorflow as tf
import os
from tqdm import tqdm
from keras.layers import Input
from keras.saving import load_model


class DQLAgent:
    """
    A class for a Deep Q-Learning agent utilizing two CNNs for learning
    """

    def __init__(self, env) -> None:
        self.REPLAY_BUFFER_SIZE = 50_000
        self.MIN_REPLAY_BUFFER_SIZE = 1_000
        self.MINIBATCH_SIZE = 64
        self.MODEL_NAME = "256x2"
        self.DISCOUNT = 0.99
        self.UPDATE_TARGET_EVERY = 5
        self.MIN_REWARD = -200
        self.env = env

        self.online_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.online_model.get_weights())

        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)

        self.target_update_counter = 0

        self.epsilon = 1
        self.EPSILON_DECAY = 0.99975
        self.MIN_EPSILON = 0.001

        self.AGGREGATE_STATS_EVERY = 500
        self.SHOW_PREVIEW = True

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=self.env.OBSERVATION_SPACE_VALUES))
        model.add(
            Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(
            learning_rate=0.001), metrics=["accuracy"])

        return model

    def update_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def get_qs(self, state):
        return self.online_model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_network(self, terminal_state):
        if len(self.replay_buffer) < self.MIN_REPLAY_BUFFER_SIZE:
            return

        minibatch = random.sample(self.replay_buffer, self.MINIBATCH_SIZE)

        current_states = np.array([transition[0]
                                  for transition in minibatch]) / 255
        current_qs_list = self.online_model.predict(current_states)

        new_current_states = np.array(
            [transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, _, terminated) in enumerate(minibatch):
            if not terminated:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.online_model.fit(np.array(X) / 255, np.array(y),
                              batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.online_model.get_weights())
            self.target_update_counter = 0

    def train(self, episodes):
        ep_rewards = [-200]

        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        for episode in tqdm(range(1, episodes + 1), unit="episode"):

            episode_reward = 0
            step = 0
            current_state = self.env.reset()

            terminated = False

            while not terminated:
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.get_qs(current_state))
                else:
                    action = np.random.randint(0, self.env.ACTION_SPACE_SIZE)

                new_state, reward, terminated = self.env.step(action)

                episode_reward += reward

                if self.SHOW_PREVIEW and episode % self.AGGREGATE_STATS_EVERY == 0:
                    self.env.render()

                self.update_replay_buffer(
                    (current_state, action, reward, new_state, terminated))
                self.train_network(terminated)

                current_state = new_state
                step += 1

            ep_rewards.append(episode_reward)

            if episode % self.AGGREGATE_STATS_EVERY == 0 or episode == 1 or episode == episodes:
                rewards_for_this_episode = ep_rewards[-self.AGGREGATE_STATS_EVERY:]

                average_reward = sum(rewards_for_this_episode) / \
                    len(rewards_for_this_episode)
                min_reward = min(rewards_for_this_episode)
                max_reward = max(rewards_for_this_episode)

                if min_reward >= self.MIN_REWARD or episode == episodes:
                    self.online_model.save(
                        f'models/{self.MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon = max(self.MIN_EPSILON, self.epsilon)

    def test(self, model_path):
        model = load_model(model_path, custom_objects=None,
                           compile=True, safe_mode=True)
        while True:

            observation = self.env.reset()
            terminated = False

            while not terminated:
                observation_reshaped = np.array(
                    observation).reshape(-1, *self.env.OBSERVATION_SPACE_VALUES) / 255.0
                action = np.argmax(model.predict(observation_reshaped))
                observation, _, terminated = self.env.step(
                    action, step_limit=False)
                self.env.render()

            time.sleep(2)
