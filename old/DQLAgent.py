from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
from keras.layers import Input
from keras.saving import load_model
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


class DQLAgent:
    """
    A class for a Deep Q-Learning agent utilizing two CNNs for learning (double deep Q-networks)
    """

    def __init__(self,
                 env,
                 replay_buffer_size,
                 min_replay_buffer_size,
                 minibatch_size,
                 discount,
                 update_target_every,
                 epsilon,
                 epsilon_decay,
                 min_epsilon,
                 model_path,
                 ) -> None:

        self.REPLAY_BUFFER_SIZE = replay_buffer_size
        self.MIN_REPLAY_BUFFER_SIZE = min_replay_buffer_size
        self.MINIBATCH_SIZE = minibatch_size
        self.MODEL_NAME = "256x2"
        self.DISCOUNT = discount
        self.UPDATE_TARGET_EVERY = update_target_every
        self.env = env

        online_model = load_model(
            model_path) if model_path else self._create_model()
        self.online_model = online_model
        self.target_model = self._create_model()
        self.target_model.set_weights(self.online_model.get_weights())

        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)

        self.target_update_counter = 0

        self.epsilon = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.MIN_EPSILON = min_epsilon

    def _create_model(self):
        model = Sequential()
        model.add(Input(shape=self.env.observation_space_shape))
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

        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=0.001),
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

        self.online_model.fit(
            np.array(X) / 255, np.array(y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False if terminal_state else None,
            callbacks=[
                WandbMetricsLogger(log_freq=10),
                WandbModelCheckpoint("models/test.keras")
            ]
        )

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.online_model.get_weights())
            self.target_update_counter = 0

    def train(self, episodes, min_save_model_episode, save_model_every, rolling_average_window=None):
        wandb.init(
            project="AvocadoRun_DQLAgent",
            config={
                "replay_buffer_size": self.REPLAY_BUFFER_SIZE,
                "min_replay_buffer_size": self.MIN_REPLAY_BUFFER_SIZE,
                "minibatch_size": self.MINIBATCH_SIZE,
                "discount": self.DISCOUNT,
                "update_target_every": self.UPDATE_TARGET_EVERY,
                "epsilon": self.epsilon,
                "epsilon_decay": self.EPSILON_DECAY,
                "min_epsilon": self.MIN_EPSILON,
            }
        )

        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        rewards_queue = deque(maxlen=rolling_average_window)
        episode_for_stats_list = []
        average_reward_list = []

        plt.ion()
        _, ax = plt.subplots(figsize=(10, 5))

        for episode in tqdm(range(1, episodes + 1), unit="episode"):

            episode_reward = 0
            current_state = self.env.reset(
                model=self.online_model, episode=episode)

            terminated = False

            while not terminated:
                if np.random.random() > self.epsilon:
                    action = np.argmax(self._get_qs(current_state))
                else:
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, terminated, _, _ = self.env.step(
                    action=action, model=self.online_model, episode=episode)

                episode_reward += reward

                self._update_replay_buffer(
                    (current_state, action, reward, new_state, terminated))

                self.target_update_counter += 1  # Updates every step
                self._train_network(terminated)

                current_state = new_state

            rewards_queue.append(episode_reward)

            if rolling_average_window and len(rewards_queue) >= rolling_average_window:

                average_reward = sum(rewards_queue) / \
                    rolling_average_window

                episode_for_stats_list.append(episode)
                average_reward_list.append(average_reward)

                ax.clear()

                ax.plot(episode_for_stats_list, average_reward_list,
                        label=f"Rolling average reward with window {rolling_average_window}", color="purple")

                ax.set_title("Reward Metrics Over Time")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Average reward")

                ax.set_ylim([-500, 100])

                ax.legend(loc="upper left")

                plt.tight_layout()
                plt.pause(0.1)

            if episode >= min_save_model_episode and episode % save_model_every == 0:
                min_reward = min(rewards_queue)
                max_reward = max(rewards_queue)

                self.online_model.save(
                    f'models/{self.MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon = max(self.MIN_EPSILON, self.epsilon)

        plt.ioff()
        plt.show()

    def test(self, episodes=10):
        for episode in range(episodes):

            observation = self.env.reset(
                episode=episode, model=self.online_model)
            terminated = False

            while not terminated:
                observation_reshaped = np.array(
                    observation).reshape(-1, *self.env.observation_space_shape) / 255.0
                action = np.argmax(
                    self.online_model.predict(observation_reshaped))

                observation, _, terminated, _, _ = self.env.step(
                    action=action, episode=episode, step_limit=False, model=self.online_model)

            time.sleep(2)
