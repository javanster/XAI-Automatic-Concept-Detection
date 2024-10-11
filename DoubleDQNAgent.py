from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, Input
from keras.api.optimizers import Adam
from collections import deque
import time
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
from keras.api.layers import Input
from keras.api.saving import load_model
import wandb
import gymnasium as gym
import os


class DoubleDQNAgent:
    def __init__(self, env, learning_rate=0.01, model_path=None):
        self.env = env

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "DoubleDQNAgent only supports discrete action spaces.")

        online_model = load_model(
            model_path) if model_path else self._create_model(learning_rate=learning_rate)
        self.online_model = online_model
        self.target_model = self._create_model(learning_rate=learning_rate)
        self.target_model.set_weights(self.online_model.get_weights())

        self.best_tumbling_window_average = float("-inf")

    def _create_model(self, learning_rate):
        model = Sequential()
        model.add(Input(shape=self.env.observation_space.shape))
        model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
        model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=learning_rate),
        )
        return model

    def _update_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def _train_network(self, terminal_state, min_replay_buffer_size, minibatch_size, discount):
        if len(self.replay_buffer) < min_replay_buffer_size:
            return

        minibatch = random.sample(self.replay_buffer, minibatch_size)

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
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.online_model.fit(
            np.array(X) / 255, np.array(y),
            batch_size=minibatch_size,
            verbose=0,
            shuffle=False if terminal_state else None,  # What function does this have?
        )

    def train(self, config, track_metrics=False):
        replay_buffer_size = config["replay_buffer_size"]
        min_replay_buffer_size = config["min_replay_buffer_size"]
        minibatch_size = config["minibatch_size"]
        discount = config["discount"]
        training_frequency = config["training_frequency"]
        update_target_every = config["update_target_every"]
        steps_to_train = config["steps_to_train"]

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        epsilon = config["starting_epsilon"]
        starting_epsilon = config["starting_epsilon"]
        prop_steps_epsilon_decay = config["prop_steps_epsilon_decay"]
        min_epsilon = config["min_epsilon"]

        num_decay_steps = steps_to_train * prop_steps_epsilon_decay

        epsilon_decay = (
            min_epsilon / starting_epsilon) ** (1 / num_decay_steps)

        if track_metrics:
            wandb.init(
                project=config["project_name"],
                config=config,
                mode="online"
            )

        if not os.path.exists("models"):
            os.makedirs("models")

        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        rewards_queue = deque(maxlen=config["episode_metrics_window"])

        steps_passed = 0
        episodes_passed = 0

        with tqdm(total=steps_to_train, unit="step") as pbar:
            while steps_passed < steps_to_train:

                current_state, _ = self.env.reset()
                episode_reward = 0
                terminated = False
                truncated = False

                while not terminated and not truncated:
                    if np.random.random() > epsilon:
                        state_reshaped = np.array(
                            current_state).reshape(-1, *current_state.shape)/255
                        q_values = self.online_model.predict(state_reshaped)[0]
                        action = np.argmax(q_values)
                    else:
                        action = np.random.randint(0, self.env.action_space.n)

                    new_state, reward, terminated, truncated, _ = self.env.step(
                        action=action)

                    steps_passed += 1
                    episode_reward += reward
                    pbar.update(1)

                    self._update_replay_buffer(
                        (current_state, action, reward, new_state, terminated))

                    if steps_passed % training_frequency == 0:
                        self._train_network(
                            terminated, min_replay_buffer_size, minibatch_size, discount)

                    if steps_passed % update_target_every == 0:
                        self.target_model.set_weights(
                            self.online_model.get_weights())

                    current_state = new_state

                    if epsilon > min_epsilon:
                        epsilon *= epsilon_decay
                        epsilon = max(min_epsilon, epsilon)

                    if track_metrics:
                        wandb.log({
                            "step": steps_passed,
                            "epsilon": epsilon,
                        })

                    if steps_passed >= steps_to_train:
                        break

                episodes_passed += 1
                rewards_queue.append(episode_reward)

                log_data = {
                    "episode": episodes_passed,
                    "episode_reward": episode_reward
                }

                if len(rewards_queue) >= config["episode_metrics_window"]:
                    average_reward = sum(rewards_queue) / \
                        config["episode_metrics_window"]
                    log_data["rolling_window_average_reward"] = average_reward

                    if episodes_passed % config["episode_metrics_window"] == 0:
                        min_reward = min(rewards_queue)
                        max_reward = max(rewards_queue)
                        log_data["tumbling_window_average_reward"] = average_reward
                        log_data["min_reward_in_tumbling_window"] = min_reward
                        log_data["max_reward_in_tumbling_window"] = max_reward

                        # Checks every average window episode whether a new best static average is reached
                        if average_reward > self.best_tumbling_window_average:
                            self.best_tumbling_window_average = average_reward
                            self.online_model.save(
                                f"models/model_{average_reward:_>9.4f}avg_{max_reward:_>9.4f}max_{min_reward:_>9.4f}min__{int(time.time())}.keras")

                if track_metrics:
                    wandb.log(log_data)

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
                q_values = self.online_model.predict(observation_reshaped)[0]
                action = np.argmax(q_values)

                observation, _, terminated, _, _ = self.env.step(
                    action=action)

                self.env.render()

            time.sleep(2)

        self.env.close()
