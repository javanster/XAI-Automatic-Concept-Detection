from shap import image_plot, GradientExplainer
import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent
from ObservationHandler import ObservationHandler
import matplotlib.pyplot as plt


env = gym.make(id="AvocadoRun-v0")

agent = DoubleDQNAgent(
    env=env,
    model_path="models/wild-glitter-14/model___99.23avg___99.90max___98.10min__1726876253.keras"
)

observation_handler = ObservationHandler(env=env)

background_observations = observation_handler.load_observations(
    file_path="data/random_observations.npy", normalize=True)

observations_to_explain = observation_handler.load_observations(
    file_path="data/do_nothing_good_move_enemy_focused_observations.npy",  normalize=True)


explainer = GradientExplainer(data=background_observations,
                              model=agent.online_model)

shap_values = explainer.shap_values(observations_to_explain)

action_dict = {
    0: "Up",
    1: "Right",
    2: "Down",
    3: "Left",
    4: "Do nothing",
}

for action in range(env.action_space.n):
    shap_values_action = shap_values[..., action]

    image_plot(shap_values_action, observations_to_explain, show=False)
    plt.suptitle(
        f"SHAP for action \"{action_dict[action]}\"", fontsize=16)
    plt.show()
