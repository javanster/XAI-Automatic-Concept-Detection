from shap import GradientExplainer
from shap_image_plot import shap_image_plot
import gymnasium as gym
import avocado_run
from DoubleDQNAgent import DoubleDQNAgent
from ObservationHandler import ObservationHandler
import matplotlib.pyplot as plt
import os

train_run_name = "wild-glitter-14"
model_name = "model___99.23avg___99.90max___98.10min__1726876253"

action_dict = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
    4: "do_nothing",
}

env = gym.make(id="AvocadoRun-v0")

agent = DoubleDQNAgent(
    env=env,
    model_path=f"models/{train_run_name}/{model_name}.keras"
)

observation_handler = ObservationHandler(env=env)

background_observations = observation_handler.load_observations(
    file_path="data/observations/random_observations_1726941159.npy", normalize=True)

actions = ["up", "right", "down", "left", "do_nothing"]
observation_focuses = ["avocado", "enemy"]

for good_action_for_obs in actions:
    for observation_focus in observation_focuses:

        observations_to_explain = observation_handler.load_observations(
            file_path=f"data/observations/{good_action_for_obs}_good_action_{observation_focus}_focused_observations.npy",  normalize=True)

        if len(observations_to_explain) > 0:
            explainer = GradientExplainer(
                data=background_observations,
                model=agent.online_model
            )

            shap_values = explainer.shap_values(observations_to_explain)
            q_values = agent.online_model.predict(observations_to_explain)

            num_actions = env.action_space.n
            shap_values_list = []

            for action in range(num_actions):
                shap_values_action = shap_values[..., action]
                shap_values_list.append(shap_values_action)

            labels = [
                f"Action \"{action_dict[action]}\"" for action in range(num_actions)]

            shap_image_plot(
                shap_values_list=shap_values_list,
                pixel_values=observations_to_explain,
                q_values=q_values,
                labels=labels,
                show=False
            )

            plt.suptitle(
                f"\"{good_action_for_obs}\" good action, {observation_focus} focused", fontsize=16)

            filename = f"{good_action_for_obs}_good_action_{observation_focus}_focused_obs.png"
            filepath = os.path.join(
                f"shap_explanations/{train_run_name}/{model_name}/", filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
