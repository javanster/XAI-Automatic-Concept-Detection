from shap import DeepExplainer
from shap_utils import shap_image_plot
from keras.api.saving import load_model
import gymnasium as gym
import avocado_run
from utils import AvocadoRunObservationHandler
import matplotlib.pyplot as plt
import os

train_run_names = ["wild_glitter_14", "eager_disco_16"]
model_names = ["best_model", "best_model"]

env = gym.make(id="AvocadoRun-v0")
action_dict = env.unwrapped.action_dict

for i in range(len(train_run_names)):
    train_run_name = train_run_names[i]
    model_name = model_names[i]

    model = load_model(f"models/{train_run_name}/{model_name}.keras")

    background_observations = AvocadoRunObservationHandler.load_observations(
        file_path="shap_data/observations/normal_environment/random_observations.npy", normalize=True)

    actions = ["up", "right", "down", "left", "do_nothing"]
    observation_focuses = ["avocado", "enemy"]

    for good_action_for_obs in actions:
        for observation_focus in observation_focuses:

            observations_to_explain = AvocadoRunObservationHandler.load_observations(
                file_path=f"shap_data/observations/normal_environment/{good_action_for_obs}_good_action_{observation_focus}_focused_observations.npy",  normalize=True)

            if len(observations_to_explain) > 0:
                explainer = DeepExplainer(
                    data=background_observations,
                    model=model
                )

                shap_values = explainer.shap_values(observations_to_explain)
                q_values = model.predict(observations_to_explain)

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
                    f"shap_explanations/normal_environment/{train_run_name}/{model_name}/", filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
