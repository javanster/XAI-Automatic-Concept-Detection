from shap import DeepExplainer
from shap_image_plot import shap_image_plot
from keras.api.saving import load_model
import gymnasium as gym
import avocado_run
from utils import ObservationHandler
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

    background_observations = ObservationHandler.load_observations(
        file_path=f"shap_data/observations/ood/1_avo_stuck/{train_run_name}/{model_name}/random_observations.npy", normalize=True)

    for i in range(0, 3):

        observations_to_explain = ObservationHandler.load_observations(
            file_path=f"shap_data/observations/ood/1_avo_stuck/{train_run_name}/{model_name}/ood_1_avo_stuck_{i}_observations.npy",  normalize=True)

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
                f"OOD observation, 0 enemies vs 1 enemy vs 2 avocado - {i}", fontsize=16)

            filename = f"ood_1_avo_stuck_{i}_obs.png"
            filepath = os.path.join(
                f"shap_explanations/ood/1_avo_stuck/{train_run_name}/{model_name}/", filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
