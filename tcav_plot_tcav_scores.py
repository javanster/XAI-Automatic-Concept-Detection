import pandas as pd
from tcav_bar_plot import tcav_barplot
import gymnasium as gym
import avocado_run

TRAIN_RUN_NAME = "eager_disco_16"
MODEL_NAME = "best_model"

tcav_scores_df = pd.read_csv(
    f'tcav_data/tcav_scores/{TRAIN_RUN_NAME}/{MODEL_NAME}/tcav_scores.csv')

env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)
action_dict = env.unwrapped.action_dict


target_classes = list(action_dict.values())

for target_class in target_classes:
    tcav_barplot(
        df=tcav_scores_df,
        target_class_name=target_class,
        show=True,
        file_path_for_saving=f"tcav_data/tcav_scores/{TRAIN_RUN_NAME}/{MODEL_NAME}/tcav_barplot_target_class_{target_class}.png"
    )
