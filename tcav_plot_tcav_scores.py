import pandas as pd
from tcav_scores_barplot import tcav_scores_barplot
import gymnasium as gym
import avocado_run

TRAIN_RUN_NAME = "dutiful_frog_68"
MODEL_NAME = "best_model"
CONCEPTS_N = 4

tcav_scores_df = pd.read_csv(
    f'tcav_explanations/tcav_scores/{TRAIN_RUN_NAME}/{MODEL_NAME}/tcav_scores.csv')

env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)
action_dict = env.unwrapped.action_dict


target_classes = list(action_dict.values())

concept_indices = sorted(tcav_scores_df['concept_index'].unique())
concept_n = len(concept_indices)

for i in range(0, concept_n, CONCEPTS_N):
    # Create a subset of the concept indices for this partition, containing max 5 unique concepts
    current_concept_indices = concept_indices[i:i + CONCEPTS_N]

    subset_df = tcav_scores_df[tcav_scores_df['concept_index'].isin(
        current_concept_indices)]

    for target_class in target_classes:
        tcav_scores_barplot(
            df=subset_df,
            target_class_name=target_class,
            show=True,
            file_path_for_saving=f"tcav_explanations/tcav_scores/{TRAIN_RUN_NAME}/{MODEL_NAME}/tcav_barplot_target_class_{target_class}_concepts_{i}-{i + CONCEPTS_N - 1}.png"
        )
