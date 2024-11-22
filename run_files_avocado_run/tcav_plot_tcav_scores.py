import pandas as pd
from tcav import tcav_scores_barplot
import gymnasium as gym
import avocado_run


def plot_tcav_scores(train_run_name, model_name, tcav_score_data_type, batch, concepts_per_plot, target_classes):
    tcav_scores_df = pd.read_csv(
        f'tcav_explanations/tcav_scores/{train_run_name}/{model_name}/{tcav_score_data_type}_observations/tcav_scores_cav_batch_{batch}.csv')

    concept_indices = sorted(tcav_scores_df['concept_index'].unique())
    concept_n = len(concept_indices)

    for i in range(0, concept_n, concepts_per_plot):
        # Create a subset of the concept indices for this partition, containing max 5 unique concepts
        current_concept_indices = concept_indices[i:i + concepts_per_plot]

        subset_df = tcav_scores_df[tcav_scores_df['concept_index'].isin(
            current_concept_indices)]

        for target_class in target_classes:
            tcav_scores_barplot(
                df=subset_df,
                target_class_name=target_class,
                show=False,
                file_path_for_saving=f"tcav_explanations/tcav_scores/{train_run_name}/{model_name}/{tcav_score_data_type}_observations/plots_cav_batch_{batch}/tcav_barplot_target_class_{target_class}_concepts_{current_concept_indices}.png"
            )


if __name__ == "__main__":
    train_run_name = "moonlit_coffin_71"
    model_name = "best_model"
    tcav_score_data_type = "model_specific"
    batch = 0
    concepts_per_plot = 4

    env = gym.make(id="AvocadoRun-v0", num_avocados=1, num_enemies=2)
    target_classes = list(env.unwrapped.action_dict.values())

    plot_tcav_scores(
        train_run_name=train_run_name,
        model_name=model_name,
        tcav_score_data_type=tcav_score_data_type,
        batch=batch,
        concepts_per_plot=concepts_per_plot,
        target_classes=target_classes
    )
