import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")


def tcav_barplot(df, target_class_name, show=True, file_path_for_saving=None):
    """
    Creates a grouped bar plot for TCAV scores for a specific target class.
    Displays the y-value (tcav_score) above each bar.
    """

    df_filtered = df[df['action'] == target_class_name]
    num_layers = len(df['layer_name'].unique())

    plt.figure(figsize=(14, 8))

    bar_plot = sns.barplot(
        data=df_filtered,
        x='concept_name',
        y='tcav_score',
        hue='layer_name',
        palette='viridis'
    )

    plt.title(
        f'TCAV Scores for target class "{target_class_name}" across concepts and layers', fontsize=16)
    plt.xlabel('Concept', fontsize=14)
    plt.ylabel('TCAV Score', fontsize=14)
    plt.ylim(0, 1)  # Assuming TCAV scores are between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Layer name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display y-values (tcav_score) above each bar
    # Not sure why the last ones need to be dropped, but it works
    for bar in bar_plot.patches[:len(bar_plot.patches) - num_layers]:
        bar_height = bar.get_height()
        if not np.isnan(bar_height):
            bar_x = bar.get_x() + bar.get_width() / 2
            plt.text(bar_x, min(0.95, bar_height + 0.02), f'{bar_height:.2f}',
                     ha='center', va='bottom', fontsize=11, color='black')

    if file_path_for_saving:
        plt.savefig(file_path_for_saving, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
