import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def tcav_barplot(df, target_class_name, show=True, file_path_for_saving=None):
    """
    Creates a grouped bar plot for TCAV scores for a specific target class.
    """
    df_filtered = df[df['action'] == target_class_name]

    plt.figure(figsize=(14, 8))

    sns.barplot(
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

    if file_path_for_saving:
        plt.savefig(file_path_for_saving, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
