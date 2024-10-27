from tcav import cassat_plot
from tcav import ConceptDetector
import matplotlib.pyplot as plt
import os


def generate_cassat_plots(train_run_name, concept_indexes, model_name, show, save):

    for ci in concept_indexes:
        if ci not in ConceptDetector.concept_name_dict.keys():
            raise ValueError(
                "All given concept indexes must be defined in ConceptDetector")

    for concept_index in concept_indexes:

        print(f"Generating CASSAT plot for concept {concept_index}")

        cassat_plot(
            data_frame_path=f"tcav_data/cassat/{train_run_name}/{model_name}/classifier_scores.csv",
            concept_index=concept_index,
            show=show,
            num_yticks=4,
            training_steps_to_show=300_000
        )

        if save:
            filename = f"cassat_plot_concept_{concept_index}.png"
            filepath = os.path.join(
                f"tcav_explanations/cassat/{train_run_name}/{model_name}/", filename)
            plt.savefig(filepath, dpi=40, bbox_inches='tight')

        plt.close()


if __name__ == "__main__":
    train_run_name = "dutiful_frog_68"
    model_name = "best_model"

    generate_cassat_plots(
        train_run_name=train_run_name,
        concept_indexes=[0, 1, 2, 3, 4, 9, 10, 11, 12],
        model_name=model_name,
        show=False,
        save=True,
    )
