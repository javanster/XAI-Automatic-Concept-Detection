from tcav_cassat_plot import tcav_cassat_plot
from ConceptDetector import ConceptDetector
import matplotlib.pyplot as plt
import os

TRAIN_RUN_NAME = "dutiful_frog_68"
MODEL_NAME = "best_model"

SHOW = False
SAVE = True


for concept_index in ConceptDetector.concept_name_dict.keys():

    print(f"Generating CASSAT plot for concept {concept_index}")

    tcav_cassat_plot(
        data_frame_path=f"tcav_data/cassat/{TRAIN_RUN_NAME}/{MODEL_NAME}/classifier_scores.csv",
        concept_index=concept_index,
        show=SHOW,
        num_yticks=4,
        training_steps_to_show=300_000
    )

    if SAVE:
        filename = f"cassat_plot_concept_{concept_index}.png"
        filepath = os.path.join(
            f"tcav_explanations/cassat/{TRAIN_RUN_NAME}/{MODEL_NAME}/", filename)
        plt.savefig(filepath, dpi=40, bbox_inches='tight')

    plt.close()
