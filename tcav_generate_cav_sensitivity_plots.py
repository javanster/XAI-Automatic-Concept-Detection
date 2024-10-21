from tcav_cav_sensitivity_plot import cav_sensitivity_plot
from ConceptDetector import ConceptDetector
import matplotlib.pyplot as plt
import os

TRAIN_RUN_NAME = "mild_cosmos_59"
MODEL_NAME = "best_model"

SHOW = False
SAVE = True


for concept_index in ConceptDetector.concept_name_dict.keys():

    print(f"Generating cav sensitivity plot for concept {concept_index}")

    cav_sensitivity_plot(
        data_file_path=f"tcav_data/cav_sensitivities_during_training/{TRAIN_RUN_NAME}/{MODEL_NAME}/model_sensitivities.csv",
        concept_index=concept_index,
        show=SHOW,
        num_yticks=6,
        training_steps_to_show=500_000
    )

    if SAVE:
        filename = f"cav_sensitivities_concept_{concept_index}.png"
        filepath = os.path.join(
            f"tcav_data/cav_sensitivities_during_training/{TRAIN_RUN_NAME}/{MODEL_NAME}/", filename)
        plt.savefig(filepath, dpi=40, bbox_inches='tight')

    plt.close()
