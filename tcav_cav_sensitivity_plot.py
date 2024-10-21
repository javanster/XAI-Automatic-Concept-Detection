import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import matplotlib as mpl


def cav_sensitivity_plot(
        data_file_path,
        concept_index,
        training_steps_to_show,
        show=True,
        num_yticks=5
):
    """
    Plots a 3D surface plot of CAV (Concept Activation Vector) sensitivities across training steps and model layers.
    Based on an implementation by Erik Storås Sommer:
    https://github.com/erikssommer/xai-concept-methods-drl/blob/main/notebooks/static_concept_presence.ipynb

    This function visualizes the CAV sensitivity values across different layers of a neural network and across specified 
    training steps for a given concept. The resulting plot displays how the sensitivity varies over time (training steps) 
    and across model layers.

    Parameters
    ----------
    data_file_path : str
        Path to the CSV file containing CAV sensitivity data. The CSV file must have columns including 
        'concept_index', 'training_step', 'layer_index', 'layer_name' and 'cav_sensitivity'.
    concept_index : int
        The index representing the concept to be visualized. This index should correspond to the 'concept_index' column 
        in the CSV file.
    training_steps_to_show : int
        Maximum number of training steps to include in the plot. The function will filter data to only include rows where 
        the 'training_step' value is less than or equal to this parameter.
    show : bool, optional, default=True
        Whether to display the plot immediately. Set this to False if you want to save the plot or perform further 
        modifications before showing it.
    num_yticks : int, optional, default=5
        The number of evenly spaced y-ticks to display along the y-axis (training steps). These ticks will be distributed 
        proportionally based on the filtered training steps.
    """

    data = pd.read_csv(data_file_path)

    concept_name = data.loc[data['concept_index'] ==
                            concept_index, 'concept_name'].unique()[0]

    filtered_data = data[(data['concept_index']
                          == concept_index) & (data['training_step'] <= training_steps_to_show)]

    training_steps = sorted(filtered_data['training_step'].unique())
    layers = sorted(filtered_data['layer_index'].unique())

    # Create a 2D grid (z) with training_steps as rows and layers as columns
    z = np.zeros((len(training_steps), len(layers)))

    # Populate the z matrix with cav_sensitivity values from the filtered data
    for i, step in enumerate(training_steps):
        for j, layer in enumerate(layers):
            value = filtered_data[(filtered_data['training_step'] == step) & (
                filtered_data['layer_index'] == layer)]['cav_sensitivity'].values
            if len(value) > 0:
                z[i, j] = value[0]

    # Plotting parameters similar to the notebook example
    mpl.style.use("seaborn-v0_8-muted")
    mpl.rcParams['figure.figsize'] = (40, 40)
    mpl.rcParams['lines.linewidth'] = 10.0
    mpl.rcParams['font.family'] = "serif"
    mpl.rcParams["axes.axisbelow"] = True
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Scale down the figure size with setting the resolution of the figure in dots-per-inch
    fig.set_dpi(40)

    # Create meshgrid for X (layer indices) and Y (training steps)
    X, Y = np.meshgrid(np.arange(len(layers)),
                       np.arange(len(training_steps)))

    # Plot the surface
    surf = ax.plot_surface(X, Y, z, cmap=cm.plasma, edgecolor="white",
                           linewidth=0.25, vmin=0.1, vmax=0.9, alpha=1)

    # Customize the plot appearance
    ax.set_zlim(0, 1.0)
    ax.set_axisbelow(False)
    ax.set_title(
        f"Concept {concept_index}: {concept_name}", fontsize=80, y=0.95)
    ax.set_xlabel("Layer", fontsize=30, labelpad=200, zorder=10)
    ax.set_ylabel("Training steps", fontsize=30, labelpad=75, zorder=10)
    ax.set_zlabel("CAV Sensitivity", fontsize=30, labelpad=75, zorder=10)
    ax.set_box_aspect(aspect=None, zoom=0.8)

    plt.xticks(fontsize=30, rotation=0)
    # Set x-tick labels to layer indices
    plt.xticks(np.arange(len(layers)), layers)
    plt.yticks(fontsize=30)
    ax.tick_params('z', labelsize=25, pad=20, reset=True)
    fig.patch.set_facecolor("white")

    ax.invert_xaxis()

    unique_layers = filtered_data[[
        'layer_index', 'layer_name']].drop_duplicates()
    unique_layers = unique_layers.sort_values(by='layer_index')

    # Create x-axis labels combining index and layer name
    x_labels = [f"{int(row['layer_index'])}: {row['layer_name']}" for _,
                row in unique_layers.iterrows()]

    # Set custom x-axis labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Move the z-axis to the other side
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)

    ax.zaxis.set_major_locator(LinearLocator(5))

    # Set y-ticks based on the range of training steps
    yticks = np.linspace(0, len(training_steps) - 1, num_yticks, dtype=int)
    ytick_labels = [training_steps[i] for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    # Format z-axis ticks
    ax.zaxis.set_major_formatter('{x:.01f}')
    ax.view_init(30, -30)

    if show:
        plt.show()
