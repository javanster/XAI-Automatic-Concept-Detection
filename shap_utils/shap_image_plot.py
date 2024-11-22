import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255, j))
cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def shap_image_plot(
    shap_values_list,
    pixel_values,
    q_values=None,
    labels=None,
    width=20,
    aspect=0.2,
    hspace=0.2,
    labelpad=None,
    show=True
):
    """Plots SHAP values for image inputs. Based on image_plot from the SHAP package:
    https://github.com/shap/shap/blob/master/shap/plots/_image.py

    Parameters
    ----------
    shap_values_list : list of numpy.array
        List of arrays of SHAP values, one for each action/output. Each array has the shape
        (num_samples x height x width x channels).

    pixel_values : numpy.array
        Array of pixel values (num_samples x height x width x 3) for each image. Supports three color channels!

    q_values : numpy.ndarray, optional
        Array of Q-values with shape (num_samples, num_actions). If provided, the function will display
        the Q-value for each action and observation under the corresponding plot. The Q-value with the
        highest value for each observation will have a yellow background to highlight it.

    labels : list
        List of names for each of the actions.

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    show : bool
        Whether matplotlib.pyplot.show() is called before returning.

    """
    num_obs = pixel_values.shape[0]
    num_actions = len(shap_values_list)

    if labels is None:
        labels = [f"Action {i}" for i in range(num_actions)]

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # Determine figure size
    fig_size = np.array([3 * (num_actions + 1), 2.5 * (num_obs + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]

    # Create subplots
    fig, axes = plt.subplots(
        nrows=num_obs, ncols=num_actions + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)

    for row in range(num_obs):
        x_curr = pixel_values[row]

        # Convert RGB to grayscale
        x_curr_gray = (
            0.2989 * x_curr[:, :, 0] +
            0.5870 * x_curr[:, :, 1] +
            0.1140 * x_curr[:, :, 2]
        )
        x_curr_disp = x_curr

        # Display original image
        axes[row, 0].imshow(x_curr_disp)
        if row == 0:
            axes[row, 0].set_title("Observation", **label_kwargs)
        axes[row, 0].axis('off')

        # Compute max_val for scaling SHAP values (over all actions for this observation)
        abs_vals = np.stack([np.abs(shap_values_list[i][row].sum(-1))
                             for i in range(num_actions)], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        for i in range(num_actions):
            # Sums shap values in all three color channels of each pixel
            sv = shap_values_list[i][row].sum(-1)

            axes[row, i + 1].imshow(x_curr_gray, cmap='gray', alpha=0.15)
            im = axes[row, i + 1].imshow(sv, cmap=cmap,
                                         vmin=-max_val, vmax=max_val)
            if row == 0:
                axes[row, i + 1].set_title(labels[i], **label_kwargs)

            # Display the Q-value under each plot
            if q_values is not None:
                q_values_for_observation = q_values[row]
                q_value = q_values[row, i]
                background_color = "yellow" if q_value == np.max(
                    q_values_for_observation) else "white"

                axes[row, i +
                     1].set_xlabel(
                    f"Q-value: {q_value:.2f}",
                    backgroundcolor=background_color,
                    **label_kwargs
                )

            # Hide ticks and spines but keep labels
            axes[row, i + 1].set_xticks([])
            axes[row, i + 1].set_yticks([])
            for spine in axes[row, i + 1].spines.values():
                spine.set_visible(False)

    # Adjust layout and add colorbar
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)

    if show:
        plt.show()
