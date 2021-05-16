"""
Author: Christopher Schicho
Project: Image Extrapolation
Version: 0.0
"""

import os
import torch
import matplotlib.pyplot as plt


def plot(inputs: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor,
         log_path: str, update: int) -> None:
    """
    Save a image of the plot of the inputs, targets and predictions to path.

    :param inputs: inputs of the neural network
    :param targets: targets corresponding to the inputs
    :param predictions: output of the neural network
    :param log_path: folder where the images should be stored
    :param update: current update number
    """
    inputs = inputs.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    plot_path = os.path.join(log_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    for i in range(len(inputs)):
        ax1.clear()
        ax1.set_axis_off()
        ax1.set_title("input")
        ax1.imshow(inputs[i, 0], cmap="gray")

        ax2.clear()
        ax2.set_axis_off()
        ax2.set_title("target")
        mask_arr = inputs[i, 1].copy()
        target = targets[i, 0][mask_arr == 0]
        mask_arr[mask_arr == 0] = target
        ax2.imshow(mask_arr, cmap="gray")

        ax3.clear()
        ax3.set_axis_off()
        ax3.set_title("prediction")
        mask_arr = inputs[i, 1].copy()
        prediction = predictions[i, 0][mask_arr == 0]
        mask_arr[mask_arr == 0] = prediction
        ax3.imshow(mask_arr, cmap="gray")

        ax4.clear()
        ax4.set_axis_off()
        ax4.set_title("prediction + input")
        input_arr = inputs[i, 0].copy()
        input_arr[input_arr == 0] = prediction
        ax4.imshow(input_arr, cmap="gray")

        fig.suptitle(f"{update:07d}_{i:03d}")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_path, f"{update:07d}_{i:03d}.png"), dpi=1000)

    del fig
