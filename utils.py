import logging
import os

import numpy as np
from matplotlib import pyplot as plt


def setup_trial(trial_name="trial_1"):
    """Set up the folder structure for a trial."""
    base_dir = f"{trial_name}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/results", exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"{base_dir}/logs/train.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Trial directory created at: {base_dir}")
    return base_dir


def visualize_k_values(predicted_k, ground_truth_k, spring_indices, epoch, sample_idx, base_dir, grid_spacing=8,
                       image_size=(256, 256)):
    """
    visualize the distribution of k values。
    """
    grid_rows = image_size[0] // grid_spacing
    grid_cols = image_size[1] // grid_spacing

    particle_positions = np.array(
        [[j * grid_spacing, i * grid_spacing] for i in range(grid_rows) for j in range(grid_cols)]
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    def plot_springs(ax, k_values, title):
        ax.set_title(title)
        for i, (start, end) in enumerate(spring_indices):
            # Get start and end positions of the spring
            x = [particle_positions[start][0], particle_positions[end][0]]
            y = [particle_positions[start][1], particle_positions[end][1]]
            color = plt.cm.coolwarm((k_values[i] - k_values.min()) / (k_values.max() - k_values.min()))
            ax.plot(x, y, color=color, linewidth=2)

        ax.scatter(particle_positions[:, 0], particle_positions[:, 1], color="blue", s=5)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plot_springs(axs[0], predicted_k, "Predicted k-values")
    plot_springs(axs[1], ground_truth_k, "Ground Truth k-values")

    plt.tight_layout()

    plot_path = os.path.join(base_dir, f"results/k_comparison_epoch_{epoch}_sample_{sample_idx}.png")
    plt.savefig(plot_path)
    plt.close()






def setup_logging(log_file_path):
    """
    Set up logging to write logs to both console and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
