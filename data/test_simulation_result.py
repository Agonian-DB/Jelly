import numpy as np
import matplotlib.pyplot as plt
import os


def plot_simulation_positions_fixed_view(data_dir, num_time_steps=10, view_width=256, view_height=256,
                                         zoom_out_factor=2):

    # Ensure the results/plots directory exists
    os.makedirs("results/test_locations", exist_ok=True)

    # Load the initial position to determine the fixed view
    initial_positions_path = os.path.join(data_dir, "positions_t0.npy")
    initial_positions = np.load(initial_positions_path)

    # Determine the bounding box for the fixed view based on initial positions
    x_min, x_max = initial_positions[:, 0].min(), initial_positions[:, 0].max()
    y_min, y_max = initial_positions[:, 1].min(), initial_positions[:, 1].max()

    # Center the bounding box within the view dimensions and apply zoom out
    x_range_centered = (x_min + x_max) / 2
    y_range_centered = (y_min + y_max) / 2

    x_min_fixed = x_range_centered - (view_width / 2) * zoom_out_factor
    x_max_fixed = x_range_centered + (view_width / 2) * zoom_out_factor
    y_min_fixed = y_range_centered - (view_height / 2) * zoom_out_factor
    y_max_fixed = y_range_centered + (view_height / 2) * zoom_out_factor

    for t in range(num_time_steps):
        # Load positions for each time step
        positions_path = os.path.join(data_dir, f"positions_t{t}.npy")
        positions = np.load(positions_path)

        # Extract x and y coordinates
        x = positions[:, 0]
        y = positions[:, 1]

        # Plot positions with the fixed view and zoom out
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, color='blue', s=10)
        plt.title(f"Particle Positions at Time {t}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(x_min_fixed, x_max_fixed)
        plt.ylim(y_min_fixed, y_max_fixed)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()  # Flip the y-axis to simulate hanging from the ceiling

        # Save plot for each time step
        plt.tight_layout()
        plt.savefig(f"results/test_locations/positions_t{t}.png")
        plt.close()
        print(f"Saved positions plot for time step {t}")


# test the npy file generation
plot_simulation_positions_fixed_view(data_dir="data/generated/sample_0/simulation", num_time_steps=20, view_width=256,
                                     view_height=256, zoom_out_factor=3)
