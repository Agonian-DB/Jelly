import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import taichi as ti
import logging

from TextureDataset import TextureDataset
from models import UNet
from taichiSimulation import TaichiSimulation, TaichiSimFunction
from utils import setup_trial, setup_logging, visualize_k_values

ti.init(arch=ti.gpu)


import time
def train_model(data_dir, grid_spacing=32, trial_name="trial_1", num_samples=100, num_epochs=10, learning_rate=1e-4,
                num_T=10,
                num_springs=480, num_steps=5000, base_channels=64, depth=5):
    """
    Train the model and save logs, models, and results in the trial folder.
    """
    # Set up trial folder
    base_dir = setup_trial(trial_name)
    log_file = os.path.join(base_dir, "training.log")
    setup_logging(log_file)  # Set up logging in trial folder

    logging.info(f"Starting training for trial: {trial_name}")
    logging.info(f"Parameters: num_samples={num_samples}, num_epochs={num_epochs}, "
                 f"learning_rate={learning_rate}, num_T={num_T}, num_springs={num_springs}, num_steps={num_steps}")

    dataset = TextureDataset(data_dir, num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = UNet(num_springs=num_springs).cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for param_group in optimizer.param_groups:
            logging.info(f"Epoch {epoch + 1}, Learning rate: {param_group['lr']:.6e}")

        for batch_idx, (images, k_matrices_gt, particle_positions) in enumerate(dataloader):
            images = images.cuda()
            particle_positions = particle_positions.squeeze(0).numpy()
            k_matrices_pred = model(images)

            spring_indices = np.load(os.path.join(data_dir, f"sample_{batch_idx}", "spring_indices.npy"))
            rest_length = np.load(os.path.join(data_dir, f"sample_{batch_idx}", "rest_lengths.npy"))

            # Load ground truth positions
            positions_gt = [
                torch.from_numpy(np.load(
                    os.path.join(data_dir, f"sample_{batch_idx}", "simulation", f"positions_t{t}.npy"))).float().to(
                    k_matrices_pred.device)
                for t in range(num_T)
            ]

            sim = TaichiSimulation(
                particle_positions=particle_positions,
                spring_indices=spring_indices,
                rest_length=rest_length,
                gravity=[0.0, 20.0],
                damping=0.99,
                num_steps=num_steps,
                num_T=num_T,
                dt=1e-3,
                device=k_matrices_pred.device
            )

            optimizer.zero_grad()
            loss = TaichiSimFunction.apply(k_matrices_pred.squeeze(0), sim, positions_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # print gradient
            #             for name, param in model.named_parameters():
            #                 if param.grad is not None:
            #                     logging.info(
            #                         f"Gradient - {name}: mean={param.grad.mean().item():.4e}, "
            #                         f"std={param.grad.std().item():.4e}, "
            #                         f"max={param.grad.max().item():.4e}, "
            #                         f"min={param.grad.min().item():.4e}"
            #                     )

            optimizer.step()

            epoch_loss += loss.item()
            logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

            predicted_k = k_matrices_pred.detach().cpu().numpy().flatten()
            ground_truth_k = k_matrices_gt.numpy().flatten()
            visualize_k_values(
                predicted_k,
                ground_truth_k,
                spring_indices,
                epoch + 1,
                batch_idx,
                base_dir,
                grid_spacing=grid_spacing  # 根据输入图片尺寸设置
            )

        logging.info(f"Epoch {epoch + 1} completed with average loss {epoch_loss / len(dataloader):.4f}")
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")
        # Save the model every 10 epochs
        if (epoch + 1) % 50 == 0:
            model_path = os.path.join(base_dir, f"models/model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved at {model_path}")

    # Save the final model
    final_model_path = os.path.join(base_dir, "models/model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    train_model("data/generated/",
                trial_name="results/trial_1", num_samples=1, num_epochs=1000, num_T=10,
                num_springs=3906, grid_spacing=8, num_steps=250000, learning_rate=1e-4)