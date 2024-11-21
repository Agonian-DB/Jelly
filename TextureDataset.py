from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

class TextureDataset(Dataset):
    def __init__(self, data_dir, num_samples, image_size=(256, 256)):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5782, 0.5775, 0.5626], std=[0.2582, 0.2526, 0.2564])
            # normalize to [-1, 1]
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.data_dir, f"sample_{idx}")
        image = plt.imread(os.path.join(sample_dir, "texture.png"))
        image = self.transform(image[:, :, :3])  # [3, H, W]

        k_matrix = np.load(os.path.join(sample_dir, "k_values.npy"))
        k_matrix = (k_matrix - k_matrix.min()) / (k_matrix.max() - k_matrix.min() + 1e-8)
        k_matrix_tensor = torch.from_numpy(k_matrix).float()

        particle_positions = np.load(os.path.join(sample_dir, "initial_positions.npy"))
        particle_positions_tensor = torch.from_numpy(particle_positions).float()

        return image, k_matrix_tensor, particle_positions_tensor