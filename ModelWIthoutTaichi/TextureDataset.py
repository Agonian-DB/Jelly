import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class SimulationDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the data folders (e.g., sample_0, sample_1, ...).
        """
        self.root_dir = root_dir
        self.samples = [os.path.join(root_dir, sample) for sample in os.listdir(root_dir) 
                        if os.path.isdir(os.path.join(root_dir, sample))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.samples[idx]

        # Load target (k_values.npy)
        target_path = os.path.join(sample_dir, 'k_values.npy')
        target = np.load(target_path)

        # Load input (simulation/positions_t0.npy to positions_t19.npy)
        input_dir = os.path.join(sample_dir, 'simulation')
        input_files = [os.path.join(input_dir, f'positions_t{i}.npy') for i in range(20)]
        input_data = np.stack([np.load(file) for file in input_files])

        # Reshape input to [time, channel=2, height=32, width=32]
        input_data = input_data.reshape(20, 1024, 2).transpose(0, 2, 1)  # [time, 2, 1024]
        input_data = input_data.reshape(20, 2, 32, 32)  # [time, 2, 32, 32]

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
