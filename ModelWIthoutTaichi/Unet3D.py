"""
refer: LIU et al.: VIDEO SUMMARIZATION THROUGH RL WITH 3D SPATIO-TEMPORAL U-Net
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import TextureDataset


class UNet3D(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),  # Output: [N, 64, 20, 32, 32]
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  # Output: [N, 64, 20, 32, 32]
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [N, 64, 10, 16, 16]

        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  # Output: [N, 128, 10, 16, 16]
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  # Output: [N, 128, 10, 16, 16]
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [N, 128, 5, 8, 8]

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),  # Output: [N, 256, 5, 8, 8]
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),  # Output: [N, 256, 5, 8, 8]
            nn.ReLU()
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # Output: [N, 128, 10, 16, 16]
        self.decoder2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),  # Output: [N, 128, 10, 16, 16]
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  # Output: [N, 128, 10, 16, 16]
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # Output: [N, 64, 20, 32, 32]
        self.decoder1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),  # Output: [N, 64, 20, 32, 32]
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  # Output: [N, 64, 20, 32, 32]
            nn.ReLU()
        )

        # Final layers
        self.final_conv = nn.Conv3d(64, output_dim, kernel_size=1)  # Output: [N, output_dim, 20, 32, 32]
        self.flatten = nn.Flatten()  # Flatten the output to [N, 20 * 32 * 32]
        self.fc1 = nn.Linear(20 * 32 * 32, 1024)  # First fully connected layer
        self.fc2 = nn.Linear(1024, 3906)  # Second fully connected layer

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # [N, 64, 20, 32, 32]
        enc2 = self.encoder2(self.pool1(enc1))  # [N, 128, 10, 16, 16]
        bottleneck = self.bottleneck(self.pool2(enc2))  # [N, 256, 5, 8, 8]

        # Decoder
        dec2 = self.decoder2(torch.cat((self.upconv2(bottleneck), enc2), dim=1))  # [N, 128, 10, 16, 16]
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))  # [N, 64, 20, 32, 32]

        # Final layers
        x = self.final_conv(dec1)  # [N, output_dim, 20, 32, 32]
        x = self.flatten(x)  # [N, 20 * 32 * 32]
        x = self.fc1(x)  # [N, 1024]
        return self.fc2(x)  # [N, 3906]

if __name__ == "__main__":
    dataset = TextureDataset.SimulationDataset(root_dir="/projectnb/ec523kb/projects/teams_Fall_2024/Team_4/taichi_11.15/data/generated_500_20")

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Hyperparameter tuning grid
    param_grid = {
        "epochs": [10, 20, 50, 100],
        "learning_rates": [0.001, 0.0005, 0.0001],
        "batch_sizes": [4, 8, 16, 32]
    }

    for epochs in param_grid["epochs"]:
        for lr in param_grid["learning_rates"]:
            for batch_size in param_grid["batch_sizes"]:

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Initialize model, loss function, and optimizer
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = UNet3D(input_channels=2, output_dim=1).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                train_loss_log = []
                val_loss_log = []

                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        inputs = inputs.permute(0, 2, 1, 3, 4).to(device)  # Reshape to [batch, channel, time, height, width]
                        targets = targets.to(device)

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    train_loss_log.append(epoch_loss / len(train_loader))

                    # Validation
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
                            targets = targets.to(device)

                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()

                    val_loss_log.append(val_loss / len(test_loader))
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss_log[-1]:.4f}, Val Loss: {val_loss_log[-1]:.4f}")

                # Save results
                result_dir = f"./FinalResults/Unet/epochs_{epochs}_lr_{lr}_batch_{batch_size}"
                os.makedirs(result_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{result_dir}/model.pth")
                np.save(f"{result_dir}/train_loss.npy", train_loss_log)
                np.save(f"{result_dir}/val_loss.npy", val_loss_log)
                print(f"Results saved for epochs={epochs}, lr={lr}, batch_size={batch_size}")

