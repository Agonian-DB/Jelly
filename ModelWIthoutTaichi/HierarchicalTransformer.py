"""refer:
1. https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html?ref=https://githubhelp.com
2. https://arxiv.org/pdf/1802.03168
3. https://arxiv.org/pdf/2403.12820v1
Split time=20 into four groups of five frames each.
Use spatial transformers on each group and combine the outputs with a temporal transformer.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import TextureDataset
class SpatialTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth):
        super(SpatialTransformerEncoder, self).__init__()
        self.patch_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        x = x.flatten(3).permute(0, 1, 3, 2).reshape(batch * time, height * width, channels)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return x.mean(dim=1).reshape(batch, time, -1)

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth):
        super(TemporalTransformerEncoder, self).__init__()
        self.temporal_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.temporal_embedding(x)
        x = self.encoder(x)
        return x.mean(dim=1)

class FactorizedTransformer(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, embed_dim, spatial_heads, temporal_heads, spatial_depth, temporal_depth, num_classes):
        super(FactorizedTransformer, self).__init__()
        self.spatial_transformer = SpatialTransformerEncoder(spatial_dim, embed_dim, spatial_heads, spatial_depth)
        self.temporal_transformer = TemporalTransformerEncoder(embed_dim, embed_dim, temporal_heads, temporal_depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        spatial_features = self.spatial_transformer(x)  # Shape: [batch, time, embed_dim]
        temporal_features = self.temporal_transformer(spatial_features)  # Shape: [batch, embed_dim]
        return self.classifier(temporal_features)

if __name__ == "__main__":
    dataset = TextureDataset.SimulationDataset(root_dir="/projectnb/ec523kb/projects/teams_Fall_2024/Team_4/taichi_11.15/data/generated_500_20")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    param_grid = {
        "epochs": [10, 20, 50, 100],
        "learning_rates": [0.001, 0.0005, 0.0001],
        "batch_sizes": [4, 8, 16, 32],
        "spatial_heads": [2, 4],
        "temporal_heads": [2, 4],
        "spatial_depth": [1, 2],
        "temporal_depth": [1, 2],
    }
    for epochs in param_grid["epochs"]:
        for lr in param_grid["learning_rates"]:
            for batch_size in param_grid["batch_sizes"]:
                for spatial_heads in param_grid["spatial_heads"]:
                    for temporal_heads in param_grid["temporal_heads"]:
                        for spatial_depth in param_grid["spatial_depth"]:
                            for temporal_depth in param_grid["temporal_depth"]:

                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                                # Initialize model, loss function, and optimizer
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                model = FactorizedTransformer(
                                    spatial_dim=2, temporal_dim=20, embed_dim=128,
                                    spatial_heads=spatial_heads, temporal_heads=temporal_heads,
                                    spatial_depth=spatial_depth, temporal_depth=temporal_depth,
                                    num_classes=3906
                                ).to(device)
                                criterion = nn.MSELoss()
                                optimizer = optim.Adam(model.parameters(), lr=lr)

                                train_loss_log = []
                                val_loss_log = []

                                for epoch in range(epochs):
                                    model.train()
                                    epoch_loss = 0
                                    for inputs, targets in train_loader:
                                        inputs = inputs.to(device)
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
                                            inputs = inputs.to(device)
                                            targets = targets.to(device)

                                            outputs = model(inputs)
                                            loss = criterion(outputs, targets)
                                            val_loss += loss.item()

                                    val_loss_log.append(val_loss / len(test_loader))

                                # Save results
                                result_dir = f"./FinalResults/Transformer/epochs_{epochs}_lr_{lr}_batch_{batch_size}_sheads_{spatial_heads}_theads_{temporal_heads}_sdepth_{spatial_depth}_tdepth_{temporal_depth}"
                                os.makedirs(result_dir, exist_ok=True)
                                torch.save(model.state_dict(), f"{result_dir}/model.pth")
                                np.save(f"{result_dir}/train_loss.npy", train_loss_log)
                                np.save(f"{result_dir}/val_loss.npy", val_loss_log)
                                print(f"Results saved for epochs={epochs}, lr={lr}, batch_size={batch_size}, sheads={spatial_heads}, theads={temporal_heads}, sdepth={spatial_depth}, tdepth={temporal_depth}")
