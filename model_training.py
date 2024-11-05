import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)


class TextureDataset(Dataset):
    def __init__(self, data_dir, num_samples, image_size=(256, 256)):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.data_dir, f"sample_{idx}")
        image = plt.imread(os.path.join(sample_dir, "texture.png"))
        image = self.transform(image[:, :, :3])  # [3, H, W]

        k_matrix = np.load(os.path.join(sample_dir, "k_values.npy"))
        k_matrix_tensor = torch.from_numpy(k_matrix)  # Flattened 1D tensor

        particle_positions = np.load(os.path.join(sample_dir, "initial_positions.npy"))
        particle_positions_tensor = torch.from_numpy(particle_positions)

        return image, k_matrix_tensor, particle_positions_tensor


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, num_springs=480):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.fc = nn.Linear(256 * 256, num_springs)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        center = self.center(self.pool4(enc4))
        dec4 = self.dec4(torch.cat([self.up4(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        out = self.out_conv(dec1)
        out = out.view(out.size(0), -1)  # Flatten to [batch_size, 256*256]
        out = self.fc(out)  # Output to [batch_size, num_springs]
        return out


@ti.data_oriented
class TaichiSimulation:
    def __init__(self, particle_positions, spring_indices, rest_length, gravity, damping, num_steps, dt, device):
        self.device = device
        self.num_particles = particle_positions.shape[0]
        self.num_springs = spring_indices.shape[0]
        self.num_steps = num_steps
        self.dt = dt
        self.gravity = gravity
        self.damping = damping

        # 定义需要梯度的场
        self.k_spring = ti.field(dtype=ti.f32, shape=self.num_springs, needs_grad=True)

        # 其他场
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.f = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=self.num_springs)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.num_springs)
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # 用于存储当前时间步的 GT 位置
        self.positions_gt_field = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)

        self.initial_positions = particle_positions.copy()
        self.x.from_numpy(self.initial_positions)
        self.springs.from_numpy(spring_indices)
        self.rest_length.from_numpy(rest_length)

    @ti.kernel
    def compute_forces(self):
        for i in range(self.num_particles):
            self.f[i] = ti.Vector(self.gravity)
            self.v[i] *= self.damping

        for i in range(self.num_springs):
            a, b = self.springs[i]
            dir = self.x[a] - self.x[b]
            length = dir.norm() + 1e-4
            force = -self.k_spring[i] * (length - self.rest_length[i]) * dir.normalized()
            self.f[a] += force
            self.f[b] -= force

    @ti.kernel
    def update_positions(self):
        for i in range(self.num_particles):
            self.v[i] += self.dt * self.f[i]
            self.x[i] += self.dt * self.v[i]

    def reset(self):
        self.x.from_numpy(self.initial_positions)
        self.v.fill(0)

    def run(self, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        for step in range(num_steps):
            self.compute_forces()
            self.update_positions()

    @ti.kernel
    def compute_loss_per_particle(self):
        for i in range(self.num_particles):
            diff = self.x[i] - self.positions_gt_field[i]
            self.loss[None] += diff.norm_sqr()

    @ti.kernel
    def reset_loss(self):
        self.loss[None] = 0.0

    def forward(self, k_spring_torch, positions_gt_torch_list):
        self.k_spring.from_torch(k_spring_torch)
        total_loss = 0.0

        for t, positions_gt_torch in enumerate(positions_gt_torch_list):
            # 每次设置当前的 GT 位置，并重置粒子位置
            self.positions_gt_field.from_numpy(positions_gt_torch.cpu().numpy())
            self.reset()

            with ti.ad.Tape(loss=self.loss):
                self.run(num_steps=t + 1)
                self.reset_loss()
                self.compute_loss_per_particle()

            grad_k_spring = self.k_spring.grad.to_torch(device=self.device)
            print(f"Timestep {t + 1}, dloss/dk:", grad_k_spring)

            grad_x = self.x.grad.to_torch(device=self.device)
            print(f"Timestep {t + 1}, dloss/dx:", grad_x)

            total_loss += self.loss[None]  # 累加损失
            self.loss[None] = 0.0  # 重置损失

        return torch.tensor(total_loss, device=self.device, requires_grad=True)

    def backward(self):
        grad_k_spring = self.k_spring.grad.to_torch(device=self.device)
        # print("d位置/dk (最后计算的梯度):", grad_k_spring)
        return grad_k_spring



class TaichiSimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k_spring_torch, sim: TaichiSimulation, positions_gt_torch_list):
        ctx.sim = sim
        ctx.device = k_spring_torch.device  # 保存设备信息
        loss = sim.forward(k_spring_torch, positions_gt_torch_list)
        ctx.save_for_backward(k_spring_torch)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_k_spring = ctx.sim.backward()
        # 将梯度移动到正确的设备上
        grad_k_spring = grad_k_spring.to(ctx.device)

        return grad_k_spring * grad_output.item(), None, None


def train_model(data_dir, num_samples=100, num_epochs=10, learning_rate=1e-4, num_T=10, num_springs=480):
    dataset = TextureDataset(data_dir, num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = UNet(in_channels=3, out_channels=4, num_springs=num_springs).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, k_matrices_gt, particle_positions) in enumerate(dataloader):
            images = images.cuda()
            particle_positions = particle_positions.squeeze(0).numpy()
            k_matrices_pred = model(images)

            spring_indices = np.load(os.path.join(data_dir, f"sample_{batch_idx}", "spring_indices.npy"))
            rest_length = np.load(os.path.join(data_dir, f"sample_{batch_idx}", "rest_lengths.npy"))

            # 加载每个时间步的真实位置
            positions_gt = [
                torch.from_numpy(np.load(os.path.join(data_dir, f"sample_{batch_idx}", "simulation", f"positions_t{t}.npy"))).float().to(k_matrices_pred.device)
                for t in range(num_T)
            ]

            sim = TaichiSimulation(
                particle_positions=particle_positions,
                spring_indices=spring_indices,
                rest_length=rest_length,
                gravity=[0.0, -9.8],
                damping=0.99,
                num_steps=num_T,  # 最大模拟步数
                dt=1e-3,
                device=k_matrices_pred.device  # 传递设备信息
            )

            optimizer.zero_grad()
            loss = TaichiSimFunction.apply(k_matrices_pred.squeeze(0), sim, positions_gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

            # 绘制预测的 k 值和真实的 k 值
            k_matrices_pred_np = k_matrices_pred.detach().cpu().numpy().flatten()
            k_matrices_gt_np = k_matrices_gt.numpy().flatten()
            plt.figure(figsize=(10, 6))
            plt.plot(k_matrices_gt_np, label='Ground Truth k', color='blue')
            plt.plot(k_matrices_pred_np, label='Predicted k', color='red')
            plt.xlabel('Spring Index')
            plt.ylabel('Spring Coefficient k')
            plt.title(f'Predicted vs Ground Truth k (Epoch {epoch + 1}, Batch {batch_idx + 1})')
            plt.legend()
            plt.show()

            # 打印 d位置/dk
            # grad_k_spring = sim.k_spring.grad.to_torch(device=sim.device)
            # print("d位置/dk (最后计算的梯度):", grad_k_spring)

        print(f"Epoch {epoch + 1} completed with average loss {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    train_model("data/generated", num_samples=10, num_epochs=10, num_T=10, num_springs=112)