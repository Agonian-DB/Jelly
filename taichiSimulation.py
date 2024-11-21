import numpy as np
import taichi as ti
import torch

@ti.data_oriented
class TaichiSimulation:
    def __init__(self, particle_positions, spring_indices, rest_length, gravity, damping, num_steps, dt, device, num_T):
        self.device = device
        self.num_particles = particle_positions.shape[0]
        self.num_springs = spring_indices.shape[0]
        self.num_steps = num_steps
        self.dt = dt
        self.gravity = gravity
        self.damping = damping
        self.num_T = num_T
        self.steps_per_frame = self.num_steps // num_T
        self.cols = int(np.sqrt(self.num_particles))

        # Define fields that require gradients
        self.k_spring = ti.field(dtype=ti.f32, shape=self.num_springs, needs_grad=True)

        # Other fields
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.f = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles, needs_grad=True)

        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=self.num_springs)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.num_springs)
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # Field to store ground truth positions for all time steps
        self.positions_gt_all = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_T, self.num_particles))

        self.initial_positions = particle_positions.copy()
        self.x.from_numpy(self.initial_positions)
        self.springs.from_numpy(spring_indices)
        self.rest_length.from_numpy(rest_length)

        initial_positions = self.initial_positions
        x_min, x_max = initial_positions[:, 0].min(), initial_positions[:, 0].max()
        y_min, y_max = initial_positions[:, 1].min(), initial_positions[:, 1].max()

        x_range_centered = (x_min + x_max) / 2
        y_range_centered = (y_min + y_max) / 2

        view_width = (x_max - x_min) * 2
        view_height = (y_max - y_min) * 2

        self.x_min_fixed = x_range_centered - view_width / 2
        self.x_max_fixed = x_range_centered + view_width / 2
        self.y_min_fixed = y_range_centered - view_height / 2
        self.y_max_fixed = y_range_centered + view_height / 2

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
            row = i // self.cols
            if row == 0:
                continue  # set the top particles fixed
            else:
                self.v[i] += self.dt * self.f[i]
                self.x[i] += self.dt * self.v[i]

    def reset(self):
        self.x.from_numpy(self.initial_positions)
        self.v.fill(0)

    @ti.kernel
    def compute_loss_per_particle(self, t: ti.i32):
        for i in range(self.num_particles):
            diff = self.x[i] - self.positions_gt_all[t, i]
            self.loss[None] += diff.norm_sqr()

    @ti.kernel
    def normalize_loss(self, num_particles: ti.i32):
        self.loss[None] /= num_particles

    def plot_positions(self, positions_pred, positions_gt, t):
        # plot the comparison of predicted particles and gt particles positions
        import matplotlib.pyplot as plt
        import os

        os.makedirs("results/plots", exist_ok=True)

        x_pred = positions_pred[:, 0]
        y_pred = positions_pred[:, 1]
        x_gt = positions_gt[:, 0]
        y_gt = positions_gt[:, 1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].scatter(x_pred, y_pred, color='red', s=10)
        axs[0].set_title(f"Predicted Positions at Time {t}")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].set_xlim(self.x_min_fixed, self.x_max_fixed)
        axs[0].set_ylim(self.y_min_fixed, self.y_max_fixed)
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].invert_yaxis()

        axs[1].scatter(x_gt, y_gt, color='blue', s=10)
        axs[1].set_title(f"Ground Truth Positions at Time {t}")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].set_xlim(self.x_min_fixed, self.x_max_fixed)
        axs[1].set_ylim(self.y_min_fixed, self.y_max_fixed)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"results/plots/positions_t{t}.png")
        plt.close()

    def step(self, t):
        # Run one simulation step and compute loss at time t
        with ti.ad.Tape(loss=self.loss):
            # Run simulation for steps_per_frame steps
            for _ in range(self.steps_per_frame):
                self.compute_forces()
                self.update_positions()

            # Compute loss
            self.compute_loss_per_particle(t)
            self.normalize_loss(self.num_particles)

        # Get gradient
        grad_k_spring = self.k_spring.grad.to_torch(device=self.device)
        loss_value = self.loss[None]

        positions_pred = self.x.to_numpy()
        positions_gt = self.positions_gt_numpy[t]

        self.plot_positions(positions_pred, positions_gt, t)

        # Clear gradients for next step
        self.loss[None] = 0.0
        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.f.grad.fill(0)
        self.k_spring.grad.fill(0)

        return grad_k_spring, loss_value

    @ti.kernel
    def save_positions(self, t: ti.i32):
        for i in range(self.num_particles):
            self.x_history[t, i] = self.x[i]

    def forward(self, k_spring_torch, positions_gt_torch_list):
        self.k_spring.from_torch(k_spring_torch)
        self.reset()
        positions_gt_numpy = np.stack([pos.cpu().numpy() for pos in positions_gt_torch_list], axis=0)
        num_T = positions_gt_numpy.shape[0]

        # load gt dataset
        self.positions_gt_all.from_numpy(positions_gt_numpy)
        self.positions_gt_numpy = positions_gt_numpy

        # save posisions for plotting
        self.x_history = ti.Vector.field(2, dtype=ti.f32, shape=(num_T, self.num_particles))

        self.loss[None] = 0.0

        # tape is used to track the gradient
        with ti.ad.Tape(loss=self.loss):
            for t in range(num_T):
                for _ in range(self.steps_per_frame):
                    self.compute_forces()
                    self.update_positions()
                self.save_positions(t)
                self.compute_loss_per_particle(t)

        total_loss = self.loss[None]

        positions_pred_all = self.x_history.to_numpy()
        for t in range(num_T):
            positions_pred = positions_pred_all[t]
            positions_gt = self.positions_gt_numpy[t]
            self.plot_positions(positions_pred, positions_gt, t)

        return torch.tensor(total_loss / num_T, device=self.device, requires_grad=True)

    def backward(self):
        grad_k_spring = self.k_spring.grad.to_torch(device=self.device)
        return grad_k_spring

class TaichiSimFunction(torch.autograd.Function):
    #pass gradient from taichi to pytorch
    @staticmethod
    def forward(ctx, k_spring_torch, sim, positions_gt_torch_list):
        # run taichi simulation and get loss
        loss = sim.forward(k_spring_torch, positions_gt_torch_list)
        ctx.sim = sim
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        sim = ctx.sim
        grad_k_spring = sim.k_spring.grad.to_torch(device=sim.device)
        return grad_output * grad_k_spring, None, None