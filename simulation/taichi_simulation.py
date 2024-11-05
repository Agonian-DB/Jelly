# simulation/taichi_simulation.py

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)


@ti.data_oriented
class TaichiSimulation:
    def __init__(self, particle_positions, spring_indices, spring_k_values,
                 positions_gt, gravity=[0.0, -9.8], damping=0.99, num_steps=100, dt=1e-3):
        self.dim = 2
        self.num_particles = particle_positions.shape[0]
        self.num_springs = spring_indices.shape[0]
        self.dt = dt
        self.num_steps = num_steps

        # 将粒子位置和弹簧数据转换为 Taichi 字段
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.num_steps + 1, self.num_particles))
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_particles)
        self.f = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_particles)

        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=self.num_springs)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.num_springs)
        self.k_spring = ti.field(dtype=ti.f32, shape=self.num_springs, needs_grad=True)

        # 声明 positions_gt 作为 Taichi 字段
        self.positions_gt = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.num_steps + 1, self.num_particles))
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # 初始化
        self.initialize(particle_positions, spring_indices, spring_k_values, positions_gt)

    def initialize(self, particle_positions, spring_indices, spring_k_values, positions_gt):
        # 初始化粒子位置
        self.x[0].from_numpy(particle_positions)
        self.v.fill(0.0)

        # 初始化弹簧
        self.springs.from_numpy(spring_indices)
        for i in range(self.num_springs):
            a, b = self.springs[i]
            self.rest_length[i] = (self.x[0, a] - self.x[0, b]).norm()
            self.k_spring[i] = spring_k_values[i]

        # 将 positions_gt 从 NumPy 数组加载到 Taichi 字段
        self.positions_gt.from_numpy(positions_gt)

    @ti.kernel
    def substep(self, t: ti.i32):
        # 计算粒子受力
        for i in range(self.num_particles):
            self.f[i] = self.mass * self.gravity

        # 计算弹簧力
        for i in range(self.num_springs):
            a, b = self.springs[i]
            x_a = self.x[t, a]
            x_b = self.x[t, b]
            k = self.k_spring[i]
            rest_length = self.rest_length[i]
            dir = x_b - x_a
            length = dir.norm()
            if length != 0:
                force = k * (length - rest_length) * dir.normalized()
                self.f[a] += force
                self.f[b] -= force

        # 更新粒子位置和速度
        for i in range(self.num_particles):
            self.v[i] += self.dt * self.f[i] / self.mass
            self.v[i] *= ti.exp(-self.dt * self.damping)
            self.x[t + 1, i] = self.x[t, i] + self.dt * self.v[i]

        # 固定顶部粒子
        for i in range(self.num_particles):
            if self.x[t + 1, i].y > 0.95:
                self.v[i] = ti.Vector([0.0, 0.0])
                self.x[t + 1, i] = ti.Vector([self.x[t + 1, i].x, 1.0])

    def run(self):
        with ti.ad.Tape(loss=self.loss):
            for t in range(self.num_steps):
                self.substep(t)
            self.compute_loss()

        # 提取粒子位置序列
        positions_over_time = [self.x[t].to_numpy() for t in range(self.num_steps + 1)]
        return positions_over_time

    @ti.kernel
    def compute_loss(self):
        self.loss[None] = 0.0
        for t in range(self.num_steps + 1):
            for i in range(self.num_particles):
                diff = self.x[t, i] - self.positions_gt[t, i]
                self.loss[None] += diff.norm_sqr()
        self.loss[None] /= (self.num_steps + 1) * self.num_particles
