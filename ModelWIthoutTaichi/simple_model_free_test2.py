"""
cwy simulation method test
"""

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

        self.x_history = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_T, self.num_particles))
        self.positions_gt_all = ti.Vector.field(2, dtype=ti.f32, shape=(self.num_T, self.num_particles))

        self.initial_positions = particle_positions.copy()
        self.x.from_numpy(self.initial_positions)
        self.springs.from_numpy(spring_indices)
        self.rest_length.from_numpy(rest_length)

    @ti.kernel
    def init_ks_field(self, val: ti.f32):
        for i in self.k_spring:
            self.k_spring[i] = val

    def init_ks(self, val):
        self.init_ks_field(val)

    @ti.kernel
    def update_ks_field(self, lr: ti.f32):
        for i in self.k_spring:
            self.k_spring[i] -= lr * self.k_spring.grad[i]

    def update_ks(self, lr):
        self.update_ks_field(lr)

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
        self.loss[None] = 0.0
        # Clear gradients
        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.f.grad.fill(0)
        self.k_spring.grad.fill(0)

    @ti.kernel
    def compute_loss_per_particle(self, t: ti.i32):
        for i in range(self.num_particles):
            diff = self.x[i] - self.positions_gt_all[t, i]
            self.loss[None] += diff.norm_sqr() #/ (self.num_particles * self.num_T)
    
    def update(self):
        # Forward step
        self.compute_forces()
        self.update_positions()

    def backward(self, t:ti.i32):
        # Backward step without ti.ad.tape
        self.compute_loss_per_particle.grad(t)
        self.update_positions.grad()
        self.compute_forces.grad()
    
def generate_target_cloth_data(num_T=50, N=10, gravity=(0.0, -9.8), damping=0.98, dt=0.01, device="cpu"):
    """
    使用一个目标材质对cloth进行模拟，并返回该模拟结果作为target_pos_data。

    参数:
        num_T: 时间步数量（或帧数）
        N: 布料大小参数，(N+1)^2为粒子数量
        gravity, damping, dt, device: cloth仿真需要的物理参数和硬件设置
    返回:
        particle_positions: 初始粒子位置(可用于后续优化的cloth)
        spring_indices, rest_length: 弹簧信息
        target_pos_data: 尺寸为(num_T, num_particles, 2)的numpy数组，表示目标cloth在每个时间步的粒子位置
    """
    ti.init(arch=ti.cpu)
    num_particles = (N+1)**2
    particle_positions = np.random.rand(num_particles, 2)*0.5
    num_springs = 20
    spring_indices = np.stack([np.random.randint(0, num_particles, 2) for _ in range(num_springs)])
    rest_length = np.ones(num_springs)*0.1

    # 定义目标材质,例如全部弹簧刚度为某固定值:
    target_ks_value = 2.0

    num_steps = 200
    cloth_target = TaichiSimulation(particle_positions, spring_indices, rest_length, gravity, damping, num_steps, dt, device, num_T)
    cloth_target.init_ks(val=target_ks_value)

    target_pos_data = np.zeros((num_T, num_particles, 2), dtype=np.float32)

    # 模拟num_T个时刻（这里相当于按时间步的间隔记录状态）
    cloth_target.reset()
    step_interval = cloth_target.steps_per_frame  # 每一帧所走的模拟子步数
    for t in range(num_T):
        for _ in range(step_interval):
            cloth_target.update()
        # 记录该帧的粒子位置到target_pos_data
        target_pos_data[t] = cloth_target.x.to_numpy()
    
    k_spring = cloth_target.k_spring.to_numpy()

    return k_spring, particle_positions, spring_indices, rest_length, target_pos_data


def randomize_cloth_material(cloth, k_min=0.1, k_max=5.0):
    """
    随机生成cloth的材料属性，如弹簧刚度分布。
    k_min与k_max控制随机化的范围。
    """
    random_ks = np.random.uniform(k_min, k_max, size=(cloth.num_springs,)).astype(np.float32)
    cloth.k_spring.from_numpy(random_ks)


def simulate_and_optimize_cloth():
    # 首先生成目标cloth的数据
    N = 10
    num_T = 50
    gravity = [0.0, -9.8]
    damping = 0.98
    dt = 0.01
    device = "cpu"

    k_spring_target, particle_positions, spring_indices, rest_length, target_pos_data = generate_target_cloth_data(
        num_T=num_T, N=N, gravity=gravity, damping=damping, dt=dt, device=device
    )

    # 用于优化的cloth实例（同样的初始拓扑和位置，但材质随机）
    num_steps = 200
    cloth = TaichiSimulation(particle_positions, spring_indices, rest_length, gravity, damping, num_steps, dt, device, num_T)

    # 上传目标位置数据
    cloth.positions_gt_all.from_numpy(target_pos_data)

    # 随机化材质参数
    randomize_cloth_material(cloth, k_min=1.5, k_max=2.3)

    lr = 1e-3
    loss_list = []
    max_frames = num_T
    max_epoch = 1000

    print("Initial k_spring:", cloth.k_spring.to_numpy())
    for epoch in range(max_epoch):
        cloth.reset()
        # 每个epoch进行一次完整的正向和反向传播
        for frame in range(max_frames):
            # 前向模拟
            cloth.update()
            # 计算该帧的loss
            cloth.loss[None] = 0.0
            cloth.compute_loss_per_particle(frame)

            # 设置loss梯度并反向传播
            cloth.loss.grad[None] = 1.0
            cloth.backward(t=frame)

            # 更新 k_spring 参数
            k_spring_value = cloth.k_spring.to_numpy() - lr * cloth.k_spring.grad.to_numpy()
            cloth.k_spring.from_numpy(k_spring_value)

            current_loss = cloth.loss[None]
            loss_list.append(current_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")
    print("target k_spring:", k_spring_target)
    print("Optimized k_spring:", cloth.k_spring.to_numpy())
    print("Final Loss:", loss_list[-1])


if __name__ == "__main__":
    simulate_and_optimize_cloth()
