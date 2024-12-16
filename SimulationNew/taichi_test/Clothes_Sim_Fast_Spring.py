import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu)  # 使用CPU后端

@ti.data_oriented
class ClothSimulation:
    def __init__(self, N, material):
        self.N = N  # 布料粒子在每个维度上的数量
        self.NV = (N + 1) * (N + 1)  # 总质点数量
        self.NE = 2 * N * (N + 1) + 2 * N * N  # 总弹簧数量
        self.dt = 1e-3  # 时间步长

        # 材质属性
        self.materials = {
            'cotton': {'stiffness': 50.0},
            'silk': {'stiffness': 10.0},
            'leather': {'stiffness': 80.0},
            'wool': {'stiffness': 30.0},
        }
        self.set_material(material)

        # 定义Taichi字段
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)
        self.old_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.NV)
        self.mass = ti.field(dtype=ti.f32, shape=self.NV)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.NV)
        self.fixed = ti.field(dtype=ti.i32, shape=self.NV)

        self.spring_indices = ti.Vector.field(2, dtype=ti.i32, shape=self.NE)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.NE)

        self.gravity = ti.Vector([0.0, -9.81, 0.0])

        self.init_positions()
        self.init_springs()
        self.init_mass()

    def set_material(self, material):
        self.ks = self.materials[material]['stiffness']

    @ti.kernel
    def init_positions(self):
        # 初始化布料的质点位置，使其形成一个完整的网格
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            idx = i * (self.N + 1) + j
            x = 0.2 + (j / self.N) * 0.6  # x 方向
            y = 0.9  # y 方向固定为 0.9
            z = 0.2 + (i / self.N) * 0.6  # z 方向
            self.pos[idx] = ti.Vector([x, y, z])
            self.old_pos[idx] = self.pos[idx]
            self.vel[idx] = ti.Vector([0.0, 0.0, 0.0])
            self.fixed[idx] = 0

        # 固定布料顶部一排质点（i = 0）
        for j in range(self.N + 1):
            idx = j  # i = 0 对应顶部
            self.fixed[idx] = 1
            self.inv_mass[idx] = 0.0

    @ti.kernel
    def init_springs(self):
        idx = 0
        N = self.N
        # 结构弹簧（横向和纵向）
        for i, j in ti.ndrange(N + 1, N):
            idx0 = i * (N + 1) + j
            idx1 = idx0 + 1
            self.spring_indices[idx] = ti.Vector([idx0, idx1])
            self.rest_length[idx] = (self.pos[idx0] - self.pos[idx1]).norm()
            idx += 1

        for i, j in ti.ndrange(N, N + 1):
            idx0 = i * (N + 1) + j
            idx1 = (i + 1) * (N + 1) + j
            self.spring_indices[idx] = ti.Vector([idx0, idx1])
            self.rest_length[idx] = (self.pos[idx0] - self.pos[idx1]).norm()
            idx += 1

        # 剪切弹簧
        for i, j in ti.ndrange(N, N):
            idx0 = i * (N + 1) + j
            idx1 = (i + 1) * (N + 1) + j + 1
            self.spring_indices[idx] = ti.Vector([idx0, idx1])
            self.rest_length[idx] = (self.pos[idx0] - self.pos[idx1]).norm()
            idx += 1

            idx0 = (i + 1) * (N + 1) + j
            idx1 = i * (N + 1) + j + 1
            self.spring_indices[idx] = ti.Vector([idx0, idx1])
            self.rest_length[idx] = (self.pos[idx0] - self.pos[idx1]).norm()
            idx += 1

    @ti.kernel
    def init_mass(self):
        for i in range(self.NV):
            if self.fixed[i] == 0:
                self.mass[i] = 1.0
                self.inv_mass[i] = 1.0 / self.mass[i]
            else:
                self.mass[i] = 0.0
                self.inv_mass[i] = 0.0

    @ti.kernel
    def apply_external_forces(self):
        for i in range(self.NV):
            if self.fixed[i] == 0:
                # 应用重力
                self.vel[i] += self.dt * self.gravity

    @ti.kernel
    def damp_velocity(self):
        damping = 0.99
        for i in range(self.NV):
            self.vel[i] *= damping

    @ti.kernel
    def semi_euler(self):
        for i in range(self.NV):
            if self.fixed[i] == 0:
                self.old_pos[i] = self.pos[i]
                self.pos[i] += self.dt * self.vel[i]

    @ti.kernel
    def constraint_projection(self):
        for s in range(self.NE):
            idx0, idx1 = self.spring_indices[s]
            p0 = self.pos[idx0]
            p1 = self.pos[idx1]
            inv_mass0 = self.inv_mass[idx0]
            inv_mass1 = self.inv_mass[idx1]
            w = inv_mass0 + inv_mass1
            if w == 0:
                continue
            rest_len = self.rest_length[s]
            dir = p1 - p0
            length = dir.norm()
            if length > 1e-6:
                dir /= length
                delta_length = length - rest_len
                correction = dir * (delta_length / w)
                if self.fixed[idx0] == 0:
                    self.pos[idx0] += correction * inv_mass0
                if self.fixed[idx1] == 0:
                    self.pos[idx1] -= correction * inv_mass1

    @ti.kernel
    def update_velocity(self):
        for i in range(self.NV):
            if self.fixed[i] == 0:
                self.vel[i] = (self.pos[i] - self.old_pos[i]) / self.dt

    def step(self):
        self.apply_external_forces()
        self.damp_velocity()
        self.semi_euler()
        # 约束求解迭代次数
        for _ in range(50):
            self.constraint_projection()
        self.update_velocity()

    def run(self):
        gui = ti.GUI("Cloth Simulation", res=(800, 800), show_gui=False)

        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)
        max_frames = 1000  # 设置要保存的帧数
        for frame in range(max_frames):
            self.step()
            gui.clear(0x112F41)
            self.render(gui)
            # 保存当前帧
            filename = os.path.join(output_dir, f'frame_{frame:05d}.png')
            gui.show(filename)
            print(f'Frame {frame} saved to {filename}')

    def render(self, gui):
        pos_np = self.pos.to_numpy()
        indices_np = self.spring_indices.to_numpy()
        lines = []
        for idx0, idx1 in indices_np:
            p0 = pos_np[idx0]
            p1 = pos_np[idx1]
            # 使用 x 和 y 坐标进行渲染
            lines.append(p0[[0, 1]])  # x, y
            lines.append(p1[[0, 1]])
        lines = np.array(lines)
        gui.lines(lines[::2], lines[1::2], color=0xAAAAAA, radius=1)
        gui.circles(pos_np[:, [0, 1]], color=0xFF0000, radius=2)

if __name__ == '__main__':
    N = 20
    material = 'cotton'  # 选择 'cotton'、'silk'、'leather' 等
    cloth_sim = ClothSimulation(N, material)
    cloth_sim.run()
