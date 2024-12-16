import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

@ti.data_oriented
class Cloth:
    def __init__(self, N, target, kd=0.6, mass_value=1.0):
        self.N = N
        self.NV = (N + 1) ** 2
        self.h=0.01

        # 可微参数
        self.ks = ti.field(ti.f32, shape=(self.N + 1, self.N + 1), needs_grad=True)
        self.kd = ti.field(ti.f32, shape=(), needs_grad=True)
        self.kf = ti.field(ti.f32, shape=(), needs_grad=True)
        self.mass_value = ti.field(ti.f32, shape=(), needs_grad=True)

        self.kd[None] = kd
        self.kf[None] = 1.0e5
        self.mass_value[None] = mass_value

        self.pos = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.vel = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.force = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)

        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.target_field = ti.Vector.field(2, ti.f32, self.NV)
        self.target_field.from_numpy(target)

        self.gravity = ti.Vector([0.0, -9.8])

        self.init_pos()

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([j / self.N, i / self.N]) + ti.Vector([0.0, 0.5])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def init_ks(self, default_ks: ti.f32):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            self.ks[i, j] = default_ks

    @ti.kernel
    def compute_force(self):
        for i in range(self.NV):
            self.force[i] = ti.Vector([0.0, 0.0])
        for i in range(self.NV):
            mass = self.mass_value[None]
            # 重力
            self.force[i] += self.gravity * mass
            # 获取二维坐标
            x, y = i // (self.N + 1), i % (self.N + 1)
            # 回拉力：将点拉回 initPos
            disp = self.pos[i] - self.initPos[i]
            self.force[i] += -self.ks[x, y] * disp
            # 阻尼力
            self.force[i] += -self.kd[None] * self.vel[i]

        # 使用kf固定顶部行的点
        for j in range(self.N + 1):
            k = 0 * (self.N + 1) + j
            top_disp = self.pos[k] - self.initPos[k]
            self.force[k] += -self.kf[None] * top_disp

    @ti.kernel
    def update_pos_vel(self):
        for i in range(self.NV):
            mass = self.mass_value[None]
            acc = self.force[i] / mass
            self.vel[i] += acc * self.h
            self.pos[i] += self.vel[i] * self.h

    @ti.kernel
    def compute_loss(self):
        for i in range(self.NV):
            diff = self.pos[i] - self.target_field[i]
            self.loss[None] += diff.norm_sqr()

    @ti.kernel
    def update_ks(self, lr: ti.f32):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            self.ks[i, j] -= lr * self.ks.grad[i, j]

    def update(self):
        self.compute_force()
        self.update_pos_vel()

    def backward(self):
        self.compute_loss.grad()
        self.update_pos_vel.grad()
        self.compute_force.grad()

    @ti.kernel
    def print_ks(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            print(f"ks[{i}, {j}] = {self.ks[i, j]}, grad = {self.ks.grad[i, j]}")

def main():
    #h = 0.01
    max_frames = 6
    N = 10

    run_id = 0
    run_dir = f"./dataset/run_{run_id}"

    # 加载目标仿真参数与数据
    initPos_target = np.load(os.path.join(run_dir, "initPos.npy"))
    material_params = np.load(os.path.join(run_dir, "material_params.npz"))
    kd_target = material_params['kd']
    mass_target = material_params['mass_value']
    kf_target = material_params['kf']

    # 加载目标每一帧的位置数据
    target_pos_data = []
    for frame in range(max_frames):
        frame_data = np.load(os.path.join(run_dir, f"pos_{frame}.npy"))
        target_pos_data.append(frame_data)

    print("Target parameters:")
    print("kd:", kd_target)
    print("mass_value:", mass_target)
    print("kf:", kf_target)

    # 初始化cloth
    cloth = Cloth(N=N, target=target_pos_data[0])

    # 初始随机参数
    default_ks = 900.0
    cloth.init_ks(default_ks)
    cloth.kd[None] = 0.5
    cloth.mass_value[None] = 1
    cloth.kf[None] = 1e5

    # 只更新 ks
    lr = 1e-3
    loss_list = []
    for _ in range(10):
        for frame in range(max_frames-1, max_frames):
            # 前向计算
            cloth.update()

            # 更新当前帧的目标位置
            cloth.target_field.from_numpy(target_pos_data[frame])

            # 计算该帧loss
            cloth.loss[None] = 0.0
            cloth.compute_loss()

            # 设置loss梯度为1
            cloth.loss.grad[None] = 1

            # 调用backward确保梯度从loss传递到ks
            cloth.backward()

            # 更新 ks 矩阵
            cloth.update_ks(lr)

            current_loss = cloth.loss[None]
            loss_list.append(current_loss)
            
            print(f"Frame {frame}, Loss: {current_loss:.4f}")
        if current_loss<1000:
                break
    print(cloth.ks.to_numpy())
    print("Final Loss:", loss_list[-1])

if __name__ == "__main__":
    main()
