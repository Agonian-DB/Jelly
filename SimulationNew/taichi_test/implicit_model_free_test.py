import argparse
import os
import numpy as np
import taichi as ti

@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2
        self.NV = (N + 1)**2
        self.NE = 2 * N * (N + 1) + 2 * N * N

        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)

        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 1000.0  # spring stiffness
        self.kd = 0.6  # damping coefficient
        self.kf = 1.0e5  # fixed point stiffness

        self.gravity = ti.Vector([0.0, -9.8])

        self.init_pos()
        self.init_edges()

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([j / self.N, i / self.N]) + ti.Vector([0.0, 0.5])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0.0, 0.0])
            self.mass[k] = 1.0

    @ti.kernel
    def init_edges(self):
        for i, j in ti.ndrange(self.N + 1, self.N):
            idx = i * self.N + j
            idx1 = i * (self.N + 1) + j
            self.spring[idx] = ti.Vector([idx1, idx1 + 1])
            self.rest_len[idx] = (self.pos[idx1] - self.pos[idx1 + 1]).norm()
        for i, j in ti.ndrange(self.N, self.N + 1):
            idx = self.N * (self.N + 1) + i * (self.N + 1) + j
            idx1 = i * (self.N + 1) + j
            idx2 = idx1 + (self.N + 1)
            self.spring[idx] = ti.Vector([idx1, idx2])
            self.rest_len[idx] = (self.pos[idx1] - self.pos[idx2]).norm()

    @ti.kernel
    def compute_force(self):
        for i in range(self.NV):
            self.force[i] = self.gravity * self.mass[i]

        for i in range(self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            disp = pos2 - pos1
            l = disp.norm()
            if l > 0:
                force = self.ks * (l - self.rest_len[i]) * disp.normalized()
                self.force[idx1] += force
                self.force[idx2] -= force

        for j in range(self.N + 1):
            idx = j
            disp = self.pos[idx] - self.initPos[idx]
            self.force[idx] += -self.kf * disp

    @ti.kernel
    def update_pos_vel(self, h: ti.f32):
        for i in range(self.NV):
            acc = self.force[i] / self.mass[i]
            self.vel[i] += acc * h
            self.pos[i] += self.vel[i] * h

    def update(self, h):
        self.compute_force()
        self.update_pos_vel(h)

    def display(self, gui, radius=5, color=0xFFFFFF):
        for i in range(self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            gui.line(begin=self.pos[idx1], end=self.pos[idx2], radius=1, color=color)
        gui.circles(self.pos.to_numpy(), radius=radius, color=color)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--use-ggui", action="store_true", help="Display with GGUI")
    args, unknowns = parser.parse_known_args()
    ti.init(arch=ti.cpu)

    h = 0.01  # timestep
    cloth = Cloth(N=10)

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    gui = ti.GUI("Cloth Simulation", res=(500, 500), show_gui=False)
    max_frames = 100

    for frame in range(max_frames):
        cloth.update(h)
        cloth.display(gui)
        filename = os.path.join(output_dir, f'frame_{frame:03d}.png')
        gui.show(filename)  # save the current frame
        print(f'Frame {frame} saved to {filename}')

if __name__ == "__main__":
    main()
