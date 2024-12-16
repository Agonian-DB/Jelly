import os
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

@ti.data_oriented
class Cloth:
    def __init__(self, N, ks=1000.0, kd=0.6, mass_value=1.0, kf=1e5):
        self.N = N
        self.NV = (N+1)**2

        self.ks = ti.field(ti.f32, shape=())
        self.kd = ti.field(ti.f32, shape=())
        self.kf = ti.field(ti.f32, shape=())
        self.mass_value = ti.field(ti.f32, shape=())

        self.ks[None] = ks
        self.kd[None] = kd
        self.kf[None] = kf
        self.mass_value[None] = mass_value

        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)

        self.gravity = ti.Vector([0.0, -9.8])

        self.init_pos()

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N+1, self.N+1):
            k = i*(self.N+1)+j
            self.pos[k] = ti.Vector([j/self.N, i/self.N]) + ti.Vector([0.0, 0.5])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        for i in range(self.NV):
            self.force[i] = ti.Vector([0.0, 0.0])
        for i in range(self.NV):
            mass = self.mass_value[None]
            self.force[i] += self.gravity * mass
            disp = self.pos[i] - self.initPos[i]
            self.force[i] += -self.ks[None]*disp
            self.force[i] += -self.kd[None]*self.vel[i]
        for j in range(self.N+1):
            k = 0*(self.N+1)+j
            top_disp = self.pos[k] - self.initPos[k]
            self.force[k] += -self.kf[None]*top_disp

    @ti.kernel
    def update_pos_vel(self, h: ti.f32):
        for i in range(self.NV):
            mass = self.mass_value[None]
            acc = self.force[i]/mass
            self.vel[i] += acc*h
            self.pos[i] += self.vel[i]*h

    def update(self, h):
        self.compute_force()
        self.update_pos_vel(h)

def save_frame_visualization(pos_np, frame, output_dir):
    """Save a visualization of the current frame."""
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_np[:, 0], pos_np[:, 1], color="blue", s=10)
    plt.title(f"Frame {frame}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"frame_{frame:03d}.png"))
    plt.close()

def main():
    h = 0.01
    max_frames = 100
    N = 10

    run_id = 0
    run_dir = f"./dataset/run_{run_id}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    viz_dir = os.path.join(run_dir, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Define target material parameters
    ks_true = 1200.0
    kd_true = 0.5
    mass_true = 1.0
    kf_true = 1e5

    cloth = Cloth(N=N, ks=ks_true, kd=kd_true, mass_value=mass_true, kf=kf_true)

    # Save initial position
    initPos_np = cloth.initPos.to_numpy()
    np.save(os.path.join(run_dir, "initPos.npy"), initPos_np)

    # Save material parameters
    np.savez(os.path.join(run_dir, "material_params.npz"),
             ks=ks_true, kd=kd_true, mass_value=mass_true, kf=kf_true)

    # Simulate and save positions and visualizations
    for frame in range(max_frames):
        cloth.update(h)
        pos_np = cloth.pos.to_numpy()
        np.save(os.path.join(run_dir, f"pos_{frame}.npy"), pos_np)
        save_frame_visualization(pos_np, frame, viz_dir)
        print(f"Saved frame {frame} and visualization")

    print("Target data and visualizations generated at:", run_dir)

if __name__ == "__main__":
    main()
