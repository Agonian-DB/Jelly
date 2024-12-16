import os
import numpy as np
import taichi as ti
import psutil

@ti.data_oriented
class Cloth:
    def __init__(self, N, target, ks=1000.0, kd=0.6, mass_value=1.0, kf=1.0e5):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1) ** 2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # number of edges

        # Declare fields with needs_grad=True for autodiff
        self.pos = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.initPos = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.vel = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.force = ti.Vector.field(2, ti.f32, self.NV, needs_grad=True)
        self.mass = ti.field(ti.f32, self.NV, needs_grad=True)
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)

        # the target field
        self.target_field = ti.Vector.field(2, ti.f32, self.NV)
        self.target_field.from_numpy(target)

        # Other fields
        self.vel_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.force_1D = ti.ndarray(ti.f32, 2 * self.NV)
        self.b = ti.ndarray(ti.f32, 2 * self.NV)
        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.indices = ti.field(ti.i32, 2 * self.NE)
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE)
        self.Jv = ti.Matrix.field(2, 2, ti.f32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)

        self.ks = ks # spring stiffness
        self.kd = kd  # damping constant
        self.kf = kf  # fix point stiffness
        self.mass_value = mass_value

        self.gravity = ti.Vector([0.0, -9.8])
        self.init_pos()
        self.init_edges()
        self.MassBuilder = ti.linalg.SparseMatrixBuilder(
            2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.DBuilder = ti.linalg.SparseMatrixBuilder(
            2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.KBuilder = ti.linalg.SparseMatrixBuilder(
            2 * self.NV, 2 * self.NV, max_num_triplets=10000)
        self.init_mass_sp(self.MassBuilder)
        self.M = self.MassBuilder.build()
        # Fixed vertices (top row)
        self.fix_vertex = [self.N * (self.N + 1) + j for j in range(self.N + 1)]
        self.num_fixed_vertices = len(self.fix_vertex)
        self.Jf = ti.Matrix.field(2, 2, ti.f32, len(self.fix_vertex))

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.pos[k] = ti.Vector([j, i]) / self.N * 0.5 + ti.Vector([0.25, 0.25])
            self.initPos[k] = self.pos[k]
            self.vel[k] = ti.Vector([0.0, 0.0])
            self.mass[k] = self.mass_value

    @ti.kernel
    def init_edges(self):
        pos, spring, N, rest_len = ti.static(
            self.pos, self.spring, self.N, self.rest_len)
        for i, j in ti.ndrange(N + 1, N):
            idx = i * N + j
            idx1 = i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
            rest_len[idx] = (pos[idx1] - pos[idx1 + 1]).norm()
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx = start + i * (N + 1) + j
            idx1 = i * (N + 1) + j
            idx2 = (i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start += N * (N + 1)
        for i, j in ti.ndrange(N, N):
            idx = start + i * N + j
            idx1 = i * (N + 1) + j
            idx2 = (i + 1) * (N + 1) + j + 1
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()
        start += N * N
        for i, j in ti.ndrange(N, N):
            idx = start + i * N + j
            idx1 = i * (N + 1) + j + 1
            idx2 = (i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
            rest_len[idx] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_mass_sp(self, M: ti.types.sparse_matrix_builder()):
        for i in range(self.NV):
            mass = self.mass[i]
            M[2 * i + 0, 2 * i + 0] += mass
            M[2 * i + 1, 2 * i + 1] += mass

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        for i in self.force:
            self.force[i] += self.gravity * self.mass[i]

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos2 - pos1
            l = dis.norm()
            if l > 0:
                dir = dis / l
                force = self.ks * (l - self.rest_len[i]) * dir
                self.force[idx1] += force
                self.force[idx2] -= force
        # Apply fix constraint gradient
        for idx in ti.static(self.fix_vertex):
            self.force[idx] += self.kf * (self.initPos[idx] - self.pos[idx])

    @ti.kernel
    def compute_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            l = dx.norm()
            if l != 0.0:
                dir = dx / l
                I = ti.Matrix.identity(ti.f32, 2)
                self.Jx[i] = (self.ks * (1 - self.rest_len[i] / l) * (I - dir.outer_product(dir))
                              - self.ks * self.rest_len[i] / l * dir.outer_product(dir))
                self.Jv[i] = self.kd * I
            else:
                self.Jx[i] = ti.Matrix.zero(ti.f32, 2, 2)
                self.Jv[i] = ti.Matrix.zero(ti.f32, 2, 2)

        # Fix point constraint Hessian
        for idx in ti.static(range(self.num_fixed_vertices)):
            self.Jf[idx] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]
        # Add fix constraint Hessian
        for idx in ti.static(range(self.num_fixed_vertices)):
            vertex_idx = self.fix_vertex[idx]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * vertex_idx + m, 2 * vertex_idx + n] += self.Jf[idx][m, n]

    @ti.kernel
    def assemble_D(self, D: ti.types.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                D[2 * idx1 + m, 2 * idx1 + n] -= self.Jv[i][m, n]
                D[2 * idx1 + m, 2 * idx2 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx1 + n] += self.Jv[i][m, n]
                D[2 * idx2 + m, 2 * idx2 + n] -= self.Jv[i][m, n]

    @ti.kernel
    def updatePosVel(self, h: ti.f32, dv: ti.types.ndarray()):
        for i in self.pos:
            self.vel[i] += ti.Vector([dv[2 * i], dv[2 * i + 1]])
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def copy_to(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.NV):
            des[2 * i] = source[i][0]
            des[2 * i + 1] = source[i][1]

    @ti.kernel
    def compute_b(
        self,
        b: ti.types.ndarray(),
        f: ti.types.ndarray(),
        Kv: ti.types.ndarray(),
        h: ti.f32,
    ):
        for i in range(2 * self.NV):
            b[i] = (f[i] + Kv[i] * h) * h

    # Define the loss function
    @ti.kernel
    def compute_loss(self):
        for i in range(self.NV):
            diff = self.pos[i] - self.target_field[i]
            self.loss[None] += diff.norm_sqr()

    def update(self, h):
        self.compute_force()
        self.compute_Jacobians()
        # Assemble global system
        #self.DBuilder.clear()
        #self.KBuilder.clear()

        self.assemble_D(self.DBuilder)
        D = self.DBuilder.build()

        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()

        A = self.M - h * D - h**2 * K

        self.copy_to(self.vel_1D, self.vel)
        self.copy_to(self.force_1D, self.force)

        # b = (force + h * K @ vel) * h
        Kv = K @ self.vel_1D
        self.compute_b(self.b, self.force_1D, Kv, h)

        # Sparse solver
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        # Solve the linear system
        dv = solver.solve(self.b)
        self.updatePosVel(h, dv)
        with ti.ad.Tape(loss=self.loss):
            self.compute_loss()

    @ti.kernel
    def spring2indices(self):
        for i in self.spring:
            self.indices[2 * i + 0] = self.spring[i][0]
            self.indices[2 * i + 1] = self.spring[i][1]


def main():
    ti.init(arch=ti.cpu)
    h = 0.01  # time step
    max_frames = 100  # number of frames
    N = 10  # number of circle in each row

    # 目标设定，这里简单设定为全部(10,10)的位置
    target_testing = np.ones(((N + 1) ** 2, 2)) * 10.0

    # 弹性参数设置
    ks = 1000.0
    kd = 0.5
    mass_value = 1.0
    kf = 1.0e5

    cloth = Cloth(N=N, target=target_testing, ks=ks, kd=kd, mass_value=mass_value, kf=kf)

    # 创建dataset目录
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")

    # 假设一次运行生成一个数据集条目，这里用 run_0 表示，如需多次生成数据，可外部循环
    run_id = 0
    run_dir = f"./dataset/run_{run_id}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # 保存初始材质信息和初始位置
    initPos_np = cloth.initPos.to_numpy()
    mass_np = cloth.mass.to_numpy()
    np.save(os.path.join(run_dir, "initPos.npy"), initPos_np)
    np.savez(os.path.join(run_dir, "material_params.npz"), 
             ks=ks, kd=kd, mass_value=mass_value, kf=kf)

    # 时间步迭代
    for frame in range(max_frames):
        cloth.update(h)
        pos_np = cloth.pos.to_numpy()
        # 保存每个时间步的位置信息
        np.save(os.path.join(run_dir, f"pos_{frame}.npy"), pos_np)
        print(f"Frame {frame} saved.")

if __name__ == "__main__":
    main()
