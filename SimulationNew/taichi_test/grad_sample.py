import taichi as ti
ti.init()

x = ti.field(dtype=ti.f32, shape=(), needs_dual=True)
y = ti.field(dtype=ti.f32, shape=(), needs_dual=True)

@ti.kernel
def compute_y():
    y[None] = ti.sin(x[None])

# `loss`: The function's output
# `param`: The input of the function
with ti.ad.FwdMode(loss=y, param=x):
    compute_y()

print('dy/dx =', y.dual[None], ' at x =', x[None])