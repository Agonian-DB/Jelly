import taichi as ti
import os

ti.init()

pixels = ti.field(ti.u8, shape=(512, 512, 3))

@ti.kernel
def paint():
    for i, j, k in pixels:
        pixels[i, j, k] = ti.random() * 255

iterations = 1000
gui = ti.GUI("Random pixels", res=512, show_gui=False)

# mainloop
for i in range(iterations):
    paint()
    gui.set_image(pixels)

    filename = f'frame_{i:05d}.png'   # create filename with suffix png
    print(f'Frame {i} is recorded in {filename}')
    gui.show(filename)  # export and show in GUI