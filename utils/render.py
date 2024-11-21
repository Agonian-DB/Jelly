# utils/render.py

import numpy as np

def render_image(positions):
    # 创建一个空白的图像
    img_width = 128
    img_height = 256
    img = np.zeros((img_height, img_width), dtype=np.uint8)

    # 将粒子的位置映射到图像坐标
    # 使用固定的坐标映射，将世界坐标直接映射到图像坐标
    # 假设世界坐标 x 和 y 的范围是 [0, 1]
    positions = positions.copy()
    positions[:, 0] = positions[:, 0] * img_width
    positions[:, 1] = positions[:, 1] * img_height
    positions = positions.astype(int)

    # 绘制粒子
    for pos in positions:
        x_pos, y_pos = pos
        if 0 <= x_pos < img_width and 0 <= y_pos < img_height:
            img[img_height - y_pos - 1, x_pos] = 255  # 反转 y 轴
    # 反转黑白颜色
    # img = 255 - img
    return img
