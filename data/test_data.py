import numpy as np
from matplotlib import pyplot as plt


def plot_jelly_model_with_texture(sample_idx, output_dir='generated_textures_and_models'):
    # 加载粒子位置、弹簧索引和弹簧 k 值
    particle_positions = np.load(f'{output_dir}/particle_positions_{sample_idx}.npy')
    spring_indices = np.load(f'{output_dir}/spring_indices_{sample_idx}.npy')
    spring_k_values = np.load(f'{output_dir}/spring_k_values_{sample_idx}.npy')

    # 加载生成的纹理图像
    texture_image = plt.imread(f'{output_dir}/texture_sample_{sample_idx}.png')

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 在左侧展示纹理图像
    ax1.imshow(texture_image)
    ax1.set_title('Generated Texture Image')
    ax1.axis('off')

    # 在右侧展示粒子和弹簧的二维模型
    ax2.set_title('Jelly Model with Spring k-values')

    # 绘制粒子位置
    ax2.scatter(particle_positions[:, 0], particle_positions[:, 1], c='b', s=50, label='Particles')

    # 反转 y 轴
    ax2.invert_yaxis()  # 检查是否是渲染时坐标系问题

    # 绘制弹簧，颜色根据 k 值区分
    norm_k_values = (spring_k_values - spring_k_values.min()) / (spring_k_values.max() - spring_k_values.min())  # 归一化 k 值
    cmap = plt.get_cmap('coolwarm')  # 颜色映射

    for idx, (i, j) in enumerate(spring_indices):
        k_value_normalized = norm_k_values[idx]
        color = cmap(k_value_normalized)
        pos_i = particle_positions[i]
        pos_j = particle_positions[j]
        ax2.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], color=color, lw=2)

    # 设置图形的轴属性
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# 调用函数
sample_idx = 3  # 选择你想要查看的样本索引
plot_jelly_model_with_texture(sample_idx)
