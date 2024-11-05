from data.data_generation import DataGenerator

def main():
    # 参数设置
    num_samples = 10
    image_size = (256, 256)
    grid_spacing = 32
    gravity = [0.0, -9.8]
    damping = 0.99
    dt = 1e-3
    num_steps = 1000
    num_T = 10  # Ground Truth 帧的数量

    data_output_dir = 'data/generated'

    # 创建数据生成器并生成数据集
    data_generator = DataGenerator(
        image_size=image_size,
        grid_spacing=grid_spacing,
        default_k=500,
        gravity=gravity,
        damping=damping,
        num_steps=num_steps,
        dt=dt,
        num_T=num_T
    )

    # 生成数据集并运行模拟
    data_generator.generate_dataset(num_samples=num_samples, output_dir=data_output_dir)

if __name__ == "__main__":
    main()
