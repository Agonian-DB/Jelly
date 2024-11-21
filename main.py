from data_generation import DataGenerator
# data generation main function


def main():
    num_samples = 10
    image_size = (256, 256)
    grid_spacing = 8
    gravity = [0.0, 20.0]
    damping = 0.99
    dt = 1e-3
    num_steps = 250000
    num_T = 20  # ground truth frame number

    data_output_dir = 'data/generated'

    data_generator = DataGenerator(
        image_size=image_size,
        grid_spacing=grid_spacing,
        default_k=10,
        gravity=gravity,
        damping=damping,
        num_steps=num_steps,
        dt=dt,
        num_T=num_T
    )
    data_generator.generate_dataset(num_samples=num_samples, output_dir=data_output_dir)

if __name__ == "__main__":
    main()
