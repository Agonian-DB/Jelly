# data/data_generation.py

import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

@ti.data_oriented
class DataGenerator:
    def __init__(self, image_size=(256, 256), grid_spacing=8, default_k=500, gravity=[0.0, -9.8], damping=0.99, num_steps=100, dt=1e-3, num_T=10):
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.grid_spacing = grid_spacing
        self.default_k = default_k
        self.gravity = ti.Vector(gravity)
        self.damping = damping
        self.num_steps = num_steps
        self.dt = dt
        self.num_T = num_T
        self.steps_per_frame = num_steps // num_T
        self.frame_time_length = dt * self.steps_per_frame

        # initialize texture
        dtd_path = 'data/dtd/images/'
        self.background_label = 'banded'
        self.background_texture = self.load_texture_images(self.background_label, dtd_path=dtd_path)
        self.textures = ['chequered', 'bubbly']

        # set the k values for corresponding texture
        k_values = np.linspace(500, 1000, len(self.textures))
        self.texture_to_k = {self.background_label: self.default_k}
        self.texture_to_k.update(dict(zip(self.textures, k_values)))

        self.texture_images = {texture: self.load_texture_images(texture, dtd_path=dtd_path) for texture in
                               self.textures}

        self.grid_rows = self.image_height // self.grid_spacing
        self.grid_cols = self.image_width // self.grid_spacing
        self.num_particles = self.grid_rows * self.grid_cols
        self.cols = self.grid_cols

        self.dim = 2

        # calculate num_springs
        num_horizontal_springs = self.grid_rows * (self.grid_cols - 1)
        num_vertical_springs = (self.grid_rows - 1) * self.grid_cols
        num_diagonal_down_right_springs = (self.grid_rows - 1) * (self.grid_cols - 1)
        num_diagonal_down_left_springs = (self.grid_rows - 1) * (self.grid_cols - 1)
        self.num_springs = (num_horizontal_springs + num_vertical_springs +
                            num_diagonal_down_right_springs + num_diagonal_down_left_springs)

        # initialize taichi files
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_particles)
        self.f = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num_particles)
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=self.num_springs)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.num_springs)
        self.k_spring = ti.field(dtype=ti.f32, shape=self.num_springs)



    def initialize_fields(self):
        self.x.fill(0)
        self.v.fill(0)
        self.f.fill(0)


    def reset_simulation_state(self, particle_positions):
        self.x.from_numpy(particle_positions)
        self.v.fill(0)
        self.f.fill(0)

    def load_texture_images(self, class_name, dtd_path='data/dtd/images/'):
        class_path = os.path.join(dtd_path, class_name)
        images = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            images.append(np.array(img))
        return images

    def get_texture_crop(self, texture_img, target_height, target_width):
        texture_height, texture_width, _ = texture_img.shape

        if texture_height >= target_height and texture_width >= target_width:
            x_rand = random.randint(0, texture_height - target_height)
            y_rand = random.randint(0, texture_width - target_width)
            texture_crop = texture_img[x_rand:x_rand+target_height, y_rand:y_rand+target_width]
        else:
            rep_x = target_height // texture_height + 1
            rep_y = target_width // texture_width + 1
            texture_tiled = np.tile(texture_img, (rep_x, rep_y, 1))
            texture_crop = texture_tiled[:target_height, :target_width]

        return texture_crop

    def is_overlapping(self, x_start, x_end, y_start, y_end, used_regions):
        for (used_x_start, used_x_end, used_y_start, used_y_end) in used_regions:
            if not (x_end <= used_x_start or x_start >= used_x_end or
                    y_end <= used_y_start or y_start >= used_y_end):
                return True
        return False

    def apply_random_shapes_in_quadrants(self):
        """divide the image into 4 parts and randomly assign shapes"""

        background_img = random.choice(self.background_texture)
        background_crop = self.get_texture_crop(background_img, self.image_height, self.image_width)
        canvas = background_crop.copy()

        shapes_info = [{
            'type': 'background',
            'texture_label': self.background_label,
            'k_value': self.texture_to_k[self.background_label],
            'mask': (0, self.image_height, 0, self.image_width)
        }]

        h_mid, w_mid = self.image_height // 2, self.image_width // 2
        quadrants = [
            (0, h_mid, 0, w_mid),
            (0, h_mid, w_mid, self.image_width),
            (h_mid, self.image_height, 0, w_mid),
            (h_mid, self.image_height, w_mid, self.image_width)
        ]

        used_regions = []

        for quadrant in quadrants:
            if random.choice([True, False]):
                texture = random.choice(self.textures)
                texture_img = random.choice(self.texture_images[texture])
                k_value = self.texture_to_k[texture]

                shape_type = random.choice(['rectangle', 'circle', 'triangle'])
                x_start, x_end, y_start, y_end = quadrant
                region_width = y_end - y_start
                region_height = x_end - x_start

                if shape_type == 'rectangle':
                    rect_height = random.randint(region_height // 2, region_height)
                    rect_width = random.randint(region_width // 2, region_width)

                    x_min = x_start
                    x_max = x_end - rect_height
                    y_min = y_start
                    y_max = y_end - rect_width

                    if x_max < x_min or y_max < y_min:
                        continue

                    rect_x_start = random.randint(x_min, x_max)
                    rect_x_end = rect_x_start + rect_height
                    rect_y_start = random.randint(y_min, y_max)
                    rect_y_end = rect_y_start + rect_width

                    # overlap
                    if self.is_overlapping(rect_x_start, rect_x_end, rect_y_start, rect_y_end, used_regions):
                        continue

                    used_regions.append((rect_x_start, rect_x_end, rect_y_start, rect_y_end))

                    texture_crop = self.get_texture_crop(texture_img, rect_height, rect_width)

                    canvas[rect_x_start:rect_x_end, rect_y_start:rect_y_end] = texture_crop

                    shapes_info.append({
                        'type': 'rectangle',
                        'position': (rect_x_start, rect_y_start),
                        'size': (rect_height, rect_width),
                        'texture_label': texture,
                        'k_value': k_value,
                        'mask': (rect_x_start, rect_x_end, rect_y_start, rect_y_end)
                    })

                elif shape_type == 'circle':

                    max_radius = min(region_width, region_height) // 4
                    radius = random.randint(max_radius // 2, max_radius)

                    x_min = x_start + radius
                    x_max = x_end - radius
                    y_min = y_start + radius
                    y_max = y_end - radius

                    if x_max < x_min or y_max < y_min:
                        continue

                    center_x = random.randint(x_min, x_max)
                    center_y = random.randint(y_min, y_max)
                    x_start_circ = center_x - radius
                    x_end_circ = center_x + radius
                    y_start_circ = center_y - radius
                    y_end_circ = center_y + radius

                    if self.is_overlapping(x_start_circ, x_end_circ, y_start_circ, y_end_circ, used_regions):
                        continue

                    used_regions.append((x_start_circ, x_end_circ, y_start_circ, y_end_circ))

                    circ_height = x_end_circ - x_start_circ
                    circ_width = y_end_circ - y_start_circ

                    texture_crop = self.get_texture_crop(texture_img, circ_height, circ_width)

                    mask = np.zeros((circ_height, circ_width), dtype=np.uint8)
                    cv2.circle(mask, (radius, radius), radius, 255, -1)

                    roi = canvas[x_start_circ:x_end_circ, y_start_circ:y_end_circ]
                    roi[mask == 255] = texture_crop[mask == 255]

                    shapes_info.append({
                        'type': 'circle',
                        'center': (center_x, center_y),
                        'radius': radius,
                        'texture_label': texture,
                        'k_value': k_value,
                        'mask': (x_start_circ, x_end_circ, y_start_circ, y_end_circ, mask)
                    })

                elif shape_type == 'triangle':
                    tri_height = random.randint(region_height // 2, region_height)
                    tri_width = random.randint(region_width // 2, region_width)

                    x_min = x_start
                    x_max = x_end - tri_height
                    y_min = y_start
                    y_max = y_end - tri_width

                    if x_max < x_min or y_max < y_min:
                        continue

                    tri_x_start = random.randint(x_min, x_max)
                    tri_y_start = random.randint(y_min, y_max)

                    pt1 = (tri_y_start + tri_width // 2, tri_x_start)
                    pt2 = (tri_y_start, tri_x_start + tri_height)
                    pt3 = (tri_y_start + tri_width, tri_x_start + tri_height)

                    triangle_cnt = np.array([pt1, pt2, pt3])

                    x_start_tri = min(pt1[1], pt2[1], pt3[1])
                    x_end_tri = max(pt1[1], pt2[1], pt3[1])
                    y_start_tri = min(pt1[0], pt2[0], pt3[0])
                    y_end_tri = max(pt1[0], pt2[0], pt3[0])

                    if self.is_overlapping(x_start_tri, x_end_tri, y_start_tri, y_end_tri, used_regions):
                        continue

                    used_regions.append((x_start_tri, x_end_tri, y_start_tri, y_end_tri))

                    tri_height = x_end_tri - x_start_tri
                    tri_width = y_end_tri - y_start_tri

                    texture_crop = self.get_texture_crop(texture_img, tri_height, tri_width)

                    mask = np.zeros((tri_height, tri_width), dtype=np.uint8)
                    triangle_pts = np.array([[pt[0] - y_start_tri, pt[1] - x_start_tri] for pt in [pt1, pt2, pt3]])
                    cv2.drawContours(mask, [triangle_pts], 0, 255, -1)

                    roi = canvas[x_start_tri:x_end_tri, y_start_tri:y_end_tri]
                    roi[mask == 255] = texture_crop[mask == 255]

                    shapes_info.append({
                        'type': 'triangle',
                        'vertices': [(pt[1], pt[0]) for pt in triangle_pts],
                        'texture_label': texture,
                        'k_value': k_value,
                        'mask': (x_start_tri, x_end_tri, y_start_tri, y_end_tri, mask)
                    })

        return canvas, shapes_info



    def get_particle_region(self, pos_x, pos_y, shapes_info):
        """return the texture label for a particle"""
        for shape in shapes_info:
            if shape['type'] == 'rectangle':
                x_start, x_end, y_start, y_end = shape['mask']
                if x_start <= pos_y < x_end and y_start <= pos_x < y_end:
                    return shape['texture_label']
            elif shape['type'] == 'circle':
                x_start, x_end, y_start, y_end, mask = shape['mask']
                rel_x = int(pos_y - x_start)
                rel_y = int(pos_x - y_start)
                if 0 <= rel_x < mask.shape[0] and 0 <= rel_y < mask.shape[1]:
                    if mask[rel_x, rel_y]:
                        return shape['texture_label']
            elif shape['type'] == 'triangle':
                x_start, x_end, y_start, y_end, mask = shape['mask']
                rel_x = int(pos_y - x_start)
                rel_y = int(pos_x - y_start)
                if 0 <= rel_x < mask.shape[0] and 0 <= rel_y < mask.shape[1]:
                    if mask[rel_x, rel_y]:
                        return shape['texture_label']
        return self.background_label

    def get_spring_k(self, region1, region2):
        """edge area smoothing"""
        k1 = self.texture_to_k.get(region1, self.default_k)
        k2 = self.texture_to_k.get(region2, self.default_k)
        return (k1 + k2) / 2

    @ti.kernel
    def substep(self):
        for i in range(self.x.shape[0]):
            self.f[i] = self.gravity
            self.v[i] *= self.damping

        for i in range(self.num_springs):
            a, b = self.springs[i]
            x_a, x_b = self.x[a], self.x[b]
            dir = x_a - x_b
            length = dir.norm() + 1e-4
            force = -self.k_spring[i] * (length - self.rest_length[i]) * (dir / length)
            self.f[a] += force
            self.f[b] -= force

        for i in range(self.x.shape[0]):
            row = i // self.cols
            if row == 0:
                continue
            else:
                acc = self.f[i]
                self.v[i] += self.dt * acc
                self.x[i] += self.dt * self.v[i]

    def run_simulation(self):
        positions_over_time = []
        for step in range(self.num_steps):
            self.substep()
            if step % self.steps_per_frame == 0:
                positions_over_time.append(self.x.to_numpy())
        return np.array(positions_over_time)

    def generate_jelly_model(self, shapes_info):
        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        particle_positions = []
        particle_regions = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                pos_x = j * self.grid_spacing
                pos_y = i * self.grid_spacing
                particle_positions.append([pos_x, pos_y])
                region_label = self.get_particle_region(pos_x, pos_y, shapes_info)
                particle_regions.append(region_label)

        particle_positions = np.array(particle_positions, dtype=np.float32)
        self.x.from_numpy(particle_positions)
        self.v.fill(0)
        self.f.fill(0)

        self.reset_simulation_state(particle_positions)


        spring_indices = []
        rest_lengths = []
        k_values = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                idx = i * grid_cols + j
                region1 = particle_regions[idx]

                if j < grid_cols - 1:
                    right = idx + 1
                    region2 = particle_regions[right]
                    spring_indices.append([idx, right])
                    rest_lengths.append(self.grid_spacing)
                    k_values.append(self.get_spring_k(region1, region2))

                if i < grid_rows - 1:
                    down = idx + grid_cols
                    region2 = particle_regions[down]
                    spring_indices.append([idx, down])
                    rest_lengths.append(self.grid_spacing)
                    k_values.append(self.get_spring_k(region1, region2))

                if i < grid_rows - 1 and j < grid_cols - 1:
                    down_right = idx + grid_cols + 1
                    region2 = particle_regions[down_right]
                    spring_indices.append([idx, down_right])
                    rest_lengths.append(np.sqrt(2) * self.grid_spacing)
                    k_values.append(self.get_spring_k(region1, region2))

                if i < grid_rows - 1 and j > 0:
                    down_left = idx + grid_cols - 1
                    region2 = particle_regions[down_left]
                    spring_indices.append([idx, down_left])
                    rest_lengths.append(np.sqrt(2) * self.grid_spacing)
                    k_values.append(self.get_spring_k(region1, region2))

        spring_indices = np.array(spring_indices, dtype=np.int32)
        rest_lengths = np.array(rest_lengths, dtype=np.float32)
        k_values = np.array(k_values, dtype=np.float32)

        spring_indices = np.array(spring_indices, dtype=np.int32)
        rest_lengths = np.array(rest_lengths, dtype=np.float32)
        k_values = np.array(k_values, dtype=np.float32)

        self.springs.from_numpy(spring_indices)
        self.rest_length.from_numpy(rest_lengths)
        self.k_spring.from_numpy(k_values)

        # print(f"Generated {self.num_springs} springs (Expected: 3906)")

    def generate_dataset(self, num_samples=5, output_dir='data/generated'):
        os.makedirs(output_dir, exist_ok=True)

        for sample_idx in range(num_samples):
            self.initialize_fields()

            texture_canvas, shapes_info = self.apply_random_shapes_in_quadrants()

            sample_output_dir = os.path.join(output_dir, f"sample_{sample_idx}")
            os.makedirs(sample_output_dir, exist_ok=True)
            texture_image_path = os.path.join(sample_output_dir, "texture.png")
            cv2.imwrite(texture_image_path, cv2.cvtColor(texture_canvas, cv2.COLOR_RGB2BGR))

            self.generate_jelly_model(shapes_info)

            initial_positions = self.x.to_numpy()
            np.save(os.path.join(sample_output_dir, "initial_positions.npy"), initial_positions)
            k_values = self.k_spring.to_numpy()
            np.save(os.path.join(sample_output_dir, "k_values.npy"), k_values)

            np.save(os.path.join(sample_output_dir, "spring_indices.npy"), self.springs.to_numpy())
            np.save(os.path.join(sample_output_dir, "rest_lengths.npy"), self.rest_length.to_numpy())

            positions_over_time = self.run_simulation()
            simulation_output_dir = os.path.join(sample_output_dir, 'simulation')
            os.makedirs(simulation_output_dir, exist_ok=True)
            for t, positions in enumerate(positions_over_time):
                np.save(os.path.join(simulation_output_dir, f"positions_t{t}.npy"), positions)

            print(f"Simulation completed for sample {sample_idx}")




