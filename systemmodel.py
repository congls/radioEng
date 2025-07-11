import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

area_size = 2  # km x km
bs_density = 1.5
ue_density = 20

num_bs = np.random.poisson(bs_density * area_size**2)
num_ue = np.random.poisson(ue_density * area_size**2)

bs_positions = np.random.uniform(0, area_size, (num_bs, 2))
ue_positions = np.random.uniform(0, area_size, (num_ue, 2))

vor = Voronoi(bs_positions)

dist_matrix = cdist(ue_positions, bs_positions)  # (num_ue, num_bs)

nearest_bs_idx = np.argmin(dist_matrix, axis=1)
nearest_distances = dist_matrix[np.arange(num_ue), nearest_bs_idx]

radius_threshold = 0.25  # km
ccu_mask = nearest_distances <= radius_threshold
ceu_mask = ~ccu_mask

plt.figure(figsize=(10, 8))
voronoi_plot_2d(vor, show_vertices=False, line_colors='blue', line_width=1.5, show_points=False)

plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='green', marker='^', s=100, label='Base Stations (BS)')

plt.scatter(ue_positions[ccu_mask][:, 0], ue_positions[ccu_mask][:, 1], c='blue', s=15, label='CCU ')
plt.scatter(ue_positions[ceu_mask][:, 0], ue_positions[ceu_mask][:, 1], c='orange', s=15, label='CEU')

for bs in bs_positions:
    circle = plt.Circle(bs, radius_threshold, color='black', fill=False, linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)

plt.xlim(0, area_size)
plt.ylim(0, area_size)
plt.xlabel("x-coordinate (km)")
plt.ylabel("y-coordinate (km)")
plt.legend(loc='upper right')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.savefig("model.png", dpi=600, bbox_inches='tight')

plt.show()