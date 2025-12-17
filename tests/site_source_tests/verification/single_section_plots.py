"""Script for creating Rx, Ry plots for a single manually specified section"""
import matplotlib.pyplot as plt
import numpy as np

import utils

# Create the rupture coordinates
segment_nztm_coords_1 = np.asarray(
    [
        [1571512.2113763038, 5179366.911408591, 0.0],
        [1571520.2113763038, 5179366.911408591, 30],
        [1571512.2113763038, 5179366.911408591 + 1e4, 0.0],
        [1571520.2113763038, 5179366.911408591 + 1e4, 30],
    ]
)

segment_nztm_coords_2 = np.asarray(
    [
        [1571512.2113763038 + 1e4, 5179366.911408591 + 2e4, 0.0],
        [1571520.2113763038 + 1e4, 5179366.911408591 + 2e4, 30],
        [1571512.2113763038, 5179366.911408591 + 1e4, 0.0],
        [1571520.2113763038, 5179366.911408591 + 1e4, 30],
    ]
)
segment_nztm_coords = np.stack((segment_nztm_coords_1, segment_nztm_coords_2), axis=-1)

# Create the mesh
x_min, x_max = (
    segment_nztm_coords[:, 0, :].ravel().min() - 3e4,
    segment_nztm_coords[:, 0, :].ravel().max() + 3e4,
)
y_min, y_max = (
    segment_nztm_coords[:, 1, :].ravel().min() - 3e4,
    segment_nztm_coords[:, 1, :].ravel().max() + 3e4,
)

p_x, p_y = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))


# Result arrays
scenario_Rx, scenario_Ry, scenario_Rrup, scenario_Rjb = utils.compute_mesh_distances(
    segment_nztm_coords, p_x, p_y, np.asarray([0]), np.asarray([0, 0])
)

# Plot the results
fig = plt.figure(figsize=(8, 8))
utils.create_distance_plot(
    segment_nztm_coords,
    p_x,
    p_y,
    scenario_Rx,
    -30,
    30,
    "coolwarm",
    plot_downdip_points=True,
    title="Rx",
    fig=fig,
)
plt.gca().set_aspect("equal")
plt.show()

fig = plt.figure(figsize=(8, 8))
utils.create_distance_plot(
    segment_nztm_coords,
    p_x,
    p_y,
    scenario_Ry,
    -30,
    30,
    "coolwarm",
    plot_downdip_points=True,
    title="Ry",
    fig=fig,
)
plt.gca().set_aspect("equal")
plt.show()

