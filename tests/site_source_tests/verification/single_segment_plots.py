"""Script for creating Rx, Ry plots for a set of manually specified segments"""
import matplotlib.pyplot as plt
import numpy as np

import seismic_hazard_analysis as sha   
import utils

# Create the rupture coordinates

# Straight North
segment_nztm_coords_1 = np.asarray(
    [
        [1571512.2113763038, 5179366.911408591, 0.0],
        [1571520.2113763038, 5179366.911408591, 30],
        [1571512.2113763038, 5179366.911408591 + 1e4, 0.0],
        [1571520.2113763038, 5179366.911408591 + 1e4, 30],
    ]
)[..., np.newaxis]

# Straight East
segment_nztm_coords_2 = np.asarray(
    [
        [1571512.2113763038, 5179366.911408591, 0.0],
        [1571512.2113763038, 5179360.911408591, 30],
        [1571512.2113763038 + 1e4, 5179366.911408591, 0.0],
        [1571512.2113763038 + 1e4, 5179360.911408591, 30],
    ]
)[..., np.newaxis]

# 45 degrees from north
segment_nztm_coords_3 = np.asarray(
    [
        [1571512.2113763038, 5179366.911408591, 0.0],
        [1571513.2113763038, 5179367.911408591, 30],
        [1571512.2113763038 + 1e4, 5179366.911408591 + 1e4, 0.0],
        [1571513.2113763038 + 1e4, 5179367.911408591 + 1e4, 30],
    ]
)[..., np.newaxis]


for segment_nztm_coords in [segment_nztm_coords_1, segment_nztm_coords_2, segment_nztm_coords_3]:
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

    # Compute segment-properties
    segment_trace_length = (
        np.linalg.norm(
            segment_nztm_coords[0, :2, :] - segment_nztm_coords[2, :2, :], axis=0
        )
        / 1e3
    )
    segment_strike, segment_strike_vec = sha.site_source.compute_segment_strike_nztm(
        segment_nztm_coords
    )

    # Result arrays
    segment_ry_values = np.zeros((p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2]))
    segment_ry_origins = np.zeros(
        (p_x.shape[0], p_x.shape[1], 2, segment_nztm_coords.shape[2])
    )
    segment_rx_values = np.zeros((p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2]))

    # Compute distances for each site
    for i in range(p_x.shape[0]):
        for j in range(p_x.shape[1]):
            # Get the current site coordinates
            site_nztm = np.array([p_x[i, j], p_y[i, j], 0])

            # Compute distances for each segment
            (
                segment_rjb,
                segment_rrup,
                segment_rx,
                segment_ry,
                segment_ry_origin,
            ) = sha.site_source.compute_segment_distances(
                segment_nztm_coords,
                segment_strike,
                segment_strike_vec,
                site_nztm,
            )

            segment_ry_values[i, j, :] = segment_ry
            segment_ry_origins[i, j, :, :] = segment_ry_origin

            segment_rx_values[i, j, :] = segment_rx


    fig = plt.figure(figsize=(8, 8))
    utils.create_distance_plot(
        segment_nztm_coords,
        p_x,
        p_y,
        segment_rx_values[:, :, 0],
        -30,
        30,
        "coolwarm",
        title="Rx",
        plot_downdip_points=True,
        fig=fig
    )
    plt.gca().set_aspect('equal')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    utils.create_distance_plot(
        segment_nztm_coords,
        p_x,
        p_y,
        segment_ry_values[:, :, 0],
        -30,
        30,
        "coolwarm",
        title="Ry",
        plot_downdip_points=True,
        fig=fig
    )
    plt.gca().set_aspect('equal')
    plt.show()

