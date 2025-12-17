import matplotlib.pyplot as plt
import numpy as np

import seismic_hazard_analysis as sha


def compute_mesh_distances(
    segment_nztm_coords: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    section_ids: np.ndarray,
    segment_section_ids: np.ndarray,
    ll: bool = False,
):
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

    # Create the result arrays
    scenario_Rx = np.full(p_x.shape, fill_value=np.nan)
    scenario_Ry = np.full(p_x.shape, fill_value=np.nan)
    scenario_Rrup = np.full(p_x.shape, fill_value=np.nan)
    scenario_Rjb = np.full(p_x.shape, fill_value=np.nan)

    segment_ry_values = np.zeros(
        (p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2])
    )
    segment_rx_values = np.zeros(
        (p_x.shape[0], p_x.shape[1], segment_nztm_coords.shape[2])
    )

    # Compute distances for each site
    for i in range(p_x.shape[0]):
        # print(f"Processing row {i}/{p_x.shape[0]}")
        for j in range(p_x.shape[1]):
            # Get the current site coordinates
            if ll:
                site_coords = np.array([p_x[i, j], p_y[i, j]])
                site_nztm = sha.site_source.site_to_nztm(site_coords)
            else:
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
            segment_rx_values[i, j, :] = segment_rx

            # Get scenario Rrup and Rjb
            scenario_Rrup[i, j] = segment_rrup.min()
            scenario_Rjb[i, j] = segment_rjb.min()

            # Compute Rx and Ry for each rupture scenario
            (
                cur_rjb,
                cur_rrup,
                cur_T,
                cur_U,
            ) = sha.site_source.compute_single_scenario_distances(
                section_ids,
                segment_nztm_coords,
                segment_strike_vec,
                segment_trace_length,
                segment_section_ids,
                segment_rjb,
                segment_rrup,
                segment_rx,
                segment_ry,
                segment_ry_origin,
            )

            scenario_Rx[i, j] = cur_T
            scenario_Ry[i, j] = cur_U

    return scenario_Rx, scenario_Ry, scenario_Rrup, scenario_Rjb


def create_distance_plot(
    segment_coords: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    z: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    plot_contours: bool = True,
    plot_downdip_points: bool = False,
    title: str = None,
    fig: plt.Figure = None,
    n_contour_lines: int = 10,
    equal_aspect: bool = True,
):
    """
    Creates a colormap scatter plot with the
    given segment coordinates and site coordinates

    Note: Plots are done in lon/lat space not NZTM
    Note II: Mainly for debugging and testing

    Parameters
    ----------
    segment_coords: np.ndarray
        The segment coordinates
        shape: [4, 2, n_segments]
    p_x: np.ndarray
        The site x-coordinates
        shape: [n_rows, n_cols]
    p_y: np.ndarray
        The site y-coordinates
        shape: [n_rows, n_cols]
    z: np.ndarray
        The z values
        shape: [n_rows, n_cols]
    vmin: float
        The minimum value for the colormap
    vmax: float
        The maximum value for the colormap
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    plt.scatter(p_x, p_y, c=z, cmap=cmap, s=2.0, vmax=vmax, vmin=vmin)
    # plt.colorbar(pad=0)

    if plot_contours:
        cs = plt.contour(p_x, p_y, z, n_contour_lines, colors="k")
        fig.gca().clabel(cs, inline=True, fontsize=10)

    for i in range(segment_coords.shape[-1]):
        plt.plot(segment_coords[::2, 0, i], segment_coords[::2, 1, i], c="b", lw=1.0)

        if plot_downdip_points:
            plt.scatter(segment_coords[1::2, 0, :], segment_coords[1::2, 1, :], c="g", s=10.0)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([p_x.min(), p_x.max()])
    plt.ylim([p_y.min(), p_y.max()])
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    if equal_aspect:
        plt.gca().set_aspect('equal')
    plt.tight_layout()

    return fig