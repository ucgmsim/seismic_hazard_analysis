import io
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import colors as mcolors

from nshmdb.nshmdb import NSHMDB
from qcore import geo

try:
    from pygmt_helper import plots, plotting

    HAS_PYGMT = True
except ImportError:
    plotting, plots = None, None
    HAS_PYGMT = False

from . import utils


def context_plot(
    site_lon: float,
    site_lat: float,
    region_expanse: float,
    nshm_source_db_ffp: Path,
    nzgmdb_event_table_ffp: Path,
    out_ffp: Path,
):
    """
    For a given site, shows the context, i.e. nearby historic
    events and source geometries based on the 2022 NSHM source model.

    Parameters
    ----------
    site_lon : float
        Longitude of the site.
    site_lat : float
        Latitude of the site.
    region_expanse : float
        The size of the region to plot around the site in km.
        E.g. if set to 100km, the region is 100km to the
        north, south, east, and west of the site,
        with the site in the center.
    nshm_source_db_ffp : Path
        Path to the NSHM source database file.
    nzgmdb_event_table_ffp : Path
        Path to the NZGMDB event table CSV file.
    out_ffp : Path
        Path to save the output figure.
    """
    if not HAS_PYGMT:
        raise ImportError(
            "pygmt_helper is not installed. Please install it to use this function."
        )

    db = NSHMDB(nshm_source_db_ffp)
    event_df = pd.read_csv(
        nzgmdb_event_table_ffp, dtype={"evid": str}, index_col="evid"
    )

    # Drop events with magnitude less than 4
    event_df = event_df.loc[event_df.mag >= 4]

    # Split historic events into magnitude bins
    mag_bins = [4, 5, 6, 9]
    labels = ["blue", "orange", "red"]
    event_df.loc[:, "mag_bin"] = pd.cut(
        event_df.mag, bins=mag_bins, include_lowest=True, labels=labels
    )

    # Create the fault objects
    fault_ids, faults = db.get_fault_ids(), {}
    for cur_id in fault_ids:
        faults[cur_id] = db.get_fault(cur_id)

    # Define the region based on the given site
    max_lat, _ = geo.ll_shift(site_lat, site_lon, region_expanse, 0)
    min_lat, _ = geo.ll_shift(site_lat, site_lon, region_expanse, 180)
    _, max_lon = geo.ll_shift(site_lat, site_lon, region_expanse, 90)
    _, min_lon = geo.ll_shift(site_lat, site_lon, region_expanse, 270)
    region = (min_lon, max_lon, min_lat, max_lat)

    # Load the mapd data

    # Create the figure
    fig = plotting.gen_region_fig(
        region=region,
        plot_kwargs={
            "topo_cmap": "geo",
            "topo_cmap_inc": 25,
            "topo_cmap_max": 3500,
            "topo_cmap_min": -3500,
            "topo_cmap_reverse": False,
            "topo_cmap_continous": True,
        },
        config_options=dict(
            MAP_FRAME_TYPE="plain",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="WSne",
        ),
        plot_roads=True,
        high_res_topo=True,
    )

    # Plot the source geometries
    for cur_fault in faults.values():
        for cur_plane in cur_fault.planes:
            corners = cur_plane.corners
            fig.plot(
                x=corners[:, 1].tolist() + [corners[0, 1]],
                y=corners[:, 0].tolist() + [corners[0, 0]],
                pen="0.5p",
            )
            fig.plot(
                x=corners[[0, 1], 1],
                y=corners[[0, 1], 0],
                pen="1p",
            )

    # Plot the historic events
    for i, (_, cur_row) in enumerate(event_df.sort_values("mag").iterrows()):
        fig.meca(
            spec={
                "strike": cur_row.strike,
                "dip": cur_row.dip,
                "rake": cur_row.rake,
                "magnitude": cur_row.mag,
                "depth": cur_row.depth,
            },
            scale=f"{0.06 * cur_row.mag}c",
            longitude=cur_row.lon,
            latitude=cur_row.lat,
            pen="0.05p,black,solid",
            compressionfill=cur_row.mag_bin,
        )

    # Plot the site
    fig.plot(
        x=site_lon,
        y=site_lat,
        style="c0.15c",
        pen="0.5p,white",
        fill="black",
    )
    fig.text(
        x=site_lon,
        y=site_lat - 0.04,
        text="Site",
        justify="BC",
        fill="white",
        # transparency=25,
        font="8p,Helvetica,black",
    )

    legend_spec = io.StringIO()
    legend_spec.write("H 12p,Helvetica-Bold Historic Events\n")
    legend_spec.write("D 0.1i 1p\n")
    legend_spec.write("S 0.1i c 0.15c blue 0.05p,black 0.4i Magnitude 4.0-5.0\n")
    legend_spec.write("S 0.1i c 0.20c orange 0.05p,black 0.4i Magnitude 5.0-6.0\n")
    legend_spec.write("S 0.1i c 0.25c red 0.05p,black 0.4i Magnitude 6.0+\n")

    fig.legend(
        spec=legend_spec,
        box="+gwhite+p1p",
    )

    fig.savefig(
        str(out_ffp),
        dpi=900,
        anti_alias=True,
    )


def disagg_plot(
    disagg: xr.DataArray,
    plot_type: utils.DisaggPlotType,
    out_ffp: Path | None = None,
    verbose: bool = True,
    mag_rounding: bool = True,
):
    """
    Creates a disaggregation 3D bar plot

    Parameters
    ----------
    disagg : xr.DataArray
        The disaggregation data to plot.
        It should have dimensions ['mag', 'dist']
        + ['tect_type'] or ['eps'],
        depending on the `plot_type`.
    plot_type : utils.DisaggPlotType
        The type of disaggregation plot to create.
    out_ffp : Path
        The file path where the plot will be saved.
    verbose : bool, default=True
        If `True`, prints progress messages during plotting.
    mag_rounding : bool, default=True
        If `True`, rounds the magnitude values to one decimal place.
        This is useful when re-producing NSHM disagg results,
        as these use mag_bin_width = 0.1999
    """
    if not HAS_PYGMT:
        raise ImportError(
            "pygmt_helper is not installed. Please install it to use this function."
        )

    z_col = utils.PLOT_TYPE_COL_MAPPING[plot_type]

    disagg_df = (
        disagg.to_dataframe().reset_index().rename(columns={"disagg": "contribution"})
    )
    disagg_df["contribution"] = disagg_df["contribution"] * 100

    disagg_df = (
        disagg_df.groupby(["mag", "dist", z_col])["contribution"].sum().reset_index()
    )

    uniform_mag_bin_width = None
    # Magnitude bin edges are given
    # Can be non-uniform
    if (disagg.attrs.get("mag_bin_edges", None)) is not None:
        # mag_bin_width = np.diff(mag_bin_edges)
        raise NotImplementedError()
    # Constant magnitude bin width
    else:
        if (mag_bin_width := disagg.attrs.get("mag_bin_width", None)) is not None:
            disagg_df["mag_bin_width"] = mag_bin_width
        else:
            mag_bin_width = np.diff(np.sort(disagg_df.mag.unique()))
            assert np.allclose(
                mag_bin_width, mag_bin_width[0]
            ), "Magnitude bin widths are not uniform."
            mag_bin_width = mag_bin_width[0]

        # Handle cases where magnitude bin width is 0.1999 (or similar)
        if mag_rounding:
            min_mag = np.round(disagg_df.mag.min() - mag_bin_width / 2, 1)
            max_mag = np.round(disagg_df.mag.max() + mag_bin_width / 2, 1)
            uniform_mag_bin_width = np.round(mag_bin_width, 1)
        else:
            min_mag = disagg_df.mag.min() - mag_bin_width / 2
            max_mag = disagg_df.mag.max() + mag_bin_width / 2
            uniform_mag_bin_width = mag_bin_width

    # Set the major and minor ticks for magnitude
    plot_kwargs = {}
    if uniform_mag_bin_width:
        if np.ceil((max_mag - min_mag) / uniform_mag_bin_width) > 10:
            plot_kwargs["mag_major_tick"] = uniform_mag_bin_width * 2
            plot_kwargs["mag_minor_tick"] = uniform_mag_bin_width
        else:
            plot_kwargs["mag_major_tick"] = uniform_mag_bin_width
            plot_kwargs["mag_minor_tick"] = uniform_mag_bin_width / 2

    uniform_dist_bin_width = None
    # Distance bin edges are given
    # Can be non-uniform
    if (dist_bin_edges := disagg.attrs.get("dist_bin_edges", None)) is not None:
        dist_bin_width = np.diff(dist_bin_edges)
        dist_bin_centres = (dist_bin_edges[:-1] + dist_bin_edges[1:]) / 2
        assert np.allclose(dist_bin_centres, disagg_df.dist.unique())
        disagg_df["dist_bin_width"] = disagg_df["dist"].map(
            dict(zip(dist_bin_centres, dist_bin_width))
        )

        min_dist, max_dist = dist_bin_edges[0], dist_bin_edges[-1]

        # Check if distance bin widths are uniform
        if np.allclose(dist_bin_width, dist_bin_width[0]):
            uniform_dist_bin_width = dist_bin_width[0]
            
    # Constant distance bin width
    else:
        if (dist_bin_width := disagg.attrs.get("dist_bin_width", None)) is not None:
            disagg_df["dist_bin_width"] = dist_bin_width
        else:
            dist_bin_width = np.diff(np.sort(disagg_df.dist.unique()))
            assert np.allclose(
                dist_bin_width, dist_bin_width[0]
            ), "Distance bin widths are not uniform."
            dist_bin_width = dist_bin_width.item()

        min_dist = disagg_df.dist.min() - dist_bin_width / 2
        max_dist = disagg_df.dist.max() + dist_bin_width / 2

        uniform_dist_bin_width = dist_bin_width

    # Set the major and minor ticks for distance
    if uniform_dist_bin_width:
        if np.ceil((max_dist - min_dist) / uniform_dist_bin_width) > 10:
            plot_kwargs["dist_major_tick"] = dist_bin_width[0] * 2
            plot_kwargs["dist_minor_tick"] = dist_bin_width[0] 
        else:
            plot_kwargs["dist_major_tick"] = dist_bin_width[0]
            plot_kwargs["dist_minor_tick"] = dist_bin_width[0] / 2

    if plot_type == utils.DisaggPlotType.TectonicType:
        category_specs = {
            "Active Shallow Crust": (None, "blue"),
            "Subduction Interface": (None, "orange"),
            "Subduction Intraslab": (None, "red"),
        }
    else:
        # User specified epsilon bins
        if (eps_bin_edges := disagg.attrs.get("eps_bin_edges", None)) is not None:
            assert np.all(np.sort(eps_bin_edges) == eps_bin_edges)
            eps_bin_centres = np.sort(disagg_df.eps.unique())
            assert eps_bin_edges.size == eps_bin_centres.size + 1
            category_colors = sns.color_palette("bwr", n_colors=eps_bin_centres.size)
            category_specs = {cur_eps: (_get_epsilon_legend_entry(i, eps_bin_edges), mcolors.to_hex(category_colors[i])) for i, cur_eps in enumerate(eps_bin_centres)}
        else:
            raise NotImplementedError()


    return plots.disagg_plot(
        disagg_df,
        (min_dist, max_dist, min_mag, max_mag),
        plots.DisaggPlotType(plot_type),
        z_col,
        category_specs,
        output_ffp=out_ffp,
        verbose=verbose,
        plot_kwargs=plot_kwargs,
    )

def _get_epsilon_legend_entry(i: int, eps_bin_edges: np.ndarray) -> str:
    """
    Returns a legend entry for the given epsilon bin index.
    """
    return f"@[{eps_bin_edges[i]:5.2f} < \\epsilon < {eps_bin_edges[i+1]:5.2f}@["