import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xarray as xr

import seismic_hazard_analysis as sha

app = typer.Typer()


@app.command("get-hcurve-stats")
def get_hcurve_stats(
    calc_id: int = typer.Argument(..., help="OpenQuake calculation ID"),
    n_procs: int = typer.Argument(
        mp.cpu_count(), help="Number of processes to use for parallel extraction"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Directory to save the extracted hazard curve statistics"
    ),
):
    """
    Extract hazard curve realisation from the OQ database,
    and computes mean and quantiles for each IM and IM level.
    """
    sha.nshm_2022.get_hcurves_stats(calc_id, n_procs, output_dir)


@app.command("compute-uhs")
def compute_uhs(
    hcurve_stats_dir: Path = typer.Argument(
        ..., help="Directory containing the hazard curve statistics"
    ),
    output_ffp: Path = typer.Argument(
        ..., help="File path to save the computed UHS DataFrame"
    ),
    excd_rates: list[float] = typer.Option(
        None,
        help="List of exceedance rates to compute UHS for. "
        "One of `excd_rates` or `rps` must be provided.",
    ),
    rps: list[float] = typer.Option(
        None,
        help="List of return periods to compute UHS for. "
        "One of `excd_rates` or `rps` must be provided.",
    ),
):
    """Compute the Uniform Hazard Spectrum (UHS) from the hazard curve statistics."""
    if not (excd_rates or rps):
        raise ValueError("One of `excd_rates` or `rps` must be provided.")
    if excd_rates and rps:
        raise ValueError("Only one of `excd_rates` or `rps` can be provided.")

    # Ensure we have both RPs and Exceedance rates
    excd_rates = (
        excd_rates
        if excd_rates is not None
        else [sha.utils.rp_to_prob(cur_rp) for cur_rp in rps]
    )
    rps = (
        rps
        if rps is not None
        else [int(np.round(sha.utils.prob_to_rp(cur_excd))) for cur_excd in excd_rates]
    )

    # Load the hazard curve statistics
    hcurve_stats_ffps = hcurve_stats_dir.glob("*statistics.csv")
    mean_hcurves = {
        sha.utils.reverse_im_file_format(ffp.name.rsplit("_", 1)[0]): pd.read_csv(
            ffp, index_col=0
        )["mean"].squeeze()
        for ffp in hcurve_stats_ffps
    }

    # Compute and save UHS
    uhs_df = sha.uhs.compute_uhs(mean_hcurves, excd_rates, rps=rps)
    uhs_df.to_csv(output_ffp)


@app.command("get-rps-im-values")
def get_rps_im_values(
    im: str = typer.Argument(..., help="IM name, e.g., 'pSA_0.1'"),
    hcurve_stats_dir: Path = typer.Argument(
        ..., help="Directory containing the hazard curve statistics"
    ),
    rps: list[float] = typer.Option(
        ..., help="Return periods for which to get the IM values. "
    ),
):
    """
    Get the IM values for the given return periods from the mean hazard curve.
    """
    # Load the hazard curve statistics
    hcurve_stats_ffp = (
        hcurve_stats_dir / f"{sha.utils.get_im_file_format(im)}_statistics.csv"
    )
    if not hcurve_stats_ffp.exists():
        raise FileNotFoundError(
            f"Hazard curve statistics file not found: {hcurve_stats_ffp}"
        )

    mean_hcurve = pd.read_csv(hcurve_stats_ffp, index_col=0)["mean"].squeeze()
    excd_rates = [sha.utils.rp_to_prob(cur_rp) for cur_rp in rps]

    im_values = sha.utils.exceedance_to_im(
        excd_rates,
        mean_hcurve.index.values,
        mean_hcurve.values,
    )

    for cur_rp, cur_im_value in zip(rps, im_values):
        print(f"{im} - {cur_rp} years: {cur_im_value:.6f}")


@app.command("extract-disagg")
def extract_disagg(
    calc_id: int = typer.Argument(..., help="OpenQuake calculation ID"),
    output_dir: Path = typer.Argument(
        ..., help="Directory to save the extracted disaggregation data"
    ),
    disagg_kind : str = typer.Option("TRT_Mag_Dist_Eps", help="Kind of disaggregation to extract")
):
    """Extract diaggregation from OQ database"""
    sha.nshm_2022.get_disagg_stats(calc_id, output_dir, disagg_kind=disagg_kind)


@app.command("context-plot")
def context_plot(
    site_lon: float = typer.Argument(..., help="Longitude of the site"),
    site_lat: float = typer.Argument(..., help="Latitude of the site"),
    region_expanse: float = typer.Argument(
        ..., help="Size of the region to plot around the site in km"
    ),
    nshm_source_db_ffp: Path = typer.Argument(
        ..., help="Path to the NSHM source database file"
    ),
    nzgmdb_event_table_ffp: Path = typer.Argument(
        ..., help="Path to the NZGMDB event table CSV file"
    ),
    out_ffp: Path = typer.Argument(..., help="Path to save the output figure"),
):
    """Plot context for a given site with nearby historic events and source geometries."""
    sha.nshm_2022.context_plot(
        site_lon,
        site_lat,
        region_expanse,
        nshm_source_db_ffp,
        nzgmdb_event_table_ffp,
        out_ffp,
    )


@app.command("disagg-plot")
def disagg_plot(
    disagg_results_ffp: list[Path] = typer.Argument(
        ..., help="Path to the disaggregation result netCDF files"
    ),
    plot_type: sha.nshm_2022.DisaggPlotType = typer.Argument(
        ..., help="Type of disaggregation plot to create", show_choices=True
    ),
    output_dir: Path = typer.Argument(
        ..., help="Directory to save the disaggregation plots"
    ),
):
    """Create a disaggregation 3D bar plot"""
    for ffp in disagg_results_ffp:
        sha.nshm_2022.disagg_plot(
            xr.open_dataarray(ffp),
            plot_type,
            output_dir / f"{ffp.stem}_{plot_type}.png",
        )


if __name__ == "__main__":
    app()
