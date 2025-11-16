from pathlib import Path

import pandas as pd
import numpy as np

import seismic_hazard_analysis as sha

def test_backarc_mask():
    bench_site_df = pd.read_csv(
        Path(__file__).parent / "nshm2022_bench_data/bench_sites_with_backarc.csv"
    )

    backarc_mask = sha.nshm_2022.get_backarc_mask(bench_site_df[["lon", "lat"]].values)

    np.testing.assert_array_equal(bench_site_df["backarc"].values, backarc_mask)