from pathlib import Path

import pandas as pd

import seismic_hazard_analysis as sha


def main():
    site_df = pd.read_csv(Path(__file__).parent / "bench_sites.csv")
    backarc_mask = sha.nshm_2022.get_backarc_mask(site_df[["lon", "lat"]].values)

    site_df["backarc"] = backarc_mask
    site_df.to_csv(Path(__file__).parent / "bench_sites_with_backarc.csv", index=False)
    

if __name__ == "__main__":
    main()