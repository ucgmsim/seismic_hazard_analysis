import enum
from pathlib import Path

import geopandas as gpd
import numpy as np

from qcore import coordinates

BACKARC_JSON_FFP = Path(__file__).parent.parent.parent / "data/NSHM2022" / "backarc.json" 

class DisaggPlotType(enum.StrEnum):
    """
    Enum for the different types of disaggregation plots.
    """
    TectonicType = "tectonic_type"
    Epsilon = "epsilon"

PLOT_TYPE_COL_MAPPING = {
    DisaggPlotType.TectonicType: "tect_type",
    DisaggPlotType.Epsilon: "eps",
}

def get_backarc_mask(locs: np.ndarray):
    """
    Computes a mask identifying each location
    that requires the backarc flag based on
    wether it is inside the backarc polygon or not

    locs: array of floats
        [lon, lat]
    """
    nztm_values = coordinates.wgs_depth_to_nztm(locs[:, ::-1])[:, [1, 0]]
    site_points_df = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(nztm_values[:, 0], nztm_values[:, 1]),
            crs="EPSG:2193",
        )
    
    backarc_region = gpd.read_file(BACKARC_JSON_FFP).to_crs("EPSG:2193")
    backarc_region["backarc"] = True

    joined = gpd.sjoin(site_points_df, backarc_region, how="left", predicate="within")
    backarc_mask = joined["backarc"].fillna(False).to_numpy(dtype=bool)
    
    return backarc_mask

