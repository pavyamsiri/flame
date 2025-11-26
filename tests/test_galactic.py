from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from flame.galactic import vrpmllpmbb_to_vxvyvz_numpy, vrpmllpmbb_to_vxvyvz_polars


@given(
    vr=st.floats(min_value=-1e9, max_value=1e9),
    pmll=st.floats(min_value=-1e9, max_value=1e9),
    pmbb=st.floats(min_value=-1e9, max_value=1e9),
    lon=st.floats(min_value=0, max_value=2 * np.pi),
    lat=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
    distance=st.floats(min_value=0, max_value=1e9),
)
def test_vrpmllpmbb_to_vxvyvz(vr: float, pmll: float, pmbb: float, lon: float, lat: float, distance: float) -> None:
    data = pl.DataFrame(
        {
            "vr": [vr],
            "pmll": [pmll * np.cos(lat)],
            "pmbb": [pmbb],
            "lon": [lon],
            "lat": [lat],
            "distance": [distance],
        }
    )

    vx_expr, vy_expr, vz_expr = vrpmllpmbb_to_vxvyvz_polars(
        pl.col("vr"), pl.col("pmll"), pl.col("pmbb"), pl.col("lon"), pl.col("lat"), pl.col("distance")
    )

    data = data.with_columns((vx_expr.alias("vx"), vy_expr.alias("vy"), vz_expr.alias("vz")))
    vr_arr = np.array([vr], dtype=np.float64)
    pmll_arr = np.array([pmll * np.cos(lat)], dtype=np.float64)
    pmbb_arr = np.array([pmbb], dtype=np.float64)
    lon_arr = np.array([lon], dtype=np.float64)
    lat_arr = np.array([lat], dtype=np.float64)
    distance_arr = np.array([distance], dtype=np.float64)
    vx, vy, vz = vrpmllpmbb_to_vxvyvz_numpy(vr_arr, pmll_arr, pmbb_arr, lon_arr, lat_arr, distance_arr)
    np.testing.assert_allclose(data["vx"].to_numpy(), vx)
    np.testing.assert_allclose(data["vy"].to_numpy(), vy)
    np.testing.assert_allclose(data["vz"].to_numpy(), vz)
