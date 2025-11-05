from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from flame.icrs import radec_to_lb_numpy, radec_to_lb_polars


@given(ra=st.floats(min_value=0, max_value=2 * np.pi), dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2))
def test_numpy_vs_polars(ra: float, dec: float) -> None:
    data = pl.DataFrame(
        {
            "ra": [ra],
            "dec": [dec],
        }
    )

    lon, lat = radec_to_lb_polars(pl.col("ra"), pl.col("dec"))

    data = data.with_columns((lon.alias("lon"), lat.alias("lat")))
    ra_arr = np.array([ra], dtype=np.float64)
    dec_arr = np.array([dec], dtype=np.float64)
    lon, lat = radec_to_lb_numpy(ra_arr, dec_arr)
    np.testing.assert_allclose(data["lon"].to_numpy(), lon)
    np.testing.assert_allclose(data["lat"].to_numpy(), lat)
