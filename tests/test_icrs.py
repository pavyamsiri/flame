from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from flame.icrs import pmrapmdec_to_pmllpmbb_numpy, pmrapmdec_to_pmllpmbb_polars, radec_to_lb_numpy, radec_to_lb_polars


@given(ra=st.floats(min_value=0, max_value=2 * np.pi), dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2))
def test_radec_to_lb_numpy_vs_polars(ra: float, dec: float) -> None:
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


@given(
    pmra=st.floats(min_value=-1e9, max_value=1e9),
    pmdec=st.floats(min_value=-1e9, max_value=1e9),
    ra=st.floats(min_value=0, max_value=2 * np.pi),
    dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
)
def test_pmrapmdec_to_pmllpmbb_numpy_vs_polars(pmra: float, pmdec: float, ra: float, dec: float) -> None:
    data = pl.DataFrame(
        {
            "ra": [ra],
            "dec": [dec],
            "pmra": [pmra * np.cos(dec)],
            "pmdec": [pmdec],
        }
    )

    pmll, pmbb = pmrapmdec_to_pmllpmbb_polars(
        pmracosdec=pl.col("pmra"), pmdec=pl.col("pmdec"), ra=pl.col("ra"), dec=pl.col("dec")
    )

    data = data.select((pmll.alias("pmll"), pmbb.alias("pmbb")))

    pmracosdec_arr = np.array([pmra * np.cos(dec)], dtype=np.float64)
    pmdec_arr = np.array([pmdec], dtype=np.float64)
    ra_arr = np.array([ra], dtype=np.float64)
    dec_arr = np.array([dec], dtype=np.float64)
    pmll, pmbb = pmrapmdec_to_pmllpmbb_numpy(pmracosdec_arr, pmdec_arr, ra=ra_arr, dec=dec_arr)
    np.testing.assert_allclose(data["pmll"].to_numpy(), pmll)
    np.testing.assert_allclose(data["pmbb"].to_numpy(), pmbb)
