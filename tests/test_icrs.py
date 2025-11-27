from __future__ import annotations

import numpy as np
import polars as pl
from galpy.util import coords as galcoords
from hypothesis import given
from hypothesis import strategies as st

from flame.icrs import pmrapmdec_to_pmllpmbb_numpy, pmrapmdec_to_pmllpmbb_polars, radec_to_lb_numpy, radec_to_lb_polars
from flame.units import AngleArray, AngleExpr


@given(ra=st.floats(min_value=0, max_value=2 * np.pi), dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2))
def test_radec_to_lb(ra: float, dec: float) -> None:
    data = pl.DataFrame(
        {
            "ra": [ra],
            "dec": [dec],
        }
    )

    fl_lon, fl_lat = radec_to_lb_polars(AngleExpr(pl.col("ra"), "rad"), AngleExpr(pl.col("dec"), "rad"))

    data = data.with_columns((fl_lon.to_radians().alias("lon"), fl_lat.to_radians().alias("lat")))
    ra_arr = np.array([ra], dtype=np.float64)
    dec_arr = np.array([dec], dtype=np.float64)
    fl_lon, fl_lat = radec_to_lb_numpy(AngleArray(ra_arr, "rad"), AngleArray(dec_arr, "rad"))

    gl_lonlat = galcoords.radec_to_lb(ra_arr, dec_arr, degree=False, epoch=None)
    gl_lon = gl_lonlat[:, 0]
    gl_lat = gl_lonlat[:, 1]

    np.testing.assert_allclose(data["lon"].to_numpy(), fl_lon.to_radians())
    np.testing.assert_allclose(data["lat"].to_numpy(), fl_lat.to_radians())
    np.testing.assert_allclose(data["lon"].to_numpy(), gl_lon)
    np.testing.assert_allclose(data["lat"].to_numpy(), gl_lat)
    np.testing.assert_allclose(fl_lon.to_radians(), gl_lon)
    np.testing.assert_allclose(fl_lat.to_radians(), gl_lat)


@given(
    pmra=st.floats(min_value=-1e9, max_value=1e9),
    pmdec=st.floats(min_value=-1e9, max_value=1e9),
    ra=st.floats(min_value=0, max_value=2 * np.pi),
    dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
)
def test_pmrapmdec_to_pmllpmbb(pmra: float, pmdec: float, ra: float, dec: float) -> None:
    data = pl.DataFrame(
        {
            "ra": [ra],
            "dec": [dec],
            "pmra": [pmra * np.cos(dec)],
            "pmdec": [pmdec],
        }
    )

    fl_pmll, fl_pmbb = pmrapmdec_to_pmllpmbb_polars(
        pmracosdec=pl.col("pmra"), pmdec=pl.col("pmdec"), ra=pl.col("ra"), dec=pl.col("dec")
    )

    data = data.select((fl_pmll.alias("pmll"), fl_pmbb.alias("pmbb")))

    pmracosdec_arr = np.array([pmra * np.cos(dec)], dtype=np.float64)
    pmdec_arr = np.array([pmdec], dtype=np.float64)
    ra_arr = np.array([ra], dtype=np.float64)
    dec_arr = np.array([dec], dtype=np.float64)
    fl_pmll, fl_pmbb = pmrapmdec_to_pmllpmbb_numpy(pmracosdec_arr, pmdec_arr, ra=ra_arr, dec=dec_arr)

    gl_pmllpmbb = galcoords.pmrapmdec_to_pmllpmbb(pmracosdec_arr, pmdec_arr, ra=ra_arr, dec=dec_arr, degree=False, epoch=None)
    gl_pmll = gl_pmllpmbb[:, 0]
    gl_pmbb = gl_pmllpmbb[:, 1]

    np.testing.assert_allclose(data["pmll"].to_numpy(), fl_pmll)
    np.testing.assert_allclose(data["pmbb"].to_numpy(), fl_pmbb)
    np.testing.assert_allclose(data["pmll"].to_numpy(), gl_pmll)
    np.testing.assert_allclose(data["pmbb"].to_numpy(), gl_pmbb)
    np.testing.assert_allclose(fl_pmll, gl_pmll)
    np.testing.assert_allclose(fl_pmbb, gl_pmbb)
