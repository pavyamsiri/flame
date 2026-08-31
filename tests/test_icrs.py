from __future__ import annotations

import numpy as np
import polars as pl
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]
from hypothesis import given
from hypothesis import strategies as st

from flame.numpy.icrs import pmrapmdec_to_pmllpmbb as pmrapmdec_to_pmllpmbb_numpy
from flame.numpy.icrs import radec_to_lb as radec_to_lb_numpy
from flame.numpy.icrs import icrs_to_galactic_cartesian as icrs_to_galactic_cartesian_numpy
from flame.numpy.icrs import icrs_to_galactic_velocity as icrs_to_galactic_velocity_numpy
from flame.polars.icrs import pmrapmdec_to_pmllpmbb as pmrapmdec_to_pmllpmbb_polars
from flame.polars.icrs import radec_to_lb as radec_to_lb_polars
from flame.polars.icrs import icrs_to_galactic_cartesian as icrs_to_galactic_cartesian_polars
from flame.polars.icrs import icrs_to_galactic_velocity as icrs_to_galactic_velocity_polars
from flame.numpy.galactic import vrpmllpmbb_to_vxvyvz as vrpmllpmbb_to_vxvyvz_numpy
from flame.numpy.geometric import spherical_to_cartesian as spherical_to_cartesian_numpy
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
        pmracosdec=pl.col("pmra"), pmdec=pl.col("pmdec"), ra=AngleExpr(pl.col("ra"), "rad"), dec=AngleExpr(pl.col("dec"), "rad")
    )

    data = data.select((fl_pmll.alias("pmll"), fl_pmbb.alias("pmbb")))

    pmracosdec_arr = np.array([pmra * np.cos(dec)], dtype=np.float64)
    pmdec_arr = np.array([pmdec], dtype=np.float64)
    ra_arr = np.array([ra], dtype=np.float64)
    dec_arr = np.array([dec], dtype=np.float64)
    fl_pmll, fl_pmbb = pmrapmdec_to_pmllpmbb_numpy(
        pmracosdec_arr, pmdec_arr, ra=AngleArray(ra_arr, "rad"), dec=AngleArray(dec_arr, "rad")
    )

    gl_pmllpmbb = galcoords.pmrapmdec_to_pmllpmbb(pmracosdec_arr, pmdec_arr, ra=ra_arr, dec=dec_arr, degree=False, epoch=None)
    gl_pmll = gl_pmllpmbb[:, 0]
    gl_pmbb = gl_pmllpmbb[:, 1]

    np.testing.assert_allclose(data["pmll"].to_numpy(), fl_pmll)
    np.testing.assert_allclose(data["pmbb"].to_numpy(), fl_pmbb)
    np.testing.assert_allclose(data["pmll"].to_numpy(), gl_pmll)
    np.testing.assert_allclose(data["pmbb"].to_numpy(), gl_pmbb)
    np.testing.assert_allclose(fl_pmll, gl_pmll)
    np.testing.assert_allclose(fl_pmbb, gl_pmbb)


@given(
    ra=st.floats(min_value=0, max_value=2 * np.pi),
    dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
    distance=st.floats(min_value=0, max_value=1e6),
)
def test_icrs_to_galactic_cartesian(ra: float, dec: float, distance: float) -> None:
    data = pl.DataFrame({"ra": [ra], "dec": [dec], "distance": [distance]})
    polars_actual = icrs_to_galactic_cartesian_polars(
        AngleExpr(pl.col("ra"), "rad"), AngleExpr(pl.col("dec"), "rad"), pl.col("distance")
    )
    data = data.with_columns(*(expression.alias(name) for expression, name in zip(polars_actual, ("x", "y", "z"), strict=True)))

    ra_array = AngleArray(np.array([ra], dtype=np.float64), "rad")
    dec_array = AngleArray(np.array([dec], dtype=np.float64), "rad")
    distance_array = np.array([distance], dtype=np.float64)
    actual = icrs_to_galactic_cartesian_numpy(ra_array, dec_array, distance_array)
    lon, lat = radec_to_lb_numpy(ra_array, dec_array)
    expected = spherical_to_cartesian_numpy(lon, lat, distance_array)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(data.select("x", "y", "z").to_numpy().T, actual)


@given(
    pmra=st.floats(min_value=-1e6, max_value=1e6),
    pmdec=st.floats(min_value=-1e6, max_value=1e6),
    radial_velocity=st.floats(min_value=-1e6, max_value=1e6),
    ra=st.floats(min_value=0, max_value=2 * np.pi),
    dec=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
    distance=st.floats(min_value=0, max_value=1e6),
)
def test_icrs_to_galactic_velocity(
    pmra: float, pmdec: float, radial_velocity: float, ra: float, dec: float, distance: float
) -> None:
    data = pl.DataFrame(
        {
            "pmra": [pmra * np.cos(dec)],
            "pmdec": [pmdec],
            "radial_velocity": [radial_velocity],
            "ra": [ra],
            "dec": [dec],
            "distance": [distance],
        }
    )
    polars_actual = icrs_to_galactic_velocity_polars(
        pl.col("radial_velocity"),
        pl.col("pmra"),
        pl.col("pmdec"),
        pl.col("distance"),
        ra=AngleExpr(pl.col("ra"), "rad"),
        dec=AngleExpr(pl.col("dec"), "rad"),
    )
    data = data.with_columns(*(expression.alias(name) for expression, name in zip(polars_actual, ("vx", "vy", "vz"), strict=True)))

    pmra_array = np.array([pmra * np.cos(dec)], dtype=np.float64)
    pmdec_array = np.array([pmdec], dtype=np.float64)
    radial_velocity_array = np.array([radial_velocity], dtype=np.float64)
    ra_array = AngleArray(np.array([ra], dtype=np.float64), "rad")
    dec_array = AngleArray(np.array([dec], dtype=np.float64), "rad")
    distance_array = np.array([distance], dtype=np.float64)
    actual = icrs_to_galactic_velocity_numpy(
        radial_velocity_array,
        pmra_array,
        pmdec_array,
        distance_array,
        ra=ra_array,
        dec=dec_array,
    )
    lon, lat = radec_to_lb_numpy(ra_array, dec_array)
    pmll, pmbb = pmrapmdec_to_pmllpmbb_numpy(pmra_array, pmdec_array, ra=ra_array, dec=dec_array)
    expected = vrpmllpmbb_to_vxvyvz_numpy(radial_velocity_array, pmll, pmbb, lon, lat, distance_array)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(data.select("vx", "vy", "vz").to_numpy().T, actual)
