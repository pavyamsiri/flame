from __future__ import annotations

import numpy as np
import polars as pl
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]
from hypothesis import given, settings
from hypothesis import strategies as st

from flame.numpy.galactic import vrpmllpmbb_to_vxvyvz as vrpmllpmbb_to_vxvyvz_numpy
from flame.numpy.geometric import spherical_to_cartesian as spherical_to_cartesian_numpy
from flame.numpy.icrs import icrs_to_galactic_cartesian as icrs_to_galactic_cartesian_numpy
from flame.numpy.icrs import icrs_to_galactic_velocity as icrs_to_galactic_velocity_numpy
from flame.numpy.icrs import lb_to_radec as lb_to_radec_numpy
from flame.numpy.icrs import pmllpmbb_to_pmrapmdec as pmllpmbb_to_pmrapmdec_numpy
from flame.numpy.icrs import pmrapmdec_to_pmllpmbb as pmrapmdec_to_pmllpmbb_numpy
from flame.numpy.icrs import radec_to_lb as radec_to_lb_numpy
from flame.polars.icrs import icrs_to_galactic_cartesian as icrs_to_galactic_cartesian_polars
from flame.polars.icrs import icrs_to_galactic_velocity as icrs_to_galactic_velocity_polars
from flame.polars.icrs import lb_to_radec as lb_to_radec_polars
from flame.polars.icrs import pmllpmbb_to_pmrapmdec as pmllpmbb_to_pmrapmdec_polars
from flame.polars.icrs import pmrapmdec_to_pmllpmbb as pmrapmdec_to_pmllpmbb_polars
from flame.polars.icrs import radec_to_lb as radec_to_lb_polars
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
    lon=st.floats(min_value=0, max_value=2 * np.pi),
    lat=st.floats(min_value=-np.pi / 2 + 1e-8, max_value=np.pi / 2 - 1e-8),
)
def test_lb_to_radec(lon: float, lat: float) -> None:
    data = pl.DataFrame({"lon": [lon], "lat": [lat]})
    polars_actual = lb_to_radec_polars(AngleExpr(pl.col("lon"), "rad"), AngleExpr(pl.col("lat"), "rad"))
    data = data.with_columns(polars_actual[0].to_radians().alias("ra"), polars_actual[1].to_radians().alias("dec"))

    lon_array = AngleArray(np.array([lon], dtype=np.float64), "rad")
    lat_array = AngleArray(np.array([lat], dtype=np.float64), "rad")
    ra, dec = lb_to_radec_numpy(lon_array, lat_array)
    galpy_ra, galpy_dec = galcoords.lb_to_radec(lon_array.to_radians(), lat_array.to_radians(), degree=False, epoch=None).T

    np.testing.assert_allclose(data["ra"].to_numpy(), ra.to_radians())
    np.testing.assert_allclose(data["dec"].to_numpy(), dec.to_radians())
    np.testing.assert_allclose(ra.to_radians(), galpy_ra)
    np.testing.assert_allclose(dec.to_radians(), galpy_dec)
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


@settings(max_examples=25, deadline=None)
@given(
    pmll=st.floats(min_value=-1e9, max_value=1e9),
    pmbb=st.floats(min_value=-1e9, max_value=1e9),
    lon=st.floats(min_value=0, max_value=2 * np.pi),
    lat=st.floats(min_value=-np.pi / 2 + 1e-8, max_value=np.pi / 2 - 1e-8),
)
def test_pmllpmbb_to_pmrapmdec(pmll: float, pmbb: float, lon: float, lat: float) -> None:
    data = pl.DataFrame({"pmll": [pmll], "pmbb": [pmbb], "lon": [lon], "lat": [lat]})
    polars_actual = pmllpmbb_to_pmrapmdec_polars(
        pl.col("pmll"), pl.col("pmbb"), lon=AngleExpr(pl.col("lon"), "rad"), lat=AngleExpr(pl.col("lat"), "rad")
    )
    data = data.select(polars_actual[0].alias("pmracosdec"), polars_actual[1].alias("pmdec"))

    pmll_array = np.array([pmll], dtype=np.float64)
    pmbb_array = np.array([pmbb], dtype=np.float64)
    lon_array = AngleArray(np.array([lon], dtype=np.float64), "rad")
    lat_array = AngleArray(np.array([lat], dtype=np.float64), "rad")
    pmracosdec, pmdec = pmllpmbb_to_pmrapmdec_numpy(pmll_array, pmbb_array, lon=lon_array, lat=lat_array)
    ra, dec = lb_to_radec_numpy(lon_array, lat_array)
    round_trip_pmll, round_trip_pmbb = pmrapmdec_to_pmllpmbb_numpy(
        pmracosdec, pmdec, ra=ra, dec=dec
    )
    galpy_pmracosdec, galpy_pmdec = galcoords.pmllpmbb_to_pmrapmdec(
        pmll_array, pmbb_array, lon_array.to_radians(), lat_array.to_radians(), degree=False, epoch=None
    ).T

    np.testing.assert_allclose(data["pmracosdec"].to_numpy(), pmracosdec)
    np.testing.assert_allclose(data["pmdec"].to_numpy(), pmdec)
    np.testing.assert_allclose(pmracosdec, galpy_pmracosdec)
    np.testing.assert_allclose(pmdec, galpy_pmdec)
    np.testing.assert_allclose(round_trip_pmll, pmll_array, atol=1e-6)
    np.testing.assert_allclose(round_trip_pmbb, pmbb_array, atol=1e-6)


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
