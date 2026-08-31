from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]

from flame.numpy.galactic import vrpmllpmbb_to_vxvyvz as vrpmllpmbb_to_vxvyvz_numpy
from flame.numpy.galactic import vxvyvz_to_vrpmllpmbb as vxvyvz_to_vrpmllpmbb_numpy
from flame.polars.galactic import vrpmllpmbb_to_vxvyvz as vrpmllpmbb_to_vxvyvz_polars
from flame.polars.galactic import vxvyvz_to_vrpmllpmbb as vxvyvz_to_vrpmllpmbb_polars
from flame.units import AngleArray, AngleExpr


@given(
    vr=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    pmll=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    pmbb=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    lon=st.floats(min_value=0, max_value=2 * np.pi, allow_subnormal=False),
    lat=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2, allow_subnormal=False),
    distance=st.floats(min_value=0, max_value=1e9, allow_subnormal=False),
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
        pl.col("vr"),
        pl.col("pmll"),
        pl.col("pmbb"),
        AngleExpr(pl.col("lon"), "rad"),
        AngleExpr(pl.col("lat"), "rad"),
        pl.col("distance"),
    )

    data = data.with_columns((vx_expr.alias("vx"), vy_expr.alias("vy"), vz_expr.alias("vz")))
    vr_arr = np.array([vr], dtype=np.float64)
    pmll_arr = np.array([pmll * np.cos(lat)], dtype=np.float64)
    pmbb_arr = np.array([pmbb], dtype=np.float64)
    lon_arr = np.array([lon], dtype=np.float64)
    lat_arr = np.array([lat], dtype=np.float64)
    distance_arr = np.array([distance], dtype=np.float64)
    fl_vx, fl_vy, fl_vz = vrpmllpmbb_to_vxvyvz_numpy(
        vr_arr,
        pmll_arr,
        pmbb_arr,
        AngleArray(lon_arr, "rad"),
        AngleArray(lat_arr, "rad"),
        distance_arr,
    )

    gl_v = galcoords.vrpmllpmbb_to_vxvyvz(vr_arr, pmll_arr, pmbb_arr, lon_arr, lat_arr, distance_arr, degree=False)
    gl_vx = gl_v[:, 0]
    gl_vy = gl_v[:, 1]
    gl_vz = gl_v[:, 2]

    np.testing.assert_allclose(data["vx"].to_numpy(), fl_vx)
    np.testing.assert_allclose(data["vy"].to_numpy(), fl_vy)
    np.testing.assert_allclose(data["vz"].to_numpy(), fl_vz)
    np.testing.assert_allclose(data["vx"].to_numpy(), gl_vx)
    np.testing.assert_allclose(data["vy"].to_numpy(), gl_vy)
    np.testing.assert_allclose(data["vz"].to_numpy(), gl_vz)
    np.testing.assert_allclose(fl_vx, gl_vx)
    np.testing.assert_allclose(fl_vy, gl_vy)
    np.testing.assert_allclose(fl_vz, gl_vz)


@given(
    vr=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    pmll=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    pmbb=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    lon=st.floats(min_value=0, max_value=2 * np.pi, allow_subnormal=False),
    lat=st.floats(min_value=-np.pi / 2 + 1e-8, max_value=np.pi / 2 - 1e-8, allow_subnormal=False),
    distance=st.floats(min_value=1e-2, max_value=1e3, allow_subnormal=False),
)
def test_vxvyvz_to_vrpmllpmbb(vr: float, pmll: float, pmbb: float, lon: float, lat: float, distance: float) -> None:
    lon_array = AngleArray(np.array([lon]), "rad")
    lat_array = AngleArray(np.array([lat]), "rad")
    expected = (np.array([vr]), np.array([pmll]), np.array([pmbb]))
    vx, vy, vz = vrpmllpmbb_to_vxvyvz_numpy(*expected, lon_array, lat_array, np.array([distance]))
    actual = vxvyvz_to_vrpmllpmbb_numpy(vx, vy, vz, lon_array, lat_array, np.array([distance]))

    data = pl.DataFrame({"vx": vx, "vy": vy, "vz": vz, "lon": [lon], "lat": [lat], "distance": [distance]})
    polars_actual = vxvyvz_to_vrpmllpmbb_polars(
        pl.col("vx"), pl.col("vy"), pl.col("vz"), AngleExpr(pl.col("lon"), "rad"), AngleExpr(pl.col("lat"), "rad"), pl.col("distance")
    )
    data = data.select(polars_actual[0].alias("vr"), polars_actual[1].alias("pmll"), polars_actual[2].alias("pmbb"))

    np.testing.assert_allclose(data.to_numpy().T, actual)
    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_vxvyvz_to_vrpmllpmbb_zero_distance_has_undefined_proper_motion() -> None:
    zero = np.array([0.0])
    lon = AngleArray(zero, "rad")
    lat = AngleArray(zero, "rad")
    vr, pmll, pmbb = vxvyvz_to_vrpmllpmbb_numpy(zero, zero, zero, lon, lat, zero)

    assert vr[0] == 0.0
    assert np.isnan(pmll[0])
    assert np.isnan(pmbb[0])
