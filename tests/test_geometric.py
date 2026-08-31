"""Tests for the module `geometric`."""

from __future__ import annotations

import numpy as np
import polars as pl
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]
from hypothesis import given
from hypothesis import strategies as st

from flame.numpy.geometric import cartesian_to_polar as cartesian_to_polar_numpy
from flame.numpy.geometric import cartesian_vec_to_polar_vec as cartesian_vec_to_polar_vec_numpy
from flame.numpy.geometric import spherical_velocity_to_cartesian as spherical_velocity_to_cartesian_numpy
from flame.polars.geometric import cartesian_to_polar as cartesian_to_polar_polars
from flame.polars.geometric import cartesian_vec_to_polar_vec as cartesian_vec_to_polar_vec_polars
from flame.polars.geometric import spherical_velocity_to_cartesian as spherical_velocity_to_cartesian_polars
from flame.units import AngleArray, AngleExpr

_ABS_TOL: float = 1e-21


@given(
    x=st.floats(min_value=-1e9, max_value=1e9),
    y=st.floats(min_value=-1e9, max_value=1e9),
)
def test_cartesian_to_polar(x: float, y: float) -> None:
    data = pl.DataFrame(
        {
            "x": [x],
            "y": [y],
        }
    )

    r_expr, phi_expr = cartesian_to_polar_polars(pl.col("x"), pl.col("y"))

    data = data.with_columns((r_expr.alias("vr"), phi_expr.to_radians().alias("vphi")))
    x_arr = np.array([x], dtype=np.float64)
    y_arr = np.array([y], dtype=np.float64)
    fl_r, fl_phi = cartesian_to_polar_numpy(x_arr, y_arr)

    zero_arr = np.zeros(1, dtype=np.float64)
    gal_r, gal_phi, _ = galcoords.rect_to_cyl(x_arr, y_arr, zero_arr)

    np.testing.assert_allclose(data["vr"].to_numpy(), fl_r, atol=_ABS_TOL)
    np.testing.assert_allclose(data["vphi"].to_numpy(), fl_phi.to_radians(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["vr"].to_numpy(), gal_r, atol=_ABS_TOL)
    np.testing.assert_allclose(data["vphi"].to_numpy(), gal_phi, atol=_ABS_TOL)
    np.testing.assert_allclose(fl_r, gal_r, atol=_ABS_TOL)
    np.testing.assert_allclose(fl_phi.to_radians(), gal_phi, atol=_ABS_TOL)


@given(
    vx=st.floats(min_value=-1e9, max_value=1e9),
    vy=st.floats(min_value=-1e9, max_value=1e9),
    phi=st.floats(min_value=0, max_value=2 * np.pi),
)
def test_cartesian_vec_to_polar_vec(vx: float, vy: float, phi: float) -> None:
    data = pl.DataFrame(
        {
            "vx": [vx],
            "vy": [vy],
            "phi": [phi],
        }
    )

    vr_expr, vphi_expr = cartesian_vec_to_polar_vec_polars(pl.col("vx"), pl.col("vy"), AngleExpr(pl.col("phi"), "rad"))

    data = data.with_columns((vr_expr.alias("vr"), vphi_expr.alias("vphi")))
    vx_arr = np.array([vx], dtype=np.float64)
    vy_arr = np.array([vy], dtype=np.float64)
    phi_arr = np.array([phi], dtype=np.float64)
    fl_vr, fl_vphi = cartesian_vec_to_polar_vec_numpy(vx_arr, vy_arr, AngleArray(phi_arr, "rad"))

    zero_arr = np.zeros(1, dtype=np.float64)
    gal_vr, gal_vphi, _ = galcoords.rect_to_cyl_vec(vx_arr, vy_arr, zero_arr, zero_arr, phi_arr, zero_arr, cyl=True)

    np.testing.assert_allclose(data["vr"].to_numpy(), fl_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), fl_vphi)
    np.testing.assert_allclose(data["vr"].to_numpy(), gal_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), gal_vphi)
    np.testing.assert_allclose(fl_vr, gal_vr)
    np.testing.assert_allclose(fl_vphi, gal_vphi)


@given(
    radial_velocity=st.floats(min_value=-1e9, max_value=1e9),
    longitude_velocity=st.floats(min_value=-1e9, max_value=1e9),
    latitude_velocity=st.floats(min_value=-1e9, max_value=1e9),
    lon=st.floats(min_value=0, max_value=2 * np.pi),
    lat=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
)
def test_spherical_velocity_to_cartesian(
    radial_velocity: float, longitude_velocity: float, latitude_velocity: float, lon: float, lat: float
) -> None:
    data = pl.DataFrame(
        {
            "radial_velocity": [radial_velocity],
            "longitude_velocity": [longitude_velocity],
            "latitude_velocity": [latitude_velocity],
            "lon": [lon],
            "lat": [lat],
        }
    )
    vx_expr, vy_expr, vz_expr = spherical_velocity_to_cartesian_polars(
        pl.col("radial_velocity"),
        pl.col("longitude_velocity"),
        pl.col("latitude_velocity"),
        AngleExpr(pl.col("lon"), "rad"),
        AngleExpr(pl.col("lat"), "rad"),
    )
    data = data.with_columns(vx_expr.alias("vx"), vy_expr.alias("vy"), vz_expr.alias("vz"))

    radial_velocity_arr = np.array([radial_velocity], dtype=np.float64)
    longitude_velocity_arr = np.array([longitude_velocity], dtype=np.float64)
    latitude_velocity_arr = np.array([latitude_velocity], dtype=np.float64)
    lon_arr = np.array([lon], dtype=np.float64)
    lat_arr = np.array([lat], dtype=np.float64)
    vx, vy, vz = spherical_velocity_to_cartesian_numpy(
        radial_velocity_arr,
        longitude_velocity_arr,
        latitude_velocity_arr,
        AngleArray(lon_arr, "rad"),
        AngleArray(lat_arr, "rad"),
    )

    expected_vx = radial_velocity * np.cos(lon) * np.cos(lat) - longitude_velocity * np.sin(lon) - latitude_velocity * np.cos(lon) * np.sin(lat)
    expected_vy = radial_velocity * np.sin(lon) * np.cos(lat) + longitude_velocity * np.cos(lon) - latitude_velocity * np.sin(lon) * np.sin(lat)
    expected_vz = radial_velocity * np.sin(lat) + latitude_velocity * np.cos(lat)

    np.testing.assert_allclose(data["vx"], vx)
    np.testing.assert_allclose(data["vy"], vy)
    np.testing.assert_allclose(data["vz"], vz)
    np.testing.assert_allclose(vx, expected_vx)
    np.testing.assert_allclose(vy, expected_vy)
    np.testing.assert_allclose(vz, expected_vz)
