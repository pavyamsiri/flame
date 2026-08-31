"""Tests for the module `geometric`."""

from __future__ import annotations

import numpy as np
import polars as pl
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]
from hypothesis import given
from hypothesis import strategies as st

from flame.numpy.geometric import cartesian_to_polar as cartesian_to_polar_numpy
from flame.numpy.geometric import cartesian_vec_to_polar_vec as cartesian_vec_to_polar_vec_numpy
from flame.numpy.geometric import cartesian_to_spherical as cartesian_to_spherical_numpy
from flame.numpy.geometric import cartesian_velocity_to_spherical as cartesian_velocity_to_spherical_numpy
from flame.numpy.geometric import cylindrical_to_cartesian as cylindrical_to_cartesian_numpy
from flame.numpy.geometric import cylindrical_velocity_to_cartesian as cylindrical_velocity_to_cartesian_numpy
from flame.numpy.geometric import cartesian_to_cylindrical as cartesian_to_cylindrical_numpy
from flame.numpy.geometric import cartesian_velocity_to_cylindrical as cartesian_velocity_to_cylindrical_numpy
from flame.numpy.geometric import spherical_to_cartesian as spherical_to_cartesian_numpy
from flame.numpy.geometric import spherical_velocity_to_cartesian as spherical_velocity_to_cartesian_numpy
from flame.polars.geometric import cartesian_to_polar as cartesian_to_polar_polars
from flame.polars.geometric import cartesian_vec_to_polar_vec as cartesian_vec_to_polar_vec_polars
from flame.polars.geometric import cartesian_to_spherical as cartesian_to_spherical_polars
from flame.polars.geometric import cartesian_velocity_to_spherical as cartesian_velocity_to_spherical_polars
from flame.polars.geometric import cylindrical_to_cartesian as cylindrical_to_cartesian_polars
from flame.polars.geometric import cylindrical_velocity_to_cartesian as cylindrical_velocity_to_cartesian_polars
from flame.polars.geometric import cartesian_to_cylindrical as cartesian_to_cylindrical_polars
from flame.polars.geometric import cartesian_velocity_to_cylindrical as cartesian_velocity_to_cylindrical_polars
from flame.polars.geometric import spherical_velocity_to_cartesian as spherical_velocity_to_cartesian_polars
from flame.units import AngleArray, AngleExpr

_ABS_TOL: float = 1e-21


def test_cartesian_to_spherical_origin_uses_zero_angles() -> None:
    zeros = np.array([0.0])
    lon, lat, distance = cartesian_to_spherical_numpy(zeros, zeros, zeros)
    data = pl.DataFrame({"x": zeros, "y": zeros, "z": zeros})
    polars_actual = cartesian_to_spherical_polars(pl.col("x"), pl.col("y"), pl.col("z"))
    data = data.select(
        polars_actual[0].to_radians().alias("lon"), polars_actual[1].to_radians().alias("lat"), polars_actual[2].alias("distance")
    )

    np.testing.assert_array_equal(lon.to_radians(), zeros)
    np.testing.assert_array_equal(lat.to_radians(), zeros)
    np.testing.assert_array_equal(distance, zeros)
    np.testing.assert_array_equal(data.to_numpy().T, (zeros, zeros, zeros))


@given(
    lon=st.floats(min_value=0, max_value=2 * np.pi, allow_subnormal=False),
    lat=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2, allow_subnormal=False),
    distance=st.floats(min_value=1e-6, max_value=1e9, allow_subnormal=False),
)
def test_cartesian_to_spherical(lon: float, lat: float, distance: float) -> None:
    lon_array = AngleArray(np.array([lon]), "rad")
    lat_array = AngleArray(np.array([lat]), "rad")
    distance_array = np.array([distance])
    x, y, z = spherical_to_cartesian_numpy(lon_array, lat_array, distance_array)
    actual_lon, actual_lat, actual_distance = cartesian_to_spherical_numpy(x, y, z)

    data = pl.DataFrame({"x": x, "y": y, "z": z})
    polars_actual = cartesian_to_spherical_polars(pl.col("x"), pl.col("y"), pl.col("z"))
    data = data.select(
        polars_actual[0].to_radians().alias("lon"), polars_actual[1].to_radians().alias("lat"), polars_actual[2].alias("distance")
    )

    np.testing.assert_allclose(data["lon"].to_numpy(), actual_lon.to_radians())
    np.testing.assert_allclose(data["lat"].to_numpy(), actual_lat.to_radians())
    np.testing.assert_allclose(data["distance"].to_numpy(), actual_distance, atol=1e-12)
    np.testing.assert_allclose(actual_distance, distance_array, atol=1e-12)
    if abs(np.cos(lat)) > 1e-12:
        np.testing.assert_allclose(np.sin(actual_lon.to_radians()), np.sin(lon_array.to_radians()))
        np.testing.assert_allclose(np.cos(actual_lon.to_radians()), np.cos(lon_array.to_radians()))
        np.testing.assert_allclose(actual_lat.to_radians(), lat_array.to_radians())


@given(
    vr=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    longitude_velocity=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    latitude_velocity=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    lon=st.floats(min_value=0, max_value=2 * np.pi, allow_subnormal=False),
    lat=st.floats(min_value=-np.pi / 2 + 1e-8, max_value=np.pi / 2 - 1e-8, allow_subnormal=False),
)
def test_cartesian_velocity_to_spherical(
    vr: float, longitude_velocity: float, latitude_velocity: float, lon: float, lat: float
) -> None:
    lon_array = AngleArray(np.array([lon]), "rad")
    lat_array = AngleArray(np.array([lat]), "rad")
    vx, vy, vz = spherical_velocity_to_cartesian_numpy(
        np.array([vr]), np.array([longitude_velocity]), np.array([latitude_velocity]), lon_array, lat_array
    )
    actual = cartesian_velocity_to_spherical_numpy(vx, vy, vz, lon_array, lat_array)

    data = pl.DataFrame({"vx": vx, "vy": vy, "vz": vz, "lon": [lon], "lat": [lat]})
    polars_actual = cartesian_velocity_to_spherical_polars(
        pl.col("vx"), pl.col("vy"), pl.col("vz"), AngleExpr(pl.col("lon"), "rad"), AngleExpr(pl.col("lat"), "rad")
    )
    data = data.select(polars_actual[0].alias("vr"), polars_actual[1].alias("longitude_velocity"), polars_actual[2].alias("latitude_velocity"))

    np.testing.assert_allclose(data.to_numpy().T, actual)
    np.testing.assert_allclose(actual, (np.array([vr]), np.array([longitude_velocity]), np.array([latitude_velocity])), atol=1e-6)


@given(
    radius=st.floats(min_value=1e-6, max_value=1e9, allow_subnormal=False),
    phi=st.floats(min_value=0, max_value=2 * np.pi, allow_subnormal=False),
    z=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    v_radius=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    v_phi=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
    vz=st.floats(min_value=-1e9, max_value=1e9, allow_subnormal=False),
)
def test_cylindrical_round_trip(radius: float, phi: float, z: float, v_radius: float, v_phi: float, vz: float) -> None:
    phi_array = AngleArray(np.array([phi]), "rad")
    radius_array = np.array([radius])
    z_array = np.array([z])
    v_radius_array = np.array([v_radius])
    v_phi_array = np.array([v_phi])
    vz_array = np.array([vz])
    x, y, actual_z = cylindrical_to_cartesian_numpy(radius_array, phi_array, z_array)
    actual_radius, actual_phi, round_trip_z = cartesian_to_cylindrical_numpy(x, y, actual_z)
    vx, vy, actual_vz = cylindrical_velocity_to_cartesian_numpy(v_radius_array, v_phi_array, vz_array, phi_array)
    actual_v_radius, actual_v_phi, round_trip_vz = cartesian_velocity_to_cylindrical_numpy(vx, vy, actual_vz, phi_array)

    data = pl.DataFrame({"radius": [radius], "phi": [phi], "z": [z], "v_radius": [v_radius], "v_phi": [v_phi], "vz": [vz]})
    position = cylindrical_to_cartesian_polars(pl.col("radius"), AngleExpr(pl.col("phi"), "rad"), pl.col("z"))
    velocity = cylindrical_velocity_to_cartesian_polars(pl.col("v_radius"), pl.col("v_phi"), pl.col("vz"), AngleExpr(pl.col("phi"), "rad"))
    data = data.with_columns(position[0].alias("x"), position[1].alias("y"), position[2].alias("actual_z"), velocity[0].alias("vx"), velocity[1].alias("vy"), velocity[2].alias("actual_vz"))
    polars_position = cartesian_to_cylindrical_polars(pl.col("x"), pl.col("y"), pl.col("actual_z"))
    polars_velocity = cartesian_velocity_to_cylindrical_polars(pl.col("vx"), pl.col("vy"), pl.col("actual_vz"), AngleExpr(pl.col("phi"), "rad"))
    data = data.select(polars_position[0].alias("radius"), polars_position[1].to_radians().alias("phi"), polars_position[2].alias("z"), polars_velocity[0].alias("v_radius"), polars_velocity[1].alias("v_phi"), polars_velocity[2].alias("vz"))

    np.testing.assert_allclose(data.to_numpy().T, (actual_radius, actual_phi.to_radians(), round_trip_z, actual_v_radius, actual_v_phi, round_trip_vz))
    np.testing.assert_allclose(actual_radius, radius_array)
    np.testing.assert_allclose(round_trip_z, z_array)
    np.testing.assert_allclose((actual_v_radius, actual_v_phi, round_trip_vz), (v_radius_array, v_phi_array, vz_array), atol=1e-6)
    if radius != 0:
        np.testing.assert_allclose(np.sin(actual_phi.to_radians()), np.sin(phi_array.to_radians()))
        np.testing.assert_allclose(np.cos(actual_phi.to_radians()), np.cos(phi_array.to_radians()))


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
