from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from galpy.util import coords as galcoords
from hypothesis import assume, given
from hypothesis import strategies as st

from flame.galactocentric import GalactocentricFrame
from flame.geometric import spherical_to_cartesian_polars

if TYPE_CHECKING:
    from typing import Literal


_ABS_TOL: float = 1e-7
_GALPY_ABS_TOL: float = 1e-3


@st.composite
def random_frame(draw: st.DrawFn, *, max_value: float = 1e8) -> GalactocentricFrame:
    """Generate a random Galactocentric frame."""
    reasonable_number = st.floats(min_value=-abs(max_value), max_value=abs(max_value), allow_nan=False, allow_infinity=False)
    sun_x: float = draw(reasonable_number)
    sun_y: float = draw(reasonable_number)
    sun_z: float = draw(reasonable_number)
    sun_vx: float = draw(reasonable_number)
    sun_vy: float = draw(reasonable_number)
    sun_vz: float = draw(reasonable_number)
    return GalactocentricFrame(
        sun_x=sun_x,
        sun_y=sun_y,
        sun_z=sun_z,
        sun_vx=sun_vx,
        sun_vy=sun_vy,
        sun_vz=sun_vz,
    )


@st.composite
def random_frame_only_x(draw: st.DrawFn, *, max_value: float = 1e8) -> GalactocentricFrame:
    """Generate a random Galactocentric frame."""
    reasonable_number = st.floats(min_value=-abs(max_value), max_value=abs(max_value), allow_nan=False, allow_infinity=False)
    sun_x: float = draw(reasonable_number)
    sun_z: float = draw(reasonable_number)
    sun_vx: float = draw(reasonable_number)
    sun_vy: float = draw(reasonable_number)
    sun_vz: float = draw(reasonable_number)
    return GalactocentricFrame(
        sun_x=sun_x,
        sun_y=0,
        sun_z=sun_z,
        sun_vx=sun_vx,
        sun_vy=sun_vy,
        sun_vz=sun_vz,
    )


@given(
    frame=random_frame(),
    u=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    v=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    w=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    handedness=st.sampled_from(["right", "left"]),
)
def test_gl_xyz_to_gc_xyz(frame: GalactocentricFrame, u: float, v: float, w: float, handedness: Literal["right", "left"]) -> None:
    gl_x = np.full(1, u)
    gl_y = np.full(1, v)
    gl_z = np.full(1, w)
    data = pl.DataFrame(
        {
            "gl_x": gl_x,
            "gl_y": gl_y,
            "gl_z": gl_z,
        }
    )

    np_x, np_y, np_z = frame.gl_xyz_to_gc_xyz_numpy(gl_x, gl_y, gl_z, handedness=handedness)
    pl_x_expr, pl_y_expr, pl_z_expr = frame.gl_xyz_to_gc_xyz_polars(
        pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"), handedness=handedness
    )

    data = data.with_columns(
        pl_x_expr.alias("gc_x"),
        pl_y_expr.alias("gc_y"),
        pl_z_expr.alias("gc_z"),
    )

    np.testing.assert_allclose(data["gc_x"], np_x, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_y"], np_y, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_z"], np_z, atol=_ABS_TOL)


@given(
    frame=random_frame(),
    x=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    y=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    z=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    handedness=st.sampled_from(["right", "left"]),
)
def test_gc_xyz_to_gl_xyz(frame: GalactocentricFrame, x: float, y: float, z: float, handedness: Literal["right", "left"]) -> None:
    gc_x = np.full(1, x)
    gc_y = np.full(1, y)
    gc_z = np.full(1, z)
    data = pl.DataFrame(
        {
            "gc_x": gc_x,
            "gc_y": gc_y,
            "gc_z": gc_z,
        }
    )

    np_x, np_y, np_z = frame.gc_xyz_to_gl_xyz_numpy(gc_x, gc_y, gc_z, handedness=handedness)
    pl_x_expr, pl_y_expr, pl_z_expr = frame.gc_xyz_to_gl_xyz_polars(
        pl.col("gc_x"), pl.col("gc_y"), pl.col("gc_z"), handedness=handedness
    )

    data = data.with_columns(
        pl_x_expr.alias("gl_x"),
        pl_y_expr.alias("gl_y"),
        pl_z_expr.alias("gl_z"),
    )

    np.testing.assert_allclose(data["gl_x"], np_x, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gl_y"], np_y, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gl_z"], np_z, atol=_ABS_TOL)


@given(
    frame=random_frame(),
    v_u=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    v_v=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    v_w=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    handedness=st.sampled_from(["right", "left"]),
)
def test_gl_vxvyvz_to_gc_vxvyvz(
    frame: GalactocentricFrame, v_u: float, v_v: float, v_w: float, handedness: Literal["right", "left"]
) -> None:
    gl_vx = np.full(1, v_u)
    gl_vy = np.full(1, v_v)
    gl_vz = np.full(1, v_w)
    data = pl.DataFrame(
        {
            "gl_vx": gl_vx,
            "gl_vy": gl_vy,
            "gl_vz": gl_vz,
        }
    )

    np_vx, np_vy, np_vz = frame.gl_vxvyvz_to_gc_vxvyvz_numpy(gl_vx, gl_vy, gl_vz, handedness=handedness)
    pl_vx_expr, pl_vy_expr, pl_vz_expr = frame.gl_vxvyvz_to_gc_vxvyvz_polars(
        pl.col("gl_vx"), pl.col("gl_vy"), pl.col("gl_vz"), handedness=handedness
    )

    data = data.with_columns(
        pl_vx_expr.alias("gc_vx"),
        pl_vy_expr.alias("gc_vy"),
        pl_vz_expr.alias("gc_vz"),
    )

    np.testing.assert_allclose(data["gc_vx"], np_vx, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vy"], np_vy, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vz"], np_vz, atol=_ABS_TOL)


@given(
    frame=random_frame(),
    vx=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    vy=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    vz=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    handedness=st.sampled_from(["right", "left"]),
)
def test_gc_vxvyvz_to_gl_vxvyvz(
    frame: GalactocentricFrame, vx: float, vy: float, vz: float, handedness: Literal["right", "left"]
) -> None:
    gc_vx = np.full(1, vx)
    gc_vy = np.full(1, vy)
    gc_vz = np.full(1, vz)
    data = pl.DataFrame(
        {
            "gc_vx": gc_vx,
            "gc_vy": gc_vy,
            "gc_vz": gc_vz,
        }
    )

    np_vx, np_vy, np_vz = frame.gc_vxvyvz_to_gl_vxvyvz_numpy(gc_vx, gc_vy, gc_vz, handedness=handedness)
    pl_vx_expr, pl_vy_expr, pl_vz_expr = frame.gc_vxvyvz_to_gl_vxvyvz_polars(
        pl.col("gc_vx"), pl.col("gc_vy"), pl.col("gc_vz"), handedness=handedness
    )

    data = data.with_columns(
        pl_vx_expr.alias("gl_vx"),
        pl_vy_expr.alias("gl_vy"),
        pl_vz_expr.alias("gl_vz"),
    )

    np.testing.assert_allclose(data["gl_vx"], np_vx, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gl_vy"], np_vy, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gl_vz"], np_vz, atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_sun_numpy(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that the Sun in the Galactic frame is at the proper coordinates in the Galactocentric frame."""
    zero_arr = np.array([0], dtype=np.float64)
    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz_numpy(zero_arr, zero_arr, zero_arr, handedness=handedness)
    gc_vx, gc_vy, gc_vz = frame.gl_vxvyvz_to_gc_vxvyvz_numpy(zero_arr, zero_arr, zero_arr, handedness=handedness)

    np.testing.assert_allclose(gc_x[0], frame.sun_x(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_y[0], frame.sun_y(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_z[0], frame.sun_z(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vx[0], frame.sun_vx(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vy[0], frame.sun_vy(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vz[0], frame.sun_vz(), atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_sun_polars(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that the Sun in the Galactic frame is at the proper coordinates in the Galactocentric frame."""
    gc_x_expr, gc_y_expr, gc_z_expr = frame.gl_xyz_to_gc_xyz_polars(
        pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"), handedness=handedness
    )
    gc_vx_expr, gc_vy_expr, gc_vz_expr = frame.gl_vxvyvz_to_gc_vxvyvz_polars(
        pl.col("gl_vx"), pl.col("gl_vy"), pl.col("gl_vz"), handedness=handedness
    )

    data = pl.DataFrame(
        {
            "gl_x": 0,
            "gl_y": 0,
            "gl_z": 0,
            "gl_vx": 0,
            "gl_vy": 0,
            "gl_vz": 0,
        }
    )

    data = data.with_columns(
        gc_x_expr.alias("gc_x"),
        gc_y_expr.alias("gc_y"),
        gc_z_expr.alias("gc_z"),
        gc_vx_expr.alias("gc_vx"),
        gc_vy_expr.alias("gc_vy"),
        gc_vz_expr.alias("gc_vz"),
    )

    np.testing.assert_allclose(data["gc_x"], frame.sun_x(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_y"], frame.sun_y(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_z"], frame.sun_z(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vx"], frame.sun_vx(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vy"], frame.sun_vy(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vz"], frame.sun_vz(), atol=_ABS_TOL)


@given(
    frame=random_frame(max_value=1e5),
    lat=st.floats(min_value=0, max_value=90),
    lon=st.floats(min_value=0, max_value=360),
    d1=st.floats(min_value=0, max_value=1e3),
    d2=st.floats(min_value=0, max_value=1e3),
    handedness=st.sampled_from(["right", "left"]),
)
def test_collinear_numpy(
    frame: GalactocentricFrame, lat: float, lon: float, d1: float, d2: float, handedness: Literal["right", "left"]
) -> None:
    """Check that points with the same longitude and latitude are collinear with the Sun."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    dist = np.array([0, d1, d2], dtype=np.float64)
    gal_x = dist * np.cos(lat) * np.cos(lon)
    gal_y = dist * np.cos(lat) * np.sin(lon)
    gal_z = dist * np.sin(lat)

    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz_numpy(gal_x, gal_y, gal_z, handedness=handedness)

    # They should be collinear with the Sun
    vec1 = np.array([gc_x[1] - gc_x[0], gc_y[1] - gc_y[0], gc_z[1] - gc_z[0]])
    vec2 = np.array([gc_x[2] - gc_x[0], gc_y[2] - gc_y[0], gc_z[2] - gc_z[0]])

    # Parallel vector via zero cross product
    cross_product = np.cross(vec1, vec2)
    np.testing.assert_allclose(cross_product, 0, atol=_ABS_TOL)


@given(
    frame=random_frame(max_value=1e5),
    lat=st.floats(min_value=0, max_value=90),
    lon=st.floats(min_value=0, max_value=360),
    d1=st.floats(min_value=0, max_value=1e3),
    d2=st.floats(min_value=0, max_value=1e3),
    handedness=st.sampled_from(["right", "left"]),
)
def test_collinear_polars(
    frame: GalactocentricFrame, lat: float, lon: float, d1: float, d2: float, handedness: Literal["right", "left"]
) -> None:
    """Check that points with the same longitude and latitude are collinear with the Sun."""
    gl_x_expr, gl_y_expr, gl_z_expr = spherical_to_cartesian_polars(
        pl.col("lon").radians(), pl.col("lat").radians(), pl.col("distance")
    )
    gc_x_expr, gc_y_expr, gc_z_expr = frame.gl_xyz_to_gc_xyz_polars(
        gl_x_expr.alias("gl_x"), gl_y_expr.alias("gl_y"), gl_z_expr.alias("gl_z"), handedness=handedness
    )
    data = pl.DataFrame(
        {
            "lon": [lon, lon, lon],
            "lat": [lat, lat, lat],
            "distance": [0.0, d1, d2],
        }
    )

    data = data.with_columns(
        gc_x_expr.alias("gc_x"),
        gc_y_expr.alias("gc_y"),
        gc_z_expr.alias("gc_z"),
    )

    # Extract the points as numpy arrays for verification
    points = data.select("gc_x", "gc_y", "gc_z").to_numpy()

    # Check they are collinear with the Sun (point at index 0)
    vec1 = points[1] - points[0]
    vec2 = points[2] - points[0]

    # Parallel vectors have zero cross product
    cross_product = np.cross(vec1, vec2)
    np.testing.assert_allclose(cross_product, 0, atol=_ABS_TOL)


@given(
    frame=random_frame_only_x(max_value=1e5),
    u=st.floats(min_value=-100, max_value=100),
    v=st.floats(min_value=-100, max_value=100),
    w=st.floats(min_value=-100, max_value=100),
)
def test_gl_xyz_to_gc_xyz_galpy(frame: GalactocentricFrame, u: float, v: float, w: float) -> None:
    assume(frame.sun_rxy() > 0)
    gl_x = np.full(1, u)
    gl_y = np.full(1, v)
    gl_z = np.full(1, w)
    data = pl.DataFrame(
        {
            "gl_x": gl_x,
            "gl_y": gl_y,
            "gl_z": gl_z,
        }
    )

    x_sign: float = 1 if frame.sun_x() >= 0 else -1

    np_x, np_y, np_z = frame.gl_xyz_to_gc_xyz_numpy(gl_x, gl_y, gl_z, handedness="right")
    pl_x_expr, pl_y_expr, pl_z_expr = frame.gl_xyz_to_gc_xyz_polars(
        pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"), handedness="right"
    )

    gal_xyz = galcoords.XYZ_to_galcenrect(
        gl_x, x_sign * gl_y, gl_z, Xsun=x_sign * frame.sun_rxy(), Zsun=frame.sun_z(), _extra_rot=True
    )
    gal_x = gal_xyz[:, 0]
    gal_y = gal_xyz[:, 1]
    gal_z = gal_xyz[:, 2]

    data = data.with_columns(
        pl_x_expr.alias("gc_x"),
        pl_y_expr.alias("gc_y"),
        pl_z_expr.alias("gc_z"),
    )

    np.testing.assert_allclose(data["gc_x"], np_x, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_y"], np_y, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_z"], np_z, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_x"], gal_x, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(data["gc_y"], gal_y, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(data["gc_z"], gal_z, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_x, gal_x, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_y, gal_y, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_z, gal_z, atol=_GALPY_ABS_TOL)
