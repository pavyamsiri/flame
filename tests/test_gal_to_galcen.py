from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from galpy.util import coords as galcoords  # pyright: ignore[reportMissingTypeStubs]
from hypothesis import assume, given
from hypothesis import strategies as st

from flame.galactocentric import (
    GALCEN_EXTRA_ROT,
    GalactocentricFrame,
    astropy_compatible_frame,
    default_galactocentric_frame,
    galpy_compatible_frame,
)
from flame.numpy.icrs import icrs_to_galactic_cartesian as icrs_to_galactic_cartesian_numpy
from flame.numpy.icrs import icrs_to_galactic_velocity as icrs_to_galactic_velocity_numpy
from flame.polars.geometric import spherical_to_cartesian as spherical_to_cartesian_polars
from flame.units import AngleArray, AngleExpr

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


def with_handedness(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> GalactocentricFrame:
    return GalactocentricFrame(
        sun_x=frame.sun_x(),
        sun_y=frame.sun_y(),
        sun_z=frame.sun_z(),
        sun_vx=frame.sun_vx(),
        sun_vy=frame.sun_vy(),
        sun_vz=frame.sun_vz(),
        handedness=handedness,
        prograde_lz_positive=frame.prograde_lz_positive,
    )


def test_cached_transforms_are_not_mutable() -> None:
    frame = GalactocentricFrame()

    with pytest.raises(AttributeError):
        setattr(frame, "transforms", frame.transforms)
    with pytest.raises(FrozenInstanceError):
        setattr(frame.numpy, "_transforms", object())
    with pytest.raises(ValueError):
        frame.transforms.pos_matrix[0, 0] = 0.0


def test_frame_metadata_presets_and_validation() -> None:
    frame = GalactocentricFrame(sun_x=-8.0, sun_y=0.0, sun_z=0.02, sun_vx=10.0, sun_vy=230.0, sun_vz=8.0)

    assert frame.handedness == "right"
    assert frame.sun_position == (-8.0, 0.0, 0.02)
    assert frame.sun_velocity == (10.0, 230.0, 8.0)
    assert default_galactocentric_frame().handedness == "right"
    assert galpy_compatible_frame().handedness == "left"
    with pytest.raises(TypeError, match=r"(?i)handedness"):
        GalactocentricFrame(handedness="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        GalactocentricFrame(sun_x=float("nan"))


def test_frame_from_cylindrical() -> None:
    right_frame = GalactocentricFrame.from_cylindrical(
        sun_radius=8.122,
        sun_z=0.0208,
        sun_v_radius=-12.9,
        sun_v_phi=245.6,
        sun_vz=7.78,
    )
    left_frame = GalactocentricFrame.from_cylindrical(
        sun_radius=8.122,
        sun_z=0.0208,
        sun_v_radius=-12.9,
        sun_v_phi=245.6,
        sun_vz=7.78,
        handedness="left",
    )

    assert right_frame.sun_position == (-8.122, 0.0, 0.0208)
    assert right_frame.sun_velocity == (12.9, -245.6, 7.78)
    assert right_frame.handedness == "right"
    assert left_frame.sun_position == (8.122, 0.0, 0.0208)
    assert left_frame.sun_velocity == (-12.9, 245.6, 7.78)
    assert left_frame.handedness == "left"
    assert GalactocentricFrame.from_cylindrical().sun_position == GalactocentricFrame().sun_position
    assert GalactocentricFrame.from_cylindrical().sun_velocity == (12.9, -245.6, -7.78)
    assert GalactocentricFrame.from_cylindrical().sun_velocity == GalactocentricFrame().sun_velocity

    with pytest.raises(ValueError, match="non-negative"):
        GalactocentricFrame.from_cylindrical(sun_radius=-1.0)


def test_default_solar_motion() -> None:
    frame = default_galactocentric_frame()
    position = np.array(frame.sun_position)
    velocity = np.array(frame.sun_velocity)

    np.testing.assert_array_equal(position, [-8.122, 0.0, -0.0208])
    np.testing.assert_array_equal(velocity, [12.9, -245.6, -7.78])
    assert np.dot(position[:2], velocity[:2]) < 0  # Moving toward the centre.
    assert np.cross(position, velocity)[2] > 0  # Counter-clockwise from +z.
    assert position[2] * velocity[2] > 0  # Moving away from the plane.

    zero = np.zeros(1, dtype=np.float64)
    np.testing.assert_array_equal(frame.numpy.position_from_galactic(zero, zero, zero), position[:, None])
    np.testing.assert_array_equal(frame.numpy.velocity_from_galactic(zero, zero, zero), velocity[:, None])
    data = pl.DataFrame({"zero": [0.0]})
    expressions = frame.polars.velocity_from_galactic(pl.col("zero"), pl.col("zero"), pl.col("zero"))
    actual = data.select(*(expr.alias(name) for expr, name in zip(expressions, ("vx", "vy", "vz"), strict=True)))
    np.testing.assert_array_equal(actual.to_numpy()[0], velocity)

    reflected = galpy_compatible_frame()
    assert reflected.sun_velocity == (-12.9, 245.6, 7.78)


def test_phase_space_from_icrs() -> None:
    frame = GalactocentricFrame()
    ra = np.array([1.0], dtype=np.float64)
    dec = np.array([-0.5], dtype=np.float64)
    distance = np.array([2.0], dtype=np.float64)
    pmracosdec = np.array([3.0], dtype=np.float64)
    pmdec = np.array([-4.0], dtype=np.float64)
    radial_velocity = np.array([5.0], dtype=np.float64)
    ra_angle = AngleArray(ra, "rad")
    dec_angle = AngleArray(dec, "rad")

    actual = frame.numpy.phase_space_from_icrs(radial_velocity, pmracosdec, pmdec, distance, ra=ra_angle, dec=dec_angle)
    gl_position = icrs_to_galactic_cartesian_numpy(ra_angle, dec_angle, distance)
    gl_velocity = icrs_to_galactic_velocity_numpy(radial_velocity, pmracosdec, pmdec, distance, ra=ra_angle, dec=dec_angle)
    expected = (*frame.numpy.position_from_galactic(*gl_position), *frame.numpy.velocity_from_galactic(*gl_velocity))
    np.testing.assert_allclose(actual, expected)

    data = pl.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "distance": distance,
            "pmracosdec": pmracosdec,
            "pmdec": pmdec,
            "radial_velocity": radial_velocity,
        }
    )
    polars_actual = frame.polars.phase_space_from_icrs(
        pl.col("radial_velocity"),
        pl.col("pmracosdec"),
        pl.col("pmdec"),
        pl.col("distance"),
        ra=AngleExpr(pl.col("ra"), "rad"),
        dec=AngleExpr(pl.col("dec"), "rad"),
    )
    data = data.with_columns(
        *(expression.alias(name) for expression, name in zip(polars_actual, ("x", "y", "z", "vx", "vy", "vz"), strict=True))
    )
    np.testing.assert_allclose(data.select("x", "y", "z", "vx", "vy", "vz").to_numpy().T, actual)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_handedness_orientation_invariants(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    frame = with_handedness(frame, handedness)
    matrix = frame.transforms.pos_matrix
    expected_determinant = 1.0 if handedness == "right" else -1.0

    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(matrix), expected_determinant, atol=1e-10)
    np.testing.assert_allclose(np.cross(matrix[:, 0], matrix[:, 1]), expected_determinant * matrix[:, 2], atol=1e-10)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_galactocentric_position_round_trip(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    frame = with_handedness(frame, handedness)
    gl_x = np.array([1.5], dtype=np.float64)
    gl_y = np.array([-2.5], dtype=np.float64)
    gl_z = np.array([3.5], dtype=np.float64)

    gc_x, gc_y, gc_z = frame.numpy.position_from_galactic(gl_x, gl_y, gl_z)
    actual = frame.numpy.position_to_galactic(gc_x, gc_y, gc_z)

    np.testing.assert_allclose(actual, (gl_x, gl_y, gl_z), atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_galactocentric_velocity_round_trip(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    frame = with_handedness(frame, handedness)
    gl_vx = np.array([1.5], dtype=np.float64)
    gl_vy = np.array([-2.5], dtype=np.float64)
    gl_vz = np.array([3.5], dtype=np.float64)

    gc_vx, gc_vy, gc_vz = frame.numpy.velocity_from_galactic(gl_vx, gl_vy, gl_vz)
    actual = frame.numpy.velocity_to_galactic(gc_vx, gc_vy, gc_vz)

    np.testing.assert_allclose(actual, (gl_vx, gl_vy, gl_vz), atol=_ABS_TOL)


@given(
    frame=random_frame(),
    u=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    v=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    w=st.floats(min_value=-1e8, max_value=1e8, allow_infinity=False, allow_nan=False),
    handedness=st.sampled_from(["right", "left"]),
)
def test_gl_xyz_to_gc_xyz(frame: GalactocentricFrame, u: float, v: float, w: float, handedness: Literal["right", "left"]) -> None:
    frame = with_handedness(frame, handedness)
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

    np_x, np_y, np_z = frame.numpy.position_from_galactic(gl_x, gl_y, gl_z)
    pl_x_expr, pl_y_expr, pl_z_expr = frame.polars.position_from_galactic(pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"))

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
    frame = with_handedness(frame, handedness)
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

    np_x, np_y, np_z = frame.numpy.position_to_galactic(gc_x, gc_y, gc_z)
    pl_x_expr, pl_y_expr, pl_z_expr = frame.polars.position_to_galactic(pl.col("gc_x"), pl.col("gc_y"), pl.col("gc_z"))

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
    frame = with_handedness(frame, handedness)
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

    np_vx, np_vy, np_vz = frame.numpy.velocity_from_galactic(gl_vx, gl_vy, gl_vz)
    pl_vx_expr, pl_vy_expr, pl_vz_expr = frame.polars.velocity_from_galactic(pl.col("gl_vx"), pl.col("gl_vy"), pl.col("gl_vz"))

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
    frame = with_handedness(frame, handedness)
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

    np_vx, np_vy, np_vz = frame.numpy.velocity_to_galactic(gc_vx, gc_vy, gc_vz)
    pl_vx_expr, pl_vy_expr, pl_vz_expr = frame.polars.velocity_to_galactic(pl.col("gc_vx"), pl.col("gc_vy"), pl.col("gc_vz"))

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
    frame = with_handedness(frame, handedness)
    zero_arr = np.array([0], dtype=np.float64)
    gc_x, gc_y, gc_z = frame.numpy.position_from_galactic(zero_arr, zero_arr, zero_arr)
    gc_vx, gc_vy, gc_vz = frame.numpy.velocity_from_galactic(zero_arr, zero_arr, zero_arr)

    np.testing.assert_allclose(gc_x[0], frame.sun_x(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_y[0], frame.sun_y(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_z[0], frame.sun_z(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vx[0], frame.sun_vx(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vy[0], frame.sun_vy(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vz[0], frame.sun_vz(), atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_sun_polars(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that the Sun in the Galactic frame is at the proper coordinates in the Galactocentric frame."""
    frame = with_handedness(frame, handedness)
    gc_x_expr, gc_y_expr, gc_z_expr = frame.polars.position_from_galactic(pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"))
    gc_vx_expr, gc_vy_expr, gc_vz_expr = frame.polars.velocity_from_galactic(pl.col("gl_vx"), pl.col("gl_vy"), pl.col("gl_vz"))

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
    frame = with_handedness(frame, handedness)
    lat = np.radians(lat)
    lon = np.radians(lon)
    dist = np.array([0, d1, d2], dtype=np.float64)
    gal_x = dist * np.cos(lat) * np.cos(lon)
    gal_y = dist * np.cos(lat) * np.sin(lon)
    gal_z = dist * np.sin(lat)

    gc_x, gc_y, gc_z = frame.numpy.position_from_galactic(gal_x, gal_y, gal_z)

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
    frame = with_handedness(frame, handedness)
    gl_x_expr, gl_y_expr, gl_z_expr = spherical_to_cartesian_polars(
        AngleExpr(pl.col("lon"), "deg"), AngleExpr(pl.col("lat"), "deg"), pl.col("distance")
    )
    gc_x_expr, gc_y_expr, gc_z_expr = frame.polars.position_from_galactic(
        gl_x_expr.alias("gl_x"), gl_y_expr.alias("gl_y"), gl_z_expr.alias("gl_z")
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
    # galpy's reference rotation becomes numerically unstable as Xsun tends to zero.
    assume(frame.sun_rxy() > 1e-3)
    frame = GalactocentricFrame(
        sun_x=abs(frame.sun_x()),
        sun_z=frame.sun_z(),
        sun_vx=frame.sun_vx(),
        sun_vy=frame.sun_vy(),
        sun_vz=frame.sun_vz(),
        handedness="left",
        prograde_lz_positive=True,
    )
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

    np_x, np_y, np_z = frame.numpy.position_from_galactic(gl_x, gl_y, gl_z)
    pl_x_expr, pl_y_expr, pl_z_expr = frame.polars.position_from_galactic(pl.col("gl_x"), pl.col("gl_y"), pl.col("gl_z"))

    gal_xyz = galcoords.XYZ_to_galcenrect(gl_x, gl_y, gl_z, Xsun=frame.sun_rxy(), Zsun=frame.sun_z(), _extra_rot=True)
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


@given(
    frame=random_frame_only_x(max_value=1e5),
    v_u=st.floats(min_value=-100, max_value=100),
    v_v=st.floats(min_value=-100, max_value=100),
    v_w=st.floats(min_value=-100, max_value=100),
)
def test_gl_vxvyvz_to_gc_vxvyvz_galpy(frame: GalactocentricFrame, v_u: float, v_v: float, v_w: float) -> None:
    # galpy's reference rotation becomes numerically unstable as Xsun tends to zero.
    assume(frame.sun_rxy() > 1e-3)
    frame = GalactocentricFrame(
        sun_x=abs(frame.sun_x()),
        sun_z=frame.sun_z(),
        sun_vx=frame.sun_vx(),
        sun_vy=frame.sun_vy(),
        sun_vz=frame.sun_vz(),
        handedness="left",
        prograde_lz_positive=True,
    )
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

    np_vx, np_vy, np_vz = frame.numpy.velocity_from_galactic(gl_vx, gl_vy, gl_vz)
    pl_vx_expr, pl_vy_expr, pl_vz_expr = frame.polars.velocity_from_galactic(pl.col("gl_vx"), pl.col("gl_vy"), pl.col("gl_vz"))

    gal_vxvyvz = galcoords.vxvyvz_to_galcenrect(
        gl_vx,
        gl_vy,
        gl_vz,
        vsun=(frame.sun_vx(), frame.sun_vy(), frame.sun_vz()),
        Xsun=frame.sun_rxy(),
        Zsun=frame.sun_z(),
        _extra_rot=True,
    )
    gal_vx = gal_vxvyvz[:, 0]
    gal_vy = gal_vxvyvz[:, 1]
    gal_vz = gal_vxvyvz[:, 2]

    data = data.with_columns(
        pl_vx_expr.alias("gc_vx"),
        pl_vy_expr.alias("gc_vy"),
        pl_vz_expr.alias("gc_vz"),
    )

    np.testing.assert_allclose(data["gc_vx"], np_vx, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vy"], np_vy, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vz"], np_vz, atol=_ABS_TOL)
    np.testing.assert_allclose(data["gc_vx"], gal_vx, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(data["gc_vy"], gal_vy, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(data["gc_vz"], gal_vz, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_vx, gal_vx, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_vy, gal_vy, atol=_GALPY_ABS_TOL)
    np.testing.assert_allclose(np_vz, gal_vz, atol=_GALPY_ABS_TOL)


@pytest.mark.parametrize("handedness", ["left", "right"])
@pytest.mark.parametrize("positive", [False, True])
def test_convention_geometry(handedness: Literal["left", "right"], positive: bool) -> None:
    frame = GalactocentricFrame(handedness=handedness, prograde_lz_positive=positive)
    cylindrical = GalactocentricFrame.from_cylindrical(handedness=handedness, prograde_lz_positive=positive)
    assert cylindrical.sun_position == frame.sun_position
    assert cylindrical.sun_velocity == frame.sun_velocity
    position = np.array(frame.sun_position)
    velocity = np.array(frame.sun_velocity)
    assert (np.cross(position, velocity)[2] > 0) == positive
    assert np.dot(position[:2], velocity[:2]) < 0
    assert position[2] * velocity[2] > 0
    h = 1 if handedness == "right" else -1
    np.testing.assert_allclose(np.linalg.det(frame.transforms.pos_matrix), h, atol=1e-10)
    north = frame.transforms.pos_matrix @ np.linalg.solve(GALCEN_EXTRA_ROT, [0.0, 0.0, 1.0])
    assert (north[2] > 0) == (frame.z_direction == "north")
    assert f"prograde_lz_positive={positive}" in repr(frame)
    assert f"Galactic {frame.z_direction}" in str(frame)
    recreated = eval(repr(frame), {"GalactocentricFrame": GalactocentricFrame})
    np.testing.assert_array_equal(recreated.transforms.pos_matrix, frame.transforms.pos_matrix)
    zero = np.zeros(1)
    np.testing.assert_array_equal(frame.numpy.velocity_from_galactic(zero, zero, zero), velocity[:, None])
    inputs = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]).T
    data = pl.DataFrame(dict(zip(("u", "v", "w"), inputs, strict=True)))
    columns = (pl.col("u"), pl.col("v"), pl.col("w"))
    for forward, inverse, polars_forward in [
        (frame.numpy.position_from_galactic, frame.numpy.position_to_galactic, frame.polars.position_from_galactic),
        (frame.numpy.velocity_from_galactic, frame.numpy.velocity_to_galactic, frame.polars.velocity_from_galactic),
    ]:
        transformed = forward(*inputs)
        np.testing.assert_allclose(inverse(*transformed), inputs, atol=1e-12)
        expressions = polars_forward(*columns)
        actual = data.select(*(expr.alias(name) for expr, name in zip(expressions, ("x", "y", "z"), strict=True)))
        np.testing.assert_allclose(actual.to_numpy().T, transformed, atol=1e-12)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("handedness", ["left", "right"])
@pytest.mark.parametrize("sun_x,sun_y,sun_z", [(-8.0, 3.0, 2.0), (8.0, 3.0, 2.0), (0.0, 8.0, 2.0)])
def test_centre_direction_for_arbitrary_solar_azimuth(
    positive: bool, handedness: Literal["left", "right"], sun_x: float, sun_y: float, sun_z: float
) -> None:
    frame = GalactocentricFrame(sun_x=sun_x, sun_y=sun_y, sun_z=sun_z, handedness=handedness, prograde_lz_positive=positive)
    # A ray toward the centre in the corrected heliocentric basis reaches the origin.
    ray = np.linalg.solve(GALCEN_EXTRA_ROT, [frame.distance_to_gc(), 0.0, 0.0])
    np.testing.assert_allclose(frame.numpy.position_from_galactic(*ray), 0.0, atol=1e-12)


def test_presets_share_observed_phase_space() -> None:
    astropy = astropy_compatible_frame()
    galpy = galpy_compatible_frame()
    preferred = GalactocentricFrame()
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]).T
    astro_position = np.array(astropy.numpy.position_from_galactic(*xyz))
    astro_velocity = np.array(astropy.numpy.velocity_from_galactic(*xyz))
    for frame, signs in [(galpy, [-1.0, 1.0, 1.0]), (preferred, [1.0, -1.0, -1.0])]:
        reflection = np.array(signs)[:, None]
        np.testing.assert_allclose(frame.numpy.position_from_galactic(*xyz), reflection * astro_position, atol=1e-12)
        np.testing.assert_allclose(frame.numpy.velocity_from_galactic(*xyz), reflection * astro_velocity, atol=1e-12)
    assert astropy.prograde_lz_positive is False
    assert galpy.prograde_lz_positive is True
    np.testing.assert_allclose(
        np.array(galpy.numpy.position_from_galactic(*xyz)).T,
        galcoords.XYZ_to_galcenrect(*xyz, Xsun=galpy.sun_x(), Zsun=galpy.sun_z()),
        atol=1e-10,
    )


def test_invalid_prograde_sign() -> None:
    with pytest.raises(TypeError, match="prograde_lz_positive"):
        GalactocentricFrame(prograde_lz_positive="positive")  # type: ignore[arg-type]
