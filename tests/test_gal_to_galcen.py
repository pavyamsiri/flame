from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from flame.galactocentric import GalactocentricFrame

if TYPE_CHECKING:
    from typing import Literal


_ABS_TOL: float = 1e-7


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


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_center(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that the GC in the Galactic frame is at the origin in the Galactocentric frame."""
    gl_x, gl_y, gl_z = (
        np.array([frame.distance_to_gc()], dtype=np.float64),
        np.array([0], dtype=np.float64),
        np.array([0], dtype=np.float64),
    )
    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz(gl_x, gl_y, gl_z, handedness=handedness)

    np.testing.assert_allclose(gc_x[0], 0, atol=_ABS_TOL)
    np.testing.assert_allclose(gc_y[0], 0, atol=_ABS_TOL)
    np.testing.assert_allclose(gc_z[0], 0, atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_sun(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that the Sun in the Galactic frame is at the proper coordinates in the Galactocentric frame."""
    zero_arr = np.array([0], dtype=np.float64)
    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz(zero_arr, zero_arr, zero_arr, handedness=handedness)
    gc_vx, gc_vy, gc_vz = frame.gl_vxvyvz_to_gc_vxvyvz(zero_arr, zero_arr, zero_arr, handedness=handedness)

    np.testing.assert_allclose(gc_x[0], frame.sun_x(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_y[0], frame.sun_y(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_z[0], frame.sun_z(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vx[0], frame.sun_vx(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vy[0], frame.sun_vy(), atol=_ABS_TOL)
    np.testing.assert_allclose(gc_vz[0], frame.sun_vz(), atol=_ABS_TOL)


@given(frame=random_frame(), handedness=st.sampled_from(["right", "left"]))
def test_opposing(frame: GalactocentricFrame, handedness: Literal["right", "left"]) -> None:
    """Check that points on opposing sides of the GC (collinear) mirror each other."""
    target_gc_x: float = frame.distance_to_gc()
    target_gc_y: float = 0
    target_gc_z: float = 0

    gal_x, gal_y, gal_z = (
        np.array([0.5 * target_gc_x, 1.5 * target_gc_x], dtype=np.float64),
        np.array([target_gc_y, target_gc_y], dtype=np.float64),
        np.array([target_gc_z, target_gc_y], dtype=np.float64),
    )
    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz(gal_x, gal_y, gal_z, handedness=handedness)

    np.testing.assert_allclose(gc_x[0], -gc_x[1], atol=_ABS_TOL)
    np.testing.assert_allclose(gc_y[0], -gc_y[1], atol=_ABS_TOL)
    np.testing.assert_allclose(gc_z[0], -gc_z[1], atol=_ABS_TOL)


@given(
    frame=random_frame(max_value=1e5),
    lat=st.floats(min_value=0, max_value=90),
    lon=st.floats(min_value=0, max_value=360),
    d1=st.floats(min_value=0, max_value=1e3),
    d2=st.floats(min_value=0, max_value=1e3),
    handedness=st.sampled_from(["right", "left"]),
)
def test_collinear(
    frame: GalactocentricFrame, lat: float, lon: float, d1: float, d2: float, handedness: Literal["right", "left"]
) -> None:
    """Check that points with the same longitude and latitude are collinear with the Sun."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    dist = np.array([0, d1, d2], dtype=np.float64)
    gal_x = dist * np.cos(lat) * np.cos(lon)
    gal_y = dist * np.cos(lat) * np.sin(lon)
    gal_z = dist * np.sin(lat)

    gc_x, gc_y, gc_z = frame.gl_xyz_to_gc_xyz(gal_x, gal_y, gal_z, handedness=handedness)

    # They should be collinear with the Sun
    vec1 = np.array([gc_x[1] - gc_x[0], gc_y[1] - gc_y[0], gc_z[1] - gc_z[0]])
    vec2 = np.array([gc_x[2] - gc_x[0], gc_y[2] - gc_y[0], gc_z[2] - gc_z[0]])

    # Parallel vector via zero cross product
    cross_product = np.cross(vec1, vec2)
    np.testing.assert_allclose(cross_product, 0, atol=_ABS_TOL)
