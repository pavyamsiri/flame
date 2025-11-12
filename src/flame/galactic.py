"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import polars as pl
from optype import numpy as onp

if TYPE_CHECKING:
    from typing import Final

_Shape = TypeVar("_Shape", bound=tuple[Any, ...])
_Float = TypeVar("_Float", bound=np.float64)


_MAS_PER_YR_TO_KM_PER_S_PER_KPC: Final[float] = 4.74047


def vrpmllpmbb_to_vxvyvz_numpy(
    vr: onp.ArrayND[_Float, _Shape],
    pmll: onp.ArrayND[_Float, _Shape],
    pmbb: onp.ArrayND[_Float, _Shape],
    lon: onp.ArrayND[_Float, _Shape],
    lat: onp.ArrayND[_Float, _Shape],
    distance: onp.ArrayND[_Float, _Shape],
) -> tuple[
    onp.ArrayND[np.float64, _Shape],
    onp.ArrayND[np.float64, _Shape],
    onp.ArrayND[np.float64, _Shape],
]:
    """Transform from spherical velocities to Cartesian velocities in the Galactic frame.

    Parameters
    ----------
    vr : Array[float]
        The line-of-sight velocity in km/s.
    pmll : Array[float]
        The proper motion in Galactic longitude corrected by cos(b) in mas/yr.
    pmbb : Array[float]
        The proper motion in Galactic latitude in mas/yr.
    lon : Array[float]
        The Galactic longitude in radians.
    lat : Array[float]
        The Galactic latitude in radians.
    distance : Array[float]
        The distance in kpc.

    Returns
    -------
    vx : Array[float]
        The velocity in the Galactic x direction in km/s.
    vy : Array[float]
        The velocity in the Galactic y direction in km/s.
    vz : Array[float]
        The velocity in the Galactic z direction in km/s.

    """
    sinl = np.sin(lon)
    cosl = np.cos(lon)
    sinb = np.sin(lat)
    cosb = np.cos(lat)

    rotation_0_0 = cosl * cosb
    rotation_1_0 = sinl * cosb
    rotation_2_0 = sinb
    rotation_0_1 = -sinl
    rotation_1_1 = cosl
    rotation_2_1 = 0
    rotation_0_2 = -cosl * sinb
    rotation_1_2 = -sinl * sinb
    rotation_2_2 = cosb

    v0 = vr
    v1 = distance * pmll * _MAS_PER_YR_TO_KM_PER_S_PER_KPC
    v2 = distance * pmbb * _MAS_PER_YR_TO_KM_PER_S_PER_KPC

    # a b c   v0
    # d e f * v1
    # g h i   v2

    vx = cast(onp.ArrayND[np.float64, _Shape], rotation_0_0 * v0 + rotation_0_1 * v1 + rotation_0_2 * v2)
    vy = cast(onp.ArrayND[np.float64, _Shape], rotation_1_0 * v0 + rotation_1_1 * v1 + rotation_1_2 * v2)
    vz = cast(onp.ArrayND[np.float64, _Shape], rotation_2_0 * v0 + rotation_2_1 * v1 + rotation_2_2 * v2)

    return (vx, vy, vz)


def vrpmllpmbb_to_vxvyvz_polars(
    vr: pl.Expr,
    pmll: pl.Expr,
    pmbb: pl.Expr,
    lon: pl.Expr,
    lat: pl.Expr,
    distance: pl.Expr,
) -> tuple[
    pl.Expr,
    pl.Expr,
    pl.Expr,
]:
    """Transform from spherical velocities to Cartesian velocities in the Galactic frame.

    Parameters
    ----------
    vr : pl.Expr
        The line-of-sight velocity in km/s.
    pmll : pl.Expr
        The proper motion in Galactic longitude corrected by cos(b) in mas/yr.
    pmbb : pl.Expr
        The proper motion in Galactic latitude in mas/yr.
    lon : pl.Expr
        The Galactic longitude in radians.
    lat : pl.Expr
        The Galactic latitude in radians.
    distance : pl.Expr
        The distance in kpc.

    Returns
    -------
    vx : pl.Expr
        The velocity in the Galactic x direction in km/s.
    vy : pl.Expr
        The velocity in the Galactic y direction in km/s.
    vz : pl.Expr
        The velocity in the Galactic z direction in km/s.

    """
    sinl = lon.sin()
    cosl = lon.cos()
    sinb = lat.sin()
    cosb = lat.cos()

    rotation_0_0 = cosl * cosb
    rotation_1_0 = sinl * cosb
    rotation_2_0 = sinb
    rotation_0_1 = -sinl
    rotation_1_1 = cosl
    rotation_2_1 = 0
    rotation_0_2 = -cosl * sinb
    rotation_1_2 = -sinl * sinb
    rotation_2_2 = cosb

    v0 = vr
    v1 = distance * pmll * _MAS_PER_YR_TO_KM_PER_S_PER_KPC
    v2 = distance * pmbb * _MAS_PER_YR_TO_KM_PER_S_PER_KPC

    # a b c   v0
    # d e f * v1
    # g h i   v2

    vx = rotation_0_0 * v0 + rotation_0_1 * v1 + rotation_0_2 * v2
    vy = rotation_1_0 * v0 + rotation_1_1 * v1 + rotation_1_2 * v2
    vz = rotation_2_0 * v0 + rotation_2_1 * v1 + rotation_2_2 * v2

    return (vx, vy, vz)
