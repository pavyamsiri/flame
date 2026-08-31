"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from optype import numpy as onp

from flame.units import AngleArray
from flame._galactic import MAS_PER_YR_TO_KM_PER_S_PER_KPC


__all__: list[str] = ["vrpmllpmbb_to_vxvyvz"]


def vrpmllpmbb_to_vxvyvz[F: np.float64, S: tuple[Any, ...]](
    vr: onp.ArrayND[F, S],
    pmll: onp.ArrayND[F, S],
    pmbb: onp.ArrayND[F, S],
    lon: AngleArray[F, S],
    lat: AngleArray[F, S],
    distance: onp.ArrayND[F, S],
) -> tuple[
    onp.ArrayND[np.float64, S],
    onp.ArrayND[np.float64, S],
    onp.ArrayND[np.float64, S],
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
    lon : AngleArray[float]
        The Galactic longitude.
    lat : AngleArray[float]
        The Galactic latitude.
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
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = np.sin(lon_radians)
    cosl = np.cos(lon_radians)
    sinb = np.sin(lat_radians)
    cosb = np.cos(lat_radians)

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
    v1 = distance * pmll * MAS_PER_YR_TO_KM_PER_S_PER_KPC
    v2 = distance * pmbb * MAS_PER_YR_TO_KM_PER_S_PER_KPC

    # a b c   v0
    # d e f * v1
    # g h i   v2

    vx = cast(onp.ArrayND[np.float64, S], rotation_0_0 * v0 + rotation_0_1 * v1 + rotation_0_2 * v2)
    vy = cast(onp.ArrayND[np.float64, S], rotation_1_0 * v0 + rotation_1_1 * v1 + rotation_1_2 * v2)
    vz = cast(onp.ArrayND[np.float64, S], rotation_2_0 * v0 + rotation_2_1 * v1 + rotation_2_2 * v2)

    return (vx, vy, vz)
