"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from optype import numpy as onp

from flame.units import AngleArray
from flame._galactic import MAS_PER_YR_TO_KM_PER_S_PER_KPC

from .geometric import spherical_velocity_to_cartesian

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
    longitude_velocity = cast(onp.ArrayND[np.float64, S], distance * pmll * MAS_PER_YR_TO_KM_PER_S_PER_KPC)
    latitude_velocity = cast(onp.ArrayND[np.float64, S], distance * pmbb * MAS_PER_YR_TO_KM_PER_S_PER_KPC)
    return spherical_velocity_to_cartesian(vr, longitude_velocity, latitude_velocity, lon, lat)
