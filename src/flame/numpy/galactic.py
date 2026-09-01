"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from flame._galactic import MAS_PER_YR_TO_KM_PER_S_PER_KPC

from .geometric import cartesian_velocity_to_spherical, spherical_velocity_to_cartesian

if TYPE_CHECKING:
    from optype import numpy as onp

    from flame.units import AngleArray

__all__: list[str] = ["vrpmllpmbb_to_vxvyvz", "vxvyvz_to_vrpmllpmbb"]


def vrpmllpmbb_to_vxvyvz[F: np.floating, S: tuple[Any, ...]](
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
    longitude_velocity = cast("onp.ArrayND[np.float64, S]", distance * pmll * MAS_PER_YR_TO_KM_PER_S_PER_KPC)
    latitude_velocity = cast("onp.ArrayND[np.float64, S]", distance * pmbb * MAS_PER_YR_TO_KM_PER_S_PER_KPC)
    vr_cast = vr.astype(np.float64)
    lon_cast = lon.astype(np.float64)
    lat_cast = lat.astype(np.float64)
    return spherical_velocity_to_cartesian(
        vr_cast,
        longitude_velocity,
        latitude_velocity,
        lon_cast,
        lat_cast,
    )


def vxvyvz_to_vrpmllpmbb[F: np.floating, S: tuple[Any, ...]](
    vx: onp.ArrayND[F, S],
    vy: onp.ArrayND[F, S],
    vz: onp.ArrayND[F, S],
    lon: AngleArray[F, S],
    lat: AngleArray[F, S],
    distance: onp.ArrayND[F, S],
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform Galactic Cartesian velocities to radial and proper-motion velocities.

    Proper motions are undefined at zero distance; those output entries are
    ``nan`` or infinite following NumPy division semantics.

    Parameters
    ----------
    vx, vy, vz : Array[float]
        Galactic Cartesian velocity components in km/s.
    lon, lat : AngleArray[float]
        Galactic longitude and latitude.
    distance : Array[float]
        Distance in kpc.

    Returns
    -------
    vr : Array[float]
        Line-of-sight velocity in km/s.
    pmll : Array[float]
        Proper motion in Galactic longitude, corrected by cos(b), in mas/yr.
    pmbb : Array[float]
        Proper motion in Galactic latitude in mas/yr.

    """
    vr, longitude_velocity, latitude_velocity = cartesian_velocity_to_spherical(vx, vy, vz, lon, lat)
    scale = distance * MAS_PER_YR_TO_KM_PER_S_PER_KPC
    with np.errstate(divide="ignore", invalid="ignore"):
        pmll = cast("onp.ArrayND[np.float64, S]", (longitude_velocity / scale).astype(np.float64))
        pmbb = cast("onp.ArrayND[np.float64, S]", (latitude_velocity / scale).astype(np.float64))
    return (vr, pmll, pmbb)
