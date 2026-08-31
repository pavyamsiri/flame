"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from optype import numpy as onp

from flame.units import AngleArray

__all__: list[str] = [
    "spherical_to_cartesian",
    "spherical_velocity_to_cartesian",
    "cartesian_to_polar",
    "cartesian_vec_to_polar_vec",
]


def spherical_to_cartesian[F: np.float64, S: tuple[Any, ...]](
    lon: AngleArray[F, S], lat: AngleArray[F, S], distance: onp.ArrayND[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : AngleArray[float]
        The longitude coordinate.
    lat : AngleArray[float]
        The latitude coordinate.
    distance : Array[float]
        The distance.

    Returns
    -------
    x : Array[f64]
        The x coordinate.
    y : Array[f64]
        The y coordinate.
    z : Array[f64]
        The z coordinate.

    """
    # Compute Cartesian coordinates in Galactic frame
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    cosb = np.cos(lat_radians)
    sinb = np.sin(lat_radians)
    cosl = np.cos(lon_radians)
    sinl = np.sin(lon_radians)
    x = cast(onp.ArrayND[np.float64, S], distance * cosb * cosl)
    y = cast(onp.ArrayND[np.float64, S], distance * cosb * sinl)
    z = cast(onp.ArrayND[np.float64, S], distance * sinb)
    return (x, y, z)


def spherical_velocity_to_cartesian[F: np.float64, S: tuple[Any, ...]](
    vr: onp.ArrayND[F, S],
    longitude_velocity: onp.ArrayND[F, S],
    latitude_velocity: onp.ArrayND[F, S],
    lon: AngleArray[F, S],
    lat: AngleArray[F, S],
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform spherical velocity components to Cartesian velocity components.

    ``longitude_velocity`` and ``latitude_velocity`` are tangential velocities
    in the directions of increasing longitude and latitude, respectively.
    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = np.sin(lon_radians)
    cosl = np.cos(lon_radians)
    sinb = np.sin(lat_radians)
    cosb = np.cos(lat_radians)

    vx = cast(
        onp.ArrayND[np.float64, S],
        vr * cosl * cosb - longitude_velocity * sinl - latitude_velocity * cosl * sinb,
    )
    vy = cast(
        onp.ArrayND[np.float64, S],
        vr * sinl * cosb + longitude_velocity * cosl - latitude_velocity * sinl * sinb,
    )
    vz = cast(onp.ArrayND[np.float64, S], vr * sinb + latitude_velocity * cosb)
    return (vx, vy, vz)


def cartesian_to_polar[F: np.float64, S: tuple[Any, ...]](
    x: onp.ArrayND[F, S], y: onp.ArrayND[F, S]
) -> tuple[onp.ArrayND[np.float64, S], AngleArray[np.float64, S]]:
    """Transform Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : Array[float]
        The x coordinate.
    y : Array[float]
        The y coordinate.

    Returns
    -------
    r : Array[f64]
        The radial coordinate.
    phi : AngleArray[f64]
        The azimuthal coordinate.

    """
    rxy = cast(onp.ArrayND[np.float64, S], np.hypot(x, y).astype(np.float64))
    phi = cast(onp.ArrayND[np.float64, S], np.arctan2(y, x).astype(np.float64))

    return (rxy, AngleArray(phi, "rad"))


def cartesian_vec_to_polar_vec[F: np.float64, S: tuple[Any, ...]](
    vx: onp.ArrayND[F, S], vy: onp.ArrayND[F, S], phi: AngleArray[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform Cartesian velocity to polar velocity.

    Parameters
    ----------
    vx : Array[float]
        The x-velocity.
    vy : Array[float]
        The y-velocity.
    phi : Array[float]
        The azimuthal coordinate in radians.

    Returns
    -------
    vr : Array[f64]
        The radial velocity.
    vphi : Array[f64]
        The tangential velocity.

    """
    phi_rad = phi.to_radians()
    cosphi = np.cos(phi_rad)
    sinphi = np.sin(phi_rad)
    vr = cast(onp.ArrayND[np.float64, S], (vx * cosphi + vy * sinphi).astype(np.float64))
    vphi = cast(onp.ArrayND[np.float64, S], (-vx * sinphi + vy * cosphi).astype(np.float64))

    return (vr, vphi)
