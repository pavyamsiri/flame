"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from flame.units import AngleArray

if TYPE_CHECKING:
    from optype import numpy as onp

__all__: list[str] = [
    "cartesian_to_cylindrical",
    "cartesian_to_polar",
    "cartesian_to_spherical",
    "cartesian_vec_to_polar_vec",
    "cartesian_velocity_to_cylindrical",
    "cartesian_velocity_to_spherical",
    "cylindrical_to_cartesian",
    "cylindrical_velocity_to_cartesian",
    "spherical_to_cartesian",
    "spherical_velocity_to_cartesian",
]


def spherical_to_cartesian[F: np.floating, S: tuple[Any, ...]](
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
    x = cast("onp.ArrayND[np.float64, S]", distance * cosb * cosl)
    y = cast("onp.ArrayND[np.float64, S]", distance * cosb * sinl)
    z = cast("onp.ArrayND[np.float64, S]", distance * sinb)
    return (x, y, z)


def cartesian_to_spherical[F: np.floating, S: tuple[Any, ...]](
    x: onp.ArrayND[F, S], y: onp.ArrayND[F, S], z: onp.ArrayND[F, S]
) -> tuple[AngleArray[np.float64, S], AngleArray[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform Cartesian coordinates to spherical coordinates.

    Longitudes are wrapped to ``[0, 2 * pi)``. At the origin the direction is
    undefined; this function returns longitude and latitude equal to zero.

    Parameters
    ----------
    x : Array[float]
        The x coordinate.
    y : Array[float]
        The y coordinate.
    z : Array[float]
        The z coordinate.

    Returns
    -------
    lon : AngleArray[f64]
        The longitude.
    lat : AngleArray[f64]
        The latitude.
    distance : Array[f64]
        The distance.

    """
    rxy = np.hypot(x, y)
    distance = cast("onp.ArrayND[np.float64, S]", np.hypot(rxy, z).astype(np.float64))
    lon: onp.ArrayND[np.float64, S] = cast("onp.ArrayND[np.float64, S]", np.mod(np.arctan2(y, x), 2 * np.pi).astype(np.float64))
    lat: onp.ArrayND[np.float64, S] = cast("onp.ArrayND[np.float64, S]", np.arctan2(z, rxy).astype(np.float64))
    return (AngleArray(lon, "rad"), AngleArray(lat, "rad"), distance)


def spherical_velocity_to_cartesian[F: np.floating, S: tuple[Any, ...]](
    vr: onp.ArrayND[F, S],
    longitude_velocity: onp.ArrayND[F, S],
    latitude_velocity: onp.ArrayND[F, S],
    lon: AngleArray[F, S],
    lat: AngleArray[F, S],
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform spherical velocity components to Cartesian velocity components.

    ``longitude_velocity`` and ``latitude_velocity`` are tangential velocities
    in the directions of increasing longitude and latitude, respectively.

    Parameters
    ----------
    vr : Array[float]
        The radial velocity.
    longitude_velocity : Array[float]
        The longitude velocity.
    latitude_velocity : Array[float]
        The latitude velocity.
    lon : AngleArray[float]
        The longitude.
    lat : AngleArray[float]
        The latitude.

    Returns
    -------
    vx : Array[f64]
        The x velocity.
    vy : Array[f64]
        The y velocity.
    vz : Array[f64]
        The z velocity.

    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = np.sin(lon_radians)
    cosl = np.cos(lon_radians)
    sinb = np.sin(lat_radians)
    cosb = np.cos(lat_radians)

    vx = cast(
        "onp.ArrayND[np.float64, S]",
        vr * cosl * cosb - longitude_velocity * sinl - latitude_velocity * cosl * sinb,
    )
    vy = cast(
        "onp.ArrayND[np.float64, S]",
        vr * sinl * cosb + longitude_velocity * cosl - latitude_velocity * sinl * sinb,
    )
    vz = cast("onp.ArrayND[np.float64, S]", vr * sinb + latitude_velocity * cosb)
    return (vx, vy, vz)


def cartesian_velocity_to_spherical[F: np.floating, S: tuple[Any, ...]](
    vx: onp.ArrayND[F, S],
    vy: onp.ArrayND[F, S],
    vz: onp.ArrayND[F, S],
    lon: AngleArray[F, S],
    lat: AngleArray[F, S],
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform Cartesian velocities to spherical velocity components.

    The returned tangential components point toward increasing longitude and
    latitude. At a pole, the longitude direction is convention-dependent.

    Parameters
    ----------
    vx, vy, vz : Array[float]
        Cartesian velocity components.
    lon, lat : AngleArray[float]
        Longitude and latitude defining the local spherical basis.

    Returns
    -------
    vr : Array[float]
        Radial velocity.
    longitude_velocity : Array[float]
        Tangential velocity toward increasing longitude.
    latitude_velocity : Array[float]
        Tangential velocity toward increasing latitude.

    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = np.sin(lon_radians)
    cosl = np.cos(lon_radians)
    sinb = np.sin(lat_radians)
    cosb = np.cos(lat_radians)
    vr = cast("onp.ArrayND[np.float64, S]", (vx * cosl * cosb + vy * sinl * cosb + vz * sinb).astype(np.float64))
    longitude_velocity = cast("onp.ArrayND[np.float64, S]", (-vx * sinl + vy * cosl).astype(np.float64))
    latitude_velocity = cast("onp.ArrayND[np.float64, S]", (-vx * cosl * sinb - vy * sinl * sinb + vz * cosb).astype(np.float64))
    return (vr, longitude_velocity, latitude_velocity)


def cartesian_to_polar[F: np.floating, S: tuple[Any, ...]](
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
    rxy = cast("onp.ArrayND[np.float64, S]", np.hypot(x, y).astype(np.float64))
    phi = cast("onp.ArrayND[np.float64, S]", np.arctan2(y, x).astype(np.float64))

    return (rxy, AngleArray(phi, "rad"))


def cartesian_vec_to_polar_vec[F: np.floating, S: tuple[Any, ...]](
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
    vr = cast("onp.ArrayND[np.float64, S]", (vx * cosphi + vy * sinphi).astype(np.float64))
    vphi = cast("onp.ArrayND[np.float64, S]", (-vx * sinphi + vy * cosphi).astype(np.float64))

    return (vr, vphi)


def cartesian_to_cylindrical[F: np.floating, S: tuple[Any, ...]](
    x: onp.ArrayND[F, S], y: onp.ArrayND[F, S], z: onp.ArrayND[F, S]
) -> tuple[onp.ArrayND[np.float64, S], AngleArray[np.float64, S], onp.ArrayND[F, S]]:
    """Transform Cartesian positions to cylindrical ``(radius, phi, z)`` coordinates.

    ``phi`` is wrapped to ``[0, 2 * pi)``. Its value at the z axis is defined as
    zero because the azimuth is otherwise undefined.

    Parameters
    ----------
    x, y, z : Array[float]
        Cartesian coordinates.

    Returns
    -------
    radius : Array[float]
        Cylindrical radius.
    phi : AngleArray[float]
        Azimuthal angle in radians.
    z : Array[float]
        Vertical coordinate.

    """
    radius, phi = cartesian_to_polar(x, y)
    phi_radians: onp.ArrayND[np.float64, S] = cast(
        "onp.ArrayND[np.float64, S]", np.mod(phi.to_radians(), 2 * np.pi).astype(np.float64)
    )
    return (radius, AngleArray(phi_radians, "rad"), z)


def cylindrical_to_cartesian[F: np.floating, S: tuple[Any, ...]](
    radius: onp.ArrayND[F, S], phi: AngleArray[F, S], z: onp.ArrayND[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[F, S]]:
    """Transform cylindrical ``(radius, phi, z)`` coordinates to Cartesian coordinates.

    Parameters
    ----------
    radius : Array[float]
        Cylindrical radius.
    phi : AngleArray[float]
        Azimuthal angle.
    z : Array[float]
        Vertical coordinate.

    Returns
    -------
    x, y, z : Array[float]
        Cartesian coordinates.

    """
    phi_radians = phi.to_radians()
    x = cast("onp.ArrayND[np.float64, S]", (radius * np.cos(phi_radians)).astype(np.float64))
    y = cast("onp.ArrayND[np.float64, S]", (radius * np.sin(phi_radians)).astype(np.float64))
    return (x, y, z)


def cartesian_velocity_to_cylindrical[F: np.floating, S: tuple[Any, ...]](
    vx: onp.ArrayND[F, S], vy: onp.ArrayND[F, S], vz: onp.ArrayND[F, S], phi: AngleArray[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[F, S]]:
    """Transform Cartesian velocities to cylindrical ``(v_radius, v_phi, v_z)``.

    Positive ``v_phi`` is toward increasing ``phi``: counter-clockwise from
    the positive x axis toward the positive y axis in a right-handed frame.

    Parameters
    ----------
    vx, vy, vz : Array[float]
        Cartesian velocity components.
    phi : AngleArray[float]
        Azimuthal angle defining the cylindrical basis.

    Returns
    -------
    v_radius : Array[float]
        Radial velocity component.
    v_phi : Array[float]
        Azimuthal velocity component.
    vz : Array[float]
        Vertical velocity component.

    """
    v_radius, v_phi = cartesian_vec_to_polar_vec(vx, vy, phi)
    return (v_radius, v_phi, vz)


def cylindrical_velocity_to_cartesian[F: np.floating, S: tuple[Any, ...]](
    v_radius: onp.ArrayND[F, S], v_phi: onp.ArrayND[F, S], vz: onp.ArrayND[F, S], phi: AngleArray[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[F, S]]:
    """Transform cylindrical ``(v_radius, v_phi, v_z)`` velocities to Cartesian velocities.

    Parameters
    ----------
    v_radius : Array[float]
        Radial velocity component.
    v_phi : Array[float]
        Azimuthal velocity component.
    vz : Array[float]
        Vertical velocity component.
    phi : AngleArray[float]
        Azimuthal angle defining the cylindrical basis.

    Returns
    -------
    vx, vy, vz : Array[float]
        Cartesian velocity components.

    """
    phi_radians = phi.to_radians()
    vx = cast("onp.ArrayND[np.float64, S]", (v_radius * np.cos(phi_radians) - v_phi * np.sin(phi_radians)).astype(np.float64))
    vy = cast("onp.ArrayND[np.float64, S]", (v_radius * np.sin(phi_radians) + v_phi * np.cos(phi_radians)).astype(np.float64))
    return (vx, vy, vz)
