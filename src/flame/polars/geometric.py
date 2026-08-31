"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

import numpy as np
import polars as pl

from flame.units import AngleExpr

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


def spherical_to_cartesian(lon: AngleExpr, lat: AngleExpr, distance: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : AngleExpr
        The longitude coordinate.
    lat : AngleExpr
        The latitude coordinate.
    distance : pl.Expr
        The distance.

    Returns
    -------
    x : pl.Expr
        The x coordinate.
    y : pl.Expr
        The y coordinate.
    z : pl.Expr
        The z coordinate.

    """
    # Compute Cartesian coordinates in Galactic frame
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    cosb = lat_radians.cos()
    sinb = lat_radians.sin()
    cosl = lon_radians.cos()
    sinl = lon_radians.sin()
    x = distance * cosb * cosl
    y = distance * cosb * sinl
    z = distance * sinb
    return (x, y, z)


def cartesian_to_spherical(x: pl.Expr, y: pl.Expr, z: pl.Expr) -> tuple[AngleExpr, AngleExpr, pl.Expr]:
    """Transform Cartesian coordinates to spherical coordinates.

    Longitudes are wrapped to ``[0, 2π)``. At the origin the direction is
    undefined; Polars' ``arctan2(0, 0)`` convention yields zero angles.
    """
    rxy = (x**2 + y**2).sqrt()
    distance = (rxy**2 + z**2).sqrt()
    lon = pl.arctan2(y, x) % (2 * np.pi)
    lat = pl.arctan2(z, rxy)
    return (AngleExpr(lon, "rad"), AngleExpr(lat, "rad"), distance)


def spherical_velocity_to_cartesian(
    vr: pl.Expr,
    longitude_velocity: pl.Expr,
    latitude_velocity: pl.Expr,
    lon: AngleExpr,
    lat: AngleExpr,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform spherical velocity components to Cartesian velocity components.

    ``longitude_velocity`` and ``latitude_velocity`` are tangential velocities
    in the directions of increasing longitude and latitude, respectively.
    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = lon_radians.sin()
    cosl = lon_radians.cos()
    sinb = lat_radians.sin()
    cosb = lat_radians.cos()

    vx = vr * cosl * cosb - longitude_velocity * sinl - latitude_velocity * cosl * sinb
    vy = vr * sinl * cosb + longitude_velocity * cosl - latitude_velocity * sinl * sinb
    vz = vr * sinb + latitude_velocity * cosb
    return (vx, vy, vz)


def cartesian_velocity_to_spherical(
    vx: pl.Expr, vy: pl.Expr, vz: pl.Expr, lon: AngleExpr, lat: AngleExpr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform Cartesian velocities to spherical velocity components.

    The returned tangential components point toward increasing longitude and
    latitude. At a pole, the longitude direction is convention-dependent.
    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = lon_radians.sin()
    cosl = lon_radians.cos()
    sinb = lat_radians.sin()
    cosb = lat_radians.cos()
    vr = vx * cosl * cosb + vy * sinl * cosb + vz * sinb
    longitude_velocity = -vx * sinl + vy * cosl
    latitude_velocity = -vx * cosl * sinb - vy * sinl * sinb + vz * cosb
    return (vr, longitude_velocity, latitude_velocity)


def cartesian_to_polar(x: pl.Expr, y: pl.Expr) -> tuple[pl.Expr, AngleExpr]:
    """Transform Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : pl.Expr
        The x coordinate.
    y : pl.Expr
        The y coordinate.

    Returns
    -------
    r : pl.Expr
        The radial coordinate.
    phi : AngleExpr
        The azimuthal coordinate.

    """
    rxy = (x**2 + y**2).sqrt()
    phi = pl.arctan2(y, x)

    return (rxy, AngleExpr(phi, "rad"))


def cartesian_vec_to_polar_vec(vx: pl.Expr, vy: pl.Expr, phi: AngleExpr) -> tuple[pl.Expr, pl.Expr]:
    """Transform Cartesian velocity to polar velocity.

    Parameters
    ----------
    vx : pl.Expr
        The x-velocity.
    vy : pl.Expr
        The y-velocity.
    phi : pl.Expr
        The azimuthal coordinate in radians.

    Returns
    -------
    vr : pl.Expr
        The radial velocity.
    vphi : pl.Expr
        The tangential velocity.

    """
    phi_rad = phi.to_radians()
    cosphi = phi_rad.cos()
    sinphi = phi_rad.sin()
    vr = vx * cosphi + vy * sinphi
    vphi = -vx * sinphi + vy * cosphi

    return (vr, vphi)


def cartesian_to_cylindrical(x: pl.Expr, y: pl.Expr, z: pl.Expr) -> tuple[pl.Expr, AngleExpr, pl.Expr]:
    """Transform Cartesian positions to cylindrical ``(radius, phi, z)`` coordinates.

    ``phi`` is wrapped to ``[0, 2π)``. Its value at the z axis is defined as
    zero because the azimuth is otherwise undefined.
    """
    radius, phi = cartesian_to_polar(x, y)
    return (radius, AngleExpr(phi.to_radians() % (2 * np.pi), "rad"), z)


def cylindrical_to_cartesian(radius: pl.Expr, phi: AngleExpr, z: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform cylindrical ``(radius, phi, z)`` coordinates to Cartesian coordinates."""
    phi_radians = phi.to_radians()
    return (radius * phi_radians.cos(), radius * phi_radians.sin(), z)


def cartesian_velocity_to_cylindrical(
    vx: pl.Expr, vy: pl.Expr, vz: pl.Expr, phi: AngleExpr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform Cartesian velocities to cylindrical ``(v_radius, v_phi, v_z)``.

    Positive ``v_phi`` is toward increasing ``phi``: counter-clockwise from
    the positive x axis toward the positive y axis in a right-handed frame.
    """
    v_radius, v_phi = cartesian_vec_to_polar_vec(vx, vy, phi)
    return (v_radius, v_phi, vz)


def cylindrical_velocity_to_cartesian(
    v_radius: pl.Expr, v_phi: pl.Expr, vz: pl.Expr, phi: AngleExpr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform cylindrical ``(v_radius, v_phi, v_z)`` velocities to Cartesian velocities."""
    phi_radians = phi.to_radians()
    return (
        v_radius * phi_radians.cos() - v_phi * phi_radians.sin(),
        v_radius * phi_radians.sin() + v_phi * phi_radians.cos(),
        vz,
    )
