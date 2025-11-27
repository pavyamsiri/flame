"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
import polars as pl
from optype import numpy as onp

from .units import AngleExpr, AngleArray

_Shape = TypeVar("_Shape", bound=tuple[Any, ...])
_Float = TypeVar("_Float", bound=np.float64)


def spherical_to_cartesian_numpy(
    lon: AngleArray[_Float, _Shape], lat: AngleArray[_Float, _Shape], distance: onp.ArrayND[_Float, _Shape]
) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
    """Transform spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : AngleArray[f64]
        The longitude coordinate.
    lat : AngleArray[f64]
        The latitude coordinate.
    distance : Array[f64]
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
    x = cast(onp.ArrayND[np.float64, _Shape], distance * cosb * cosl)
    y = cast(onp.ArrayND[np.float64, _Shape], distance * cosb * sinl)
    z = cast(onp.ArrayND[np.float64, _Shape], distance * sinb)
    return (x, y, z)


def spherical_to_cartesian_polars(lon: AngleExpr, lat: AngleExpr, distance: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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


def cartesian_vec_to_polar_vec_numpy(
    vx: onp.ArrayND[_Float, _Shape], vy: onp.ArrayND[_Float, _Shape], phi: AngleArray[_Float, _Shape]
) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
    """Transform Cartesian velocity to polar velocity.

    Parameters
    ----------
    vx : Array[f64]
        The x-velocity.
    vy : Array[f64]
        The y-velocity.
    phi : Array[f64]
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
    vr = cast(onp.ArrayND[np.float64, _Shape], (vx * cosphi + vy * sinphi).astype(np.float64))
    vphi = cast(onp.ArrayND[np.float64, _Shape], (-vx * sinphi + vy * cosphi).astype(np.float64))

    return (vr, vphi)


def cartesian_vec_to_polar_vec_polars(vx: pl.Expr, vy: pl.Expr, phi: AngleExpr) -> tuple[pl.Expr, pl.Expr]:
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
