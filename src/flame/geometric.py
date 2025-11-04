"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from optype import numpy as onp


def spherical_to_cartesian_numpy(
    lon: onp.ArrayND[np.float64], lat: onp.ArrayND[np.float64], distance: onp.ArrayND[np.float64]
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]]:
    """Transform spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : Array[f64]
        The longitude coordinate in radians, in range [0, 2*pi].
    lat : Array[f64]
        The latitude coordinate in radians, in range [0, pi].
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
    cosb = np.cos(lat)
    sinb = np.sin(lat)
    cosl = np.cos(lon)
    sinl = np.sin(lon)
    x = distance * cosb * cosl
    y = distance * cosb * sinl
    z = distance * sinb
    return (x, y, z)


def spherical_to_cartesian_polars(lon: pl.Expr, lat: pl.Expr, distance: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : pl.Expr
        The longitude coordinate in radians, in range [0, 2*pi].
    lat : pl.Expr
        The latitude coordinate in radians, in range [0, pi].
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
    cosb = lat.cos()
    sinb = lat.sin()
    cosl = lon.cos()
    sinl = lon.sin()
    x = distance * cosb * cosl
    y = distance * cosb * sinl
    z = distance * sinb
    return (x, y, z)
