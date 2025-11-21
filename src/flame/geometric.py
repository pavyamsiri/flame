"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
import polars as pl
from optype import numpy as onp

_Shape = TypeVar("_Shape", bound=tuple[Any, ...])
_Float = TypeVar("_Float", bound=np.float64)


def spherical_to_cartesian_numpy(
    lon: onp.ArrayND[_Float, _Shape], lat: onp.ArrayND[_Float, _Shape], distance: onp.ArrayND[_Float, _Shape]
) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
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
    x = cast(onp.ArrayND[np.float64, _Shape], distance * cosb * cosl)
    y = cast(onp.ArrayND[np.float64, _Shape], distance * cosb * sinl)
    z = cast(onp.ArrayND[np.float64, _Shape], distance * sinb)
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
