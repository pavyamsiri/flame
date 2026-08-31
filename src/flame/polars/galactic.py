"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

import polars as pl

from flame.units import AngleExpr
from flame._galactic import MAS_PER_YR_TO_KM_PER_S_PER_KPC


__all__: list[str] = ["vrpmllpmbb_to_vxvyvz"]


def vrpmllpmbb_to_vxvyvz(
    vr: pl.Expr,
    pmll: pl.Expr,
    pmbb: pl.Expr,
    lon: AngleExpr,
    lat: AngleExpr,
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
    lon : AngleExpr
        The Galactic longitude.
    lat : AngleExpr
        The Galactic latitude.
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
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    sinl = lon_radians.sin()
    cosl = lon_radians.cos()
    sinb = lat_radians.sin()
    cosb = lat_radians.cos()

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

    vx = rotation_0_0 * v0 + rotation_0_1 * v1 + rotation_0_2 * v2
    vy = rotation_1_0 * v0 + rotation_1_1 * v1 + rotation_1_2 * v2
    vz = rotation_2_0 * v0 + rotation_2_1 * v1 + rotation_2_2 * v2

    return (vx, vy, vz)
