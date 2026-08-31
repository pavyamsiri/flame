"""Routines to transform from and to the Galactic frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flame._galactic import MAS_PER_YR_TO_KM_PER_S_PER_KPC

from .geometric import cartesian_velocity_to_spherical, spherical_velocity_to_cartesian

if TYPE_CHECKING:
    import polars as pl

    from flame.units import AngleExpr

__all__: list[str] = ["vrpmllpmbb_to_vxvyvz", "vxvyvz_to_vrpmllpmbb"]


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
    longitude_velocity = distance * pmll * MAS_PER_YR_TO_KM_PER_S_PER_KPC
    latitude_velocity = distance * pmbb * MAS_PER_YR_TO_KM_PER_S_PER_KPC
    return spherical_velocity_to_cartesian(vr, longitude_velocity, latitude_velocity, lon, lat)


def vxvyvz_to_vrpmllpmbb(
    vx: pl.Expr, vy: pl.Expr, vz: pl.Expr, lon: AngleExpr, lat: AngleExpr, distance: pl.Expr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Transform Galactic Cartesian velocities to radial and proper-motion velocities.

    Proper motions are undefined at zero distance; the corresponding Polars
    expressions evaluate according to its floating-point division semantics.
    """
    vr, longitude_velocity, latitude_velocity = cartesian_velocity_to_spherical(vx, vy, vz, lon, lat)
    scale = distance * MAS_PER_YR_TO_KM_PER_S_PER_KPC
    return (vr, longitude_velocity / scale, latitude_velocity / scale)
