"""Routines to transform between different geometric coordinate systems."""

from __future__ import annotations


import polars as pl

from flame.units import AngleExpr

__all__: list[str] = ["spherical_to_cartesian", "cartesian_to_polar", "cartesian_vec_to_polar_vec"]


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
