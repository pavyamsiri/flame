"""Routines to transform from and to the equatorial ICRS frame via polars."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

from flame._icrs import DEC_NGP, RA_NGP, TRANSFORMATION_MATRIX
from flame.units import AngleExpr


__all__: list[str] = ["radec_to_lb", "pmrapmdec_to_pmllpmbb"]


def radec_to_lb(ra: AngleExpr, dec: AngleExpr) -> tuple[AngleExpr, AngleExpr]:
    """Transform from equatorial ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra : AngleExpr
        The right ascension.
    dec : AngleExpr
        The declination.

    Returns
    -------
    lon : AngleExpr
        The longitude.
    lat : AngleExpr
        The latitude.

    """
    ra_radians = ra.to_radians()
    dec_radians = dec.to_radians()
    cosdec_cosra = dec_radians.cos() * ra_radians.cos()
    cosdec_sinra = dec_radians.cos() * ra_radians.sin()
    sindec = dec_radians.sin()

    gl_x = cast(
        pl.Expr,
        TRANSFORMATION_MATRIX[0, 0] * cosdec_cosra
        + TRANSFORMATION_MATRIX[0, 1] * cosdec_sinra
        + TRANSFORMATION_MATRIX[0, 2] * sindec,
    )
    gl_y = cast(
        pl.Expr,
        TRANSFORMATION_MATRIX[1, 0] * cosdec_cosra
        + TRANSFORMATION_MATRIX[1, 1] * cosdec_sinra
        + TRANSFORMATION_MATRIX[1, 2] * sindec,
    )
    gl_z = cast(
        pl.Expr,
        TRANSFORMATION_MATRIX[2, 0] * cosdec_cosra
        + TRANSFORMATION_MATRIX[2, 1] * cosdec_sinra
        + TRANSFORMATION_MATRIX[2, 2] * sindec,
    )

    gl_z = pl.when(gl_z > +1.0).then(+1.0).otherwise(gl_z)
    gl_z = pl.when(gl_z < -1.0).then(-1.0).otherwise(gl_z)

    lat = gl_z.arcsin()
    lon = pl.arctan2(gl_y / lat.cos(), gl_x / lat.cos())
    lon = pl.when(lon < 0.0).then(lon + 2 * np.pi).otherwise(lon)
    return (AngleExpr(lon, "rad"), AngleExpr(lat, "rad"))


def pmrapmdec_to_pmllpmbb(pmracosdec: pl.Expr, pmdec: pl.Expr, *, ra: AngleExpr, dec: AngleExpr) -> tuple[pl.Expr, pl.Expr]:
    """Transform from equatorial ICRS proper motions to Galactic proper motions.

    Parameters
    ----------
    pmracosdec : pl.Expr
        The proper motion in right ascension corrected by cos(dec) in mas/yr.
    pmdec : pl.Expr
        The proper motion in declination in mas/yr.
    ra : AngleExpr
        The right ascension in radians.
    dec : AngleExpr
        The declination in radians.

    Returns
    -------
    pmll : pl.Expr
        The proper motion in longitude corrected by cos(b) in mas/yr.
    pmbb : pl.Expr
        The proper motion in latitude in mas/yr.

    """
    ra_radians = ra.to_radians()
    dec_radians = dec.to_radians()
    # Add epsilon to handle pole
    treated_dec = pl.when(dec_radians == DEC_NGP).then(dec_radians + 1e-16).otherwise(dec_radians)
    sindec_ngp: float = np.sin(DEC_NGP)
    cosdec_ngp: float = np.cos(DEC_NGP)
    sindec = treated_dec.sin()
    cosdec = treated_dec.cos()
    sinrarangp = (ra_radians - RA_NGP).sin()
    cosrarangp = (ra_radians - RA_NGP).cos()
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = (cosphi**2.0 + sinphi**2.0).sqrt()
    cosphi /= norm
    sinphi /= norm

    pmll = cosphi * pmracosdec + sinphi * pmdec
    pmbb = -sinphi * pmracosdec + cosphi * pmdec
    return (pmll, pmbb)
