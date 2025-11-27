"""Routines to transform from and to the equatorial ICRS frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import polars as pl
from optype import numpy as onp

from .units import AngleArray, AngleExpr

if TYPE_CHECKING:
    from typing import Final

_Shape = TypeVar("_Shape", bound=tuple[Any, ...])
_Float = TypeVar("_Float", bound=np.float64)


# These constants were obtained from galpy's `get_epoch_angles`.
THETA: Final[float] = 2.1455668515225916
DEC_NGP: Final[float] = 0.4734773249532947
RA_NGP: Final[float] = 3.366032882941063


def radec_to_lb_numpy(
    ra: AngleArray[_Float, _Shape], dec: AngleArray[_Float, _Shape]
) -> tuple[AngleArray[np.float64, _Shape], AngleArray[np.float64, _Shape]]:
    """Transform from equatorial ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra : AngleArray[f64]
        The right ascension.
    dec : AngleArray[f64]
        The declination.

    Returns
    -------
    lon : AngleArray[f64]
        The longitude.
    lat : AngleArray[f64]
        The latitude.

    """
    dec_radians = dec.to_radians()
    ra_radians = ra.to_radians()
    transform = _get_transformation_matrix()
    xyz = cast(
        onp.Array2D[np.float64],
        np.array(
            [
                np.cos(dec_radians) * np.cos(ra_radians),
                np.cos(dec_radians) * np.sin(ra_radians),
                np.sin(dec_radians),
            ],
            dtype=np.float64,
        ),
    )
    gl_xyz = np.dot(transform, xyz)
    # Clamp
    gl_xyz[2][gl_xyz[2] > +1.0] = +1.0
    gl_xyz[2][gl_xyz[2] < -1.0] = -1.0

    lat = np.arcsin(gl_xyz[2])
    lon = np.arctan2(gl_xyz[1] / np.cos(lat), gl_xyz[0] / np.cos(lat))
    lon[lon < 0.0] += 2 * np.pi
    return (AngleArray(lon, "rad"), AngleArray(lat, "rad"))


def radec_to_lb_polars(
    ra: AngleExpr,
    dec: AngleExpr,
) -> tuple[AngleExpr, AngleExpr]:
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
    transform = _get_transformation_matrix()
    cosdec_cosra = dec_radians.cos() * ra_radians.cos()
    cosdec_sinra = dec_radians.cos() * ra_radians.sin()
    sindec = dec_radians.sin()

    gl_x = cast(pl.Expr, transform[0, 0] * cosdec_cosra + transform[0, 1] * cosdec_sinra + transform[0, 2] * sindec)
    gl_y = cast(pl.Expr, transform[1, 0] * cosdec_cosra + transform[1, 1] * cosdec_sinra + transform[1, 2] * sindec)
    gl_z = cast(pl.Expr, transform[2, 0] * cosdec_cosra + transform[2, 1] * cosdec_sinra + transform[2, 2] * sindec)

    gl_z = pl.when(gl_z > +1.0).then(+1.0).otherwise(gl_z)
    gl_z = pl.when(gl_z < -1.0).then(-1.0).otherwise(gl_z)

    lat = gl_z.arcsin()
    lon = pl.arctan2(gl_y / lat.cos(), gl_x / lat.cos())
    lon = pl.when(lon < 0.0).then(lon + 2 * np.pi).otherwise(lon)
    return (AngleExpr(lon, "rad"), AngleExpr(lat, "rad"))


def pmrapmdec_to_pmllpmbb_numpy(
    pmracosdec: onp.ArrayND[_Float, _Shape],
    pmdec: onp.ArrayND[_Float, _Shape],
    *,
    ra: AngleArray[_Float, _Shape],
    dec: AngleArray[_Float, _Shape],
) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
    """Transform from equatorial ICRS proper motions to Galactic proper motions.

    Parameters
    ----------
    pmracosdec : Array[float]
        The proper motion in right ascension corrected by cos(dec) in mas/yr.
    pmdec : Array[float]
        The proper motion in declination in mas/yr.
    ra : AngleArray[float]
        The right ascension in radians.
    dec : AngleArray[float]
        The declination in radians.

    Returns
    -------
    pmllcosb : Array[float]
        The proper motion in longitude corrected by cos(b) in mas/yr.
    pmbb : Array[float]
        The proper motion in latitude in mas/yr.

    """
    ra_radians = ra.to_radians()
    dec_radians = dec.to_radians()

    treated_dec = np.copy(dec_radians).astype(np.float64)
    # Add epsilon to handle pole
    treated_dec[treated_dec == DEC_NGP] += 1e-16
    sindec_ngp: float = np.sin(DEC_NGP)
    cosdec_ngp: float = np.cos(DEC_NGP)
    sindec = np.sin(treated_dec)
    cosdec = np.cos(treated_dec)
    sinrarangp = np.sin(ra_radians - RA_NGP)
    cosrarangp = np.cos(ra_radians - RA_NGP)
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = np.sqrt(cosphi**2.0 + sinphi**2.0)
    cosphi /= norm
    sinphi /= norm

    pmll: onp.ArrayND[np.float64, _Shape] = cast(
        onp.ArrayND[np.float64, _Shape], (cosphi * pmracosdec + sinphi * pmdec).astype(np.float64)
    )
    pmbb: onp.ArrayND[np.float64, _Shape] = cast(
        onp.ArrayND[np.float64, _Shape], (-sinphi * pmracosdec + cosphi * pmdec).astype(np.float64)
    )

    return (pmll, pmbb)


def pmrapmdec_to_pmllpmbb_polars(
    pmracosdec: pl.Expr,
    pmdec: pl.Expr,
    *,
    ra: AngleExpr,
    dec: AngleExpr,
) -> tuple[pl.Expr, pl.Expr]:
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


def _get_transformation_matrix() -> onp.Array2D[np.float64]:
    theta: float = THETA
    dec_ngp: float = DEC_NGP
    ra_ngp: float = RA_NGP
    theta_matrix = cast(
        onp.Array2D[np.float64],
        np.array(
            [
                [np.cos(theta), np.sin(theta), 0.0],
                [np.sin(theta), -np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    dec_matrix = cast(
        onp.Array2D[np.float64],
        np.array(
            [
                [-np.sin(dec_ngp), 0.0, np.cos(dec_ngp)],
                [0.0, 1.0, 0.0],
                [np.cos(dec_ngp), 0.0, np.sin(dec_ngp)],
            ],
            dtype=np.float64,
        ),
    )
    ra_matrix = cast(
        onp.Array2D[np.float64],
        np.array(
            ((np.cos(ra_ngp), np.sin(ra_ngp), 0.0), (-np.sin(ra_ngp), np.cos(ra_ngp), 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
    )
    return cast(onp.Array2D[np.float64], theta_matrix @ dec_matrix @ ra_matrix)
