"""Routines to transform from and to the equatorial ICRS frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from optype import numpy as onp

from flame._icrs import DEC_NGP, RA_NGP, THETA
from flame.units import AngleArray

if TYPE_CHECKING:
    from typing import Final


__all__: list[str] = ["radec_to_lb", "pmrapmdec_to_pmllpmbb"]


_THETA_MATRIX: Final[onp.Array2D[np.float64]] = np.array(
    [
        [np.cos(THETA), np.sin(THETA), 0.0],
        [np.sin(THETA), -np.cos(THETA), 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_DEC_MATRIX: Final[onp.Array2D[np.float64]] = np.array(
    [
        [-np.sin(DEC_NGP), 0.0, np.cos(DEC_NGP)],
        [0.0, 1.0, 0.0],
        [np.cos(DEC_NGP), 0.0, np.sin(DEC_NGP)],
    ],
    dtype=np.float64,
)
_RA_MATRIX: Final[onp.Array2D[np.float64]] = np.array(
    ((np.cos(RA_NGP), np.sin(RA_NGP), 0.0), (-np.sin(RA_NGP), np.cos(RA_NGP), 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
_TRANSFORMATION_MATRIX: Final[onp.Array2D[np.float64]] = _THETA_MATRIX @ _DEC_MATRIX @ _RA_MATRIX


def radec_to_lb[F: np.float64, S: tuple[Any, ...]](
    ra: AngleArray[F, S], dec: AngleArray[F, S]
) -> tuple[AngleArray[np.float64, S], AngleArray[np.float64, S]]:
    """Transform from equatorial ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra : AngleArray[float]
        The right ascension.
    dec : AngleArray[float]
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
    xyz: onp.Array2D[np.float64] = np.array(
        [
            np.cos(dec_radians) * np.cos(ra_radians),
            np.cos(dec_radians) * np.sin(ra_radians),
            np.sin(dec_radians),
        ],
        dtype=np.float64,
    )
    gl_xyz = np.dot(_TRANSFORMATION_MATRIX, xyz)
    # Clamp
    gl_xyz[2][gl_xyz[2] > +1.0] = +1.0
    gl_xyz[2][gl_xyz[2] < -1.0] = -1.0

    lat = np.arcsin(gl_xyz[2])
    lon = np.arctan2(gl_xyz[1] / np.cos(lat), gl_xyz[0] / np.cos(lat))
    lon[lon < 0.0] += 2 * np.pi
    return (AngleArray(lon, "rad"), AngleArray(lat, "rad"))


def pmrapmdec_to_pmllpmbb[F: np.float64, S: tuple[Any, ...]](
    pmracosdec: onp.ArrayND[F, S], pmdec: onp.ArrayND[F, S], *, ra: AngleArray[F, S], dec: AngleArray[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
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

    pmll: onp.ArrayND[np.float64, S] = cast(onp.ArrayND[np.float64, S], (cosphi * pmracosdec + sinphi * pmdec).astype(np.float64))
    pmbb: onp.ArrayND[np.float64, S] = cast(
        onp.ArrayND[np.float64, S], (-sinphi * pmracosdec + cosphi * pmdec).astype(np.float64)
    )

    return (pmll, pmbb)
