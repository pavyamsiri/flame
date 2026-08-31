"""Routines to transform from and to the equatorial ICRS frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from flame._icrs import DEC_NGP, RA_NGP, TRANSFORMATION_MATRIX
from flame.units import AngleArray

from .galactic import vrpmllpmbb_to_vxvyvz
from .geometric import spherical_to_cartesian

if TYPE_CHECKING:
    from optype import numpy as onp

__all__: list[str] = [
    "icrs_to_galactic_cartesian",
    "icrs_to_galactic_velocity",
    "lb_to_radec",
    "pmllpmbb_to_pmrapmdec",
    "pmrapmdec_to_pmllpmbb",
    "radec_to_lb",
]


def icrs_to_galactic_cartesian[F: np.float64, S: tuple[Any, ...]](
    ra: AngleArray[F, S], dec: AngleArray[F, S], distance: onp.ArrayND[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform from equatorial ICRS coordinates to Galactic Cartesian coordinates.

    Parameters
    ----------
    ra : AngleArray[float]
        The right ascension.
    dec : AngleArray[float]
        The declination.
    distance : Array[float]
        The distance.

    Returns
    -------
    x : Array[f64]
        The Galactic Cartesian x coordinate.
    y : Array[f64]
        The Galactic Cartesian y coordinate.
    z : Array[f64]
        The Galactic Cartesian z coordinate.

    """
    lon, lat = radec_to_lb(ra, dec)
    x, y, z = spherical_to_cartesian(lon, lat, distance)
    return x, y, z


def icrs_to_galactic_velocity[F: np.float64, S: tuple[Any, ...]](
    vr: onp.ArrayND[F, S],
    pmracosdec: onp.ArrayND[F, S],
    pmdec: onp.ArrayND[F, S],
    distance: onp.ArrayND[F, S],
    *,
    ra: AngleArray[F, S],
    dec: AngleArray[F, S],
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform from equatorial ICRS coordinates and proper motions to Galactic Cartesian velocities.

    Parameters
    ----------
    vr : Array[float]
        The radial velocity in km/s.
    pmracosdec : Array[float]
        The proper motion in right ascension corrected by cos(dec) in mas/yr.
    pmdec : Array[float]
        The proper motion in declination in mas/yr.
    distance : Array[float]
        The distance.
    ra : AngleArray[float]
        The right ascension in radians.
    dec : AngleArray[float]
        The declination in radians.

    Returns
    -------
    vx : Array[f64]
        The Galactic Cartesian x velocity.
    vy : Array[f64]
        The Galactic Cartesian y velocity.
    vz : Array[f64]
        The Galactic Cartesian z velocity.

    """
    lon, lat = radec_to_lb(ra, dec)
    pmll, pmbb = pmrapmdec_to_pmllpmbb(pmracosdec, pmdec, ra=ra, dec=dec)
    return vrpmllpmbb_to_vxvyvz(vr, pmll, pmbb, lon, lat, distance)


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
    gl_xyz = np.dot(TRANSFORMATION_MATRIX, xyz)
    # Clamp
    gl_xyz[2][gl_xyz[2] > +1.0] = +1.0
    gl_xyz[2][gl_xyz[2] < -1.0] = -1.0

    lat = np.arcsin(gl_xyz[2])
    lon = np.arctan2(gl_xyz[1] / np.cos(lat), gl_xyz[0] / np.cos(lat))
    lon[lon < 0.0] += 2 * np.pi
    lon = np.mod(lon, 2 * np.pi)
    return (AngleArray(lon, "rad"), AngleArray(lat, "rad"))


def lb_to_radec[F: np.float64, S: tuple[Any, ...]](
    lon: AngleArray[F, S], lat: AngleArray[F, S]
) -> tuple[AngleArray[np.float64, S], AngleArray[np.float64, S]]:
    """Transform Galactic coordinates to equatorial ICRS coordinates.

    Parameters
    ----------
    lon : AngleArray[float]
        The Galactic longitude.
    lat : AngleArray[float]
        The Galactic latitude.

    Returns
    -------
    ra : AngleArray[f64]
        The right ascension in the range `[0, 2pi)`.
    dec : AngleArray[f64]
        The declination.

    """
    lon_radians = lon.to_radians()
    lat_radians = lat.to_radians()
    xyz: onp.Array2D[np.float64] = np.array(
        [
            np.cos(lat_radians) * np.cos(lon_radians),
            np.cos(lat_radians) * np.sin(lon_radians),
            np.sin(lat_radians),
        ],
        dtype=np.float64,
    )
    icrs_xyz = np.dot(TRANSFORMATION_MATRIX.T, xyz)
    icrs_xyz[2][icrs_xyz[2] > +1.0] = +1.0
    icrs_xyz[2][icrs_xyz[2] < -1.0] = -1.0

    dec = np.arcsin(icrs_xyz[2])
    ra = np.arctan2(icrs_xyz[1] / np.cos(dec), icrs_xyz[0] / np.cos(dec))
    ra[ra < 0.0] += 2 * np.pi
    ra = np.mod(ra, 2 * np.pi)
    return (AngleArray(ra, "rad"), AngleArray(dec, "rad"))


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

    pmll: onp.ArrayND[np.float64, S] = cast("onp.ArrayND[np.float64, S]", (cosphi * pmracosdec + sinphi * pmdec).astype(np.float64))
    pmbb: onp.ArrayND[np.float64, S] = cast(
        "onp.ArrayND[np.float64, S]", (-sinphi * pmracosdec + cosphi * pmdec).astype(np.float64)
    )

    return (pmll, pmbb)


def pmllpmbb_to_pmrapmdec[F: np.float64, S: tuple[Any, ...]](
    pmll: onp.ArrayND[F, S], pmbb: onp.ArrayND[F, S], *, lon: AngleArray[F, S], lat: AngleArray[F, S]
) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
    """Transform Galactic proper motions to equatorial ICRS proper motions.

    Parameters
    ----------
    pmll : Array[float]
        The proper motion in longitude corrected by cos(b) in mas/yr.
    pmbb : Array[float]
        The proper motion in latitude in mas/yr.
    lon : AngleArray[float]
        The Galactic longitude in radians.
    lat : AngleArray[float]
        The Galactic latitude in radians.

    Returns
    -------
    pmracosdec : Array[f64]
        The proper motion in right ascension corrected by cos(dec) in mas/yr.
    pmdec : Array[f64]
        The proper motion in declination in mas/yr.

    """
    ra, dec = lb_to_radec(lon, lat)
    ra_radians = ra.to_radians()
    dec_radians = dec.to_radians()

    treated_dec = np.copy(dec_radians).astype(np.float64)
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

    pmracosdec: onp.ArrayND[np.float64, S] = cast(
        "onp.ArrayND[np.float64, S]", (cosphi * pmll - sinphi * pmbb).astype(np.float64)
    )
    pmdec: onp.ArrayND[np.float64, S] = cast(
        "onp.ArrayND[np.float64, S]", (sinphi * pmll + cosphi * pmbb).astype(np.float64)
    )
    return (pmracosdec, pmdec)
