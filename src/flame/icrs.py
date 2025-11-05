"""Routines to transform from and to the equatorial ICRS frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from optype import numpy as onp

if TYPE_CHECKING:
    from typing import TypeVar, Any, Final

    _Shape = TypeVar("_Shape", bound=Any)
    _Float = TypeVar("_Float", bound=np.float64)


# These constants were obtained from galpy's `get_epoch_angles`.
THETA: Final[float] = 2.1455668515225916
DEC_NGP: Final[float] = 0.4734773249532947
RA_NGP: Final[float] = 3.366032882941063


def radec_to_lb_numpy(
    ra: onp.ArrayND[_Float, _Shape], dec: onp.ArrayND[_Float, _Shape]
) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
    """Transform from equatorial ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra : Array[float]
        The right ascension in radians.
    dec : Array[float]
        The declination in radians.

    Returns
    -------
    lon : Array[float]
        The longitude in radians.
    lat : Array[float]
        The latitude in radians.

    """
    transform = _get_transformation_matrix()
    xyz = cast(
        onp.Array2D[np.float64],
        np.array(
            [
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
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
    return (lon, lat)


def radec_to_lb_polars(
    ra: pl.Expr,
    dec: pl.Expr,
) -> tuple[pl.Expr, pl.Expr]:
    """Transform from equatorial ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra : pl.Expr
        The right ascension in radians.
    dec : pl.Expr
        The declination in radians.

    Returns
    -------
    lon : pl.Expr
        The longitude in radians.
    lat : pl.Expr
        The latitude in radians.

    """
    transform = _get_transformation_matrix()
    cosdec_cosra = dec.cos() * ra.cos()
    cosdec_sinra = dec.cos() * ra.sin()
    sindec = dec.sin()

    gl_x = cast(pl.Expr, transform[0, 0] * cosdec_cosra + transform[0, 1] * cosdec_sinra + transform[0, 2] * sindec)
    gl_y = cast(pl.Expr, transform[1, 0] * cosdec_cosra + transform[1, 1] * cosdec_sinra + transform[1, 2] * sindec)
    gl_z = cast(pl.Expr, transform[2, 0] * cosdec_cosra + transform[2, 1] * cosdec_sinra + transform[2, 2] * sindec)

    gl_z = pl.when(gl_z > +1.0).then(+1.0).otherwise(gl_z)
    gl_z = pl.when(gl_z < -1.0).then(-1.0).otherwise(gl_z)

    lat = gl_z.arcsin()
    lon = pl.arctan2(gl_y / lat.cos(), gl_x / lat.cos())
    lon = pl.when(lon < 0.0).then(lon + 2 * np.pi).otherwise(lon)
    return (lon, lat)


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


# if __name__ == "__main__":
#     data = pl.DataFrame(
#         {
#             "ra": [0],
#             "dec": [0],
#         }
#     )

#     lon, lat = radec_to_lb_polars(pl.col("ra"), pl.col("dec"))

#     print(data.with_columns((lon.alias("lon"), lat.alias("lat"))))
#     print(radec_to_lb_numpy(np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)))
