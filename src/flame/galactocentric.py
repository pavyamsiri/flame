"""Routines to transform to and from the Galactocentric frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any, Literal, TypeVar

    from optype import numpy as onp

    _Shape = TypeVar("_Shape", bound=Any)
    _Float = TypeVar("_Float", bound=np.float64)

DEFAULT_SUN_X: Final[float] = -8.122
DEFAULT_SUN_Y: Final[float] = 0
DEFAULT_SUN_Z: Final[float] = 0.0208
DEFAULT_SUN_VX: Final[float] = 12.9
DEFAULT_SUN_VY: Final[float] = 245.6
DEFAULT_SUN_VZ: Final[float] = 7.78


class GalactocentricFrame:
    """Definition of the Galactocentric frame."""

    def __init__(
        self,
        sun_x: float = DEFAULT_SUN_X,
        sun_y: float = DEFAULT_SUN_Y,
        sun_z: float = DEFAULT_SUN_Z,
        sun_vx: float = DEFAULT_SUN_VX,
        sun_vy: float = DEFAULT_SUN_VY,
        sun_vz: float = DEFAULT_SUN_VZ,
    ) -> None:
        self._sun_x: float = sun_x
        self._sun_y: float = sun_y
        self._sun_z: float = sun_z
        self._sun_vx: float = sun_vx
        self._sun_vy: float = sun_vy
        self._sun_vz: float = sun_vz

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(sun_x={self._sun_x},sun_y={self._sun_y},sun_z={self._sun_z},sun_vx={self._sun_vx},sun_vy={self._sun_vy},sun_vz={self._sun_vz})"

    def sun_x(self) -> float:
        return self._sun_x

    def sun_y(self) -> float:
        return self._sun_y

    def sun_z(self) -> float:
        return self._sun_z

    def sun_vx(self) -> float:
        return self._sun_vx

    def sun_vy(self) -> float:
        return self._sun_vy

    def sun_vz(self) -> float:
        return self._sun_vz

    def distance_to_gc(self) -> float:
        return np.sqrt(self._sun_x**2 + self._sun_y**2 + self._sun_z**2)

    def gl_xyz_to_gc_xyz_numpy(
        self,
        u: onp.ArrayND[_Float, _Shape],
        v: onp.ArrayND[_Float, _Shape],
        w: onp.ArrayND[_Float, _Shape],
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
        """Transform Cartesian coordinates in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        u : Array[f64]
            The x coordinate in the Galactic frame.
        v : Array[f64]
            The y coordinate in the Galactic frame.
        w : Array[f64]
            The z coordinate in the Galactic frame.
        handedness : "left" or "right"
            The handedness of the coordinate system.

        Returns
        -------
        gc_x : Array[f64]
            The x coordinate in the Galactocentric frame.
        gc_y : Array[f64]
            The y coordinate in the Galactocentric frame.
        gc_z : Array[f64]
            The z coordinate in the Galactocentric frame.
        """

        sun_x: float = self._sun_x
        sun_y: float = self._sun_y
        sun_z: float = self._sun_z
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)

        gc_r: onp.ArrayND[np.float64, _Shape]
        gc_y: onp.ArrayND[np.float64, _Shape]
        gc_z: onp.ArrayND[np.float64, _Shape]
        if handedness == "right":
            gc_r = -u + sun_distance  # pyright: ignore[reportAssignmentType]
            gc_y = v.astype(np.float64)
            gc_z = w.astype(np.float64)
        else:
            gc_r = -u + sun_distance  # pyright: ignore[reportAssignmentType]
            gc_y = -v
            gc_z = w

        if np.isclose(sun_distance, 0):
            rotated_gc_r = gc_r
            rotated_gc_z = gc_z
        else:
            cosb: float = sun_r / sun_distance
            sinb: float = sun_z / sun_distance

            rotated_gc_r = cosb * gc_r - sinb * gc_z
            rotated_gc_z = sinb * gc_r + cosb * gc_z  # pyright: ignore[reportAssignmentType]

        rotated_gc_x: onp.ArrayND[np.float64, _Shape]
        rotated_gc_y: onp.ArrayND[np.float64, _Shape]
        rotated_gc_z: onp.ArrayND[np.float64, _Shape]
        if np.isclose(sun_r, 0):
            rotated_gc_x = rotated_gc_r  # pyright: ignore[reportAssignmentType]
            rotated_gc_y = gc_y
            rotated_gc_z = rotated_gc_z
        else:
            cosl: float = sun_x / sun_r
            sinl: float = sun_y / sun_r

            rotated_gc_x = cosl * rotated_gc_r - sinl * gc_y  # pyright: ignore[reportAssignmentType]
            rotated_gc_y = sinl * rotated_gc_r + cosl * gc_y  # pyright: ignore[reportAssignmentType]

        return (rotated_gc_x, rotated_gc_y, rotated_gc_z)

    def gl_xyz_to_gc_xyz_polars(
        self,
        u: pl.Expr,
        v: pl.Expr,
        w: pl.Expr,
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Transform Cartesian coordinates in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        u : pl.Expr
            The x coordinate in the Galactic frame.
        v : pl.Expr
            The y coordinate in the Galactic frame.
        w : pl.Expr
            The z coordinate in the Galactic frame.
        handedness : "left" or "right"
            The handedness of the coordinate system.

        Returns
        -------
        gc_x : pl.Expr
            The x coordinate in the Galactocentric frame.
        gc_y : pl.Expr
            The y coordinate in the Galactocentric frame.
        gc_z : pl.Expr
            The z coordinate in the Galactocentric frame.
        """

        sun_x: float = self._sun_x
        sun_y: float = self._sun_y
        sun_z: float = self._sun_z
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)

        if handedness == "right":
            gc_r = -u + sun_distance
            gc_y = v
            gc_z = w
        else:
            gc_r = -u + sun_distance
            gc_y = -v
            gc_z = w

        if np.isclose(sun_distance, 0):
            rotated_gc_r = gc_r
            rotated_gc_z = gc_z
        else:
            cosb: float = sun_r / sun_distance
            sinb: float = sun_z / sun_distance

            rotated_gc_r = cosb * gc_r - sinb * gc_z
            rotated_gc_z = sinb * gc_r + cosb * gc_z

        if np.isclose(sun_r, 0):
            rotated_gc_x = rotated_gc_r
            rotated_gc_y = gc_y
            rotated_gc_z = rotated_gc_z
        else:
            cosl: float = sun_x / sun_r
            sinl: float = sun_y / sun_r

            rotated_gc_x = cosl * rotated_gc_r - sinl * gc_y
            rotated_gc_y = sinl * rotated_gc_r + cosl * gc_y

        return (rotated_gc_x, rotated_gc_y, rotated_gc_z)

    def gl_vxvyvz_to_gc_vxvyvz_numpy(
        self,
        v_u: onp.ArrayND[_Float, _Shape],
        v_v: onp.ArrayND[_Float, _Shape],
        v_w: onp.ArrayND[_Float, _Shape],
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
        """Transform Cartesian coordinates in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        vx : Array[f64]
            The x-velocity in the Galactic frame.
        vy : Array[f64]
            The y-velocity in the Galactic frame.
        vz : Array[f64]
            The z-velocity in the Galactic frame.
        handedness : "left" or "right"
            The handedness of the coordinate system.

        Returns
        -------
        gc_vx : Array[f64]
            The x-velocity in the Galactocentric frame.
        gc_vy : Array[f64]
            The y-velocity in the Galactocentric frame.
        gc_vz : Array[f64]
            The z-velocity in the Galactocentric frame.
        """

        sun_x: float = self._sun_x
        sun_y: float = self._sun_y
        sun_z: float = self._sun_z
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)
        sun_vx: float = self._sun_vx
        sun_vy: float = self._sun_vy
        sun_vz: float = self._sun_vz

        if handedness == "right":
            gc_vr = -v_u
            gc_vy = v_v
            gc_vz = v_w
        else:
            gc_vr = -v_u
            gc_vy = -v_v
            gc_vz = v_w

        if np.isclose(sun_distance, 0):
            rotated_gc_vr = gc_vr
            rotated_gc_vz = gc_vz
        else:
            cosb: float = sun_r / sun_distance
            sinb: float = sun_z / sun_distance

            rotated_gc_vr = cosb * gc_vr - sinb * gc_vz
            rotated_gc_vz = sinb * gc_vr + cosb * gc_vz

        if np.isclose(sun_r, 0):
            rotated_gc_vx = rotated_gc_vr
            rotated_gc_vy = gc_vy
            rotated_gc_vz = rotated_gc_vz
        else:
            cosl: float = sun_x / sun_r
            sinl: float = sun_y / sun_r

            rotated_gc_vx = cosl * rotated_gc_vr - sinl * gc_vy
            rotated_gc_vy = sinl * rotated_gc_vr + cosl * gc_vy
        return (rotated_gc_vx + sun_vx, rotated_gc_vy + sun_vy, rotated_gc_vz + sun_vz)  # pyright: ignore[reportReturnType]

    def gl_vxvyvz_to_gc_vxvyvz_polars(
        self,
        v_u: pl.Expr,
        v_v: pl.Expr,
        v_w: pl.Expr,
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Transform Cartesian coordinates in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        vx : pl.Expr
            The x-velocity in the Galactic frame.
        vy : pl.Expr
            The y-velocity in the Galactic frame.
        vz : pl.Expr
            The z-velocity in the Galactic frame.
        handedness : "left" or "right"
            The handedness of the coordinate system.

        Returns
        -------
        gc_vx : pl.Expr
            The x-velocity in the Galactocentric frame.
        gc_vy : pl.Expr
            The y-velocity in the Galactocentric frame.
        gc_vz : pl.Expr
            The z-velocity in the Galactocentric frame.

        """

        sun_x: float = self._sun_x
        sun_y: float = self._sun_y
        sun_z: float = self._sun_z
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)
        sun_vx: float = self._sun_vx
        sun_vy: float = self._sun_vy
        sun_vz: float = self._sun_vz

        if handedness == "right":
            gc_vr = -v_u
            gc_vy = v_v
            gc_vz = v_w
        else:
            gc_vr = -v_u
            gc_vy = -v_v
            gc_vz = v_w

        if np.isclose(sun_distance, 0):
            rotated_gc_vr = gc_vr
            rotated_gc_vz = gc_vz
        else:
            cosb: float = sun_r / sun_distance
            sinb: float = sun_z / sun_distance

            rotated_gc_vr = cosb * gc_vr - sinb * gc_vz
            rotated_gc_vz = sinb * gc_vr + cosb * gc_vz

        if np.isclose(sun_r, 0):
            rotated_gc_vx = rotated_gc_vr
            rotated_gc_vy = gc_vy
            rotated_gc_vz = rotated_gc_vz
        else:
            cosl: float = sun_x / sun_r
            sinl: float = sun_y / sun_r

            rotated_gc_vx = cosl * rotated_gc_vr - sinl * gc_vy
            rotated_gc_vy = sinl * rotated_gc_vr + cosl * gc_vy
        return (rotated_gc_vx + sun_vx, rotated_gc_vy + sun_vy, rotated_gc_vz + sun_vz)
