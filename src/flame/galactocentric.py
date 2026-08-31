"""Routines to transform to and from the Galactocentric frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from dataclasses import dataclass

import numpy as np
import polars as pl
from optype import numpy as onp

if TYPE_CHECKING:
    from typing import Any, Literal, TypeVar

    _Shape = TypeVar("_Shape", bound=Any)
    _Float = TypeVar("_Float", bound=np.float64)

DEFAULT_SUN_X: Final[float] = -8.122
DEFAULT_SUN_Y: Final[float] = 0
DEFAULT_SUN_Z: Final[float] = 0.0208
DEFAULT_SUN_VX: Final[float] = 12.9
DEFAULT_SUN_VY: Final[float] = 245.6
DEFAULT_SUN_VZ: Final[float] = 7.78

# These constants were obtained from galpy's `galcen_extra_rot`.
GALCEN_EXTRA_ROT: Final[onp.Array2D[np.float64]] = np.array(
    [
        [1.00000000e00, 9.30225997e-07, -1.43136054e-06],
        [-9.30228099e-07, 1.00000000e00, -1.46880340e-06],
        [1.43135917e-06, 1.46880473e-06, 1.00000000e00],
    ],
    dtype=np.float64,
)

# A 180 degree rotation around the z axis to preserve right-handed convention.
_RIGHT_HANDED_ORIENTATION: Final[onp.Array2D[np.float64]] = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

# A flip along the x-axis to get the left-handed convention.
_LEFT_HANDED_ORIENTATION: Final[onp.Array2D[np.float64]] = np.array(
    [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.float64,
)


@dataclass
class _AffineTransforms:
    pos_matrix: onp.Array2D[np.float64]
    pos_translation: onp.Array2D[np.float64]
    pos_matrix_inv: onp.Array2D[np.float64]
    pos_translation_inv: onp.Array2D[np.float64]
    vel_matrix: onp.Array2D[np.float64]
    vel_translation: onp.Array2D[np.float64]
    vel_matrix_inv: onp.Array2D[np.float64]
    vel_translation_inv: onp.Array2D[np.float64]


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
        *,
        handedness: Literal["left", "right"] = "right",
    ) -> None:
        assert handedness in ("left", "right"), "`handedness `must be either `left` or `right`"
        self._sun_x: float = sun_x
        self._sun_y: float = sun_y
        self._sun_z: float = sun_z
        self._sun_vx: float = sun_vx
        self._sun_vy: float = sun_vy
        self._sun_vz: float = sun_vz
        self._handedness: Literal["left", "right"] = handedness

        self.transforms: _AffineTransforms
        self._update_transforms()

    def _update_transforms(self) -> None:
        self.transforms = GalactocentricFrame._compute_affine_transform(
            self._sun_x,
            self._sun_y,
            self._sun_z,
            self._sun_vx,
            self._sun_vy,
            self._sun_vz,
            self._handedness,
        )

    @staticmethod
    def _compute_affine_transform(
        sun_x: float,
        sun_y: float,
        sun_z: float,
        sun_vx: float,
        sun_vy: float,
        sun_vz: float,
        handedness: Literal["left", "right"],
    ) -> _AffineTransforms:
        sun_r: float = np.hypot(sun_x, sun_y)
        x_sign = 1 if sun_x >= 0 else -1

        lon = np.arctan2(sun_y, sun_x)
        cosl = np.cos(lon)
        sinl = np.sin(lon)

        lat = x_sign * np.arctan2(sun_z, sun_r)
        cosb = np.cos(lat)
        sinb = np.sin(lat)

        rxy_matrix = np.array(
            [
                [cosl, -sinl, 0],
                [sinl, cosl, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        rxz_matrix = np.array(
            [
                [cosb, 0, -sinb],
                [0, 1, 0],
                [sinb, 0, cosb],
            ],
            dtype=np.float64,
        )

        orientation_matrix = _RIGHT_HANDED_ORIENTATION if handedness == "right" else _LEFT_HANDED_ORIENTATION
        pos_matrix = rxz_matrix @ orientation_matrix @ GALCEN_EXTRA_ROT @ rxy_matrix.T
        pos_translation = np.array([sun_x, sun_y, sun_z])
        pos_matrix_inv = np.linalg.inv(pos_matrix)
        pos_translation_inv = -pos_matrix_inv @ pos_translation

        vel_matrix = pos_matrix
        vel_matrix_inv = pos_matrix_inv
        vel_translation = np.array([sun_vx, sun_vy, sun_vz])
        vel_translation_inv = -vel_matrix_inv @ vel_translation

        return _AffineTransforms(
            pos_matrix=pos_matrix,
            pos_translation=pos_translation,
            pos_matrix_inv=pos_matrix_inv,
            pos_translation_inv=pos_translation_inv,
            vel_matrix=vel_matrix,
            vel_translation=vel_translation,
            vel_matrix_inv=vel_matrix_inv,
            vel_translation_inv=vel_translation_inv,
        )

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

    def sun_rxy(self) -> float:
        return np.sqrt(self._sun_x**2 + self._sun_y**2)

    def gl_xyz_to_gc_xyz_numpy(
        self,
        u: onp.ArrayND[_Float, _Shape],
        v: onp.ArrayND[_Float, _Shape],
        w: onp.ArrayND[_Float, _Shape],
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

        Returns
        -------
        gc_x : Array[f64]
            The x coordinate in the Galactocentric frame.
        gc_y : Array[f64]
            The y coordinate in the Galactocentric frame.
        gc_z : Array[f64]
            The z coordinate in the Galactocentric frame.
        """
        transform = self.transforms.pos_matrix
        translation = self.transforms.pos_translation

        gc_x = transform[0, 0] * u + transform[0, 1] * v + transform[0, 2] * w + translation[0]
        gc_y = transform[1, 0] * u + transform[1, 1] * v + transform[1, 2] * w + translation[1]
        gc_z = transform[2, 0] * u + transform[2, 1] * v + transform[2, 2] * w + translation[2]

        # print("AFFINE NO TRANSLATION")
        # print(f"X' = {gc_x - translation[0]}")
        # print(f"Y' = {gc_y - translation[1]}")
        # print(f"Z' = {gc_z - translation[2]}")

        # print("AFFINE TRANSLATION")
        # print(f"Tx' = {translation[0]}")
        # print(f"Ty' = {translation[1]}")
        # print(f"Tz' = {translation[2]}")

        return (gc_x, gc_y, gc_z)

    def gc_xyz_to_gl_xyz_numpy(
        self,
        gc_x: onp.ArrayND[_Float, _Shape],
        gc_y: onp.ArrayND[_Float, _Shape],
        gc_z: onp.ArrayND[_Float, _Shape],
    ) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
        """Transform Cartesian coordinates in the Galactocentic frame to the Galactic frame.

        Parameters
        ----------
        x : Array[f64]
            The x coordinate in the Galactocentric frame.
        y : Array[f64]
            The y coordinate in the Galactocentric frame.
        z : Array[f64]
            The z coordinate in the Galactocentric frame.

        Returns
        -------
        gl_x : Array[f64]
            The x coordinate in the Galactic frame.
        gl_y : Array[f64]
            The y coordinate in the Galactic frame.
        gl_z : Array[f64]
            The z coordinate in the Galactic frame.
        """

        transform = self.transforms.pos_matrix_inv
        translation = self.transforms.pos_translation_inv

        gl_x = transform[0, 0] * gc_x + transform[0, 1] * gc_y + transform[0, 2] * gc_z + translation[0]
        gl_y = transform[1, 0] * gc_x + transform[1, 1] * gc_y + transform[1, 2] * gc_z + translation[1]
        gl_z = transform[2, 0] * gc_x + transform[2, 1] * gc_y + transform[2, 2] * gc_z + translation[2]
        return (gl_x, gl_y, gl_z)

    def gl_xyz_to_gc_xyz_polars(
        self,
        u: pl.Expr,
        v: pl.Expr,
        w: pl.Expr,
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

        Returns
        -------
        gc_x : pl.Expr
            The x coordinate in the Galactocentric frame.
        gc_y : pl.Expr
            The y coordinate in the Galactocentric frame.
        gc_z : pl.Expr
            The z coordinate in the Galactocentric frame.
        """
        transform = self.transforms.pos_matrix
        translation = self.transforms.pos_translation

        gc_x = transform[0, 0] * u + transform[0, 1] * v + transform[0, 2] * w + translation[0]
        gc_y = transform[1, 0] * u + transform[1, 1] * v + transform[1, 2] * w + translation[1]
        gc_z = transform[2, 0] * u + transform[2, 1] * v + transform[2, 2] * w + translation[2]

        return (gc_x, gc_y, gc_z)

    def gc_xyz_to_gl_xyz_polars(
        self,
        x: pl.Expr,
        y: pl.Expr,
        z: pl.Expr,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Transform Cartesian coordinates in the Galactocentric frame to the Galactic frame.

        Parameters
        ----------
        x : pl.Expr
            The x coordinate in the Galactocentric frame.
        y : pl.Expr
            The y coordinate in the Galactocentric frame.
        z : pl.Expr
            The z coordinate in the Galactocentric frame.
        handedness : "left" or "right"
            The handedness of the coordinate system.

        Returns
        -------
        gl_x : pl.Expr
            The x coordinate in the Galactic frame.
        gl_y : pl.Expr
            The y coordinate in the Galactic frame.
        gl_z : pl.Expr
            The z coordinate in the Galactic frame.
        """
        transform = self.transforms.pos_matrix_inv
        translation = self.transforms.pos_translation_inv

        gl_x = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z + translation[0]
        gl_y = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z + translation[1]
        gl_z = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z + translation[2]

        return (gl_x, gl_y, gl_z)

    def gl_vxvyvz_to_gc_vxvyvz_numpy(
        self,
        v_u: onp.ArrayND[_Float, _Shape],
        v_v: onp.ArrayND[_Float, _Shape],
        v_w: onp.ArrayND[_Float, _Shape],
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
        Returns
        -------
        gc_vx : Array[f64]
            The x-velocity in the Galactocentric frame.
        gc_vy : Array[f64]
            The y-velocity in the Galactocentric frame.
        gc_vz : Array[f64]
            The z-velocity in the Galactocentric frame.
        """
        transform = self.transforms.vel_matrix
        translation = self.transforms.vel_translation

        gc_x = transform[0, 0] * v_u + transform[0, 1] * v_v + transform[0, 2] * v_w + translation[0]
        gc_y = transform[1, 0] * v_u + transform[1, 1] * v_v + transform[1, 2] * v_w + translation[1]
        gc_z = transform[2, 0] * v_u + transform[2, 1] * v_v + transform[2, 2] * v_w + translation[2]

        return (gc_x, gc_y, gc_z)

    def gc_vxvyvz_to_gl_vxvyvz_numpy(
        self,
        vx: onp.ArrayND[_Float, _Shape],
        vy: onp.ArrayND[_Float, _Shape],
        vz: onp.ArrayND[_Float, _Shape],
    ) -> tuple[onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape], onp.ArrayND[np.float64, _Shape]]:
        """Transform Cartesian velocities in the Galactocentric frame to the Galactic frame.

        Parameters
        ----------
        vx : Array[f64]
            The x-velocity in the Galactocentric frame.
        vy : Array[f64]
            The y-velocity in the Galactocentric frame.
        vz : Array[f64]
            The z-velocity in the Galactocentric frame.
        Returns
        -------
        gl_vx : Array[f64]
            The x-velocity in the Galactic frame.
        gl_vy : Array[f64]
            The y-velocity in the Galactic frame.
        gl_vz : Array[f64]
            The z-velocity in the Galactic frame.
        """
        transform = self.transforms.vel_matrix_inv
        translation = self.transforms.vel_translation_inv

        gl_x = transform[0, 0] * vx + transform[0, 1] * vy + transform[0, 2] * vz + translation[0]
        gl_y = transform[1, 0] * vx + transform[1, 1] * vy + transform[1, 2] * vz + translation[1]
        gl_z = transform[2, 0] * vx + transform[2, 1] * vy + transform[2, 2] * vz + translation[2]

        return (gl_x, gl_y, gl_z)

    def gl_vxvyvz_to_gc_vxvyvz_polars(
        self,
        v_u: pl.Expr,
        v_v: pl.Expr,
        v_w: pl.Expr,
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
        Returns
        -------
        gc_vx : pl.Expr
            The x-velocity in the Galactocentric frame.
        gc_vy : pl.Expr
            The y-velocity in the Galactocentric frame.
        gc_vz : pl.Expr
            The z-velocity in the Galactocentric frame.

        """
        transform = self.transforms.vel_matrix
        translation = self.transforms.vel_translation

        gc_x = transform[0, 0] * v_u + transform[0, 1] * v_v + transform[0, 2] * v_w + translation[0]
        gc_y = transform[1, 0] * v_u + transform[1, 1] * v_v + transform[1, 2] * v_w + translation[1]
        gc_z = transform[2, 0] * v_u + transform[2, 1] * v_v + transform[2, 2] * v_w + translation[2]

        return (gc_x, gc_y, gc_z)

    def gc_vxvyvz_to_gl_vxvyvz_polars(
        self,
        vx: pl.Expr,
        vy: pl.Expr,
        vz: pl.Expr,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Transform Cartesian velocities in the Galactocentric frame to the Galactic frame.

        Parameters
        ----------
        vx : pl.Expr
            The x-velocity in the Galactocentric frame.
        vy : pl.Expr
            The y-velocity in the Galactocentric frame.
        vz : pl.Expr
            The z-velocity in the Galactocentric frame.
        Returns
        -------
        gl_vx : pl.Expr
            The x-velocity in the Galactic frame.
        gl_vy : pl.Expr
            The y-velocity in the Galactic frame.
        gl_vz : pl.Expr
            The z-velocity in the Galactic frame.
        """
        transform = self.transforms.vel_matrix_inv
        translation = self.transforms.vel_translation_inv

        gl_x = transform[0, 0] * vx + transform[0, 1] * vy + transform[0, 2] * vz + translation[0]
        gl_y = transform[1, 0] * vx + transform[1, 1] * vy + transform[1, 2] * vz + translation[1]
        gl_z = transform[2, 0] * vx + transform[2, 1] * vy + transform[2, 2] * vz + translation[2]

        return (gl_x, gl_y, gl_z)
