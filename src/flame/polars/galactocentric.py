"""Adapter to handle Galactocentric coordinate transformations using polars."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import polars as pl

if TYPE_CHECKING:
    from flame._galactocentric import AffineTransforms

__all__: list[str] = ["PolarsGalactocentricAdapter"]


@dataclass(frozen=True, slots=True)
class PolarsGalactocentricAdapter:
    _transforms: AffineTransforms

    def position_from_galactic(self, u: pl.Expr, v: pl.Expr, w: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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
        transform = self._transforms.pos_matrix
        translation = self._transforms.pos_translation

        gc_x = transform[0, 0] * u + transform[0, 1] * v + transform[0, 2] * w + translation[0]
        gc_y = transform[1, 0] * u + transform[1, 1] * v + transform[1, 2] * w + translation[1]
        gc_z = transform[2, 0] * u + transform[2, 1] * v + transform[2, 2] * w + translation[2]

        return (gc_x, gc_y, gc_z)

    def position_to_galactic(self, x: pl.Expr, y: pl.Expr, z: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Transform Cartesian coordinates in the Galactocentric frame to the Galactic frame.

        Parameters
        ----------
        x : pl.Expr
            The x coordinate in the Galactocentric frame.
        y : pl.Expr
            The y coordinate in the Galactocentric frame.
        z : pl.Expr
            The z coordinate in the Galactocentric frame.

        Returns
        -------
        gl_x : pl.Expr
            The x coordinate in the Galactic frame.
        gl_y : pl.Expr
            The y coordinate in the Galactic frame.
        gl_z : pl.Expr
            The z coordinate in the Galactic frame.
        """
        transform = self._transforms.pos_matrix_inv
        translation = self._transforms.pos_translation_inv

        gl_x = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2] * z + translation[0]
        gl_y = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2] * z + translation[1]
        gl_z = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2] * z + translation[2]

        return (gl_x, gl_y, gl_z)

    def velocity_from_galactic(self, v_u: pl.Expr, v_v: pl.Expr, v_w: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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
        transform = self._transforms.vel_matrix
        translation = self._transforms.vel_translation

        gc_x = transform[0, 0] * v_u + transform[0, 1] * v_v + transform[0, 2] * v_w + translation[0]
        gc_y = transform[1, 0] * v_u + transform[1, 1] * v_v + transform[1, 2] * v_w + translation[1]
        gc_z = transform[2, 0] * v_u + transform[2, 1] * v_v + transform[2, 2] * v_w + translation[2]

        return (gc_x, gc_y, gc_z)

    def velocity_to_galactic(self, vx: pl.Expr, vy: pl.Expr, vz: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
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
        transform = self._transforms.vel_matrix_inv
        translation = self._transforms.vel_translation_inv

        gl_x = transform[0, 0] * vx + transform[0, 1] * vy + transform[0, 2] * vz + translation[0]
        gl_y = transform[1, 0] * vx + transform[1, 1] * vy + transform[1, 2] * vz + translation[1]
        gl_z = transform[2, 0] * vx + transform[2, 1] * vy + transform[2, 2] * vz + translation[2]

        return (gl_x, gl_y, gl_z)
