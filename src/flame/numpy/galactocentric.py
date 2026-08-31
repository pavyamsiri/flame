"""Adapter to handle Galactocentric coordinate transformations using numpy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp

    from flame._galactocentric import AffineTransforms

__all__: list[str] = ["NumpyGalactocentricAdapter"]


@dataclass(frozen=True, slots=True)
class NumpyGalactocentricAdapter:
    _transforms: AffineTransforms

    def position_from_galactic[F: np.float64, S: tuple[Any, ...]](
        self, u: onp.ArrayND[F, S], v: onp.ArrayND[F, S], w: onp.ArrayND[F, S]
    ) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
        """Transform Cartesian coordinates in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        u : Array[float]
            The x coordinate in the Galactic frame.
        v : Array[float]
            The y coordinate in the Galactic frame.
        w : Array[float]
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
        transform = self._transforms.pos_matrix
        translation = self._transforms.pos_translation

        gc_x = transform[0, 0] * u + transform[0, 1] * v + transform[0, 2] * w + translation[0]
        gc_y = transform[1, 0] * u + transform[1, 1] * v + transform[1, 2] * w + translation[1]
        gc_z = transform[2, 0] * u + transform[2, 1] * v + transform[2, 2] * w + translation[2]

        return (gc_x, gc_y, gc_z)

    def position_to_galactic[F: np.float64, S: tuple[Any, ...]](
        self, gc_x: onp.ArrayND[F, S], gc_y: onp.ArrayND[F, S], gc_z: onp.ArrayND[F, S]
    ) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
        """Transform Cartesian coordinates in the Galactocentic frame to the Galactic frame.

        Parameters
        ----------
        x : Array[float]
            The x coordinate in the Galactocentric frame.
        y : Array[float]
            The y coordinate in the Galactocentric frame.
        z : Array[float]
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
        transform = self._transforms.pos_matrix_inv
        translation = self._transforms.pos_translation_inv

        gl_x = transform[0, 0] * gc_x + transform[0, 1] * gc_y + transform[0, 2] * gc_z + translation[0]
        gl_y = transform[1, 0] * gc_x + transform[1, 1] * gc_y + transform[1, 2] * gc_z + translation[1]
        gl_z = transform[2, 0] * gc_x + transform[2, 1] * gc_y + transform[2, 2] * gc_z + translation[2]
        return (gl_x, gl_y, gl_z)

    def velocity_from_galactic[F: np.float64, S: tuple[Any, ...]](
        self,
        v_u: onp.ArrayND[F, S],
        v_v: onp.ArrayND[F, S],
        v_w: onp.ArrayND[F, S],
    ) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
        """Transform Cartesian velocities in the Galactic frame to the Galactocentric frame.

        Parameters
        ----------
        vx : Array[float]
            The x-velocity in the Galactic frame.
        vy : Array[float]
            The y-velocity in the Galactic frame.
        vz : Array[float]
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
        transform = self._transforms.vel_matrix
        translation = self._transforms.vel_translation

        gc_x = transform[0, 0] * v_u + transform[0, 1] * v_v + transform[0, 2] * v_w + translation[0]
        gc_y = transform[1, 0] * v_u + transform[1, 1] * v_v + transform[1, 2] * v_w + translation[1]
        gc_z = transform[2, 0] * v_u + transform[2, 1] * v_v + transform[2, 2] * v_w + translation[2]

        return (gc_x, gc_y, gc_z)

    def velocity_to_galactic[F: np.float64, S: tuple[Any, ...]](
        self, vx: onp.ArrayND[F, S], vy: onp.ArrayND[F, S], vz: onp.ArrayND[F, S]
    ) -> tuple[onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S], onp.ArrayND[np.float64, S]]:
        """Transform Cartesian velocities in the Galactocentric frame to the Galactic frame.

        Parameters
        ----------
        vx : Array[float]
            The x-velocity in the Galactocentric frame.
        vy : Array[float]
            The y-velocity in the Galactocentric frame.
        vz : Array[float]
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
        transform = self._transforms.vel_matrix_inv
        translation = self._transforms.vel_translation_inv

        gl_x = transform[0, 0] * vx + transform[0, 1] * vy + transform[0, 2] * vz + translation[0]
        gl_y = transform[1, 0] * vx + transform[1, 1] * vy + transform[1, 2] * vz + translation[1]
        gl_z = transform[2, 0] * vx + transform[2, 1] * vy + transform[2, 2] * vz + translation[2]

        return (gl_x, gl_y, gl_z)
