from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from optype import numpy as onp

DEFAULT_SUN_X: float = -8.122
DEFAULT_SUN_Y: float = 0
DEFAULT_SUN_Z: float = 0.0208
DEFAULT_SUN_VX: float = 12.9
DEFAULT_SUN_VY: float = 245.6
DEFAULT_SUN_VZ: float = 7.78


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

    def gl_xyz_to_gc_xyz(
        self,
        u: onp.ArrayND[np.float64],
        v: onp.ArrayND[np.float64],
        w: onp.ArrayND[np.float64],
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]]:
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
        z_sgn: float = 1 if sun_z >= 0 else -1
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)

        if handedness == "right":
            gc_r = -u + sun_distance
            gc_y = z_sgn * v
            gc_z = z_sgn * w
        else:
            gc_r = -u + sun_distance
            gc_y = -z_sgn * v
            gc_z = z_sgn * w

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

    def gl_vxvyvz_to_gc_vxvyvz(
        self,
        v_u: onp.ArrayND[np.float64],
        v_v: onp.ArrayND[np.float64],
        v_w: onp.ArrayND[np.float64],
        *,
        handedness: Literal["right", "left"] = "right",
    ) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.float64]]:
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
        z_sgn: float = 1 if sun_z >= 0 else -1
        sun_r: float = np.hypot(sun_x, sun_y)
        sun_distance: float = np.hypot(sun_r, sun_z)
        sun_vx: float = self._sun_vx
        sun_vy: float = self._sun_vy
        sun_vz: float = self._sun_vz

        if handedness == "right":
            gc_vr = -v_u
            gc_vy = z_sgn * v_v
            gc_vz = z_sgn * v_w
        else:
            gc_vr = -v_u
            gc_vy = -z_sgn * v_v
            gc_vz = z_sgn * v_w

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
