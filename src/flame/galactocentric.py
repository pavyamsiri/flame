"Routines to transform to and from the Galactocentric frame."

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from optype import numpy as onp
from flame._galactocentric import AffineTransforms
from flame.numpy.galactocentric import NumpyGalactocentricAdapter
from flame.polars.galactocentric import PolarsGalactocentricAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


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


class GalactocentricFrame:
    """Definition of the Galactocentric frame."""

    __slots__: Sequence[str] = (
        "_sun_x",
        "_sun_y",
        "_sun_z",
        "_sun_vx",
        "_sun_vy",
        "_sun_vz",
        "_handedness",
        "_transforms",
        "numpy",
        "polars",
    )

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

        self._transforms: AffineTransforms = GalactocentricFrame._compute_affine_transform(
            self._sun_x,
            self._sun_y,
            self._sun_z,
            self._sun_vx,
            self._sun_vy,
            self._sun_vz,
            self._handedness,
        )
        self.numpy: NumpyGalactocentricAdapter = NumpyGalactocentricAdapter(_transforms=self._transforms)
        self.polars: PolarsGalactocentricAdapter = PolarsGalactocentricAdapter(_transforms=self._transforms)

    @staticmethod
    def _compute_affine_transform(
        sun_x: float,
        sun_y: float,
        sun_z: float,
        sun_vx: float,
        sun_vy: float,
        sun_vz: float,
        handedness: Literal["left", "right"],
    ) -> AffineTransforms:
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

        return AffineTransforms(
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
