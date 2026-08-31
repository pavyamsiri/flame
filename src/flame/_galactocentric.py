"""Private module to share common classes for use in Galactocentric transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp

__all__: list[str] = ["AffineTransforms"]


@dataclass(frozen=True, slots=True)
class AffineTransforms:
    pos_matrix: onp.Array2D[np.float64]
    pos_translation: onp.Array2D[np.float64]
    pos_matrix_inv: onp.Array2D[np.float64]
    pos_translation_inv: onp.Array2D[np.float64]
    vel_matrix: onp.Array2D[np.float64]
    vel_translation: onp.Array2D[np.float64]
    vel_matrix_inv: onp.Array2D[np.float64]
    vel_translation_inv: onp.Array2D[np.float64]

    def __post_init__(self) -> None:
        for array in (
            self.pos_matrix,
            self.pos_translation,
            self.pos_matrix_inv,
            self.pos_translation_inv,
            self.vel_matrix,
            self.vel_translation,
            self.vel_matrix_inv,
            self.vel_translation_inv,
        ):
            array.setflags(write=False)
