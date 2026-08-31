"""Private module for sharing constants."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from typing import Final
    from optype import numpy as onp

__all__: list[str] = [
    "DEC_NGP",
    "RA_NGP",
    "THETA",
]


# These constants were obtained from galpy's `get_epoch_angles`.
THETA: Final[float] = 2.1455668515225916
DEC_NGP: Final[float] = 0.4734773249532947
RA_NGP: Final[float] = 3.366032882941063


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
TRANSFORMATION_MATRIX: Final[onp.Array2D[np.float64]] = _THETA_MATRIX @ _DEC_MATRIX @ _RA_MATRIX
