"""Private module for sharing constants."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

__all__: list[str] = [
    "DEC_NGP",
    "RA_NGP",
    "THETA",
]


# These constants were obtained from galpy's `get_epoch_angles`.
THETA: Final[float] = 2.1455668515225916
DEC_NGP: Final[float] = 0.4734773249532947
RA_NGP: Final[float] = 3.366032882941063
