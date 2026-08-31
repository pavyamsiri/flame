"""Private module for sharing constants."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

__all__: list[str] = [
    "MAS_PER_YR_TO_KM_PER_S_PER_KPC",
]


MAS_PER_YR_TO_KM_PER_S_PER_KPC: Final[float] = 4.74047
