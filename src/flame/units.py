"""Module to represent unit-ful quantities."""

from __future__ import annotations

from typing import Any, Literal, TypeVar, cast

import numpy as np
import polars as pl
from optype import numpy as onp

_Shape = TypeVar("_Shape", bound=tuple[Any, ...])
_Float = TypeVar("_Float", bound=np.float64)


class AngleExpr:
    def __init__(self, data: pl.Expr, units: Literal["deg", "rad"]) -> None:
        self.data: pl.Expr = data
        self.units: Literal["deg", "rad"] = units

    def to_radians(self) -> pl.Expr:
        match self.units:
            case "deg":
                return self.data.radians()
            case "rad":
                return self.data

    def to_degrees(self) -> pl.Expr:
        match self.units:
            case "deg":
                return self.data
            case "rad":
                return self.data.degrees()


class AngleArray[F: np.float64, S: tuple[Any, ...]]:
    def __init__(self, data: onp.ArrayND[F, S], units: Literal["deg", "rad"]) -> None:
        self.data: onp.ArrayND[F, S] = data
        self.units: Literal["deg", "rad"] = units

    def to_radians(self) -> onp.ArrayND[np.float64, S]:
        match self.units:
            case "deg":
                return cast(onp.ArrayND[np.float64, S], np.radians(self.data).astype(np.float64))
            case "rad":
                return self.data.astype(np.float64)

    def to_degrees(self) -> onp.ArrayND[np.float64, S]:
        match self.units:
            case "deg":
                return self.data.astype(np.float64)
            case "rad":
                return cast(onp.ArrayND[np.float64, S], np.degrees(self.data).astype(np.float64))
