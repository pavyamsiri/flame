"""Module to represent unit-ful quantities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    import polars as pl
    from optype import numpy as onp


class AngleExpr:
    """A Polars expression carrying an angular unit annotation."""

    def __init__(self, data: pl.Expr, units: Literal["deg", "rad"]) -> None:
        self.data: pl.Expr = data
        self.units: Literal["deg", "rad"] = units

    def alias(self, name: str) -> AngleExpr:
        """Return an equivalent expression with a new output column name.

        Parameters
        ----------
        name : str
            Output column name.

        Returns
        -------
        AngleExpr
            Expression with the specified alias and the same angular unit.

        """
        return AngleExpr(self.data.alias(name), units=self.units)

    def to_radians(self) -> pl.Expr:
        """Return the underlying expression converted to radians.

        Returns
        -------
        pl.Expr
            Expression whose values are in radians.

        """
        match self.units:
            case "deg":
                return self.data.radians()
            case "rad":
                return self.data

    def to_degrees(self) -> pl.Expr:
        """Return the underlying expression converted to degrees.

        Returns
        -------
        pl.Expr
            Expression whose values are in degrees.

        """
        match self.units:
            case "deg":
                return self.data
            case "rad":
                return self.data.degrees()


class AngleArray[F: np.float64, S: tuple[Any, ...]]:
    """A NumPy array carrying an angular unit annotation."""

    def __init__(self, data: onp.ArrayND[F, S], units: Literal["deg", "rad"]) -> None:
        self.data: onp.ArrayND[F, S] = data
        self.units: Literal["deg", "rad"] = units

    def to_radians(self) -> onp.ArrayND[np.float64, S]:
        """Return the underlying array converted to radians with ``float64`` precision.

        Returns
        -------
        Array[float64]
            Array whose values are in radians.

        """
        match self.units:
            case "deg":
                return cast("onp.ArrayND[np.float64, S]", np.radians(self.data).astype(np.float64))
            case "rad":
                return self.data.astype(np.float64)

    def to_degrees(self) -> onp.ArrayND[np.float64, S]:
        """Return the underlying array converted to degrees with ``float64`` precision.

        Returns
        -------
        Array[float64]
            Array whose values are in degrees.

        """
        match self.units:
            case "deg":
                return self.data.astype(np.float64)
            case "rad":
                return cast("onp.ArrayND[np.float64, S]", np.degrees(self.data).astype(np.float64))
