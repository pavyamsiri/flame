"""Tests for the module `geometric`."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from galpy.util import coords as galcoords

from flame.geometric import (
    cartesian_vec_to_polar_vec_numpy,
    cartesian_vec_to_polar_vec_polars,
    cartesian_to_polar_numpy,
    cartesian_to_polar_polars,
)
from flame.units import AngleArray, AngleExpr


_ABS_TOL: float = 1e-21


@given(
    x=st.floats(min_value=-1e9, max_value=1e9),
    y=st.floats(min_value=-1e9, max_value=1e9),
)
def test_cartesian_to_polar(x: float, y: float) -> None:
    data = pl.DataFrame(
        {
            "x": [x],
            "y": [y],
        }
    )

    r_expr, phi_expr = cartesian_to_polar_polars(pl.col("x"), pl.col("y"))

    data = data.with_columns((r_expr.alias("vr"), phi_expr.to_radians().alias("vphi")))
    x_arr = np.array([x], dtype=np.float64)
    y_arr = np.array([y], dtype=np.float64)
    fl_r, fl_phi = cartesian_to_polar_numpy(x_arr, y_arr)

    zero_arr = np.zeros(1, dtype=np.float64)
    gal_r, gal_phi, _ = galcoords.rect_to_cyl(x_arr, y_arr, zero_arr)

    np.testing.assert_allclose(data["vr"].to_numpy(), fl_r, atol=_ABS_TOL)
    np.testing.assert_allclose(data["vphi"].to_numpy(), fl_phi.to_radians(), atol=_ABS_TOL)
    np.testing.assert_allclose(data["vr"].to_numpy(), gal_r, atol=_ABS_TOL)
    np.testing.assert_allclose(data["vphi"].to_numpy(), gal_phi, atol=_ABS_TOL)
    np.testing.assert_allclose(fl_r, gal_r, atol=_ABS_TOL)
    np.testing.assert_allclose(fl_phi.to_radians(), gal_phi, atol=_ABS_TOL)


@given(
    vx=st.floats(min_value=-1e9, max_value=1e9),
    vy=st.floats(min_value=-1e9, max_value=1e9),
    phi=st.floats(min_value=0, max_value=2 * np.pi),
)
def test_cartesian_vec_to_polar_vec(vx: float, vy: float, phi: float) -> None:
    data = pl.DataFrame(
        {
            "vx": [vx],
            "vy": [vy],
            "phi": [phi],
        }
    )

    vr_expr, vphi_expr = cartesian_vec_to_polar_vec_polars(pl.col("vx"), pl.col("vy"), AngleExpr(pl.col("phi"), "rad"))

    data = data.with_columns((vr_expr.alias("vr"), vphi_expr.alias("vphi")))
    vx_arr = np.array([vx], dtype=np.float64)
    vy_arr = np.array([vy], dtype=np.float64)
    phi_arr = np.array([phi], dtype=np.float64)
    fl_vr, fl_vphi = cartesian_vec_to_polar_vec_numpy(vx_arr, vy_arr, AngleArray(phi_arr, "rad"))

    zero_arr = np.zeros(1, dtype=np.float64)
    gal_vr, gal_vphi, _ = galcoords.rect_to_cyl_vec(vx_arr, vy_arr, zero_arr, zero_arr, phi_arr, zero_arr, cyl=True)

    np.testing.assert_allclose(data["vr"].to_numpy(), fl_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), fl_vphi)
    np.testing.assert_allclose(data["vr"].to_numpy(), gal_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), gal_vphi)
    np.testing.assert_allclose(fl_vr, gal_vr)
    np.testing.assert_allclose(fl_vphi, gal_vphi)
