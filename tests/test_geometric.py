"""Tests for the module `geometric`."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from galpy.util import coords as galcoords

from flame.geometric import cartesian_vec_to_polar_vec_numpy, cartesian_vec_to_polar_vec_polars


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

    vr_expr, vphi_expr = cartesian_vec_to_polar_vec_polars(pl.col("vx"), pl.col("vy"), pl.col("phi"))

    data = data.with_columns((vr_expr.alias("vr"), vphi_expr.alias("vphi")))
    vx_arr = np.array([vx], dtype=np.float64)
    vy_arr = np.array([vy], dtype=np.float64)
    phi_arr = np.array([phi], dtype=np.float64)
    fl_vr, fl_vphi = cartesian_vec_to_polar_vec_numpy(vx_arr, vy_arr, phi_arr)

    zero_arr = np.zeros(1, dtype=np.float64)
    gal_vr, gal_vphi, _ = galcoords.rect_to_cyl_vec(vx_arr, vy_arr, zero_arr, zero_arr, phi_arr, zero_arr, cyl=True)

    np.testing.assert_allclose(data["vr"].to_numpy(), fl_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), fl_vphi)
    np.testing.assert_allclose(data["vr"].to_numpy(), gal_vr)
    np.testing.assert_allclose(data["vphi"].to_numpy(), gal_vphi)
    np.testing.assert_allclose(fl_vr, gal_vr)
    np.testing.assert_allclose(fl_vphi, gal_vphi)
