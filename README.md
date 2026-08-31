# flame

`flame` transforms ICRS, Galactic, and Galactocentric coordinates with
explicit NumPy and Polars APIs.

## NumPy: ICRS phase space to a right-handed Galactocentric frame

```python
import numpy as np

from flame import GalactocentricFrame
from flame.units import AngleArray

frame = GalactocentricFrame(handedness="right")
x, y, z, vx, vy, vz = frame.numpy.phase_space_from_icrs(
    np.array([20.0]),  # vr, km/s
    np.array([3.0]),   # pm_ra_cosdec, mas/yr
    np.array([-2.0]),  # pm_dec, mas/yr
    np.array([1.5]),   # distance, kpc
    ra=AngleArray(np.array([1.0]), "rad"),
    dec=AngleArray(np.array([0.2]), "rad"),
)
```

At zero heliocentric distance, the position becomes the frame's configured
Solar position. Use `galpy_compatible_frame()` when a left-handed galpy axis
orientation is required.

## Polars: dataframe pipeline

```python
import polars as pl

from flame import galpy_compatible_frame
from flame.units import AngleExpr

result = data.with_columns(
    *(
        expr.alias(name)
        for expr, name in zip(
            galpy_compatible_frame().polars.phase_space_from_icrs(
                pl.col("vr"), pl.col("pmracosdec"), pl.col("pmdec"), pl.col("distance"),
                ra=AngleExpr(pl.col("ra"), "rad"),
                dec=AngleExpr(pl.col("dec"), "rad"),
            ),
            ("x", "y", "z", "vx", "vy", "vz"),
            strict=True,
        )
    )
)
```

See the `flame.galactocentric` module docstring for axis directions and units.
