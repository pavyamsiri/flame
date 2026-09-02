# flame

`flame` transforms ICRS, Galactic, and Galactocentric coordinates with
explicit NumPy and Polars APIs.

The default frame is right-handed with positive prograde `Lz = x*vy - y*vx`.
It places the Sun at `(-8.122, 0, -0.0208)` kpc with Solar velocity
`(+12.9, -245.6, -7.78)` km/s. Its +z axis points toward Galactic south:
the Sun moves inward, counter-clockwise viewed from +z, and away from the
plane. Cartesian arguments are signed components; omitted values follow the
selected convention.

```python
from flame import GalactocentricFrame, astropy_compatible_frame, galpy_compatible_frame

frame = GalactocentricFrame(handedness="right", prograde_lz_positive=True)
print(frame)  # Includes handedness, prograde Lz sign, and the +z direction.
```

| Convention | Handedness | Prograde Lz | Solar x | +z |
|---|---|---|---|---|
| Package default | Right | Positive | Negative | South |
| `astropy_compatible_frame()` | Right | Negative | Negative | North |
| `galpy_compatible_frame()` | Left | Positive | Positive | North |

Both options can be combined freely. With omitted Solar x, handedness selects
its side of the centre; explicit Solar positions may use any azimuth.
The presets match axis conventions and use the package Solar parameters.
Astropy defines its 8.122 kpc distance in three dimensions, whereas this
package uses 8.122 kpc as the cylindrical Solar radius.
In left-handed coordinates, `x*vy - y*vx` is a coordinate expression rather
than the physical axial-vector component along +z.

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

For cylindrical Solar parameters, `sun_v_radius` is positive outward and
`sun_v_phi` is positive prograde, independently of the selected convention.
The Sun is placed at negative x for right-handed frames and positive x for
left-handed frames. Explicit `sun_z` and `sun_vz` remain signed components.

```python
frame = GalactocentricFrame.from_cylindrical(
    sun_radius=8.122,     # kpc
    sun_z=-0.0208,       # kpc, north is negative z in this convention
    sun_v_radius=-12.9,  # km/s, inward
    sun_v_phi=245.6,     # km/s, prograde
    sun_vz=-7.78,        # km/s, northward
    handedness="right",
    prograde_lz_positive=True,
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
