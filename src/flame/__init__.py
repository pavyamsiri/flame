from __future__ import annotations

import numpy as np
from flame.galactocentric import GalactocentricFrame


def run_test(frame: GalactocentricFrame) -> None:
    gl_vx, gl_vy, gl_vz = (
        np.array(
            [
                0,
            ],
            dtype=np.float64,
        ),
        np.array([0], dtype=np.float64),
        np.array([0], dtype=np.float64),
    )
    for handedness in ("left", "right"):
        print(f"Calculating with handedness: {handedness}...")
        gc_vx, gc_vy, gc_vz = frame.gl_vxvyvz_to_gc_vxvyvz(gl_vx, gl_vy, gl_vz, handedness=handedness)

        gc_vx = np.round(gc_vx, decimals=10)
        gc_vy = np.round(gc_vy, decimals=10)
        gc_vz = np.round(gc_vz, decimals=10)

        print("Sun:")
        print(f"vx={frame.sun_vx()}")
        print(f"vy={frame.sun_vy()}")
        print(f"vz={frame.sun_vz()}")

        print("Galactic:")
        print(gl_vx)
        print(gl_vy)
        print(gl_vz)

        print("Galactocentric:")
        print(gc_vx)
        print(gc_vy)
        print(gc_vz)
        print()

        np.testing.assert_allclose(gc_vx[0], frame.sun_vx())
        np.testing.assert_allclose(gc_vy[0], frame.sun_vy())
        np.testing.assert_allclose(gc_vz[0], frame.sun_vz())


if __name__ == "__main__":
    # sun_x = 1.0
    # sun_y = 0.0
    # sun_z = 100.0
    # sun_vx = 123.0
    # sun_vy = 250.0
    # sun_vz = 50.0
    sun_x = 0.0
    sun_y = 0.0
    sun_z = 0.0
    sun_vx = 0.0
    sun_vy = 0.0
    sun_vz = 1.0
    frame = GalactocentricFrame(
        sun_x=sun_x,
        sun_y=sun_y,
        sun_z=sun_z,
        sun_vx=sun_vx,
        sun_vy=sun_vy,
        sun_vz=sun_vz,
    )
    run_test(frame)
