from __future__ import annotations

import numpy as np
from flame.galactocentric import GalactocentricFrame


def run_test(frame: GalactocentricFrame) -> None:
    target_gc_x = np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)
    target_gc_y = 0
    target_gc_z = 0

    gal_x, gal_y, gal_z = (
        np.array([target_gc_x, 0.5 * target_gc_x, 1.5 * target_gc_x, -target_gc_x], dtype=np.float64),
        np.array([target_gc_y, target_gc_y, target_gc_y, target_gc_y], dtype=np.float64),
        np.array([target_gc_z, target_gc_z, target_gc_z, target_gc_z], dtype=np.float64),
    )
    gc_x, gc_y, gc_z = frame.gal_to_galcen_cartesian(gal_x, gal_y, gal_z)

    gc_x = np.round(gc_x, decimals=10)
    gc_y = np.round(gc_y, decimals=10)
    gc_z = np.round(gc_z, decimals=10)

    print("Sun:")
    print(f"{sun_x=}")
    print(f"{sun_y=}")
    print(f"{sun_z=}")

    print("Galactic:")
    print(gal_x)
    print(gal_y)
    print(gal_z)
    print(f"Gal Distance = {np.sqrt(gal_x**2 + gal_y**2 + gal_z**2)}")

    print("Galactocentric:")
    print(gc_x)
    print(gc_y)
    print(gc_z)
    print(f"GC Distance = {np.sqrt(gc_x**2 + gc_y**2 + gc_z**2)}")

    assert np.isclose(gc_x[0], 0)
    assert np.isclose(gc_y[0], 0)
    assert np.isclose(gc_z[0], 0)

    assert np.isclose(gc_x[1], -gc_x[2])
    assert np.isclose(gc_y[1], -gc_y[2])
    assert np.isclose(gc_z[1], -gc_z[2])


def collinear(frame: GalactocentricFrame) -> None:
    lat = 0.0
    lon = 63.0
    d1 = 22.0
    d2 = 277.0

    lat = np.radians(lat)
    lon = np.radians(lon)

    gal_x1 = d1 * np.cos(lat) * np.cos(lon)
    gal_y1 = d1 * np.cos(lat) * np.sin(lon)
    gal_z1 = d1 * np.sin(lat)

    gal_x2 = d2 * np.cos(lat) * np.cos(lon)
    gal_y2 = d2 * np.cos(lat) * np.sin(lon)
    gal_z2 = d2 * np.sin(lat)

    gc_x, gc_y, gc_z = frame.gal_to_galcen_cartesian(
        np.array([gal_x1, gal_x2, 0]), np.array([gal_y1, gal_y2, 0]), np.array([gal_z1, gal_z2, 0])
    )

    # They should be collinear with the Sun
    vec1 = np.array([gc_x[0] - gc_x[2], gc_y[0] - gc_y[2], gc_z[0] - gc_z[2]])
    vec2 = np.array([gc_x[1] - gc_x[2], gc_y[1] - gc_y[2], gc_z[1] - gc_z[2]])

    # Check if vectors are parallel (cross product should be zero)
    cross_product = np.cross(vec1, vec2)
    print("Galactic:")
    print(f"P1: x = {gal_x1:.05f}, y = {gal_y1:.05f}, z = {gal_z1:.05f}")
    print(f"P2: x = {gal_x2:.05f}, y = {gal_y2:.05f}, z = {gal_z2:.05f}")
    print("Galactocentric:")
    print(f"P1: x = {gc_x[0]:.05f}, y = {gc_y[0]:.05f}, z = {gc_z[0]:.05f}")
    print(f"P2: x = {gc_x[1]:.05f}, y = {gc_y[1]:.05f}, z = {gc_z[1]:.05f}")
    print(f"Sun: x = {gc_x[2]:.05f}, y = {gc_y[2]:.05f}, z = {gc_z[2]:.05f}")
    print("Displacement wrt Sun:")
    print(f"D1: x = {vec1[0]:.05f}, y = {vec1[1]:.05f}, z = {vec1[2]:.05f}")
    print(f"D2: x = {vec2[0]:.05f}, y = {vec2[2]:.05f}, z = {vec2[2]:.05f}")
    print(cross_product)
    np.testing.assert_allclose(cross_product, 0, atol=1e-10)


def sun(frame: GalactocentricFrame) -> None:
    gc_x, gc_y, gc_z = frame.gal_to_galcen_cartesian(np.array([0]), np.array([0]), np.array([0]))

    print("Galactic:")
    print(f"SUN: x = {0:.05f}, y = {0:.05f}, z = {0:.05f}")
    print("Galactocentric:")
    print(f"SUN: x = {gc_x[0]:.05f}, y = {gc_y[0]:.05f}, z = {gc_z[0]:.05f}")

    np.testing.assert_allclose(gc_x[0], frame.sun_x(), atol=1e-10)
    np.testing.assert_allclose(gc_y[0], frame.sun_y(), atol=1e-10)
    np.testing.assert_allclose(gc_z[0], frame.sun_z(), atol=1e-10)


if __name__ == "__main__":
    sun_x = 0.0
    sun_y = 1.0
    sun_z = 274.0
    frame = GalactocentricFrame(sun_x=sun_x, sun_y=sun_y, sun_z=sun_z)
    collinear(frame)
