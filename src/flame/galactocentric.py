"""Routines to transform to and from configurable Galactocentric frames.

Each frame selects Cartesian handedness and the sign of ``x*vy - y*vx``
for prograde motion. Explicit Solar Cartesian parameters are signed components
in the selected frame; omitted parameters follow its axis convention.

The default is right-handed with positive prograde Lz, with the Sun at
``(-8.122, 0, -0.0208)`` kpc and velocity ``(12.9, -245.6, -7.78)`` km/s.
Its +z axis points toward Galactic south. The Sun moves inward, prograde,
and away from the Galactic plane. Positive prograde Lz describes a coordinate
cross product; in left-handed coordinates it is not the physical axial-vector
component along +z.

The Astropy preset is right-handed with negative prograde Lz and the Sun at
negative x. The galpy preset is left-handed with positive prograde Lz and the
Sun at positive x. Both presets point +z toward Galactic north. Their Solar
parameters remain configurable through the frame constructors.

Distances are measured in kpc, velocities in km/s, and proper motions in
mas/yr. NumPy functions support normal broadcasting, while Polars functions
return lazy expressions.
"""

from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING, Final, Self, TypeGuard, override

import numpy as np

from flame._galactocentric import AffineTransforms
from flame.numpy.galactocentric import NumpyGalactocentricAdapter
from flame.polars.galactocentric import PolarsGalactocentricAdapter

__all__: list[str] = [
    "AffineTransforms",
    "GalactocentricFrame",
    "astropy_compatible_frame",
    "default_galactocentric_frame",
    "galpy_compatible_frame",
    "gravity2022_frame",
]

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from optype import numpy as onp

type _Handedness = Literal["left", "right"]

DEFAULT_SUN_X: Final[float] = -8.122
DEFAULT_SUN_Y: Final[float] = 0
DEFAULT_SUN_Z: Final[float] = -0.0208
DEFAULT_SUN_VX: Final[float] = 12.9
DEFAULT_SUN_VY: Final[float] = -245.6
DEFAULT_SUN_VZ: Final[float] = -7.78

# These constants were obtained from galpy's `galcen_extra_rot`.
GALCEN_EXTRA_ROT: Final[onp.Array2D[np.float64]] = np.array(
    [
        [1.00000000e00, 9.30225997e-07, -1.43136054e-06],
        [-9.30228099e-07, 1.00000000e00, -1.46880340e-06],
        [1.43135917e-06, 1.46880473e-06, 1.00000000e00],
    ],
    dtype=np.float64,
)


def _is_handedness(value: str) -> TypeGuard[_Handedness]:
    """Check handedness is `left` or `right`."""
    return value in ("left", "right")


def _validate_handedness(value: str) -> _Handedness:
    """Validate handedness is `left` or `right`.

    Parameters
    ----------
    value : str
        The handedness string.

    Returns
    -------
    "left" or "right"
        The validated handedness.

    """
    if not _is_handedness(value):
        msg = f"Handedness must be either `left` or `right` not {value}."
        raise TypeError(msg)
    return value


def _validate_prograde_sign(value: object) -> bool:
    """Validate the coordinate angular-momentum sign option."""
    if not isinstance(value, bool):
        raise TypeError("prograde_lz_positive must be a bool")
    return value


class GalactocentricFrame:
    """Configure Solar parameters and Cartesian orientation.

    Parameters
    ----------
    sun_x, sun_y, sun_z
        Signed Solar Cartesian position in kpc. Omitted x is negative for
        right-handed frames and positive for left-handed frames. Omitted z
        places the Sun 20.8 pc toward Galactic north in the selected axes.
    sun_vx, sun_vy, sun_vz
        Signed Solar Cartesian velocity in km/s. Omitted components use
        inward radial speed 12.9, prograde speed 245.6, and northward speed
        7.78, expressed in the selected axes at the Solar azimuth.
    handedness
        Cartesian handedness, either ``"right"`` (default) or ``"left"``.
    prograde_lz_positive
        Whether observed prograde motion has positive ``x*vy - y*vx``.
        Defaults to True. Together with handedness this determines whether
        +z points toward Galactic north or south. Explicit Solar components
        are never reflected automatically.

    """

    __slots__: Sequence[str] = (
        "_handedness",
        "_prograde_lz_positive",
        "_sun_vx",
        "_sun_vy",
        "_sun_vz",
        "_sun_x",
        "_sun_y",
        "_sun_z",
        "_transforms",
        "numpy",
        "polars",
    )

    def __init__(
        self,
        sun_x: float | None = None,
        sun_y: float = DEFAULT_SUN_Y,
        sun_z: float | None = None,
        sun_vx: float | None = None,
        sun_vy: float | None = None,
        sun_vz: float | None = None,
        *,
        handedness: Literal["left", "right"] = "right",
        prograde_lz_positive: bool = True,
    ) -> None:
        validated_handedness = _validate_handedness(handedness)
        prograde_lz_positive = _validate_prograde_sign(prograde_lz_positive)
        h = 1 if validated_handedness == "right" else -1
        rotation_sign = 1 if prograde_lz_positive else -1
        north_sign = -h * rotation_sign
        sun_x = h * DEFAULT_SUN_X if sun_x is None else sun_x
        sun_z = north_sign * abs(DEFAULT_SUN_Z) if sun_z is None else sun_z
        radius = hypot(sun_x, sun_y)
        radial_x = sun_x / radius if radius else -h
        radial_y = sun_y / radius if radius else 0.0
        sun_vx = (-abs(DEFAULT_SUN_VX) * radial_x - rotation_sign * abs(DEFAULT_SUN_VY) * radial_y) if sun_vx is None else sun_vx
        sun_vy = (-abs(DEFAULT_SUN_VX) * radial_y + rotation_sign * abs(DEFAULT_SUN_VY) * radial_x) if sun_vy is None else sun_vy
        sun_vz = north_sign * abs(DEFAULT_SUN_VZ) if sun_vz is None else sun_vz
        parameters = (sun_x, sun_y, sun_z, sun_vx, sun_vy, sun_vz)
        if not all(np.isfinite(parameter) for parameter in parameters):
            msg = "Solar position and velocity parameters must be finite"
            raise ValueError(msg)
        self._sun_x: float = sun_x
        self._sun_y: float = sun_y
        self._sun_z: float = sun_z
        self._sun_vx: float = sun_vx
        self._sun_vy: float = sun_vy
        self._sun_vz: float = sun_vz
        self._handedness: Literal["left", "right"] = validated_handedness
        self._prograde_lz_positive: bool = prograde_lz_positive

        self._transforms: AffineTransforms = GalactocentricFrame._compute_affine_transform(
            self._sun_x,
            self._sun_y,
            self._sun_z,
            self._sun_vx,
            self._sun_vy,
            self._sun_vz,
            self._handedness,
            self._prograde_lz_positive,
        )
        self.numpy: NumpyGalactocentricAdapter = NumpyGalactocentricAdapter(_transforms=self._transforms)
        self.polars: PolarsGalactocentricAdapter = PolarsGalactocentricAdapter(_transforms=self._transforms)

    @classmethod
    def from_cylindrical(
        cls,
        sun_radius: float = -DEFAULT_SUN_X,
        sun_z: float | None = None,
        sun_v_radius: float = -DEFAULT_SUN_VX,
        sun_v_phi: float = -DEFAULT_SUN_VY,
        sun_vz: float | None = None,
        *,
        handedness: Literal["left", "right"] = "right",
        prograde_lz_positive: bool = True,
    ) -> Self:
        """Create a frame from the Solar cylindrical phase-space coordinates.

        The Sun is placed at negative x for right-handed frames and positive
        x for left-handed frames. ``sun_v_radius`` is positive outward and
        ``sun_v_phi`` is positive in the observed prograde direction.
        Explicit ``sun_z`` and ``sun_vz`` are signed Cartesian components;
        omitted values follow the selected north/south orientation.

        Parameters
        ----------
        sun_radius
            Solar cylindrical radius in kpc. It must be non-negative.
        sun_z
            Solar height above the Galactic plane in kpc.
        sun_v_radius
            Solar radial velocity in km/s, positive away from the Galactic
            centre.
        sun_v_phi
            Solar azimuthal velocity in km/s, positive prograde.
        sun_vz
            Solar vertical velocity in km/s.
        handedness
            Handedness of the Cartesian Galactocentric axes.
        prograde_lz_positive
            Whether prograde motion has positive coordinate Lz.

        Returns
        -------
        GalactocentricFrame
            A frame with the equivalent Cartesian Solar parameters.

        Raises
        ------
        ValueError
            If ``sun_radius`` is negative.

        """
        if sun_radius < 0:
            msg = "Solar cylindrical radius must be non-negative"
            raise ValueError(msg)
        validated_handedness = _validate_handedness(handedness)
        x_sign = -1 if validated_handedness == "right" else 1
        rotation_sign = 1 if prograde_lz_positive else -1
        sun_vy = x_sign * rotation_sign * sun_v_phi
        return cls(
            sun_x=x_sign * sun_radius,
            sun_y=0.0,
            sun_z=sun_z,
            sun_vx=x_sign * sun_v_radius,
            sun_vy=sun_vy,
            sun_vz=sun_vz,
            handedness=validated_handedness,
            prograde_lz_positive=prograde_lz_positive,
        )

    @staticmethod
    def _compute_affine_transform(
        sun_x: float,
        sun_y: float,
        sun_z: float,
        sun_vx: float,
        sun_vy: float,
        sun_vz: float,
        handedness: Literal["left", "right"],
        prograde_lz_positive: bool,
    ) -> AffineTransforms:
        sun_r = hypot(sun_x, sun_y)
        h = 1 if handedness == "right" else -1
        rotation_sign = 1 if prograde_lz_positive else -1
        radial_x = sun_x / sun_r if sun_r else -h
        radial_y = sun_y / sun_r if sun_r else 0.0
        latitude = np.arctan2(sun_z, sun_r)
        cosb, sinb = np.cos(latitude), np.sin(latitude)

        # Columns map the centre, rotation, and north directions in the
        # corrected heliocentric Galactic basis into the configured axes.
        toward_centre = np.array([-cosb * radial_x, -cosb * radial_y, -sinb])
        along_rotation = rotation_sign * np.array([-radial_y, radial_x, 0.0])
        toward_north = h * np.cross(toward_centre, along_rotation)
        pos_matrix = np.column_stack((toward_centre, along_rotation, toward_north)) @ GALCEN_EXTRA_ROT
        pos_translation = np.array([sun_x, sun_y, sun_z])
        pos_matrix_inv: onp.Array2D[np.float64] = np.linalg.inv(pos_matrix).astype(np.float64)
        pos_translation_inv = -pos_matrix_inv @ pos_translation

        vel_matrix = pos_matrix
        vel_matrix_inv = pos_matrix_inv
        vel_translation = np.array([sun_vx, sun_vy, sun_vz])
        vel_translation_inv = -vel_matrix_inv @ vel_translation

        return AffineTransforms(
            pos_matrix=pos_matrix,
            pos_translation=pos_translation,
            pos_matrix_inv=pos_matrix_inv,
            pos_translation_inv=pos_translation_inv,
            vel_matrix=vel_matrix,
            vel_translation=vel_translation,
            vel_matrix_inv=vel_matrix_inv,
            vel_translation_inv=vel_translation_inv,
        )

    @override
    def __repr__(self) -> str:
        """Return a reconstructible representation of the frame parameters.

        Returns
        -------
        str
            Representation that can be used to recreate this frame.

        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}(sun_x={self._sun_x}, sun_y={self._sun_y}, sun_z={self._sun_z}, "
            f"sun_vx={self._sun_vx}, sun_vy={self._sun_vy}, sun_vz={self._sun_vz}, "
            f"handedness={self._handedness!r}, prograde_lz_positive={self._prograde_lz_positive!r})"
        )

    @property
    def handedness(self) -> Literal["left", "right"]:
        """Return the handedness of the Galactocentric Cartesian axes.

        Returns
        -------
        {"left", "right"}
            Axis handedness.

        """
        return self._handedness

    @property
    def prograde_lz_positive(self) -> bool:
        """Whether prograde motion has positive coordinate Lz.

        Returns
        -------
        bool
            Sign convention for ``x*vy - y*vx``.

        """
        return self._prograde_lz_positive

    @property
    def z_direction(self) -> Literal["north", "south"]:
        """Return the Galactic pole toward which positive z points.

        Returns
        -------
        {"north", "south"}
            Polar axis orientation.

        """
        north = (self._handedness == "left") == self._prograde_lz_positive
        return "north" if north else "south"

    @override
    def __str__(self) -> str:
        """Describe the frame and its coordinate convention.

        Returns
        -------
        str
            Parameters, handedness, prograde Lz sign, and polar orientation.

        """
        sign = "> 0" if self._prograde_lz_positive else "< 0"
        return f"{self!r} [prograde Lz {sign}; +z: Galactic {self.z_direction}]"

    @property
    def transforms(self) -> AffineTransforms:
        """Return the read-only affine transforms used by this frame.

        Returns
        -------
        AffineTransforms
            Cached forward and inverse position and velocity transforms.

        """
        return self._transforms

    @property
    def sun_position(self) -> tuple[float, float, float]:
        """Return the Solar Galactocentric position.

        Returns
        -------
        tuple of float
            Solar ``(x, y, z)`` position in kpc.

        """
        return (self._sun_x, self._sun_y, self._sun_z)

    @property
    def sun_velocity(self) -> tuple[float, float, float]:
        """Return the Solar Galactocentric velocity.

        Returns
        -------
        tuple of float
            Solar ``(vx, vy, vz)`` velocity in km/s.

        """
        return (self._sun_vx, self._sun_vy, self._sun_vz)

    def sun_x(self) -> float:
        """Return the Solar Galactocentric x coordinate.

        Returns
        -------
        float
            Solar x coordinate in kpc.

        """
        return self._sun_x

    def sun_y(self) -> float:
        """Return the Solar Galactocentric y coordinate.

        Returns
        -------
        float
            Solar y coordinate in kpc.

        """
        return self._sun_y

    def sun_z(self) -> float:
        """Return the Solar Galactocentric z coordinate.

        Returns
        -------
        float
            Solar z coordinate in kpc.

        """
        return self._sun_z

    def sun_vx(self) -> float:
        """Return the Solar Galactocentric x velocity.

        Returns
        -------
        float
            Solar x velocity in km/s.

        """
        return self._sun_vx

    def sun_vy(self) -> float:
        """Return the Solar Galactocentric y velocity.

        Returns
        -------
        float
            Solar y velocity in km/s.

        """
        return self._sun_vy

    def sun_vz(self) -> float:
        """Return the Solar Galactocentric z velocity.

        Returns
        -------
        float
            Solar z velocity in km/s.

        """
        return self._sun_vz

    def distance_to_gc(self) -> float:
        """Return the three-dimensional Solar distance from the Galactic centre.

        Returns
        -------
        float
            Solar distance in kpc.

        """
        return np.sqrt(self._sun_x**2 + self._sun_y**2 + self._sun_z**2)

    def sun_rxy(self) -> float:
        """Return the Solar cylindrical distance from the Galactic centre.

        Returns
        -------
        float
            Solar cylindrical distance in kpc.

        """
        return np.sqrt(self._sun_x**2 + self._sun_y**2)


def default_galactocentric_frame() -> GalactocentricFrame:
    """Create the package-default right-handed Galactocentric frame.

    Returns
    -------
    GalactocentricFrame
        Frame with the package-default Solar parameters.

    """
    return GalactocentricFrame()


def galpy_compatible_frame() -> GalactocentricFrame:
    """Create a Galactocentric frame compatible with galpy's axis orientation.

    Returns
    -------
    GalactocentricFrame
        Left-handed, north-up frame with positive prograde Lz and the
        Sun at positive x.

    """
    return GalactocentricFrame(handedness="left", prograde_lz_positive=True)


def astropy_compatible_frame() -> GalactocentricFrame:
    """Create a frame with Astropy's default Galactocentric axis orientation.

    Returns
    -------
    GalactocentricFrame
        Right-handed, north-up frame with negative prograde Lz and the
        Sun at negative x. The adopted cylindrical Solar radius is 8.122 kpc;
        Astropy uses 8.122 kpc as the three-dimensional Solar distance.

    """
    return GalactocentricFrame(handedness="right", prograde_lz_positive=False)


def gravity2022_frame() -> GalactocentricFrame:
    """Create the right-handed frame described by the 2022 GC-distance model.

    The Solar Galactocentric distance is 8.277 kpc, the Solar height is
    20.8 pc, and the cylindrical Solar velocity is
    ``(V_R, V_phi, V_z) = (-11.1, -251.5, 8.6)`` km/s. The distance is treated
    as the three-dimensional Sun--GC distance, so the cylindrical radius is
    computed as ``sqrt(8.277**2 - 0.0208**2)`` kpc.

    Returns
    -------
    GalactocentricFrame
        Right-handed, north-up frame with negative prograde Lz.

    """
    sun_distance = 8.277
    sun_z = 0.0208
    sun_radius = np.sqrt(sun_distance**2 - sun_z**2)
    return GalactocentricFrame.from_cylindrical(
        sun_radius=sun_radius,
        sun_z=sun_z,
        sun_v_radius=-11.1,
        sun_v_phi=251.5,
        sun_vz=8.6,
        handedness="right",
        prograde_lz_positive=False,
    )
